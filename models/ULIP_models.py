'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''

# Modified from github.com/openai/CLIP
from collections import OrderedDict

import timm
from torch import nn
from models.pointnet2.pointnet2 import Pointnet2_Ssg
from data.dataset_3d import  *

from models import losses
from torch.nn.parameter import Parameter
from easydict import EasyDict

import open_clip
from pprint import pprint
from torchsummary import summary

from thop import profile

# from src.models.factory import create_model
# from src.zeroshot_utils.zeroshot_path import zero_shot_classifier

import pandas as pd


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class GeneLIP_WITH_IMAGE(nn.Module):
    def __init__(self, gene_encoder, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.gene_encoder = gene_encoder

        self.gene_projection = nn.Parameter(torch.empty(kwargs.omic_feat_dims, 512))
        nn.init.normal_(self.gene_projection, std=512 ** -0.5)

        self.tune_visual = kwargs.args.tune_visual
        if self.tune_visual.lower() == 'adapter' :
            print("Use visual adapter!")
            self.adapter = Adapter(kwargs.vision_width,4)


    def encode_image(self, image):
        image_feat = self.visual(image)

        if self.tune_visual.lower() == 'adapter':
            x = self.adapter(image_feat)
            ratio = 0.2
            image_feat = ratio * x + (1 - ratio) * image_feat
        
        x = image_feat @ self.image_projection

        return x

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]   torch.Size([5, 77, 512])
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) # 1,77,512
        # bz, context_length, embed_dim

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection  # 1,512

        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # def encode_pc(self, pc):
    #     pc_feat = self.point_encoder(pc)
    #     pc_embed = pc_feat @ self.pc_projection
    #     return pc_embed

    def encode_omic(self, x_omic):
        omic_feat = self.gene_encoder(x_omic=x_omic)[0] # featues, out, pred, omic_grads
        omic_embed = omic_feat @ self.gene_projection # dimension: 128 -> 512
        return omic_embed

    def forward(self, x_omic, text, image=None):
        # For omic, image ==> torch.Size([1, 3, 512, 512])    
        # For pc, image ==> torch.Size([1, 3, 224, 224])

        
        text_embed_all = []
        for i in range(text.shape[0]):  #(1,1,77)
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)

        '''
        07/15 adjusted from pc to omic
        '''
        # pc_embed = self.encode_pc(pc)
        omic_embed = self.encode_omic(x_omic)
        

        if image is not None:
            image_embed = self.encode_image(image)
            return {'text_embed': text_embed_all,
                    'omic_embed': omic_embed,
                    'image_embed': image_embed,
                    'logit_scale': self.logit_scale.exp()}

        else:
            return {'text_embed': text_embed_all,
                    'omic_embed': omic_embed,
                    'logit_scale': self.logit_scale.exp()}

class GeneLIP_WITH_BIOMEDCLIP(nn.Module):
    def __init__(self, gene_encoder, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        # self.context_length = kwargs.context_length
        # self.vision_width = kwargs.vision_width

        # self.visual = kwargs.vision_model
        # self.text = kwargs.text_model

        self.visual = kwargs.vl_model.visual # trucnk + head
        self.text = kwargs.vl_model.text



        # self.transformer = Transformer(
        #     width=kwargs.transformer_width,
        #     layers=kwargs.transformer_layers,
        #     heads=kwargs.transformer_heads,
        #     attn_mask=self.build_attention_mask(),
        # ) 
        self.kwargs = kwargs

        # self.vocab_size = kwargs.vocab_size
        # self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        # self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        # self.ln_final = LayerNorm(kwargs.transformer_width)

        # self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim)) # 768 -> 512
        # self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim)) # 512 -> 512

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.image_encoder = self.visual.trunk
        # self.text_encoder = self.text.transformer

        self.image_projection = self.visual.head.proj
        # self.text_projection = self.text.proj
        self.logit_scale = kwargs.vl_model.logit_scale

        # self.initialize_parameters()

        self.gene_encoder = gene_encoder

        self.gene_projection = nn.Parameter(torch.empty(kwargs.omic_feat_dims, 512))
        nn.init.normal_(self.gene_projection, std=512 ** -0.5)

        self.tune_visual = kwargs.args.tune_visual
        if self.tune_visual.lower() == 'adapter' :
            print("Use visual adapter!")
            self.adapter = Adapter(kwargs.vision_width,4)


    def encode_image(self, image):
        image_feat = self.image_encoder(image)

        if self.tune_visual.lower() == 'adapter':
            x = self.adapter(image_feat)
            ratio = 0.2
            image_feat = ratio * x + (1 - ratio) * image_feat
        
        # x = image_feat @ self.image_projection
        x = self.image_projection( image_feat)

        return x

    def encode_text(self, text):
        if len(text.shape) ==3:
            if text.shape[0] ==1 and text.shape[1]==1:
                text = text[0]
            else:
                raise ValueError("text must have 2 dimension, or 3 dimension like [1,1,77]")
        # x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        # x = x + self.positional_embedding
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x)

        # x = self.text_encoder(text)

        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        x = self.text(text)

        # self.text(text[0]).shape   1,512


        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # def encode_pc(self, pc):
    #     pc_feat = self.point_encoder(pc)
    #     pc_embed = pc_feat @ self.pc_projection
    #     return pc_embed

    def encode_omic(self, x_omic):
        omic_feat = self.gene_encoder(x_omic=x_omic)[0]
        omic_embed = omic_feat @ self.gene_projection
        return omic_embed

    def forward(self, x_omic, text, image=None):
        # For omic, image ==> torch.Size([1, 3, 512, 512])    
        # For pc, image ==> torch.Size([1, 3, 224, 224])
    
        text_embed_all = []
        for i in range(text.shape[0]):  #(1,1,77)
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)

        '''
        07/15 adjusted from pc to omic
        '''
        # pc_embed = self.encode_pc(pc)
        omic_embed = self.encode_omic(x_omic)
        

        if image is not None:
            image_embed = self.encode_image(image)
            return {'text_embed': text_embed_all,
                    'omic_embed': omic_embed,
                    'image_embed': image_embed,
                    'logit_scale': self.logit_scale.exp()}

        else:
            return {'text_embed': text_embed_all,
                    'omic_embed': omic_embed,
                    'logit_scale': self.logit_scale.exp()}


class GeneLIP_WITH_QUILTCLIP(nn.Module):
    def __init__(self, gene_encoder, vl_model, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        # self.visual = kwargs.vision_model

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        # self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.gene_encoder = gene_encoder

        self.gene_projection = nn.Parameter(torch.empty(kwargs.omic_feat_dims, 512))
        nn.init.normal_(self.gene_projection, std=512 ** -0.5)

        self.image_projection = vl_model.visual.proj

        self.logit_scale = vl_model.logit_scale

        self.visual = vl_model.visual
        self.visual.proj = None
        # vision_model = timm.create_model('vit_base_patch32_224', num_classes=0)


        self.tune_visual = kwargs.args.tune_visual
        if self.tune_visual.lower() == 'adapter' :
            print("Use visual adapter!")
            self.adapter = Adapter(kwargs.vision_width,4)

    def encode_image(self, image):
        image_feat = self.visual(image)  #[253,512]

        if self.tune_visual.lower() == 'adapter':
            x = self.adapter(image_feat)
            ratio = 0.2
            image_feat = ratio * x + (1 - ratio) * image_feat
        
        x = image_feat @ self.image_projection

        return x

    def encode_text(self, text):
        # Eval [5,77]   Train [1,5,77]
        # if len(text.shape) >2:
        #   text = text.squeeze(0)
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]   torch.Size([5, 77, 512])
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) # 1,77,512
        # bz, context_length, embed_dim

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection  # 1,512

        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # def encode_pc(self, pc):
    #     pc_feat = self.point_encoder(pc)
    #     pc_embed = pc_feat @ self.pc_projection
    #     return pc_embed

    def encode_omic(self, x_omic):
        omic_feat = self.gene_encoder(x_omic=x_omic)[0] # featues, out, pred, omic_grads
        omic_embed = omic_feat @ self.gene_projection # dimension: 128 -> 512
        return omic_embed

    def forward(self, image, gene, cls_label):
        # For omic, image ==> torch.Size([1, 3, 512, 512])    
        # For pc, image ==> torch.Size([1, 3, 224, 224])

        
        # text_embed_all = []
        # for i in range(text.shape[0]):  #(1,1,77)
        #     text_for_one_sample = text[i]
        #     text_embed = self.encode_text(text_for_one_sample)
        #     text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        #     text_embed = text_embed.mean(dim=0)
        #     text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        #     text_embed_all.append(text_embed)

        # text_embed_all = torch.stack(text_embed_all)

        '''
        07/15 adjusted from pc to omic
        '''
        # pc_embed = self.encode_pc(pc)
        # omic_embed = self.encode_omic(x_omic)
        
        image_embed = self.encode_image(image)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        # pc_embed = self.encode_pc(pc)
        # omic_embed = self.encode_omic(x_omic)

        return { 
              'image_embed': image_embed,
              'gene_embed': None,
              'logit_scale': self.logit_scale.exp()
                }


# class GeneLIP_WITH_QUILTCLIP(nn.Module):
#     def __init__(self, gene_encoder, **kwargs):
#         # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
#         super().__init__()

#         kwargs = EasyDict(kwargs)
#         self.kwargs = kwargs
#         # self.context_length = kwargs.context_length
#         # self.vision_width = kwargs.vision_width

#         # self.visual = kwargs.vision_model
#         # self.text = kwargs.text_model

#         self.visual = kwargs.vl_model.visual # trucnk + head
#         self.image_projection = self.visual.proj


#         self.transformer = kwargs.vl_model.transformer
#         self.token_embedding = kwargs.vl_model.token_embedding
#         self.positional_embedding = kwargs.vl_model.positional_embedding
#         self.ln_final = kwargs.vl_model.ln_final

#         self.text_projection = kwargs.vl_model.text_projection


#         # self.transformer = Transformer(
#         #     width=kwargs.transformer_width,
#         #     layers=kwargs.transformer_layers,
#         #     heads=kwargs.transformer_heads,
#         #     attn_mask=self.build_attention_mask(),
#         # ) 
  

#         # self.vocab_size = kwargs.vocab_size
#         # self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
#         # self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
#         # self.ln_final = LayerNorm(kwargs.transformer_width)

#         # self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim)) # 768 -> 512
#         # self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim)) # 512 -> 512

#         # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#         # self.image_encoder = self.visual
#         # self.text_encoder = self.text.transformer

        
#         # self.text_projection = self.text.proj
#         self.logit_scale = kwargs.vl_model.logit_scale

#         # self.initialize_parameters()

#         self.gene_encoder = gene_encoder

#         self.gene_projection = nn.Parameter(torch.empty(kwargs.omic_feat_dims, 512))
#         nn.init.normal_(self.gene_projection, std=512 ** -0.5)

#         self.tune_visual = kwargs.args.tune_visual
#         if self.tune_visual.lower() == 'adapter' :
#             print("Use visual adapter!")
#             self.adapter = Adapter(kwargs.vision_width,4)


#     def encode_image(self, image):
#         image_feat = self.visual(image)

#         if self.tune_visual.lower() == 'adapter':
#             x = self.adapter(image_feat)
#             ratio = 0.2
#             image_feat = ratio * x + (1 - ratio) * image_feat
        
#         x = image_feat @ self.image_projection
#         # x = self.image_projection( image_feat)

#         return x

#     def encode_text(self, text):
#         x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]   torch.Size([5, 77, 512])
#         x = x + self.positional_embedding
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x) # 1,77,512

#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection


#         return x

#     def build_attention_mask(self):
#         # lazily create causal attention mask, with full attention between the vision tokens
#         # pytorch uses additive attention mask; fill with -inf
#         mask = torch.empty(self.context_length, self.context_length)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # zero out the lower diagonal
#         return mask

#     def initialize_parameters(self):
#         nn.init.normal_(self.token_embedding.weight, std=0.02)
#         nn.init.normal_(self.positional_embedding, std=0.01)

#         proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
#         attn_std = self.transformer.width ** -0.5
#         fc_std = (2 * self.transformer.width) ** -0.5
#         for block in self.transformer.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

#         nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
#         nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

#     # def encode_pc(self, pc):
#     #     pc_feat = self.point_encoder(pc)
#     #     pc_embed = pc_feat @ self.pc_projection
#     #     return pc_embed

#     def encode_omic(self, x_omic):
#         omic_feat = self.gene_encoder(x_omic=x_omic)[0]
#         omic_embed = omic_feat @ self.gene_projection
#         return omic_embed

#     def forward(self, x_omic, text, image=None):
#         # For omic, image ==> torch.Size([1, 3, 512, 512])    
#         # For pc, image ==> torch.Size([1, 3, 224, 224])
    
#         text_embed_all = []
#         for i in range(text.shape[0]):  #(1,1,77)
#             text_for_one_sample = text[i]
#             text_embed = self.encode_text(text_for_one_sample)
#             text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
#             text_embed = text_embed.mean(dim=0)
#             text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
#             text_embed_all.append(text_embed)

#         text_embed_all = torch.stack(text_embed_all)

#         '''
#         07/15 adjusted from pc to omic
#         '''
#         # pc_embed = self.encode_pc(pc)
#         omic_embed = self.encode_omic(x_omic)
        

#         if image is not None:
#             image_embed = self.encode_image(image)
#             return {'text_embed': text_embed_all,
#                     'omic_embed': omic_embed,
#                     'image_embed': image_embed,
#                     'logit_scale': self.logit_scale.exp()}

#         else:
#             return {'text_embed': text_embed_all,
#                     'omic_embed': omic_embed,
#                     'logit_scale': self.logit_scale.exp()}


class GeneLIP_WITH_MIZERO(nn.Module):
    def __init__(self, gene_encoder, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        # self.context_length = kwargs.context_length
        # self.vision_width = kwargs.vision_width

        # self.visual = kwargs.vision_model
        # self.text = kwargs.text_model

        self.visual = kwargs.vl_model.visual # trucnk + head
        self.text = kwargs.vl_model.encode_text

        self.tokenizer = kwargs.tokenizer



        # self.transformer = Transformer(
        #     width=kwargs.transformer_width,
        #     layers=kwargs.transformer_layers,
        #     heads=kwargs.transformer_heads,
        #     attn_mask=self.build_attention_mask(),
        # ) 
        self.kwargs = kwargs

        # self.vocab_size = kwargs.vocab_size
        # self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        # self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        # self.ln_final = LayerNorm(kwargs.transformer_width)

        # self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim)) # 768 -> 512
        # self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim)) # 512 -> 512

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.image_encoder = self.visual.head
        # self.text_encoder = self.text.transformer

        self.image_projection = self.visual.head.proj
        # self.text_projection = self.text.proj
        self.logit_scale = kwargs.vl_model.logit_scale

        # self.initialize_parameters()

        self.gene_encoder = gene_encoder

        self.gene_projection = nn.Parameter(torch.empty(kwargs.omic_feat_dims, 512))
        nn.init.normal_(self.gene_projection, std=512 ** -0.5)

        self.tune_visual = kwargs.args.tune_visual
        if self.tune_visual.lower() == 'adapter' :
            print("Use visual adapter!")
            self.adapter = Adapter(kwargs.vision_width,4)


    def encode_image(self, image):
        image_feat = self.image_encoder(image)

        if self.tune_visual.lower() == 'adapter':
            x = self.adapter(image_feat)
            ratio = 0.2
            image_feat = ratio * x + (1 - ratio) * image_feat
        
        # x = image_feat @ self.image_projection
        x = self.image_projection( image_feat)

        return x
    
    

    def encode_text(self, text, attention_mask):
        if len(text.shape) ==3:
            if text.shape[0] ==1 and text.shape[1]==1:
                text = text[0]
            else:
                raise ValueError("text must have 2 dimension, or 3 dimension like [1,1,77]")
        # x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        # x = x + self.positional_embedding
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x)

        # x = self.text_encoder(text)

        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        # texts, attention_mask = tokenize(self.tokenizer, texts) 
        # texts = torch.from_numpy(np.array(texts)).cuda(non_blocking=True)
        # attention_mask = torch.from_numpy(np.array(attention_mask)).cuda()
        # class_embeddings = utils.get_model(model).encode_text(texts, attention_mask=attention_mask)

        # x = self.text(text)

        x = self.text(text,attention_mask=attention_mask )

        # self.text(text[0]).shape   1,512


        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # def encode_pc(self, pc):
    #     pc_feat = self.point_encoder(pc)
    #     pc_embed = pc_feat @ self.pc_projection
    #     return pc_embed

    def encode_omic(self, x_omic):
        omic_feat = self.gene_encoder(x_omic=x_omic)[0]
        omic_embed = omic_feat @ self.gene_projection
        return omic_embed

    def forward(self, x_omic, text, image=None):
        # For omic, image ==> torch.Size([1, 3, 512, 512])    
        # For pc, image ==> torch.Size([1, 3, 224, 224])
    
        text_embed_all = []
        for i in range(text.shape[0]):  #(1,1,77)
            text_for_one_sample = text[i]
            text_for_one_sample, attention_mask = tokenize(self.tokenizer, text_for_one_sample) 
            text_for_one_sample = torch.from_numpy(np.array(text_for_one_sample)).cuda( non_blocking=True)
            attention_mask = torch.from_numpy(np.array(attention_mask)).cuda()
            text_embed = self.encode_text(text_for_one_sample, attention_mask=attention_mask)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)

        '''
        07/15 adjusted from pc to omic
        '''
        # pc_embed = self.encode_pc(pc)
        omic_embed = self.encode_omic(x_omic)
        

        if image is not None:
            image_embed = self.encode_image(image)
            return {'text_embed': text_embed_all,
                    'omic_embed': omic_embed,
                    'image_embed': image_embed,
                    'logit_scale': self.logit_scale.exp()}

        else:
            return {'text_embed': text_embed_all,
                    'omic_embed': omic_embed,
                    'logit_scale': self.logit_scale.exp()}


class ULIP_WITH_IMAGE(nn.Module):
    def __init__(self, point_encoder, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.point_encoder = point_encoder

        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 512))
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

        # self.use_visual_adapter = kwargs.args.use_visual_adapter
        # if self.use_visual_adapter:
        #     print("Use visual adapter!")
        #     self.adapter = Adapter(kwargs.vision_width,4)


    def encode_image(self, image):
        image_feat = self.visual(image)

        # if self.use_visual_adapter:
        #     x = self.adapter(image_feat)
        #     ratio = 0.2
        #     image_feat = ratio * x + (1 - ratio) * image_feat

        x = image_feat @ self.image_projection

        return x

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_pc(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    def encode_omic(self, x_omic):
        omic_feat = self.point_encoder(x_omic=x_omic)[0]
        omic_embed = omic_feat @ self.pc_projection
        return omic_embed

    def forward(self, pc, text, image=None):
        # For omic, image ==> torch.Size([1, 3, 512, 512])    
        # For pc, image ==> torch.Size([1, 3, 224, 224])
    
        text_embed_all = []
        for i in range(text.shape[0]):  #(1,1,77)
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)

        '''
        07/15 adjusted from pc to omic
        '''
        # pc_embed = self.encode_pc(pc)
        pc_embed = self.encode_omic(pc)
        

        if image is not None:
            image_embed = self.encode_image(image)
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'image_embed': image_embed,
                    'logit_scale': self.logit_scale.exp()}

        else:
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'logit_scale': self.logit_scale.exp()}


def get_loss(args):
    # return losses.ULIPWithImageLoss()
    # return losses.GeneLIPWithImageLoss(args)
    return losses.CITEImageLoss(args)


def get_metric_names(model):
    # return ['loss', 'ulip_loss', 'ulip_pc_image_acc', 'ulip_pc_text_acc']
    # return ['loss', 'ulip_loss', 'ulip_omic_image_acc', 'ulip_omic_text_acc']
    return ['loss', 'ulip_loss', 'ulip_omic_image_matching_acc', 'ulip_omic_text_matching_acc', 'ulip_image_text_matching_acc']

# {'loss': loss, 'ulip_loss': loss, 'ulip_omic_image_matching_acc': omic_image_acc, 'ulip_omic_text_matching_acc': omic_text_acc, 'ulip_image_text_matching_acc': image_text_acc}

def ULIP_PN_SSG(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    point_encoder = Pointnet2_Ssg()
    pc_feat_dims = 256
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model

def ULIP_PN_MLP(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointmlp.pointMLP import pointMLP
    point_encoder = pointMLP()
    pc_feat_dims = 256
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, 
                            point_encoder=point_encoder, 
                            vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters(): # 把slip的参数往
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model

'''
2023/07/13 基因多模态模型
'''

def ULIP_GENE_SNN(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0, prompt_type=args.tune_visual)
    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    # from models.pointmlp.pointMLP import pointMLP
    # point_encoder = pointMLP()
    # pc_feat_dims = 256
    # =====================================================================

    from models.gene.SNN import SNN
    omic_feat_dims = 128

    snn = SNN()

    model = GeneLIP_WITH_IMAGE(embed_dim=512, vision_width=768, 
                            gene_encoder=snn, 
                            vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, 
                            omic_feat_dims=omic_feat_dims, args=args)

    if not args.evaluate:
        # load the pretrained model
        
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))

        # pretrain_slip_model = torch.load('./data/initialize_models/clip_base_25ep.pt', map_location=torch.device('cpu'))

        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}
        
        # print('###########')

        for name, param in model.named_parameters(): # 把slip的参数往
            if name not in pretrain_slip_model_params: # 不在SLIP的参数
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]
            
            if args.tune_visual == "fine-tune" and 'visual' in name:
                pass
            else:
                param.requires_grad = False

            
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

        for name, param in model.named_parameters():
            if (args.fix_gene or args.wo_gene) and 'gene' in name:
                param.requires_grad = False

            if param.requires_grad:
                print('Update parameters {}'.format(name))

    return model

'''
===========================================================================
Layer (type:depth-idx)                             Param #
===========================================================================
├─VisionTransformer: 1-1                           --
|    └─PatchEmbed: 2-1                             --
|    |    └─Conv2d: 3-1                            (590,592)
|    |    └─Identity: 3-2                          --
|    └─Dropout: 2-2                                --
|    └─Sequential: 2-3                             --
|    |    └─Block: 3-3                             (7,087,872)
|    |    └─Block: 3-4                             (7,087,872)
|    |    └─Block: 3-5                             (7,087,872)
|    |    └─Block: 3-6                             (7,087,872)
|    |    └─Block: 3-7                             (7,087,872)
|    |    └─Block: 3-8                             (7,087,872)
|    |    └─Block: 3-9                             (7,087,872)
|    |    └─Block: 3-10                            (7,087,872)
|    |    └─Block: 3-11                            (7,087,872)
|    |    └─Block: 3-12                            (7,087,872)
|    |    └─Block: 3-13                            (7,087,872)
|    |    └─Block: 3-14                            (7,087,872)
|    └─LayerNorm: 2-4                              (1,536)
|    └─Identity: 2-5                               --
|    └─Identity: 2-6                               --
├─Transformer: 1-2                                 --
|    └─Sequential: 2-7                             --
|    |    └─ResidualAttentionBlock: 3-15           (3,152,384)
|    |    └─ResidualAttentionBlock: 3-16           (3,152,384)
|    |    └─ResidualAttentionBlock: 3-17           (3,152,384)
|    |    └─ResidualAttentionBlock: 3-18           (3,152,384)
|    |    └─ResidualAttentionBlock: 3-19           (3,152,384)
|    |    └─ResidualAttentionBlock: 3-20           (3,152,384)
|    |    └─ResidualAttentionBlock: 3-21           (3,152,384)
|    |    └─ResidualAttentionBlock: 3-22           (3,152,384)
|    |    └─ResidualAttentionBlock: 3-23           (3,152,384)
|    |    └─ResidualAttentionBlock: 3-24           (3,152,384)
|    |    └─ResidualAttentionBlock: 3-25           (3,152,384)
|    |    └─ResidualAttentionBlock: 3-26           (3,152,384)
├─Embedding: 1-3                                   (25,296,896)
├─LayerNorm: 1-4                                   (1,024)
├─SNN: 1-5                                         --
|    └─LogSoftmax: 2-8                             --
|    └─Sequential: 2-9                             --
|    |    └─Sequential: 3-27                       5,184
|    |    └─Sequential: 3-28                       3,120
|    |    └─Sequential: 3-29                       1,568
|    |    └─Sequential: 3-30                       4,224
|    └─ReLU: 2-10                                  --
|    └─Sequential: 2-11                            --
|    |    └─Linear: 3-31                           387
===========================================================================
Total params: 148,787,603
Trainable params: 14,483
Non-trainable params: 148,773,120
===========================================================================
'''


'''
BiomedCLIP

=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─TimmModel: 1-1                         --
|    └─VisionTransformer: 2-1            --
|    |    └─PatchEmbed: 3-1              590,592
|    |    └─Dropout: 3-2                 --
|    |    └─Sequential: 3-3              85,054,464
|    |    └─LayerNorm: 3-4               1,536
|    |    └─Identity: 3-5                --
|    |    └─Identity: 3-6                --
|    └─Sequential: 2-2                   --
|    |    └─Dropout: 3-7                 --
|    |    └─Linear: 3-8                  393,216
├─HFTextEncoder: 1-2                     --
|    └─BertModel: 2-3                    --
|    |    └─BertEmbeddings: 3-9          23,837,184
|    |    └─BertEncoder: 3-10            85,054,464
|    └─ClsLastHiddenStatePooler: 2-4     --
|    └─Sequential: 2-5                   --
|    |    └─Linear: 3-11                 491,520
|    |    └─GELU: 3-12                   --
|    |    └─Linear: 3-13                 327,680
=================================================================
Total params: 195,750,656
Trainable params: 195,750,656
Non-trainable params: 0
=================================================================

SLIP

===========================================================================
Layer (type:depth-idx)                             Param #
===========================================================================
├─VisionTransformer: 1-1                           --
|    └─PatchEmbed: 2-1                             --
|    |    └─Conv2d: 3-1                            590,592
|    |    └─Identity: 3-2                          --
|    └─Dropout: 2-2                                --
|    └─Sequential: 2-3                             --
|    |    └─Block: 3-3                             7,087,872
|    |    └─Block: 3-4                             7,087,872
|    |    └─Block: 3-5                             7,087,872
|    |    └─Block: 3-6                             7,087,872
|    |    └─Block: 3-7                             7,087,872
|    |    └─Block: 3-8                             7,087,872
|    |    └─Block: 3-9                             7,087,872
|    |    └─Block: 3-10                            7,087,872
|    |    └─Block: 3-11                            7,087,872
|    |    └─Block: 3-12                            7,087,872
|    |    └─Block: 3-13                            7,087,872
|    |    └─Block: 3-14                            7,087,872
|    └─LayerNorm: 2-4                              1,536
|    └─Identity: 2-5                               --
|    └─Identity: 2-6                               --
├─Transformer: 1-2                                 --
|    └─Sequential: 2-7                             --
|    |    └─ResidualAttentionBlock: 3-15           3,152,384
|    |    └─ResidualAttentionBlock: 3-16           3,152,384
|    |    └─ResidualAttentionBlock: 3-17           3,152,384
|    |    └─ResidualAttentionBlock: 3-18           3,152,384
|    |    └─ResidualAttentionBlock: 3-19           3,152,384
|    |    └─ResidualAttentionBlock: 3-20           3,152,384
|    |    └─ResidualAttentionBlock: 3-21           3,152,384
|    |    └─ResidualAttentionBlock: 3-22           3,152,384
|    |    └─ResidualAttentionBlock: 3-23           3,152,384
|    |    └─ResidualAttentionBlock: 3-24           3,152,384
|    |    └─ResidualAttentionBlock: 3-25           3,152,384
|    |    └─ResidualAttentionBlock: 3-26           3,152,384
├─Embedding: 1-3                                   25,296,896
├─LayerNorm: 1-4                                   1,024
├─SNN: 1-5                                         --
|    └─LogSoftmax: 2-8                             --
|    └─Sequential: 2-9                             --
|    |    └─Sequential: 3-27                       5,184
|    |    └─Sequential: 3-28                       3,120
|    |    └─Sequential: 3-29                       1,568
|    |    └─Sequential: 3-30                       4,224
|    └─ReLU: 2-10                                  --
|    └─Sequential: 2-11                            --
|    |    └─Linear: 3-31                           387
===========================================================================
Total params: 148,787,603
Trainable params: 148,787,603
Non-trainable params: 0
===========================================================================


'''

def ULIP_GENE_SNN_BiomedCLIP(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    # from models.pointmlp.pointMLP import pointMLP
    # point_encoder = pointMLP()
    # pc_feat_dims = 256
    # =====================================================================


    biomedclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    # Compose(
    # RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=bicubic)
    # <function _convert_to_rgb at 0x7f64da8a1a70>
    # ToTensor()
    # Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    # )
    # Compose(
    # Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
    # CenterCrop(size=(224, 224))
    # <function _convert_to_rgb at 0x7f64da8a1a70>
    # ToTensor()
    # Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    # )

    for name, param in biomedclip.named_parameters():
      print('parameters {}'.format(name))
    '''
    - logit_scale
    - visual
      - trunk
      - hdead
    - text
      - transformer
      - proj
    '''
  
    from models.gene.SNN import SNN
    omic_feat_dims = 128

    snn = SNN()

    if not args.ori_biomedclip:
        print('###### Embed Biomedclip into ULIP framework')
        model = GeneLIP_WITH_BIOMEDCLIP(
            embed_dim=512, 
            # vision_width=768, 
            gene_encoder=snn, 
            vl_model=biomedclip,
            # vision_model=biomedclip.visual,
            # text_model = biomedclip.text,
            # context_length=77, vocab_size=49408,
            # transformer_width=512, transformer_heads=8, transformer_layers=12, 
            omic_feat_dims=omic_feat_dims, 
            args=args)
    else:
        print('##### Use whole BioMedCLIP in an original fashion.. ')
        model = biomedclip 

    # if not args.evaluate:
    #     # load the pretrained model
        
    #     pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
    #     pretrain_slip_model_params = pretrain_slip_model['state_dict']
    #     pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
    #                                   pretrain_slip_model_params.items()}
        
    #     '''
    #     positional_embedding
    #     image_proj
    #     text_proj
    #     logit_scale
    #     visual.-
    #     transformer.-
    #     token_embedding.weight
    #     ln_final.weight
    #     ln_final.bias
    #     image_mlp.-

        
    #     '''
        
    #     print('###########')

    #     for name, param in model.named_parameters(): # 把slip的参数往
    #         if name not in pretrain_slip_model_params:
    #             continue

    #         if isinstance(pretrain_slip_model_params[name], Parameter):
    #             param_new = pretrain_slip_model_params[name].data
    #         else:
    #             param_new = pretrain_slip_model_params[name]

    #         param.requires_grad = False
    #         print('load {} and freeze'.format(name))
    #         param.data.copy_(param_new)
    for name, param in model.named_parameters(): # 把slip的参数往
          if 'gene' in name or 'omic' in name:
              param.requires_grad = True
          else:
              param.requires_grad = False

    for name, param in model.named_parameters():
      if not param.requires_grad:
          print('Freeze parameters {}'.format(name))
              
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('Update parameters {}'.format(name))
  
  

    return model


def ULIP_GENE_SNN_QuiltCLIP(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    # from models.pointmlp.pointMLP import pointMLP
    # point_encoder = pointMLP()
    # pc_feat_dims = 256
    # =====================================================================


    biomedclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
    tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')

    # Compose(
    # RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=bicubic)
    # <function _convert_to_rgb at 0x7f64da8a1a70>
    # ToTensor()
    # Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    # )
    # Compose(
    # Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
    # CenterCrop(size=(224, 224))
    # <function _convert_to_rgb at 0x7f64da8a1a70>
    # ToTensor()
    # Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    # )

    for name, param in biomedclip.named_parameters():
        print('parameters {}'.format(name))
  
    from models.gene.SNN import SNN
    omic_feat_dims = 128

    snn = SNN()

    # vision_model = timm.create_model('vit_base_patch32_224', num_classes=0)


    model = GeneLIP_WITH_QUILTCLIP(embed_dim=512, vision_width=768, 
                            gene_encoder=snn, 
                            vl_model=biomedclip,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, 
                            omic_feat_dims=omic_feat_dims, args=args)


    # if not args.ori_biomedclip:
    #     print('###### Embed Quilt into ULIP framework')

    # GeneLIP_WITH_IMAGE

    # model = GeneLIP_WITH_QUILTCLIP(
    #     embed_dim=512, 
    #     # vision_width=768, 
    #     gene_encoder=snn, 
    #     vl_model=biomedclip,
    #     # vision_model=biomedclip.visual,
    #     # text_model = biomedclip.text,
    #     # context_length=77, vocab_size=49408,
    #     # transformer_width=512, transformer_heads=8, transformer_layers=12, 
    #     omic_feat_dims=omic_feat_dims, 
    #     args=args)
    # else:
    #     print('##### Use whole BioMedCLIP in an original fashion.. ')
    #     model = biomedclip 

    if not args.evaluate:
        # load the pretrained model
        
        # pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = biomedclip.state_dict()
        # pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}
        '''
        visual encoder没有载入....
        '''
        
        for name, param in model.named_parameters(): # 把slip的参数往
            # if 'visual' in name:
            #     pass
            #     print('a')
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

        for name, param in model.named_parameters(): # 把slip的参数往
              if 'gene' in name or 'omic' in name:
                  param.requires_grad = True
              else:
                  param.requires_grad = False

        for name, param in model.named_parameters():
          if not param.requires_grad:
              print('Freeze parameters {}'.format(name))
                  
        for name, param in model.named_parameters():
            if param.requires_grad:
                print('Update parameters {}'.format(name))
  

    return model


'''
CustomTextCLIP(
  (visual): TimmModel(
    (trunk): VisionTransformer(
      (patch_embed): PatchEmbed(
        (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
        (norm): Identity()
      )
      (pos_drop): Dropout(p=0.0, inplace=False)
      (blocks): Sequential(
        (0): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): Identity()
          (drop_path1): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (ls2): Identity()
          (drop_path2): Identity()
        )
        (1): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): Identity()
          (drop_path1): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (ls2): Identity()
          (drop_path2): Identity()
        )
        (2): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): Identity()
          (drop_path1): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (ls2): Identity()
          (drop_path2): Identity()
        )
        (3): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): Identity()
          (drop_path1): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (ls2): Identity()
          (drop_path2): Identity()
        )
        (4): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): Identity()
          (drop_path1): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (ls2): Identity()
          (drop_path2): Identity()
        )
        (5): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): Identity()
          (drop_path1): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (ls2): Identity()
          (drop_path2): Identity()
        )
        (6): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): Identity()
          (drop_path1): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (ls2): Identity()
          (drop_path2): Identity()
        )
        (7): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): Identity()
          (drop_path1): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (ls2): Identity()
          (drop_path2): Identity()
        )
        (8): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): Identity()
          (drop_path1): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (ls2): Identity()
          (drop_path2): Identity()
        )
        (9): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): Identity()
          (drop_path1): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (ls2): Identity()
          (drop_path2): Identity()
        )
        (10): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): Identity()
          (drop_path1): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (ls2): Identity()
          (drop_path2): Identity()
        )
        (11): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): Identity()
          (drop_path1): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (drop1): Dropout(p=0.0, inplace=False)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop2): Dropout(p=0.0, inplace=False)
          )
          (ls2): Identity()
          (drop_path2): Identity()
        )
      )
      (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (fc_norm): Identity()
      (head): Identity()
    )
    (head): Sequential(
      (drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=512, bias=False)
    )
  )
  (text): HFTextEncoder(
    (transformer): BertModel(
      (embeddings): BertEmbeddings(
        (word_embeddings): Embedding(30522, 768, padding_idx=0)
        (position_embeddings): Embedding(512, 768)
        (token_type_embeddings): Embedding(2, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): BertEncoder(
        (layer): ModuleList(
          (0): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (1): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (2): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (3): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (4): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (5): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (6): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (7): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (8): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (9): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (10): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (11): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
    )
    (pooler): ClsLastHiddenStatePooler()
    (proj): Sequential(
      (0): Linear(in_features=768, out_features=640, bias=False)
      (1): GELU()
      (2): Linear(in_features=640, out_features=512, bias=False)
    )
  )
)
'''

def ULIP_GENE_SNN_MIZERO(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    # from models.pointmlp.pointMLP import pointMLP
    # point_encoder = pointMLP()
    # pc_feat_dims = 256
    # =====================================================================

    # biomedclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    # tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    def clean_state_dict_ctranspath(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'attn_mask' in k:
                continue
            new_state_dict[k.replace('module.', '')] = v
        return new_state_dict

    state_dict = torch.load('/data/cxli/code/MI-Zero/src/checkpoint/ctranspath_448_bioclinicalbert/checkpoints/epoch_50.pt', map_location='cpu')['state_dict']

    args.model_checkpoint = "/data/cxli/code/MI-Zero/src/checkpoint/ctranspath_448_bioclinicalbert/checkpoints/epoch_50.pt"
    args.model_name = args.model_checkpoint.split('/')[-3]
    model = create_model(model_name=args.model_name, device='cuda', override_image_size=None)
    if args.model_checkpoint is not None: # load PPTCLIP checkpoint if applicable
        if os.path.exists(args.model_checkpoint):
            state_dict = torch.load(args.model_checkpoint, map_location='cpu')['state_dict']
            state_dict = clean_state_dict_ctranspath(state_dict)
            missing_keys, _ = model.load_state_dict(state_dict, strict=False)
            assert pd.Series(missing_keys).str.contains('attn_mask').all() # only modules with attn_mask are not loaded
            logging.info(f'Checkpoint {args.model_checkpoint} loaded successfully')
        else:
            logging.error(f'Cannot find model checkpoint {args.model_checkpoint}')
            return 1
    

    # Compose(
    # RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=bicubic)
    # <function _convert_to_rgb at 0x7f64da8a1a70>
    # ToTensor()
    # Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    # )
    # Compose(
    # Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
    # CenterCrop(size=(224, 224))
    # <function _convert_to_rgb at 0x7f64da8a1a70>
    # ToTensor()
    # Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    # )

    # for name, param in biomedclip.named_parameters():
    #   print('parameters {}'.format(name))
    '''
    - logit_scale
    - visual
      - trunk
      - hdead
    - text
      - transformer
      - proj
    '''
  
    from models.gene.SNN import SNN
    omic_feat_dims = 128

    snn = SNN()

    model = GeneLIP_WITH_MIZERO(
            embed_dim=512, 
            # vision_width=768, 
            gene_encoder=snn, 
            vl_model=model,
            tokenizer=args.tokenizer,
            # vision_model=biomedclip.visual,
            # text_model = biomedclip.text,
            # context_length=77, vocab_size=49408,
            # transformer_width=512, transformer_heads=8, transformer_layers=12, 
            omic_feat_dims=omic_feat_dims, 
            args=args)

    # if not args.ori_biomedclip:
    #     print('###### Embed Biomedclip into ULIP framework')
    #     model = GeneLIP_WITH_BIOMEDCLIP(
    #         embed_dim=512, 
    #         # vision_width=768, 
    #         gene_encoder=snn, 
    #         vl_model=biomedclip,
    #         # vision_model=biomedclip.visual,
    #         # text_model = biomedclip.text,
    #         # context_length=77, vocab_size=49408,
    #         # transformer_width=512, transformer_heads=8, transformer_layers=12, 
    #         omic_feat_dims=omic_feat_dims, 
    #         args=args)
    # else:
    #     print('##### Use whole BioMedCLIP in an original fashion.. ')
    #     model = biomedclip 

    # if not args.evaluate:
    #     # load the pretrained model
        
    #     pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
    #     pretrain_slip_model_params = pretrain_slip_model['state_dict']
    #     pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
    #                                   pretrain_slip_model_params.items()}
        
    #     '''
    #     positional_embedding
    #     image_proj
    #     text_proj
    #     logit_scale
    #     visual.-
    #     transformer.-
    #     token_embedding.weight
    #     ln_final.weight
    #     ln_final.bias
    #     image_mlp.-

        
    #     '''
        
    #     print('###########')

    #     for name, param in model.named_parameters(): # 把slip的参数往
    #         if name not in pretrain_slip_model_params:
    #             continue

    #         if isinstance(pretrain_slip_model_params[name], Parameter):
    #             param_new = pretrain_slip_model_params[name].data
    #         else:
    #             param_new = pretrain_slip_model_params[name]

    #         param.requires_grad = False
    #         print('load {} and freeze'.format(name))
    #         param.data.copy_(param_new)
    for name, param in model.named_parameters(): # 把slip的参数往
          if 'gene' in name or 'omic' in name:
              param.requires_grad = True
          else:
              param.requires_grad = False

    for name, param in model.named_parameters():
      if not param.requires_grad:
          print('Freeze parameters {}'.format(name))
              
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('Update parameters {}'.format(name))
  
  

    return model



def ULIP_PointBERT(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointbert.point_encoder import PointTransformer
    config_addr = './models/pointbert/PointTransformer_8192point.yaml'
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer(config.model, args=args)
    pc_feat_dims = 768
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model

def ULIP_PN_NEXT(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointnext.pointnext import PointNEXT
    point_encoder = PointNEXT()
    pc_feat_dims = 256
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model


def ULIP_CUSTOMIZED(args):
  
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # This is a sample template to pre-train your customized 3D backbones, please modify this part accordingly!
    from models.customized_backbone.customized_backbone import CUSTOMIZED_BACKBONE
    point_encoder = CUSTOMIZED_BACKBONE()
    # We assume you might have different point cloud output feature dimension,
    # we added a projecting layer to unify the point cloud output dimension before doing the multimodal alignment,
    # please change the output feature dimension here.
    pc_feat_dims = 512
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model


def tokenize(tokenizer, texts):
    tokens = tokenizer.batch_encode_plus(texts, 
                                        max_length = 64,
                                        add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                        return_token_type_ids=False,
                                        truncation = True,
                                        padding = 'max_length',
                                        return_attention_mask=True)
    return tokens['input_ids'], tokens['attention_mask']