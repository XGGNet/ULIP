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
from collections import OrderedDict

import dgl
from scipy.stats import pearsonr

import torch.nn.functional as F

import torch as th

import torch

# from coop import *

from graph_models import (
    # GCN,
    GAT,
    NTPoolGCN,
    GIN,
    HGT,
    HEATNet2,
    HEATNet4,
    HeteroRGCN,
    # HEATNet4_v1,
)
from graph_models.HEATNet4 import *



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

        self.args=  args  = kwargs.args

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

        self.tokenizer = tokenizer = self.args.tokenizer


        self.tune_visual = kwargs.args.tune_visual
        if self.tune_visual.lower() == 'adapter' :
            print("Use visual adapter!")
            self.adapter = Adapter(kwargs.vision_width,4)

        if self.args.only_vis_cls_head:
            self.cls_head = nn.Linear(512, 3)

        
        self.use_text_prompt = self.args.use_text_prompt 
        
        
        labels = ['II', 'III', 'IV']
        if args.text_mode == 'sentence':
            templates = ['A pathology slide with WHO grade {} gliomas']
            self.text_sentence = [templates[0].format(l) for l in labels] 
        if args.text_mode == 'description':
            self.text_description = {
              'A pathology slide with WHO grade II gliomas':
              [
              'Infiltrative growth pattern',
              'Relatively uniform cells with round or oval nuclei and minimal pleomorphism',
              'Low mitotic activity',
              'Absence of microvascular proliferation',
              'Absence of necrosis',

              # 'A pathology slide with grade II gliomas'
              ],

              'A pathology slide with WHO grade III gliomas':
              [
              # "Increased cellularity compared to grade II gliomas",
              # "Mild to moderate nuclear atypia and pleomorphism.",
              # "Higher mitotic activity compared to grade II gliomas.",
              # "Absence or minimal microvascular proliferation.",
              # "Absence or focal necrosis.",
              "Increased cellularity",
              "Mild to moderate nuclear atypia and pleomorphism.",
              "Higher mitotic activity.",
              "Absence or minimal microvascular proliferation.",
              "Absence or focal necrosis.",

              # 'A pathology slide with grade III gliomas'
              ],

              'A pathology slide with WHO grade IV gliomas':
              [
              "Highly cellular and pleomorphic tumor cells",
              "Marked nuclear atypia and pleomorphism.",
              "High mitotic activity",
              "Prominent microvascular proliferation",
              "Presence of necrosis, often with pseudopalisading pattern (tumor cells surrounding necrotic areas).",

              # 'A pathology slide with grade IV gliomas'
              ]
            }
            # 把key的字符串加在其value的每一个字符串前面
            # caption_candidate = {k: [k + ', which has ' + i for i in v] for k, v in caption_candidate.items()}

        if self.use_text_prompt:
            cfg = {'N_CTX': 16 , 'CLASS_TOKEN_POSITION': 'end'}
            # n_prompt = 1 if self.text_prompt=='sentence' else 5
            # n_cls =  3
            # if self.text_mode  == 'sentence':
            #     self.prompt_learner
            # elif self.text_mode == 'description':

            if args.text_mode == 'sentence':
                self.prompt_learner = PromptLearner(cfg=cfg, args=args, n_cls=3, n_des = 5,  clip_model=vl_model, tokenizer=tokenizer,  text = self.text_sentence)
            elif args.text_mode == 'description':
                self.prompt_learner = PromptLearner(cfg=cfg, args=args, n_cls=3, n_des = 5,  clip_model=vl_model, tokenizer=tokenizer,  text = self.text_description)

            self.text_token = self.prompt_learner.text_token
        else:
            if args.text_mode == 'sentence':
                # texts.app[t.format(l) for t in templates]
                text_token = tokenizer(text_sentence).cuda()
            elif args.text_mode == 'description':
                text_token = OrderedDict()
                for k, v in self.text_description.items():
                    tokens = tokenizer(v).cuda() #[5,77]
                    text_token[k] = tokens
            self.text_token = text_token

        self.text_mode = args.text_mode
                
              
    def encode_image(self, image):
        image_feat = self.visual(image)  #[253,512]
      
        if self.tune_visual.lower() == 'adapter':
            x = self.adapter(image_feat)
            ratio = 0.2
            image_feat = ratio * x + (1 - ratio) * image_feat
        
        x = image_feat @ self.image_projection

        return x

    def encode_text(self, text_token, text_embed=None):
        # Eval [5,77]   Train [1,5,77]
        # if len(text.shape) >2:
        #   text = text.squeeze(0)
        if self.use_text_prompt: # prompt tuning时, 从prompt embedding开始
            x = text_embed + self.positional_embedding
        else:
            x = self.token_embedding(text_token)  # [batch_size, n_ctx, d_model]   torch.Size([5, 77, 512])
            x = x + self.positional_embedding

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) # 1,77,512
        # bz, context_length, embed_dim

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text_token.argmax(dim=-1)] @ self.text_projection  # 1,512

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

    def forward_visual_cls_head(self, image):
        image_embed = self.encode_image(image)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        logits = self.cls_head(image_embed)

        return { 
              'logits': logits
                }

    def forward(self, image, gene):
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

        # pc_embed = self.encode_pc(pc)
        # omic_embed = self.encode_omic(x_omic)
        
        image_embed = self.encode_image(image)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        if self.use_text_prompt: 
            if self.text_mode == 'sentence':
                learnble_text_embed = self.prompt_learner()
                text_embed = self.encode_text( text_token=self.text_token, text_embed=learnble_text_embed )
            elif self.text_mode == 'description':
                text_embed = OrderedDict()
                learnble_text_embed = self.prompt_learner() # 5个一组.., [15, 512]
                for k, v in self.text_description.items():
                    text_embed_output = self.encode_text( text_token=self.text_token[k], text_embed=learnble_text_embed[k] )
                    text_embed_output = text_embed_output / text_embed_output.norm(dim=-1, keepdim=True)
                    text_embed[k] = text_embed_output
        else:
            if self.text_mode == 'sentence':
                text_embed = self.encode_text(text_token = self.text_token)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            elif self.text_mode == 'description':
                  text_embed = OrderedDict()
                  for k, v in self.text_description.items():
                      text_embed_output = self.encode_text( text_token=self.text_token[k] )
                      text_embed_output = text_embed_output / text_embed_output.norm(dim=-1, keepdim=True)
                      text_embed[k] = text_embed_output

        # pc_embed = self.encode_pc(pc)
        # omic_embed = self.encode_omic(x_omic)

        return { 
              'image_embed': image_embed,
              'text_embed': text_embed,
              'gene_embed': None,
              'logit_scale': self.logit_scale.exp()
                }


class GeneLIP_WITH_QUILTCLIP_GeneLM(nn.Module):
    def __init__(self, gene_encoder, vl_model, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        # self.visual = kwargs.vision_model

        self.args=  args  = kwargs.args

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

        # self.omic_to_embedding = nn.Parameter(torch.empty(80, kwargs.omic_feat_dims))
        # nn.init.normal_(self.omic_to_embedding, std=kwargs.omic_feat_dims ** -0.5)

        self.args =  kwargs.args
        
        if self.args.gene_lm != 'snn':

          self.omic_to_embedding = nn.Linear(80, kwargs.omic_feat_dims)

        self.omic_projection = nn.Parameter(torch.empty(kwargs.omic_feat_dims, 512))
        nn.init.normal_(self.omic_projection, std=512 ** -0.5)

        self.image_projection = vl_model.visual.proj

        self.logit_scale = vl_model.logit_scale

        self.visual = vl_model.visual
        self.visual.proj = None
        # vision_model = timm.create_model('vit_base_patch32_224', num_classes=0)

        self.tokenizer = tokenizer = self.args.tokenizer

        self.text_mode = args.text_mode


        self.tune_visual = kwargs.args.tune_visual
        if self.tune_visual.lower() == 'adapter' :
            print("Use visual adapter!")
            self.adapter = Adapter(kwargs.vision_width,4)

        if self.args.only_vis_cls_head:
            self.cls_head = nn.Linear(512, 3)

        
        self.use_text_prompt = self.args.use_text_prompt 
        
        
        labels = ['II', 'III', 'IV']
        if args.text_mode == 'sentence':
            templates = ['A pathology slide with WHO grade {} gliomas']
            self.text_sentence = [templates[0].format(l) for l in labels] 
        if args.text_mode == 'description':
            self.text_description = {
              'A pathology slide with WHO grade II gliomas':
              [
              'Infiltrative growth pattern',
              'Relatively uniform cells with round or oval nuclei and minimal pleomorphism',
              'Low mitotic activity',
              'Absence of microvascular proliferation',
              'Absence of necrosis',

              # 'A pathology slide with grade II gliomas'
              ],

              'A pathology slide with WHO grade III gliomas':
              [
              # "Increased cellularity compared to grade II gliomas",
              # "Mild to moderate nuclear atypia and pleomorphism.",
              # "Higher mitotic activity compared to grade II gliomas.",
              # "Absence or minimal microvascular proliferation.",
              # "Absence or focal necrosis.",
              "Increased cellularity",
              "Mild to moderate nuclear atypia and pleomorphism.",
              "Higher mitotic activity.",
              "Absence or minimal microvascular proliferation.",
              "Absence or focal necrosis.",

              # 'A pathology slide with grade III gliomas'
              ],

              'A pathology slide with WHO grade IV gliomas':
              [
              "Highly cellular and pleomorphic tumor cells",
              "Marked nuclear atypia and pleomorphism.",
              "High mitotic activity",
              "Prominent microvascular proliferation",
              "Presence of necrosis, often with pseudopalisading pattern (tumor cells surrounding necrotic areas).",

              # 'A pathology slide with grade IV gliomas'
              ]
            }
            # 把key的字符串加在其value的每一个字符串前面
            # caption_candidate = {k: [k + ', which has ' + i for i in v] for k, v in caption_candidate.items()}

        if self.use_text_prompt:
            cfg = {'N_CTX': 16 , 'CLASS_TOKEN_POSITION': 'end'}
            # n_prompt = 1 if self.text_prompt=='sentence' else 5
            # n_cls =  3
            # if self.text_mode  == 'sentence':
            #     self.prompt_learner
            # elif self.text_mode == 'description':

            if args.text_mode == 'sentence':
                self.prompt_learner = PromptLearner(cfg=cfg, args=args, n_cls=3, n_des = 5,  clip_model=vl_model, tokenizer=tokenizer,  text = self.text_sentence)
            elif args.text_mode == 'description':
                self.prompt_learner = PromptLearner(cfg=cfg, args=args, n_cls=3, n_des = 5,  clip_model=vl_model, tokenizer=tokenizer,  text = self.text_description)

            self.text_token = self.prompt_learner.text_token
        else:
            if args.text_mode == 'sentence':
                # texts.app[t.format(l) for t in templates]
                text_token = tokenizer(text_sentence).cuda()
            elif args.text_mode == 'description':
                text_token = OrderedDict()
                for k, v in self.text_description.items():
                    tokens = tokenizer(v).cuda() #[5,77]
                    text_token[k] = tokens
            self.text_token = text_token
                
              
    def encode_image(self, image):
        image_feat = self.visual(image)  #[253,512]
      
        if self.tune_visual.lower() == 'adapter':
            x = self.adapter(image_feat)
            ratio = 0.2
            image_feat = ratio * x + (1 - ratio) * image_feat
        
        x = image_feat @ self.image_projection

        return x

    def encode_text(self, text_token, text_embed=None):
        # Eval [5,77]   Train [1,5,77]
        # if len(text.shape) >2:
        #   text = text.squeeze(0)
        if self.use_text_prompt: # prompt tuning时, 从prompt embedding开始
            x = text_embed + self.positional_embedding
        else:
            x = self.token_embedding(text_token)  # [batch_size, n_ctx, d_model]   torch.Size([5, 77, 512])
            x = x + self.positional_embedding

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) # 1,77,512
        # bz, context_length, embed_dim

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text_token.argmax(dim=-1)] @ self.text_projection  # 1,512

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
        if self.args.gene_lm == 'snn':
            omic_feat = self.gene_encoder(x_omic=x_omic)[0]
            omic_embed = omic_feat @ self.omic_projection

        else:
            omic_embed = self.omic_to_embedding(x_omic).unsqueeze(1) # [N, len_seq=1, n_embed]

            if self.args.gene_lm == 'geneformer':
              omic_feat = self.gene_encoder(omic_embed)[0] # N, len_seq=1, n_embed 
            elif self.args.gene_lm == 'dnabert':
              omic_feat = self.gene_encoder(omic_embed,attention_mask=torch.ones_like(omic_embed)[:,:,0], output_all_encoded_layers=False,subset_mask=None)[0]
            elif self.args.gene_lm == 'gpn':
              omic_feat = self.gene_encoder(omic_embed)
              
            omic_feat = torch.mean(omic_feat, dim=1) # 把seq进行mean
            omic_embed = omic_feat @ self.omic_projection # dimension: n_embed -> n_clip_embed

        return omic_embed

    def forward_visual_cls_head(self, image):
        image_embed = self.encode_image(image)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        logits = self.cls_head(image_embed)

        return { 
              'logits': logits
                }


    def forward(self, image, gene):
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

        # pc_embed = self.encode_pc(pc)
        # omic_embed = self.encode_omic(x_omic)
        
        image_embed = self.encode_image(image)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        if self.use_text_prompt: 
            if self.text_mode == 'sentence':
                learnble_text_embed = self.prompt_learner()
                text_embed = self.encode_text( text_token=self.text_token, text_embed=learnble_text_embed )
            elif self.text_mode == 'description':
                text_embed = OrderedDict()
                learnble_text_embed = self.prompt_learner() # 5个一组.., [15, 512]
                for k, v in self.text_description.items():
                    text_embed_output = self.encode_text( text_token=self.text_token[k], text_embed=learnble_text_embed[k] )
                    text_embed_output = text_embed_output / text_embed_output.norm(dim=-1, keepdim=True)
                    text_embed[k] = text_embed_output
        else:
            if self.text_mode == 'sentence':
                text_embed = self.encode_text(text_token = self.text_token)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            elif self.text_mode == 'description':
                  text_embed = OrderedDict()
                  for k, v in self.text_description.items():
                      text_embed_output = self.encode_text( text_token=self.text_token[k] )
                      text_embed_output = text_embed_output / text_embed_output.norm(dim=-1, keepdim=True)
                      text_embed[k] = text_embed_output

        omic_embed = self.encode_omic(gene)
        omic_embed = omic_embed / omic_embed.norm(dim=-1, keepdim=True)

        # pc_embed = self.encode_pc(pc)
        # omic_embed = self.encode_omic(x_omic)

        return { 
              'image_embed': image_embed,
              'text_embed': text_embed,
              'omic_embed': omic_embed,
              'logit_scale': self.logit_scale.exp()
                }


class GeneLIP_WITH_QUILTCLIP_GeneLM_Graph(nn.Module):
    def __init__(self, gene_encoder, vl_model, gnn_model, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        # self.visual = kwargs.vision_model

        self.args=  args  = kwargs.args

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

        # self.omic_to_embedding = nn.Parameter(torch.empty(80, kwargs.omic_feat_dims))
        # nn.init.normal_(self.omic_to_embedding, std=kwargs.omic_feat_dims ** -0.5)

        self.args =  kwargs.args
        
        if self.args.gene_lm != 'snn':

          self.omic_to_embedding = nn.Linear(80, kwargs.omic_feat_dims)

        self.omic_projection = nn.Parameter(torch.empty(kwargs.omic_feat_dims, 512))
        nn.init.normal_(self.omic_projection, std=512 ** -0.5)

        self.image_projection = vl_model.visual.proj

        self.logit_scale = vl_model.logit_scale

        self.visual = vl_model.visual
        self.visual.proj = None
        # vision_model = timm.create_model('vit_base_patch32_224', num_classes=0)

        self.tokenizer = tokenizer = self.args.tokenizer

        self.text_mode = args.text_mode


        self.tune_visual = kwargs.args.tune_visual
        if self.tune_visual.lower() == 'adapter' :
            print("Use visual adapter!")
            self.adapter = Adapter(kwargs.vision_width,4)

        if self.args.only_vis_cls_head:
            self.cls_head = nn.Linear(512, 3)

        '''
        Temp
        '''
        # self.cls_head = nn.Linear(512, 3)
        
        self.use_text_prompt = self.args.use_text_prompt 
        
        
        labels = ['II', 'III', 'IV']
        if args.text_mode == 'sentence':
            templates = ['A pathology slide with WHO grade {} gliomas']
            self.text_sentence = [templates[0].format(l) for l in labels] 
        if args.text_mode == 'description':
            self.text_description = {
              'A pathology slide with WHO grade II gliomas':
              [
              'Infiltrative growth pattern',
              'Relatively uniform cells with round or oval nuclei and minimal pleomorphism',
              'Low mitotic activity',
              'Absence of microvascular proliferation',
              'Absence of necrosis',

              # 'A pathology slide with grade II gliomas'
              ],

              'A pathology slide with WHO grade III gliomas':
              [
              # "Increased cellularity compared to grade II gliomas",
              # "Mild to moderate nuclear atypia and pleomorphism.",
              # "Higher mitotic activity compared to grade II gliomas.",
              # "Absence or minimal microvascular proliferation.",
              # "Absence or focal necrosis.",
              "Increased cellularity",
              "Mild to moderate nuclear atypia and pleomorphism.",
              "Higher mitotic activity.",
              "Absence or minimal microvascular proliferation.",
              "Absence or focal necrosis.",

              # 'A pathology slide with grade III gliomas'
              ],

              'A pathology slide with WHO grade IV gliomas':
              [
              "Highly cellular and pleomorphic tumor cells",
              "Marked nuclear atypia and pleomorphism.",
              "High mitotic activity",
              "Prominent microvascular proliferation",
              "Presence of necrosis, often with pseudopalisading pattern (tumor cells surrounding necrotic areas).",

              # 'A pathology slide with grade IV gliomas'
              ]
            }
            # 把key的字符串加在其value的每一个字符串前面
            # caption_candidate = {k: [k + ', which has ' + i for i in v] for k, v in caption_candidate.items()}

        if self.use_text_prompt:
            cfg = {'N_CTX': 16 , 'CLASS_TOKEN_POSITION': 'end'}
            # n_prompt = 1 if self.text_prompt=='sentence' else 5
            # n_cls =  3
            # if self.text_mode  == 'sentence':
            #     self.prompt_learner
            # elif self.text_mode == 'description':

            if args.text_mode == 'sentence':
                self.prompt_learner = PromptLearner(cfg=cfg, args=args, n_cls=3, n_des = 5,  clip_model=vl_model, tokenizer=tokenizer,  text = self.text_sentence)
            elif args.text_mode == 'description':
                self.prompt_learner = PromptLearner(cfg=cfg, args=args, n_cls=3, n_des = 5,  clip_model=vl_model, tokenizer=tokenizer,  text = self.text_description)

            self.text_token = self.prompt_learner.text_token
        else:
            if args.text_mode == 'sentence':
                # texts.app[t.format(l) for t in templates]
                text_token = tokenizer(text_sentence).cuda()
            elif args.text_mode == 'description':
                text_token = OrderedDict()
                for k, v in self.text_description.items():
                    tokens = tokenizer(v).cuda() #[5,77]
                    text_token[k] = tokens
            self.text_token = text_token


        # self.gnn = my_gcn()
        
        self.gnn = HEATNet4_v1(
            in_dim=512,
            hidden_dim=512//2,
            out_dim=3,
            n_layers=2,
            n_heads=4,
            # node_dict = {'image':0},
            node_dict={'gene': 0, 'image': 1, 'text': 2}, # TBD
            dropuout=0.0,
            graph_pooling_type='mean'
        )

        # self.gnn = my_gcn()

        # self.gnn = GCN(in_feats=512, n_hidden=256, n_classes=3, n_layers=1, activation=F.relu, dropout=0.0) 
                
              
    def encode_image(self, image):
        image_feat = self.visual(image)  #[253,512]
      
        if self.tune_visual.lower() == 'adapter':
            x = self.adapter(image_feat)
            ratio = 0.2
            image_feat = ratio * x + (1 - ratio) * image_feat
        
        x = image_feat @ self.image_projection

        return x

    def encode_text(self, text_token, text_embed=None):
        # Eval [5,77]   Train [1,5,77]
        # if len(text.shape) >2:
        #   text = text.squeeze(0)
        if self.use_text_prompt: # prompt tuning时, 从prompt embedding开始
            x = text_embed + self.positional_embedding
        else:
            x = self.token_embedding(text_token)  # [batch_size, n_ctx, d_model]   torch.Size([5, 77, 512])
            x = x + self.positional_embedding

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) # 1,77,512
        # bz, context_length, embed_dim

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text_token.argmax(dim=-1)] @ self.text_projection  # 1,512

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
        if self.args.gene_lm == 'snn':
            omic_feat = self.gene_encoder(x_omic=x_omic)[0]
            omic_embed = omic_feat @ self.omic_projection

        else:
            omic_embed = self.omic_to_embedding(x_omic).unsqueeze(1) # [N, len_seq=1, n_embed]

            if self.args.gene_lm == 'geneformer':
              omic_feat = self.gene_encoder(omic_embed)[0] # N, len_seq=1, n_embed 
            elif self.args.gene_lm == 'dnabert':
              omic_feat = self.gene_encoder(omic_embed,attention_mask=torch.ones_like(omic_embed)[:,:,0], output_all_encoded_layers=False,subset_mask=None)[0]
            elif self.args.gene_lm == 'gpn':
              omic_feat = self.gene_encoder(omic_embed)
              
            omic_feat = torch.mean(omic_feat, dim=1) # 把seq进行mean
            omic_embed = omic_feat @ self.omic_projection # dimension: n_embed -> n_clip_embed

        return omic_embed

    def forward_visual_cls_head(self, image):
        image_embed = self.encode_image(image)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        logits = self.cls_head(image_embed)

        return { 'logits': logits}


    def find_index(self,lst,x):
        indexes = []
        for i in range(len(lst)):
            if lst[i] == x:
                indexes.append(i)
        return indexes


    def forward(self, image, gene, label,case_name):

        image_embed = self.encode_image(image)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        if self.use_text_prompt: 
            if self.text_mode == 'sentence':
                learnble_text_embed = self.prompt_learner()
                text_embed = self.encode_text( text_token=self.text_token, text_embed=learnble_text_embed )
            elif self.text_mode == 'description':
                text_embed = OrderedDict()
                learnble_text_embed = self.prompt_learner() # 5个一组.., [15, 512]
                for k, v in self.text_description.items():
                    text_embed_output = self.encode_text( text_token=self.text_token[k], text_embed=learnble_text_embed[k] )
                    text_embed_output = text_embed_output / text_embed_output.norm(dim=-1, keepdim=True)
                    text_embed[k] = text_embed_output
        else:
            if self.text_mode == 'sentence':
                text_embed = self.encode_text(text_token = self.text_token)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            elif self.text_mode == 'description':
                  text_embed = OrderedDict()
                  for k, v in self.text_description.items():
                      text_embed_output = self.encode_text( text_token=self.text_token[k] )
                      text_embed_output = text_embed_output / text_embed_output.norm(dim=-1, keepdim=True)
                      text_embed[k] = text_embed_output

      
        omic_embed = self.encode_omic(gene)
        omic_embed = omic_embed / omic_embed.norm(dim=-1, keepdim=True) # dim=512

        # graph_logits = self.cls_head(image_embed)

        # constuct graph

        case_name_base = []
        for name in case_name:
            if name not in case_name_base:
              case_name_base.append(name)
              # 
              # 531 for train


        # case_name_base = set(case_name) # 531
        # # 引用case_name
        # case_name_base = list(case_name_base)
        
        graph_logits = []
        for case_base in case_name_base:
            indexes = self.find_index(case_name, case_base)

            # image_node_feat = image_embed[indexes].detach().cpu()
            # gene_node_feat = omic_embed[ indexes[0] ].detach().cpu()

            image_node_feat = image_embed[indexes]
            gene_node_feat = omic_embed[ indexes[0] ]

            text_node_feat = torch.zeros(15,512).to('cuda')
            if len(image_node_feat.shape)==1:
                image_node_feat = image_node_feat.unsqueeze(0)
            if len(gene_node_feat.shape)==1:
                gene_node_feat = gene_node_feat.unsqueeze(0)
            cnt =0
            for key, value in text_embed.items():
                for i in range(len(value)):
                    text_node_feat[cnt] = value[i] # .detach().cpu()
                    cnt += 1  

            n_patch = len(indexes)

            #     graph_logit = self.cls_head( image_node_feat )
            #     graph_logits.append(graph_logit)
            # graph_logits = torch.cat(graph_logits,dim=0) # 1072,3

            edge_0 = []
            edge_1 = []
            edge_sim = []
            for i in range(n_patch):
                for j in range(n_patch):
                    edge_0.append( i )
                    edge_1.append( j )
                    edge_sim.append( torch.cosine_similarity(image_node_feat[i:i+1], image_node_feat[j:j+1]) )

            edge_2 = []
            edge_3 = []
            edge_sim2 = []
            for i in range(15):
                for j in range(15):
                    edge_2.append( i )
                    edge_3.append( j )
                    edge_sim2.append( torch.cosine_similarity(text_node_feat[i:i+1], text_node_feat[j:j+1]) )


            # 531 subjects
            graph_data = dgl.heterograph( 
                    {
                    ('image', 'image_gene', 'gene'): (   th.tensor( [i for i in range(0,n_patch)]),   th.tensor([0]*n_patch)  ), # n_patch
                    ('gene', 'image_gene', 'image'): (   th.tensor([0]*n_patch),  th.tensor( [i for i in range(0,n_patch)])), # n_patch
                    ('image', 'image_text', 'text'): (   th.tensor(  [num for num in range(0, n_patch) for _ in range(15)] ), th.tensor( list(range(0,15))*n_patch )   ), # n_patch*15
                    ('text', 'image_text', 'image'): (    th.tensor( list(range(0,15))*n_patch ),  th.tensor(  [num for num in range(0, n_patch) for _ in range(15)] )  ), # n_patch*15
                    ('image', 'image_image', 'image'): ( th.tensor(edge_0), th.tensor(edge_1)  ), # n_patch
                    ('text', 'text_text', 'text'): ( th.tensor(edge_2), th.tensor(edge_3)  ), 
                    # ('gene', 'gene_text', 'text'): (  th.tensor([0]*15),  th.tensor( [i for i in range(0,15)]) ), # n_patch
                    }
                ).to('cuda')
            graph_data.nodes['image'].data['feat'] = torch.tensor(image_node_feat)
            graph_data.nodes['text'].data['feat'] = torch.tensor(text_node_feat)
            graph_data.nodes['gene'].data['feat'] = torch.tensor(gene_node_feat)

            graph_data.edges[('image', 'image_image', 'image')].data['sim'] = torch.tensor(edge_sim).to('cuda')
            graph_data.edges[('text', 'text_text', 'text')].data['sim'] = torch.tensor(edge_sim2).to('cuda')

            # X = image_node_feat
            # H = (torch.ones( (n_patch, n_patch) ) - torch.eye( n_patch )).cuda()

            # edge_sim = []
            # for i in range(1):
            #     for j in range(15):
            #         # corr = pearsonr(image_node_feat[i].detach().cpu(), text_node_feat[j].detach().cpu())[0]
            #         corr = torch.cosine_similarity(gene_node_feat[i:i+1], text_node_feat[j:j+1])
            #         edge_sim.append(corr)
            # # edge_sim = edge_sim * 2
            # # graph_data['image_text'].edata.update({'sim': torch.tensor(edge_sim)})
            # graph_data.edges[('gene', 'gene_text', 'text')].data['sim'] =  torch.tensor(edge_sim).to('cuda')


            '''
            Reminder:

            这里的type是4类, 不是想象中的两类...
            '''

            # image-gene, n_patch*1
            # image-text, n_patch*15

            # edge_sim = []
            # for i in range(n_patch):
            #     corr = pearsonr(image_node_feat[i].detach().cpu(), gene_node_feat[0].detach().cpu())[0]
            #     # cos相似度
  
            #     edge_sim.append(corr)

            edge_sim = torch.cosine_similarity(image_node_feat, gene_node_feat)

            # repeat edge_sim
            # graph_data.edata[('image', 'image_gene', 'gene')].update( {'sim': torch.tensor(edge_sim)} )
            graph_data.edges[('image', 'image_gene', 'gene')].data['sim'] =  torch.tensor(edge_sim).to('cuda')
            graph_data.edges[('gene', 'image_gene', 'image')].data['sim'] =  torch.tensor(edge_sim).to('cuda')


            edge_sim = []
            for i in range(n_patch):
                for j in range(15):
                    # corr = pearsonr(image_node_feat[i].detach().cpu(), text_node_feat[j].detach().cpu())[0]
                    corr = torch.cosine_similarity(image_node_feat[i:i+1], text_node_feat[j:j+1])
                    edge_sim.append(corr)
            # edge_sim = edge_sim * 2
            # graph_data['image_text'].edata.update({'sim': torch.tensor(edge_sim)})

            # edge_sim = torch.cosine_similarity(image_node_feat, text_node_feat)

            graph_data.edges[('image', 'image_text', 'text')].data['sim'] =  torch.tensor(edge_sim).to('cuda')
            graph_data.edges[('text', 'image_text', 'image')].data['sim'] =  torch.tensor(edge_sim).to('cuda')

            # graph_logit = self.gnn(graph_data.to('cuda'), features = image_node_feat)
            graph_logit = self.gnn(graph_data.to('cuda'))
            # graph_logit = self.gnn(X, H)

            graph_logits.append(graph_logit)

            # for i in range(n_patch):
            #     graph_logits.append(graph_logit)
        
        graph_logits = torch.cat(graph_logits,dim=0)
            
        return { 
              'image_embed': image_embed, # [1072, 512]
              'text_embed': text_embed, # 3 * [5,512]
              'omic_embed': omic_embed, # [1072, 512]
              'logit_scale': self.logit_scale.exp(),  # 100
              'graph_logits': graph_logits
              }


class PromptLearner(nn.Module):
    def __init__(self, cfg, args, n_cls, n_des, clip_model, tokenizer, text):
        super().__init__()
        # n_cls = len(classnames)
        n_ctx = cfg['N_CTX']
        # ctx_init = cfg['CTX_INIT']

        # dtype = clip_model.dtype
        dtype = clip_model.visual.conv1.weight.dtype

        self.use_text_prompt = args.use_text_prompt
        self.text_mode = args.text_mode

        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if ctx_init:
        #     # use given words to initialize context vectors
        #     ctx_init = ctx_init.replace("_", " ")
        #     n_ctx = len(ctx_init.split(" "))
        #     prompt = clip.tokenize(ctx_init)
        #     with torch.no_grad():
        #         embedding = clip_model.token_embedding(prompt).type(dtype)
        #     ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        #     prompt_prefix = ctx_init

        # else:
            # random initialization
        # if cfg['CSC']:
        #     print("Initializing class-specific contexts")
        #     ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        # else:

            # print("Initializing a generic context")

        self.text_mode = args.text_mode

        if self.text_mode == 'sentence':
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # [n_pt, 16, 512]   待学习的embedding
            nn.init.normal_(ctx_vectors, std=0.02)

        elif self.text_mode == 'description':
            ctx_vectors = torch.empty(n_des, n_ctx, ctx_dim, dtype=dtype) # [n_pt, 16, 512]   待学习的embedding
            nn.init.normal_(ctx_vectors, std=0.02)

        text_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{text_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.text = text

        '''
        0810 下午: 先只code description 的设定, 不然太久了...
        '''
        # classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(tokenizer(name)) for name in classnames]

        if self.text_mode == 'sentence':
            text_sentence = [text_prefix + " " + sen for sen in self.text]
            text_token = torch.cat([tokenizer(p) for p in text_sentence])
            with torch.no_grad():
                text_embedding = clip_model.token_embedding(text_token).type(dtype)
            embed_prefix = text_embedding[:, :1, :].type(dtype).to(self.ctx.device)  # [3,1,512]
            embed_suffix = text_embedding[:, 1 + n_ctx :, :].type(dtype).to(self.ctx.device)   # [3, *, 512]

        if self.text_mode == 'description':
            text_description = OrderedDict() 
            text_token = OrderedDict()  
            text_embed = OrderedDict() 
            embed_prefix = OrderedDict() 
            embed_suffix = OrderedDict() 
            for k, v in self.text.items():
                text_description[k] = [text_prefix + " " + des for des in v]
                text_token[k] =  torch.cat([tokenizer(text_des) for text_des in text_description[k]])
                with torch.no_grad():
                    text_embed[k] = clip_model.token_embedding(text_token[k]).type(dtype) # [5, 77, 512]
                embed_prefix[k] = text_embed[k][:, :1, :].type(dtype).to(self.ctx.device)  # [5,1,512]
                embed_suffix[k] = text_embed[k][:, 1 + n_ctx :, :].type(dtype).to(self.ctx.device)   # [5, *, 512]
            
            # self.description_dict = description_dict
            

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # self.register_buffer("embed_prefix", embed_prefix)  # SOS 
        # self.register_buffer("embed_suffix", embed_suffix)  # CLS, EOS

        self.embed_prefix = embed_prefix
        self.embed_suffix = embed_suffix

        self.n_cls = n_cls
        self.n_des = n_des
        self.n_ctx = n_ctx
        self.text_token = text_token  # torch.Tensor
        # self.name_lens = name_lens
        self.class_token_position = cfg['CLASS_TOKEN_POSITION']

    def forward(self):
        ctx = self.ctx # 待学习embed   [5,16,512]
        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(0).expand(self.n_des, -1, -1)

        prefix = self.embed_prefix
        suffix = self.embed_suffix

        if self.class_token_position == "end":
            learnable_embed = {} 
            if self.text_mode == 'sentence':
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

                learnable_embed = torch.cat(
                      [
                          prefix.to(ctx.device),  # (n_des, 1, dim)
                          ctx,     # (n_des, n_ctx, dim)
                          suffix.to(ctx.device),  # (n_des, *, dim)
                      ],
                      dim=1,
                  )

            elif self.text_mode == 'description':
                for k, v in self.text.items():
                  cls_index = list(self.text.keys()).index(k)

                  learnable_embed[k] = torch.cat(
                      [
                          prefix[k].to(ctx.device),  # (n_des, 1, dim)
                          ctx,     # (n_des, n_ctx, dim)
                          suffix[k].to(ctx.device),  # (n_des, *, dim)
                      ],
                      dim=1,
                  )

        # elif self.class_token_position == "middle":
        #     half_n_ctx = self.n_ctx // 2
        #     prompts = []
        #     for i in range(self.n_cls):
        #         name_len = self.name_lens[i]
        #         prefix_i = prefix[i : i + 1, :, :]
        #         class_i = suffix[i : i + 1, :name_len, :]
        #         suffix_i = suffix[i : i + 1, name_len:, :]
        #         ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
        #         ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
        #         prompt = torch.cat(
        #             [
        #                 prefix_i,     # (1, 1, dim)
        #                 ctx_i_half1,  # (1, n_ctx//2, dim)
        #                 class_i,      # (1, name_len, dim)
        #                 ctx_i_half2,  # (1, n_ctx//2, dim)
        #                 suffix_i,     # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #         prompts.append(prompt)
        #     prompts = torch.cat(prompts, dim=0)

        # elif self.class_token_position == "front":
        #     prompts = []
        #     for i in range(self.n_cls):
        #         name_len = self.name_lens[i]
        #         prefix_i = prefix[i : i + 1, :, :]
        #         class_i = suffix[i : i + 1, :name_len, :]
        #         suffix_i = suffix[i : i + 1, name_len:, :]
        #         ctx_i = ctx[i : i + 1, :, :]
        #         prompt = torch.cat(
        #             [
        #                 prefix_i,  # (1, 1, dim)
        #                 class_i,   # (1, name_len, dim)
        #                 ctx_i,     # (1, n_ctx, dim)
        #                 suffix_i,  # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #         prompts.append(prompt)
        #     prompts = torch.cat(prompts, dim=0)

        # else:
        #     raise ValueError

        return learnable_embed # dict



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
    # if args.only_vis_cls_head:
    #     return losses.VisClsLoss(args)
    # elif args.gene_lm is not None:
    #     return losses.CITEImage_ContGene_Loss(args)
    # else:
    #     return losses.CITEImageLoss(args)

    return losses.CITEImage_ContGene_Graph_Loss(args)


def get_metric_names(args):
    # return ['loss', 'ulip_loss', 'ulip_pc_image_acc', 'ulip_pc_text_acc']
    # return ['loss', 'ulip_loss', 'ulip_omic_image_acc', 'ulip_omic_text_acc']
    # return ['loss', 'ulip_loss', 'ulip_omic_image_matching_acc', 'ulip_omic_text_matching_acc', 'ulip_image_text_matching_acc']

    if args.gene_lm:
        return ['loss','image_text_cls_loss', 'omic_text_cls_loss', 'image_omic_cont_loss', 'graph_cls_loss','image_text_cls_acc', 'omic_text_cls_acc', 'image_omic_cont_acc', 'graph_cls_acc']
    else:
        return ['loss', 'image_text_matching_acc']

    # return ['loss', 'image_text_matching_acc']

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
            param.requires_grad = False
            for tune_name in ['gene', 'omic', 'adapter', 'prompt', 'cls_head']:
                if tune_name in name:
                    param.requires_grad = True
                    break
              # else:
              #     param.requires_grad = False

        # for name, param in model.named_parameters():
        #   if not param.requires_grad:
        #       print('Freeze parameters {}'.format(name))
                  
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print('Update parameters {}'.format(name))
  

    return model


def ULIP_GENE_LM_QuiltCLIP(args):
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
  
    # from models.gene.SNN import SNN
    # omic_feat_dims = 128
  
    # snn = SNN()


    from transformers import AutoTokenizer, AutoModelForMaskedLM
    import torch
    from transformers import AutoTokenizer, AutoModel

    if args.gene_lm == 'dnabert':
        gene_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        # AutoModelForMaskedLM
        gene_encoder =  gene_model.encoder
        omic_feat_dims = 768

    elif args.gene_lm == 'geneformer':
        gene_model = AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer")
        gene_encoder = gene_model.bert.encoder
        omic_feat_dims = 256

    elif args.gene_lm == 'gpn':
        import gpn.model
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from sklearn.preprocessing import StandardScaler
        import torch
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
        gene_model = AutoModel.from_pretrained("songlab/gpn-brassicales")
        gene_encoder = gene_model.encoder
        omic_feat_dims = 512
    elif args.gene_lm == 'snn':
        from models.gene.SNN import SNN
        gene_encoder = SNN()
        if args.pt_snn:
          pt_param = torch.load('/data/cxli/code/MultiModal-learning/MICCAI-2022/checkpoints/TCGA_GBMLGG/grad_k1_0812/stage1_pathomic_teacher/stage1_pathomic_teacher_1_best.pt')['model_state_dict']

          print('Loaded model from {}'.format('/data/cxli/code/MultiModal-learning/MICCAI-2022/checkpoints/TCGA_GBMLGG/grad_k1_0812/stage1_pathomic_teacher/stage1_pathomic_teacher_1_best.pt'))
          
          omic_pt_param = {}
          for name, param in pt_param.items():
              if 'omic_net' in name:
                omic_pt_param[name.replace('omic_net.', '')] = pt_param[name]

          gene_encoder.load_state_dict(omic_pt_param, strict=False)
        omic_feat_dims = 128

    # vision_model = timm.create_model('vit_base_patch32_224', num_classes=0)


    model = GeneLIP_WITH_QUILTCLIP_GeneLM(embed_dim=512, vision_width=768, 
                            gene_encoder=gene_encoder, 
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
            param.requires_grad = False
            for tune_name in ['omic', 'adapter', 'prompt', 'cls_head']:
                if tune_name in name:
                    param.requires_grad = True
                    break
              # else:
              #     param.requires_grad = False

        # for name, param in model.named_parameters():
        #   if not param.requires_grad:
        #       print('Freeze parameters {}'.format(name))
                  
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print('Update parameters {}'.format(name))
  

    return model


def ULIP_GENE_LM_QuiltCLIP_Graph(args):

    biomedclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
    tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')

    for name, param in biomedclip.named_parameters():
        print('parameters {}'.format(name))
  

    from transformers import AutoTokenizer, AutoModelForMaskedLM
    import torch
    from transformers import AutoTokenizer, AutoModel

    if args.gene_lm == 'dnabert':
        gene_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        # AutoModelForMaskedLM
        gene_encoder =  gene_model.encoder
        omic_feat_dims = 768

    elif args.gene_lm == 'geneformer':
        gene_model = AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer")
        gene_encoder = gene_model.bert.encoder
        omic_feat_dims = 256

    elif args.gene_lm == 'gpn':
        import gpn.model
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from sklearn.preprocessing import StandardScaler
        import torch
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
        gene_model = AutoModel.from_pretrained("songlab/gpn-brassicales")
        gene_encoder = gene_model.encoder
        omic_feat_dims = 512
    elif args.gene_lm == 'snn':
        from models.gene.SNN import SNN
        gene_encoder = SNN()
        if args.pt_snn:
          pt_param = torch.load('/data/cxli/code/MultiModal-learning/MICCAI-2022/checkpoints/TCGA_GBMLGG/grad_k1_0812/stage1_pathomic_teacher/stage1_pathomic_teacher_1_best.pt')['model_state_dict']

          print('Loaded model from {}'.format('/data/cxli/code/MultiModal-learning/MICCAI-2022/checkpoints/TCGA_GBMLGG/grad_k1_0812/stage1_pathomic_teacher/stage1_pathomic_teacher_1_best.pt'))
          
          omic_pt_param = {}
          for name, param in pt_param.items():
              if 'omic_net' in name:
                omic_pt_param[name.replace('omic_net.', '')] = pt_param[name]

          gene_encoder.load_state_dict(omic_pt_param, strict=False)
          omic_feat_dims = 128
    
    gnn = HEATNet4_v2(
            in_dim=512,
            hidden_dim=512//2,
            out_dim=3,
            n_layers=2,
            n_heads=4,
            node_dict = {'image':0},
            # node_dict={'gene': 0, 'image': 1, 'text': 2}, # TBD
            dropuout=0.2,
            graph_pooling_type='mean'
        )


    model = GeneLIP_WITH_QUILTCLIP_GeneLM_Graph(embed_dim=512, vision_width=768, 
                            gene_encoder=gene_encoder, 
                            vl_model=biomedclip,
                            gnn_model = gnn,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, 
                            omic_feat_dims=omic_feat_dims, args=args)

    
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
            param.requires_grad = False
            for tune_name in ['omic', 'adapter', 'prompt', 'cls_head', 'gnn']:
                if tune_name in name:
                    param.requires_grad = True
                    break
              # else:
              #     param.requires_grad = False

        # for name, param in model.named_parameters():
        #   if not param.requires_grad:
        #       print('Freeze parameters {}'.format(name))
                  
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print('Update parameters {}'.format(name))
  

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