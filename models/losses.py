'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

class ULIPWithImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        pc_embed = outputs['pc_embed']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # gather features from all GPUs
        pc_embed_all, text_embed_all, image_embed_all = \
            utils.all_gather_batch([pc_embed, text_embed, image_embed])

        # cosine similarity as logits
        logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        logits_per_pc_image = logit_scale * pc_embed @ image_embed_all.t()
        logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()

        loss = (F.cross_entropy(logits_per_pc_text, self.labels) +  F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
            (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2
        # label is the sample index in the batch
        # the first term is the contrastive loss between gene and text
        # the second term is the contrastive loss between gene and image

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_pc_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_pc_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_image_acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ulip_loss': loss, 'ulip_pc_image_acc': pc_image_acc, 'ulip_pc_text_acc': pc_text_acc}


class GeneLIPWithImageLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

        self.tune_visual = args.tune_visual
        self.wo_gene = args.wo_gene

        self.args = args

        if self.tune_visual.lower() != 'none':
            print('## Perform visual tuning')
        else:
            print('## No perform visual tuning')


    def forward(self, outputs):


        omic_embed = outputs['omic_embed']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        logit_scale = outputs['logit_scale'] # 76 |  
        local_batch_size = omic_embed.size(0) 

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=omic_embed.device
            )
            self.last_local_batch_size = local_batch_size
        # labels: 0~1072

        # normalized features
        omic_embed = F.normalize(omic_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # gather features from all GPUs
        omic_embed_all, text_embed_all, image_embed_all = \
            utils.all_gather_batch([omic_embed, text_embed, image_embed])

        # cosine similarity as logits
        logits_per_omic_text = logit_scale * omic_embed @ text_embed_all.t()
        logits_per_text_omic = logit_scale * text_embed @ omic_embed_all.t()
        logits_per_omic_image = logit_scale * omic_embed @ image_embed_all.t()
        logits_per_image_omic = logit_scale * image_embed @ omic_embed_all.t()

        logits_per_image_text = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text_image = logit_scale * text_embed @ image_embed_all.t()

        #   [1072, 1072]


        # label is the sample index in the batch
        # the first term is the contrastive loss between gene and text
        # the second term is the contrastive loss between gene and image

        '''
        # Found bug 0717 loss有点问题, 感觉 omic和text的对齐会差...
        '''
        # loss = (F.cross_entropy(logits_per_omic_image, self.labels) +  F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
        #     (F.cross_entropy(logits_per_omic_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2

        if self.args.half_label_and_pair:
            assert logits_per_omic_text.shape[0]==1072

            bz = len(logits_per_omic_text)
            loss = (F.cross_entropy(logits_per_omic_text[:bz//2], self.labels[:bz//2]) +  F.cross_entropy(logits_per_text_omic[:bz//2], self.labels[:bz//2])) / 2 + \
                    (F.cross_entropy(logits_per_omic_image[:bz//2], self.labels[:bz//2]) + F.cross_entropy(logits_per_image_omic[:bz//2], self.labels[:bz//2])) / 2
            
            if self.wo_gene:
                loss = loss* 0.0
            
            if self.tune_visual.lower() != 'none':
                loss += ( F.cross_entropy(logits_per_image_text[:bz//2], self.labels[:bz//2]) + F.cross_entropy(logits_per_text_image[:bz//2], self.labels[:bz//2]) ) / 2

            loss += (F.cross_entropy(logits_per_omic_image[bz//2:], self.labels[bz//2:]) + F.cross_entropy(logits_per_image_omic[bz//2:], self.labels[bz//2:])) / 2
        
        else:
            
            loss = (F.cross_entropy(logits_per_omic_text, self.labels) +  F.cross_entropy(logits_per_text_omic, self.labels)) / 2 + \
                    (F.cross_entropy(logits_per_omic_image, self.labels) + F.cross_entropy(logits_per_image_omic, self.labels)) / 2
            
            if self.wo_gene:
                loss = loss* 0.0
            
            if self.tune_visual.lower() != 'none':
                loss += ( F.cross_entropy(logits_per_image_text, self.labels) + F.cross_entropy(logits_per_text_image, self.labels) ) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_omic_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            omic_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_omic_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            omic_image_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_image_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            image_text_acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ulip_loss': loss, 'ulip_omic_image_matching_acc': omic_image_acc, 'ulip_omic_text_matching_acc': omic_text_acc, 'ulip_image_text_matching_acc': image_text_acc}


class CITEImageLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

        self.tune_visual = args.tune_visual
        self.wo_gene = args.wo_gene

        self.args = args

        if self.tune_visual.lower() != 'none':
            print('## Perform visual tuning')
        else:
            print('## No perform visual tuning')

        if args.text_mode == 'sentence':
            self.text_features = args.text_features
        elif args.text_mode == 'description':
            self.text_description_features = args.text_description_features


    def forward(self, outputs):

        # omic_embed = outputs['omic_embed']
        # text_embed = outputs['text_embed']
        image_embed = outputs['image_embed'] # [B, 512]
        logit_scale = outputs['logit_scale'] 
        cls_label = outputs['cls_label']
        local_batch_size = image_embed.size(0) 


        '''
        self-supervised labels
        '''
        # if local_batch_size != self.last_local_batch_size:
        #     self.labels = local_batch_size * utils.get_rank() + torch.arange(
        #         local_batch_size, device=omic_embed.device
        #     )
        #     self.last_local_batch_size = local_batch_size
        # labels: 0~1072

        # normalized features

        # omic_embed = F.normalize(omic_embed, dim=-1, p=2)
        # text_embed = F.normalize(text_embed, dim=-1, p=2)
        # image_embed = F.normalize(image_embed, dim=-1, p=2)

        # gather features from all GPUs
        # omic_embed_all, text_embed_all, image_embed_all = \
        #     utils.all_gather_batch([omic_embed, text_embed, image_embed])

        image_embed_all = \
        utils.all_gather_batch([image_embed])

        # cosine similarity as logits
        # logits_per_omic_text = logit_scale * omic_embed @ text_embed_all.t()
        # logits_per_text_omic = logit_scale * text_embed @ omic_embed_all.t()
        
        # logits_per_omic_image = logit_scale * omic_embed @ image_embed_all.t()
        # logits_per_image_omic = logit_scale * image_embed @ omic_embed_all.t()

        # logits_per_image_text = logit_scale * image_embed @ text_embed_all.t()
        # logits_per_text_image = logit_scale * text_embed @ image_embed_all.t()

        #   [1072, 1072]

        # Cross-entropy
        if self.args.text_mode == 'sentence':
            logits = (logit_scale * path_features @ text_features.t())
        elif self.args.text_mode == 'description':
            logits =  torch.zeros((local_batch_size, len(text_description_features))).cuda() # [B,3]
            for k, text_features in text_description_features.items():
                logits[:, list(caption_candidate.keys()).index(k) ] = (logit_scale * path_features @ text_features.t()).mean(dim=-1)

        text_cls_loss = F.cross_entropy(logits, cls_label)
        loss = text_cls_loss

        # loss = F.cross_entropy(logits_per_omic_text, self.labels) +  F.cross_entropy(logits_per_text_omic, self.labels)) / 2 + \
        #         (F.cross_entropy(logits_per_omic_image, self.labels) + F.cross_entropy(logits_per_image_omic, self.labels)) / 2
        
        # if self.wo_gene:
        #     loss = loss* 0.0
        
        # if self.tune_visual.lower() != 'none':
        #     loss += ( F.cross_entropy(logits_per_image_text, self.labels) + F.cross_entropy(logits_per_text_image, self.labels) ) / 2



        # label is the sample index in the batch
        # the first term is the contrastive loss between gene and text
        # the second term is the contrastive loss between gene and image

        '''
        # Found bug 0717 loss有点问题, 感觉 omic和text的对齐会差...
        '''
        # loss = (F.cross_entropy(logits_per_omic_image, self.labels) +  F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
        #     (F.cross_entropy(logits_per_omic_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2

        # if self.args.half_label_and_pair:
        #     assert logits_per_omic_text.shape[0]==1072

        #     bz = len(logits_per_omic_text)
        #     loss = (F.cross_entropy(logits_per_omic_text[:bz//2], self.labels[:bz//2]) +  F.cross_entropy(logits_per_text_omic[:bz//2], self.labels[:bz//2])) / 2 + \
        #             (F.cross_entropy(logits_per_omic_image[:bz//2], self.labels[:bz//2]) + F.cross_entropy(logits_per_image_omic[:bz//2], self.labels[:bz//2])) / 2
            
        #     if self.wo_gene:
        #         loss = loss* 0.0
            
        #     if self.tune_visual.lower() != 'none':
        #         loss += ( F.cross_entropy(logits_per_image_text[:bz//2], self.labels[:bz//2]) + F.cross_entropy(logits_per_text_image[:bz//2], self.labels[:bz//2]) ) / 2

        #     loss += (F.cross_entropy(logits_per_omic_image[bz//2:], self.labels[bz//2:]) + F.cross_entropy(logits_per_image_omic[bz//2:], self.labels[bz//2:])) / 2
        
        # else:
        
        # loss = (F.cross_entropy(logits_per_omic_text, self.labels) +  F.cross_entropy(logits_per_text_omic, self.labels)) / 2 + \
        #         (F.cross_entropy(logits_per_omic_image, self.labels) + F.cross_entropy(logits_per_image_omic, self.labels)) / 2
        
        # # if self.wo_gene:
        # #     loss = loss* 0.0
        
        # if self.tune_visual.lower() != 'none':
        #     loss += ( F.cross_entropy(logits_per_image_text, self.labels) + F.cross_entropy(logits_per_text_image, self.labels) ) / 2

        # compute accuracy
        with torch.no_grad():
            # pred = torch.argmax(logits_per_omic_image, dim=-1)
            # correct = pred.eq(self.labels).sum()
            # omic_text_acc = 100 * correct / local_batch_size

            # pred = torch.argmax(logits_per_omic_image, dim=-1)
            # correct = pred.eq(self.labels).sum()
            # omic_image_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits, dim=-1)
            correct = pred.eq(cls_label).sum()
            image_text_acc = 100 * correct / local_batch_size

        # return {'loss': loss, 'ulip_loss': loss, 'ulip_omic_image_matching_acc': omic_image_acc, 'ulip_omic_text_matching_acc': omic_text_acc, 'ulip_image_text_matching_acc': image_text_acc}
        return {'loss': loss, 'image_text_matching_acc': image_text_acc}
