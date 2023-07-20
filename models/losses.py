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

        if self.tune_visual.lower() != 'none':
            print('## Perform visual tuning')
        else:
            print('## No perform visual tuning')


    def forward(self, outputs):
        omic_embed = outputs['omic_embed']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = omic_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=omic_embed.device
            )
            self.last_local_batch_size = local_batch_size

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


        # label is the sample index in the batch
        # the first term is the contrastive loss between gene and text
        # the second term is the contrastive loss between gene and image

        '''
        # Found bug 0717 loss有点问题, 感觉 omic和text的对齐会差...
        '''
        # loss = (F.cross_entropy(logits_per_omic_image, self.labels) +  F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
        #     (F.cross_entropy(logits_per_omic_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2


        loss = (F.cross_entropy(logits_per_omic_text, self.labels) +  F.cross_entropy(logits_per_text_omic, self.labels)) / 2 + \
            (F.cross_entropy(logits_per_omic_image, self.labels) + F.cross_entropy(logits_per_image_omic, self.labels)) / 2
        
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
