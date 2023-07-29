"""
Author: Xing Xiaohan
Date: 2021.12.31
Construct dataset for the mean teacher framework.
Transform the pathology image twice as input for the student and mean teacher model, respectively.
Currently, use the same omic input for the student and mean teacher.
Memory banks are also included for the CRD loss.
"""
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets, transforms

# from utils import mixed_collate


def omic_transform(omic_data, transform='drop', rate=0.2):
    # print("original_data:", omic_data) ## [bs, dim]
    mask = np.random.binomial(1, rate, omic_data.shape)

    if transform == "drop":
        omic_data = omic_data * (1.0 - mask)
 
    ### the vime augmentation is from the paper: 
    ### VIME: Extending the success of self- And semi-supervised learning to tabular domain
    elif transform == "vime":
        no, dim = omic_data.shape  
        # Randomly (and column-wise) shuffle data
        x_bar = np.zeros([no, dim])
        for i in range(dim):
            idx = np.random.permutation(no)
            x_bar[:, i] = omic_data[idx, i]            
        # Corrupt samples
        omic_data = omic_data * (1-mask) + x_bar * mask
    
    return torch.tensor(omic_data).type(torch.FloatTensor)



def pathomic_dataset(opt, data):

    train_dataset = Pathomic_InstanceSample(opt, data, split='train', mode=opt.mode)
    print("number of training samples:", len(train_dataset))  # number of training samples: 1072
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=custom_data_loader, batch_size=opt.batch_size,  # 16
    #     num_workers=4, shuffle=True, collate_fn=mixed_collate, drop_last=True)
    
    n_data = len(train_dataset) # 1072

    test_dataset = PathomicDataset(opt, data, split='test', mode=opt.mode)
    # print("number of testing samples:", len(test_data_loader)) # number of testing samples: 253
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_data_loader, batch_size=opt.batch_size, 
    #     num_workers=4, shuffle=False, collate_fn=mixed_collate)

    return train_dataset, test_dataset, n_data



def pathomic_patches_dataset(opt, data):
    """
    Load the test set, each ROI image corresponds to 9 patch inputs with the size of 512*512.
    """
    test_dataset = PathomicDataset(opt, data, split='test', mode=opt.mode)
    print("number of testing patches:", len(test_dataset))
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_data_loader, batch_size=opt.batch_size, 
    #     num_workers=4, shuffle=False, collate_fn=mixed_collate)

    return test_dataset


#################################
# Dataloader without memory bank
#################################
class PathomicDataset(Dataset):
    def __init__(self, opt, data, split, mode='omic'):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        # print(data[split]['x_path'])
        self.X_path = data[split]['x_path']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']
        self.mode = mode

        self.split = split

        self.opt = opt
        
        if opt.label_dim == 2:
            ### 改成二分类，将标签中的1改为0, 标签中的2改为1
            label = self.g.astype(int)
            label[label == 1] = 0
            label[label == 2] = 1
            self.g = label

        # self.transforms = transforms.Compose([
        #                     transforms.RandomCrop(opt.input_size_path),
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if opt.normalization == 'clip':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        elif opt.normalization == 'biomedclip':
            normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        elif opt.normalization == 'data':
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


        self.train_transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomCrop(opt.input_size_path),
                            # transforms.Resize([opt.input_size_path, opt.input_size_path]),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                            transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
                            normalize
                            ])
        
        self.test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            normalize
                                                 ])

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor)

        if self.mode == "path" or self.mode == 'pathpath':
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            # print(single_X_path, self.transforms(single_X_path).shape)
            return (self.transforms(single_X_path), 0, 0, single_e, single_t, single_g)
        elif self.mode == "omic" or self.mode == 'omicomic':
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathomic":
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            # print(single_X_path, self.transforms(single_X_path).shape)


        if self.opt.text_mode == 'sentence':
            grading_name = {0: 'II', 1: 'III', 2: 'IV'}
            # caption = f'A pathology slide with WHO grading {grading_name[ int(single_g) ]}'
            # templates = ['A pathology slide with grade {} gliomas']
            caption = f'A pathology slide with grade {grading_name[ int(single_g) ]} gliomas'
        elif self.opt.text_mode == 'description':
            caption_candidate = {
                'A pathology slide with grade II gliomas':
                [
                "The cells tend to be relatively uniform in size and shape, and they may be arranged in a pattern that resembles the normal organization of tissue.",
                "The cells have a relatively low rate of division (mitotic rate) and may be surrounded by normal brain tissue.",
                "The tumor may have a well-defined border between the tumor and the surrounding tissue."
                ],
                'A pathology slide with grade III gliomas':
                [
                "The cells tend to be more variable in size and shape, and they may show signs of abnormal division (mitotic figures).",
                "The cells may be arranged in a more irregular pattern and may infiltrate the surrounding brain tissue.",
                "There may be areas of dead tissue (necrosis) within the tumor."
                ],
                ' A pathology slide with grade IV gliomas':
                [
                'The cells tend to be highly abnormal in appearance and may be very variable in size and shape, with large, irregular nuclei.',
                'There may be a high degree of mitotic activity, with many cells dividing rapidly.',
                'The tumor may have a very irregular border and may infiltrate extensively into the surrounding tissue.'
                'There may be areas of necrosis within the tumor.'
                ]
            }
            caption = list(caption_candidate.values())[int(single_g)]
            

            
            
        # tokenized_captions.append(self.tokenizer(caption))
        # tokenized_captions = torch.stack(tokenized_captions) 
        if self.split == 'train':
            return (self.train_transforms(single_X_path), caption , single_X_omic, single_e, single_t, single_g)
        elif self.split == 'test':
            return (self.test_transforms(single_X_path), caption, single_X_omic, single_e, single_t, single_g)

    def __len__(self):
        return len(self.X_path)


#################################
# Dataloader with memory bank
#################################
class Pathomic_InstanceSample(Dataset):
    def __init__(self, opt, data, split, mode='omic', is_sample=True):
        super(Pathomic_InstanceSample, self).__init__()

        '''
        对比样本
        '''
        self.p = opt.nce_p  # 300 正样本
        self.k = opt.nce_k  # 700 负样本

        self.is_sample = is_sample

        self.pos_mode = opt.pos_mode # 构造正样本的方式

        self.X_path = data[split]['x_path']
        self.X_omic = data[split]['x_omic']  # 
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']

        self.tokenizer = opt.tokenizer
        self.model =opt.model
        # self.train_transform = opt.train_transform

        self.mode = mode  # 'pathomic'
        self.task = opt.task # 'grad'

        self.split = split
        
        # self.transforms = TransformTwice(transforms.Compose([
        #                     transforms.RandomHorizontalFlip(0.5),
        #                     transforms.RandomVerticalFlip(0.5),
        #                     transforms.RandomCrop(opt.input_size_path), # 512
        #                     # transforms.Resize([opt.input_size_path, opt.input_size_path]),
        #                     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        if opt.normalization == 'clip':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        elif opt.normalization == 'biomedclip':
            normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        elif opt.normalization == 'data':
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
        self.train_transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomCrop(opt.input_size_path),
                            # transforms.Resize([opt.input_size_path, opt.input_size_path]),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                            transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            normalize
                            ])
        
        self.test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            normalize
                            ])

        self.num_samples = len(self.X_path) # 1072

        if opt.task == "grad":
            ## newly added, 2021/12/22
            if opt.label_dim == 3:  # enter here
                num_classes = 3
                label = self.g.astype(int)
            
            elif opt.label_dim == 2:
                ### 改成二分类，将标签中的1改为0, 标签中的2改为1
                num_classes = 2
                label = self.g.astype(int)
                print("original labels:", label)
                label[label == 1] = 0
                label[label == 2] = 1
                print("modified labels:", label)
                self.g = label

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(self.num_samples): # self.num_samples: 1072, len(label) = 1072
                self.cls_positive[label[i]].append(i) # 按类别划分

            
            # 收集所有不属于同一类别的sample id
            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])
            

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]  # 取值范围 0~1071   315+340+417
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]  # 取值范围 0~1071   757+723+655

            # self.cls_positive = np.asarray(self.cls_positive)
            # self.cls_negative = np.asarray(self.cls_negative)
            
            
            # print("positive:", self.cls_positive)
            # print("negative:", self.cls_negative)

            self.opt = opt

    def __getitem__(self, index):

        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor) # tensor(1.)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor) # tensor(395.)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor) # tensor(2)  grading

        single_X_path = Image.open(self.X_path[index]).convert('RGB')
        single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor) 
        # Whole ==> 1024,1024     Patch ==> 512,512


        # return (self.transforms(single_X_path), 0, single_X_omic, single_e, single_t, single_g)

        # print("sample index and label:", index, single_g)

        if self.task == "surv": # 生存预测
            pos_idx = index
            all_neg_idx = list(range(0, self.num_samples))
            all_neg_idx.remove(index)
            replace = True if self.k > len(all_neg_idx) else False
            neg_idx = np.random.choice(all_neg_idx, self.k, replace=replace)

        elif self.task == "grad": # grading分类
            
            '''
            选择构造正样本的模式
            '''
            if self.pos_mode == 'exact':
                pos_idx = np.asarray([index])

            elif self.pos_mode == 'relax':
                pos_idx = np.asarray([np.random.choice(self.cls_positive[single_g], 1)[0]])
                # print("anchor:", index, "pos_idx:", pos_idx)

            elif self.pos_mode == 'multi_pos': # enter here 
                # print("==============multiple positive pairs===============")
                # print("total number of positive samples:", self.cls_positive[single_g].shape)
                replace = True if self.p > len(self.cls_positive[single_g]) else False
                pos_idx = np.random.choice(self.cls_positive[single_g], self.p, replace=replace)
                # self.p: 300 是正样本数量;  single_g: 2 是类别, self.cls_positive[single_g] 是同类别的样本ID

                pos_idx[0] = index ### make sure the sample is selected as positive pair.

            else:
                raise NotImplementedError(self.pos_mode)

            replace = True if self.k > len(self.cls_negative[single_g]) else False
            neg_idx = np.random.choice(self.cls_negative[single_g], self.k, replace=replace)

            '''
            Caption for grading
            '''
            # captions = []
            tokenized_captions = []
            # grading_name = {0: 'II', 1: 'III', 2: 'IV'}
            # # caption = f'A pathology slide with WHO grading {grading_name[ int(single_g) ]}'
            # caption = f'A pathology slide with grade {grading_name[ int(single_g) ]} gliomas'

            # if 'mizero' not in self.model.lower():
            #     tokenized_captions.append(self.tokenizer(caption))
            #     tokenized_captions = torch.stack(tokenized_captions) 
            # else:
            #     tokenized_captions = caption

            if self.opt.text_mode == 'sentence':
                grading_name = {0: 'II', 1: 'III', 2: 'IV'}
                # caption = f'A pathology slide with WHO grading {grading_name[ int(single_g) ]}'
                # templates = ['A pathology slide with grade {} gliomas']
                caption = f'A pathology slide with grade {grading_name[ int(single_g) ]} gliomas'
            elif self.opt.text_mode == 'description':
                caption_candidate = {
                    'A pathology slide with grade II gliomas':
                    [
                    "The cells tend to be relatively uniform in size and shape, and they may be arranged in a pattern that resembles the normal organization of tissue.",
                    "The cells have a relatively low rate of division (mitotic rate) and may be surrounded by normal brain tissue.",
                    "The tumor may have a well-defined border between the tumor and the surrounding tissue."
                    ],
                    'A pathology slide with grade III gliomas':
                    [
                    "The cells tend to be more variable in size and shape, and they may show signs of abnormal division (mitotic figures).",
                    "The cells may be arranged in a more irregular pattern and may infiltrate the surrounding brain tissue.",
                    "There may be areas of dead tissue (necrosis) within the tumor."
                    ],
                    ' A pathology slide with grade IV gliomas':
                    [
                    'The cells tend to be highly abnormal in appearance and may be very variable in size and shape, with large, irregular nuclei.',
                    'There may be a high degree of mitotic activity, with many cells dividing rapidly.',
                    'The tumor may have a very irregular border and may infiltrate extensively into the surrounding tissue.'
                    'There may be areas of necrosis within the tumor.'
                    ]
                }
                caption = list(caption_candidate.values())[int(single_g)]

            tokenized_captions = self.opt.tokenizer(caption).unsqueeze(0)
                

        # print("sample index:", index)
        # print("positive index:", pos_idx)
        # print("negative index:", neg_idx)
        sample_idx = np.hstack((pos_idx, neg_idx))

        # print("sample label:", single_g)
        # for i in range(len(sample_idx)):
        #     idx = sample_idx[i]
        #     print("contrast sample label:", self.g[idx])

        # return (self.transforms(single_X_path), tokenized_captions, single_X_omic, single_e, single_t, single_g, index, sample_idx)
        # path_img, x_graph, x_omic, censor, survtime, grade ==> path_img, caption, x_omic, censor, survtime, grade

        if self.split == 'train':
            return (self.train_transforms(single_X_path), tokenized_captions , single_X_omic, single_e, single_t, single_g, index, sample_idx)
        elif self.split == 'test':
            return (self.test_transforms(single_X_path), tokenized_captions, single_X_omic, single_e, single_t, single_g, index, sample_idx)


    def __len__(self):
        return len(self.X_path)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def tokenize(tokenizer, texts):
    tokens = tokenizer.batch_encode_plus(texts, 
                                         max_length = 64,
                                         add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                         return_token_type_ids=False,
                                         truncation = True,
                                         padding = 'max_length',
                                         return_attention_mask=True)
    return tokens['input_ids'], tokens['attention_mask']