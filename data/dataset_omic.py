'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''

import random

import torch
import numpy as np
import torch.utils.data as data

import yaml
from easydict import EasyDict

from utils.io import IO
from utils.build import DATASETS
from utils.logger import *
from utils.build import build_dataset_from_cfg
import json
from tqdm import tqdm
import pickle
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class Dataset_OMIC(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        self.generate_from_raw_data = False
        split = config.subset
        self.subset = config.subset

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        '''
        ###  shape_ids
        {'train': ['airplane_0001', 'airplane_0002', 'airplane_0003', 'airplane_0004', 'airplane_0005', 'airplane_0006', 'airplane_0007', 'airplane_0008', 'airplane_0009', ...], 'test': ['airplane_0627', 'airplane_0628', 'airplane_0629', 'airplane_0630', 'airplane_0631', 'airplane_0632', 'airplane_0633', 'airplane_0634', 'airplane_0635', ...]}

        len(shape_ids['train']) = 9843
        len(shape_ids['test']) = 2468
        '''

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]] # len(shape_names) = 2468
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print_log('The size of %s data is %d' % (split, len(self.datapath)), logger='ModelNet')

        if self.uniform: # enrer here
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
            # '/data/cxli/data/3d_point_cloud/modelnet40_normal_resampled/modelnet40_test_8192pts_fps.dat'
        else:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data: # true
            if not os.path.exists(self.save_path):
                # make sure you have raw data in the path before you enable generate_from_raw_data=True.
                if self.generate_from_raw_data:
                    print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ModelNet')
                    self.list_of_points = [None] * len(self.datapath)
                    self.list_of_labels = [None] * len(self.datapath)

                    for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                        fn = self.datapath[index]
                        cls = self.classes[self.datapath[index][0]]
                        cls = np.array([cls]).astype(np.int32)
                        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                        if self.uniform:
                            point_set = farthest_point_sample(point_set, self.npoints)
                            print_log("uniformly sampled out {} points".format(self.npoints))
                        else:
                            point_set = point_set[0:self.npoints, :]

                        self.list_of_points[index] = point_set
                        self.list_of_labels[index] = cls

                    with open(self.save_path, 'wb') as f:
                        pickle.dump([self.list_of_points, self.list_of_labels], f)
                else:
                    # no pre-processed dataset found and no raw data found, then load 8192 points dataset then do fps after.
                    self.save_path = os.path.join(self.root,
                                                  'modelnet%d_%s_%dpts_fps.dat' % (
                                                  self.num_category, split, 8192))
                    print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                    print_log('since no exact points pre-processed dataset found and no raw data found, load 8192 pointd dataset first, then do fps to {} after, the speed is excepted to be slower due to fps...'.format(self.npoints), logger='ModelNet')
                    with open(self.save_path, 'rb') as f:
                        self.list_of_points, self.list_of_labels = pickle.load(f)

            else:
                print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)
                    # len(self.list_of_points) = 2468, element.shape= (8192, 6)
                    # len(self.list_of_labels) = 2468, element: [label]

        self.shape_names_addr = os.path.join(self.root, 'modelnet40_shape_names.txt')
        with open(self.shape_names_addr) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        self.shape_names = lines

        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        if  self.npoints < point_set.shape[0]:
            point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.use_height: # false
            self.gravity_dim = 1
            height_array = point_set[:, self.gravity_dim:self.gravity_dim + 1] - point_set[:,
                                                                            self.gravity_dim:self.gravity_dim + 1].min()
            point_set = np.concatenate((point_set, height_array), axis=1)

        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        label_name = self.shape_names[int(label)]

        return current_points, label, label_name


import collections.abc as container_abcs
int_classes = int
# from torch._six import string_classes

import re
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')

def customized_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(batch, list):
        batch = [example for example in batch if example[4] is not None]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return customized_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: customized_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customized_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [customized_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict() 
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

# class Dataset_OMIC():
#     def __init__(self, args, tokenizer, dataset_type, train_transform=None):
#         # if dataset_type == 'train':
#         #     self.dataset_name = args.pretrain_dataset_name # shapenet
#         # elif dataset_type == 'val':
#         #     self.dataset_name = args.validate_dataset_name # modelnet40
#         # else:
#         #     raise ValueError("not supported dataset type.")
        
#         # with open('./data/dataset_catalog.json', 'r') as f:
#         #     self.dataset_catalog = json.load(f)
#         #     self.dataset_usage = self.dataset_catalog[self.dataset_name]['usage']
#         #     self.dataset_split = self.dataset_catalog[self.dataset_name][self.dataset_usage]
#         #     self.dataset_config_dir = self.dataset_catalog[self.dataset_name]['config']
        
#         '''
#         ### self.dataset_catalog 
#         {'shapenet': {'config': './data/ShapeNet-55.yaml', 'train': 'train', 'test': 'test', 'usage': 'train'}, 'modelnet40': {'config': './data/ModelNet40.yaml', 'train': 'train', 'test': 'test', 'usage': 'test'}}
        
#         ### self.dataset_usage      test
#         ### self.dataset_split      test
#         ### self.dataset_config_dir     ./data/ModelNet40.yaml
#         '''

#         self.tokenizer = tokenizer
#         self.train_transform = train_transform
#         # self.pretrain_dataset_prompt = args.pretrain_dataset_prompt # shapenet_64
#         # self.validate_dataset_prompt = args.validate_dataset_prompt # modelnet40_64  这里的64是指text_template有64种
#         self.build_omic_dataset(args, self.dataset_config_dir)

#     def build_omic_dataset(self, args, config):
#         config = cfg_from_yaml_file(config)
#         config.tokenizer = self.tokenizer
#         config.train_transform = self.train_transform
#         config.pretrain_dataset_prompt = self.pretrain_dataset_prompt
#         config.validate_dataset_prompt = self.validate_dataset_prompt
#         config.args = args
#         config.use_height = args.use_height
#         config.npoints = args.npoints
#         config_others = EasyDict({'subset': self.dataset_split, 'whole': True})
#         '''
#         {'NAME': 'ModelNet', 'DATA_PATH': '/data/cxli/data/3d_p..._resampled', 'NUM_CATEGORY': 40, 'USE_NORMALS': False, 'tokenizer': <utils.tokenizer.Sim...6c89c61d0>, 'train_transform': None, 'pretrain_dataset_prompt': 'shapenet_64', 'validate_dataset_prompt': 'modelnet40_64', 'args': Namespace(CE_grads=F...ld_size=1), 'use_height': False, 'npoints': 8192}
#         '''
#         self.dataset = build_dataset_from_cfg(config, config_others)
