'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * Changed from SLIP
 * https://github.com/facebookresearch/SLIP
 * By Le Xue
'''
import argparse
from collections import OrderedDict
import math
import time
import wandb

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import collections

import torch.nn.functional as F

from data.dataset_3d import *

from utils.utils import get_dataset
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from data.dataset_3d import customized_collate_fn

from data_loaders_MT import pathomic_dataset, pathomic_patches_dataset
# from options import parse_args
# from train_test_MT import train, test

from pdb import set_trace as st

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, recall_score, confusion_matrix

import open_clip

from transformers import AutoTokenizer

'''
CUDA_VISIBLE_DEVICES=0 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/gene_GBMLGG_vis_adapter --input_size_path 224 --use_visual_adapter --batch_size 64 

no_vis_tuning

CUDA_VISIBLE_DEVICES=2 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/gene_GBMLGG_0719/fix_vis --input_size_path 224 --train_bz 1072  --test_bz 1500 --test_mode patch --tune_visual none


vis_adapter

CUDA_VISIBLE_DEVICES=3 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/gene_GBMLGG_0719/vis_adapter --input_size_path 224 --train_bz 1072  --test_bz 1500 --test_mode patch --tune_visual adapter


vis_prompt

# deep_prompt
CUDA_VISIBLE_DEVICES=0 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/gene_GBMLGG_0719/vis_prompt --input_size_path 224 --train_bz 192  --test_bz 1500 --test_mode patch --tune_visual prompt 

# shallow_promot
CUDA_VISIBLE_DEVICES=1 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/gene_GBMLGG_0719/vis_shallow_prompt --input_size_path 224 --train_bz 192  --test_bz 1500 --test_mode patch --tune_visual prompt 


fine-tune
CUDA_VISIBLE_DEVICES=1 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/gene_GBMLGG_0719/vis_fine-tune --input_size_path 224 --train_bz 1024  --test_bz 1500 --test_mode patch --tune_visual fine-tune 


BioMedCLIP
CUDA_VISIBLE_DEVICES=0 python main_omic.py --model ULIP_GENE_SNN_BiomedCLIP --lr 1e-3 --output_dir ./outputs/gene_GBMLGG_0719/fix_vis_biomedclip --input_size_path 224 --train_bz 96 --test_bz 1500 --test_mode patch --tune_visual none




CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=8 main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/gene_GBMLGG_0719/vis_prompt --input_size_path 224 --train_bz 1072  --test_bz 1500 --test_mode patch --tune_visual prompt 


>> 0729
CUDA_VISIBLE_DEVICES=0 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/gene_GBMLGG/fix_gene_ft_vis --input_size_path 224 --train_bz 128 --test_bz 256 --test_mode patch --tune_visual fine-tune --normalization data --text_mode sentence --fix_gene

CUDA_VISIBLE_DEVICES=1 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/gene_GBMLGG/fix_gene_vis_adapter --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence --fix_gene



'''


def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP training and evaluation', add_help=False)
    # Data
    parser.add_argument_group('ULIP')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='shapenet', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--validate_dataset_name', default='modelnet40', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--use_height', action='store_true', help='whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    # Model
    parser.add_argument('--model', default='ULIP_PN_SSG', type=str)
    # Training
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    # parser.add_argument('--batch-size', default=64, type=int,
    #                     help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')

    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    
    # parser.add_argument('--evaluate_3d', action='store_true', help='eval 3d only')
    parser.add_argument('--evaluate', action='store_true')

    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')

    parser.add_argument('--test_ckpt_addr', default='', help='the ckpt to test 3d zero shot')


    parser.add_argument_group('OMIC')
    parser.add_argument('--tSVD_mode', type=str, default="path", help="[path, omic, pathomic]")
    parser.add_argument('--tSVD_loss', type=str, default="False")
    parser.add_argument('--n_views', type=int, default=4, help='number of views for tSVD constraint')
    parser.add_argument('--Lambda_global', type=float, default=0.05, metavar='N',
                    help='the trade-off parameter of losses')
    parser.add_argument('--mu', type=float, default=1e-5, metavar='N',
                        help='the scalar mu')
    parser.add_argument('--max_mu', type=float, default=1, metavar='N',
                        help='the maximum of mu')
    parser.add_argument('--pho', type=float, default=1.1, metavar='N',
                        help='the scalar pho')
    parser.add_argument('--aux_iter', type=int, default=1, metavar='N',
                        help='when to update auxiliary variable')
    parser.add_argument('--proto_beta', type=float, default=0.5, metavar='N',
                        help='moving weight for updating the prototypes')

    parser.add_argument('--orth_loss', type=str, default="False", 
        help='whether to regularize the multi-modal feature to be orthogonal.')
    parser.add_argument('--student_customize', type=str, default="False", 
        help='whether mask the KD loss according to the similarity of KD gradient and CE loss gradient.')
    parser.add_argument('--assign_weights', type=str, default="False", 
        help='whether assign weights to different KD losses according to the gradient similarity.')
    parser.add_argument('--distill', type=str, default='kd', 
                        choices=['kd', 'feats_KL', 'hint', 'attention', 'similarity','correlation', 'vid', 
                        'crd', 'kdsvd', 'fsp', 'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--kd_T', type=float, default=1, help='temperature for KD distillation')
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')
    
    parser.add_argument('--cut_fuse_grad', default=False, action="store_true", 
        help='whether cut the gradients from the fuse branch to single modality.')
    parser.add_argument('--select_pos_mode', type=str, default='random', help='the rule to select positive pairs for CRD')
    parser.add_argument('--select_pos_pairs', default=True, action="store_true", 
        help='whether to select positive pairs for the CRD loss.')
    parser.add_argument('--select_neg_pairs', type=str, default="True", 
        help='whether to select negative pairs for the CRD loss.')
    parser.add_argument('--CE_grads', default=False, action="store_true", 
        help='whether use the gradients of CE loss to assign teacher weights.')
    # parser.add_argument('--fixed_model', type=str, default='pathomic_self_MT_KD', help='mode')
    parser.add_argument('--fixed_model', type=str, default='1023_pathomic_MT', help='mode')
    # parser.add_argument('--fixed_model', type=str, default='0322_pofusion_path_omic_4views_tsvd_lam0.1', help='mode')

    parser.add_argument('--svm_norm', default=False, action="store_true", help='if use norm when compute with svm')
    parser.add_argument('--grad_place', type=str,  default='feat', help='where to compare gradients.')

    ### The method to transform the omic data. xxh, 2022/01/01
    parser.add_argument('--omic_transform', type=str,  default='drop', help='[drop, vime]')

    ### whether return feature gradients. xxh, 2021/12/16
    parser.add_argument('--return_grad', type=str,  default='False', help='whether to return gradients.')

    ### for knowledge distillation
    parser.add_argument('--start_KD', type=int, default=10, help='which epoch start to employ KD in the model')
    parser.add_argument('--pred_distill', type=int, default=1, help='whether to use pred KD loss')
    parser.add_argument('--num_teachers', type=int, default=1, help='number of teacher for each single modality')
    parser.add_argument('--KD_weight', type=float, default=1.0, help='the weight for the KD loss')
    parser.add_argument('--KD_type', type=str,  default='KD', help='[KD, CRD, CRD_KD].')
    parser.add_argument('--sample_KD', type=str,  default='False', help='whether to select some samples for KD.')
    parser.add_argument('--global_step', type=int,  default=0, help='global_step')
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument('--consistency_rampup', type=float,  default=10, help='consistency_rampup')
    parser.add_argument('--which_teacher', type=str,  default='fuse', 
        help='[fuse, self_EMA], when num_teachers=1, choose one teacher for distillation')

    ### for CRD loss
    parser.add_argument('--CRD_distill', type=int, default=1, help='whether use the CRD loss')
    parser.add_argument('--CRD_mode', type=str, default="sup", choices=['sup', 'unsup'])
    parser.add_argument('--CRD_weight', type=float, default=0.1, help='the weight for the SP loss')

    parser.add_argument('--s_dim', type=int, default=128, help='feature dim of the student model')
    parser.add_argument('--t_dim', type=int, default=128, help='feature dim of the EMA teacher')
    parser.add_argument('--feat_dim', type=int, default=128, help='reduced feature dimension')
    parser.add_argument('--pos_mode', default='multi_pos', type=str, choices=['exact', 'relax', 'multi_pos'])
    parser.add_argument('--nce_p', default=300, type=int, help='number of positive samples for NCE')
    parser.add_argument('--nce_p2', default=10, type=int, help='number of positive samples for NCE')
    parser.add_argument('--nce_k', default=700, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_k2', default=512, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--n_data', default=1024, type=int, help='number of training samples')

    ### for similarity-preserving (SP) loss
    parser.add_argument('--SP_distill', type=int, default=0, help='whether to use SP loss')
    parser.add_argument('--SP_weight', type=float, default=1.0, help='the weight for the SP loss')

    ### for supervised contrastive loss
    parser.add_argument('--supcon_distill', type=int, default=0, help='whether to use supcon loss')
    parser.add_argument('--supcon_weight', type=float, default=1.0, help='the weight for the supcon loss')

    ### common params
    parser.add_argument('--dataroot', default='/data/cxli/code/MultiModal-learning/MICCAI-2022/data/TCGA_GBMLGG/', help="datasets")
    # parser.add_argument('--dataroot', default='/data/cxli/data/TCGA_GBMLGG', help="datasets")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/TCGA_GBMLGG', help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='grad_15', help='name of the project. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--mode', type=str, default='pathomic', help='mode')
    parser.add_argument('--model_name', type=str, default='omic', help='mode')

    # 0712 
    parser.add_argument('--use_vgg_features', type=int, default=0, help='Use pretrained embeddings')
    # parser.add_argument('--use_vgg_features', type=int, default=1, help='Use pretrained embeddings')
    
    parser.add_argument('--use_rnaseq', type=int, default=0, help='Use RNAseq data.')
    
    parser.add_argument('--task', type=str, default='grad', help='surv | grad')
    parser.add_argument('--useRNA', type=int, default=0) # Doesn't work at the moment...:(
    parser.add_argument('--useSN', type=int, default=1)
    parser.add_argument('--act_type', type=str, default='LSM', help='activation function')
    parser.add_argument('--input_size_omic', type=int, default=80, help="input_size for omic vector")

    # parser.add_argument('--input_size_path', type=int, default=512, help="input_size for path images")
    parser.add_argument('--input_size_path', type=int, default=512, help="input_size for path images")

    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--save_at', type=int, default=20, help="adsfasdf")
    parser.add_argument('--label_dim', type=int, default=3, help='size of output')
    parser.add_argument('--measure', default=1, type=int, help='disables measure while training (make program faster)')
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--print_every', default=0, type=int)

    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--beta2', type=float, default=0.999, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--lr_policy', default='linear', type=str, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--lr_decay_iters', default=10, type=int, help='decay lr after 20 epochs')
    parser.add_argument('--finetune', default=1, type=int, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--final_lr', default=0.1, type=float, help='Used for AdaBound')
    parser.add_argument('--reg_type', default='omic', type=str, help="regularization type")
    parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=30, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--epoch_count', type=int, default=1, help='start of epoch')

    parser.add_argument('--train_bz', type=int, default=256)
    parser.add_argument('--test_bz', type=int, default=256)

    parser.add_argument('--lambda_cox', type=float, default=1)
    parser.add_argument('--lambda_reg', type=float, default=3e-4)
    parser.add_argument('--lambda_nll', type=float, default=1)

    parser.add_argument('--fusion_type', type=str, default="pofusion", help='concat|pofusion|LMF|HFB|GPDBN|mmdynamics')
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--use_bilinear', type=int, default=1)
    parser.add_argument('--path_gate', type=int, default=1)
    parser.add_argument('--omic_gate', type=int, default=1)
    parser.add_argument('--path_dim', type=int, default=128)
    parser.add_argument('--omic_dim', type=int, default=128)
    parser.add_argument('--path_scale', type=int, default=1)
    parser.add_argument('--omic_scale', type=int, default=1)
    parser.add_argument('--mmhid', type=int, default=128)

    parser.add_argument('--init_type', type=str, default='max', help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='0 - 0.25. Increasing dropout_rate helps overfitting. Some people have gone as high as 0.5. You can try adding more regularization')
    parser.add_argument('--use_edges', default=1, type=float, help='Using edge_attr')
    parser.add_argument('--pooling_ratio', default=0.2, type=float, help='pooling ratio for SAGPOOl')

    '''
    Temporary comments to address conflicting
    '''
    # parser.add_argument('--lr', default=0.0005, type=float, help='5e-4 for Adam | 1e-3 for AdaBound')

    parser.add_argument('--weight_decay', default=4e-4, type=float, help='Used for Adam. L2 Regularization on weights. I normally turn this off if I am using L1. You should try')
    parser.add_argument('--GNN', default='GCN', type=str, help='GCN | GAT | SAG. graph conv mode for pooling')
    parser.add_argument('--patience', default=0.005, type=float)

    parser.add_argument('--test_mode', type=str, default='full', choices=['full', 'patch'])

    parser.add_argument('--tune_visual', type=str, default='none', choices=['none', 'fine-tune','adapter', 'shallow_prompt', 'deep_prompt'])

    parser.add_argument('--normalization', type=str, default='data', choices=['clip', 'biomedclip', 'data'])

    parser.add_argument('--ori_biomedclip', action = 'store_true')

    parser.add_argument('--text_mode', type=str, default='sentence', choices=['sentence', 'description'])

    parser.add_argument('--fix_gene', action='store_true', default=False)

    

    return parser

best_acc1 = 0

def main(args):


    utils.init_distributed_mode(args)

    global best_acc1

    if utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='ULIP', id=wandb_id, config=args, reinit=True, entity='lxue')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.evaluate:
        zero_stats = test_zeroshot_pathomic(args)
        print(zero_stats)
        return
    
    if 'biomed' in args.model.lower():

        tokenizer =  open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224') 
        biomedclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    # elif 'mizero' in args.model.lower():
    #     args.model_checkpoint = "/data/cxli/code/MI-Zero/src/checkpoint/ctranspath_448_bioclinicalbert/checkpoints/epoch_50.pt"
    #     args.model_name = args.model_checkpoint.split('/')[-3]
    #     tokenizer = load_pretrained_tokenizer(args.model_name)
        # pass
    else:
        tokenizer = SimpleTokenizer()
    args.tokenizer = tokenizer

    # create model
    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(args=args)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200, find_unused_parameters=False)


    # define loss function (criterion) and optimizer
    criterion = models.get_loss(args).cuda(args.gpu)

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            print('in optimizer freeze {}'.format(n))
            continue  # frozen weights
        print('update parameters {}'.format(n)) 
        # 原始版本 ~ 观察到只更新 point_encoder部分 
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1']
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from the latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    cudnn.benchmark = True


    # Data loading code
    print("=> creating dataset")



    '''
    OMIC
    '''
    if not os.path.exists(args.checkpoints_dir): os.makedirs(args.checkpoints_dir)
    if not os.path.exists(os.path.join(args.checkpoints_dir, args.exp_name)): os.makedirs(os.path.join(args.checkpoints_dir, args.exp_name))
    if not os.path.exists(os.path.join(args.checkpoints_dir, args.exp_name, args.model_name)): os.makedirs(os.path.join(args.checkpoints_dir, args.exp_name, args.model_name))

    ### Initiate Data
    ignore_missing_histype = 1 if 'grad' in args.task else 0  # opt.task = 'grad'
    ignore_missing_moltype = 1 if 'omic' in args.mode else 0  # opt.mode = 'pathomic'

    use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if args.use_vgg_features else ('_', 'all_st')
    use_rnaseq = '_rnaseq' if args.use_rnaseq else ''

    # data_cv_path = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
    data_cv_path = '%s/splits_5cv_2022/gbmlgg5cv_%s_%d_%d_%d%s.pkl' % (args.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, args.use_vgg_features, use_rnaseq)
    print("Loading %s" % data_cv_path)
    # './data/TCGA_GBMLGG/splits_5cv_2022/gbmlgg5cv_all_st_1_1_0.pkl'
    data_cv = pickle.load(open(data_cv_path, 'rb'))
    data_cv_splits = data_cv['cv_splits']

    results, results_path, results_omic = [], [], []
    rocauc_fuse_all, ap_fuse_all, f1_micro_fuse_all, f1_gradeIV_fuse_all = [], [], [], []
    rocauc_path_all, ap_path_all, f1_micro_path_all, f1_gradeIV_path_all = [], [], [], []
    rocauc_omic_all, ap_omic_all, f1_micro_omic_all, f1_gradeIV_omic_all = [], [], [], []

    ### 读取裁剪之后的每张ROI对应的9个patches.
    roi_dir = 'all_st_patches_512'
    # data_cv_path_patches = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
    data_cv_path_patches = '%s/splits_5cv_2022/gbmlgg5cv_%s_%d_%d_%d%s.pkl' % (args.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, args.use_vgg_features, use_rnaseq)
    # './data/TCGA_GBMLGG/splits_5cv_2022/gbmlgg5cv_all_st_patches_512_1_1_0.pkl'
    # 512, 1, 1, 0
    print("Loading %s" % data_cv_path_patches)
    data_cv_patches = pickle.load(open(data_cv_path_patches, 'rb'))
    data_cv_splits_patches = data_cv_patches['cv_splits']
    # 每个split的训练集和测试集

    k = 1
    data = data_cv_splits[1] # 先只取单折, 进行debug调通


    train_dataset, test_dataset, n_data = pathomic_dataset(args, data) 
    # len(train_dataset) = 1072, len(test_dataset) = 253
    data_patches = data_cv_splits_patches[k]
    test_patches_dataset = pathomic_patches_dataset(args, data_patches)
    # len(test_patches_dataset) = 2277


    '''
    Temporary comments for debug
    '''
    if args.test_mode == 'full':
        val_dataset = test_dataset
    elif args.test_mode == 'patch' or args.test_mode == 'patches':
        val_dataset = test_patches_dataset
    else:
        raise ValueError('Invalid test mode')
    
    # val_dataset = test_patches_dataset # 在patches上测试
    # val_dataset = test_dataset


    # 似乎是 训练的前面阶段用whole image 测试, 后期再用 patches测试
    # 这里我们暂时先用whole image; 先调通再说


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) 
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    # len = 1072
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.train_bz, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
    #     collate_fn=customized_collate_fn)

    # len = 2277
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.test_bz, shuffle=(val_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    '''
    0719 train_loader shuffle=True,    val_loader shuffle=False
    '''
    # len = 1072
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_bz, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        collate_fn=customized_collate_fn)

    # len = 2277
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_bz, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
        # len(val_loader) = 5

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    print(args)

    print("=> beginning training")

    best_epoch = -1

    best_ap = 0
    best_auc = 0

    for epoch in range(args.start_epoch, args.epochs):

        '''
        Temp debug
        '''
        val_stats = test_zeroshot_pathomic_core(val_loader, model, tokenizer, args)


        '''
        END
        '''

        
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)
        val_stats = {"acc1": -1}

        if epoch % 1 == 0:

            val_stats = test_zeroshot_pathomic_core(val_loader, model, tokenizer, args)

            main_modality = ['omic', 'path', 'mm'][2]

            print(val_stats)

            acc1 = val_stats[main_modality]["acc1"]
            ap = val_stats[main_modality]["ap"]
            auc = val_stats[main_modality]["rocauc"]

            is_best = acc1 > best_acc1
            if is_best:
                best_epoch = epoch
                best_ap = ap
                best_auc = auc

            best_acc1 = max(acc1, best_acc1)

            if is_best or epoch % 50 == 0:
                print("=> saving checkpoint")
                utils.save_on_master({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_acc1': best_acc1,
                        'args': args,
                    }, is_best, args.output_dir)

            if epoch + 1 == args.epochs:
                print("=> saving last checkpoint")
                utils.save_on_master({
                    'epoch': 'last',
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_acc1': best_acc1,
                    'args': args,
                }, is_best, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'best_acc1': best_acc1,
                     'ap': best_ap,
                     'auc': best_auc,
                     'best_epoch': best_epoch,
                     }

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
                # wandb.watch(model)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]

        # (x_path, x_grph, x_omic, censor, survtime, grade, index, sample_idx) = inputs

        '''
        TimeStamp
        '''

        # pc = inputs[3] # (8092,3)train
        # texts = inputs[2] # 从大类里找个 Only类别名 文本

        # image = inputs[4] #[3,224,224]
        (x_path, x_texts, x_omic, censor, survtime, grade, index, sample_idx) = inputs 

        inputs = [x_omic, x_texts, x_path]

        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        # Temporary debug
        # print('\n ######## Debug for backward ##########\n')
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)


        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step

        '''
        Temporary comments for debug
        '''

        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale to [0, 100]

        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052) # 4.3307  |  4.4696
        logit_scale = utils.get_model(model).logit_scale.exp().item() # 76.0 | 87

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.train_bz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                        'scaler': scaler.get_scale(),
                        'logit': logit_scale})
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def test_zeroshot_pathomic_core(test_loader, model, tokenizer, args=None):

    import torch
    from PIL import Image

    from lavis.models import load_model_and_preprocess
    from lavis.processors import load_processor

    # import timm
    # from tokenizer import SimpleTokenizer

    batch_time = AverageMeter('Time', ':6.3f')
    
    omic_top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    omic_rocauc = AverageMeter('ROCAUC', ':6.2f')
    omic_ap = AverageMeter('AP', ':6.2f')

    path_top1 = AverageMeter('Acc@1', ':6.2f')
    path_rocauc = AverageMeter('ROCAUC', ':6.2f')
    path_ap = AverageMeter('AP', ':6.2f')

    mm_top1 = AverageMeter('Acc@1', ':6.2f')
    mm_rocauc = AverageMeter('ROCAUC', ':6.2f')
    mm_ap = AverageMeter('AP', ':6.2f')

    # 512,512 -> 224, 224  ==> 3*3
    if args.test_mode == 'full':
        sw_ratio = 1024//224 + 1
    elif args.test_mode == 'patch' or args.test_mode == 'patches':
        sw_ratio = 512//224 + 1

    progress = ProgressMeter(
    len(test_loader)*sw_ratio * sw_ratio,
    [batch_time, omic_top1],
    prefix='Test: ')


    # progress = ProgressMeter(
    #     len(test_loader),
    #     [batch_time, top1, top5],
    #     prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    # print('=> encoding captions')
    # with open(os.path.join("./data", 'templates.json')) as f:
    #     templates = json.load(f)[args.validate_dataset_prompt]

    # templates
    # ['a point cloud model of {}.', 'There is a {} in the scene.', 'There is the {} in the scene.', 'a photo of a {} in the scene.', 'a photo of the {} in...the scene.', 'a photo of one {} in...the scene.', 'itap of a {}.', 'itap of my {}.', 'itap of the {}.', 'a photo of a {}.', 'a photo of my {}.', 'a photo of the {}.', 'a photo of one {}.', 'a photo of many {}.', ...]

    # with open(os.path.join("./data", 'labels.json')) as f:
    #     labels = json.load(f)[args.validate_dataset_name]

    # labels
    # ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', ...]


    # grading_name = {0: 'II', 1: 'III', 2: 'IV'}
    # # caption = f'A pathology slide with WHO grading {grading_name[ int(single_g) ]} '
    # caption = f'WHO grading {grading_name[ int(single_g) ]}'

    # labels = ['II', 'III', 'IV']
    # if args.text_mode == 'sentence':
    #     templates = ['A pathology slide with grade {} gliomas'] 
    # if args.text_mode == 'description':

        # caption_candidate = {
        #     'A pathology slide with grade II gliomas':
        #     [
        #     "The cells tend to be relatively uniform in size and shape, and they may be arranged in a pattern that resembles the normal organization of tissue.",
        #     "The cells have a relatively low rate of division (mitotic rate) and may be surrounded by normal brain tissue.",
        #     "The tumor may have a well-defined border between the tumor and the surrounding tissue.",
        #     'A pathology slide with grade II gliomas'
        #     ],
        #     'A pathology slide with grade III gliomas':
        #     [
        #     "The cells tend to be more variable in size and shape, and they may show signs of abnormal division (mitotic figures).",
        #     "The cells may be arranged in a more irregular pattern and may infiltrate the surrounding brain tissue.",
        #     "There may be areas of dead tissue (necrosis) within the tumor.",
        #     'A pathology slide with grade III gliomas'
        #     ],
        #     'A pathology slide with grade IV gliomas':
        #     [
        #     'The cells tend to be highly abnormal in appearance and may be very variable in size and shape, with large, irregular nuclei.',
        #     'There may be a high degree of mitotic activity, with many cells dividing rapidly.',
        #     'The tumor may have a very irregular border and may infiltrate extensively into the surrounding tissue.',
        #     'There may be areas of necrosis within the tumor.',
        #     'A pathology slide with grade IV gliomas'
        #     ]
        # }

        # caption_candidate = {
        #     'A pathology slide with grade II gliomas':
        #     [
        #     'Infiltrative growth pattern',
        #     'Relatively uniform cells with round or oval nuclei and minimal pleomorphism',
        #     'Low mitotic activity',
        #     'Absence of microvascular proliferation',
        #     'Absence of necrosis',
        #     # 'A pathology slide with grade II gliomas'
        #     ],

        #     'A pathology slide with grade III gliomas':
        #     [
        #     # "Increased cellularity compared to grade II gliomas",
        #     # "Mild to moderate nuclear atypia and pleomorphism.",
        #     # "Higher mitotic activity compared to grade II gliomas.",
        #     # "Absence or minimal microvascular proliferation.",
        #     # "Absence or focal necrosis.",
        #     "Increased cellularity",
        #     "Mild to moderate nuclear atypia and pleomorphism.",
        #     "Higher mitotic activity.",
        #     "Absence or minimal microvascular proliferation.",
        #     "Absence or focal necrosis.",
        #     # 'A pathology slide with grade III gliomas'
        #     ],

        #     'A pathology slide with grade IV gliomas':
        #     [
        #     "Highly cellular and pleomorphic tumor cells",
        #     "Marked nuclear atypia and pleomorphism.",
        #     "High mitotic activity",
        #     "Prominent microvascular proliferation",
        #     "Presence of necrosis, often with pseudopalisading pattern (tumor cells surrounding necrotic areas).",
        #     # 'A pathology slide with grade IV gliomas'
        #     ]
        # }

    
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)


    captions = []
    captions.append('ecrosis')
    captions.append('microvascular proliferation')
    captions.append('mitotic activity')
    # captions.append('nuclear atypia and pleomorphism')
    captions.append('Relatively uniform cells with round or oval nuclei and minimal pleomorphism') # inverse
    # captions.append('Highly cellular and pleomorphic tumor cells')
    captions.append('Infiltrative growth pattern') # inverse


    with torch.no_grad():
        # text_features = []
        # for id, l in enumerate(labels):
        #     try:
        #         templates = list(caption_candidate.values())[id]
        #     except:
        #         pass

        #     texts = [t.format(l) for t in templates]
        #     # if 'mizero' in args.model.lower():
        #     #     texts, attention_mask = tokenize(tokenizer, texts) 
        #     #     texts = torch.from_numpy(np.array(texts)).cuda(args.gpu, non_blocking=True)
        #     #     attention_mask = torch.from_numpy(np.array(attention_mask)).cuda()
        #     #     class_embeddings = utils.get_model(model).encode_text(texts, attention_mask=attention_mask)
        #     # else: 
        #     texts = tokenizer(texts).cuda(args.gpu, non_blocking=True) # torch.Size([1, 77]) # [3,77]
        #     if len(texts.shape) < 2:
        #         texts = texts[None, ...]

        #     class_embeddings = utils.get_model(model).encode_text(texts) # 调用SLIP () | biomedCLIP (512)

        #     class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        #     class_embeddings = class_embeddings.mean(dim=0)
        #     class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        #     text_features.append(class_embeddings)
        # text_features = torch.stack(text_features, dim=0)  # 一样的  # 调用SLIP (3,512) | biomedCLIP (3,512)
        texts = []
        for ind_text in range(len(captions)):
            text = text_processors['eval'](captions[ind_text])
            texts.append(text)

        end = time.time()
        per_class_stats = collections.defaultdict(int)

        omic_per_class_correct_top1 = collections.defaultdict(int)
        path_per_class_correct_top1 = collections.defaultdict(int)
        mm_per_class_correct_top1 = collections.defaultdict(int)
        
        # per_class_correct_top5 = collections.defaultdict(int)

        for i, inputs in enumerate(test_loader):
            # (pc, target, target_name)
            x_path_full, x_text , x_omic, single_e, single_t, single_g = inputs

            # target_name = x_text
            target = single_g

            target_name = []
        
            for g in single_g:
                grade = ['II', 'III', 'IV' ]
                name = f'Grade {grade}'
                per_class_stats[name] += 1
                target_name.append(name)

            # pc = pc.cuda(args.gpu, non_blocking=True)

            sw_cnt = 0
            temp_record = 0
            for h in range(0, x_path_full.shape[-2], args.input_size_path):
                for w in range(0, x_path_full.shape[-1], args.input_size_path):

                    if h+args.input_size_path <= x_path_full.shape[-2]:
                        h_start = h
                        h_end = h+args.input_size_path
                    else:
                        h_start = x_path_full.shape[-2] - args.input_size_path
                        h_end = x_path_full.shape[-2]
                    
                    if w+args.input_size_path <= x_path_full.shape[-1]:
                        w_start = w
                        w_end = w+args.input_size_path
                    else:
                        w_start = x_path_full.shape[-1] - args.input_size_path
                        w_end = x_path_full.shape[-1]
                        
                    
                    x_path = x_path_full[:, :, h_start:h_end, w_start:w_end]
                    temp_record += x_path.sum()

                    x_path = x_path.cuda(args.gpu, non_blocking=True)

                    # encode pathology
                    if sw_cnt == 0:
                        path_features = utils.get_model(model).encode_image(x_path)
                        path_features = path_features / path_features.norm(dim=-1, keepdim=True)
                    else:
                        _path_features = utils.get_model(model).encode_image(x_path)
                        _path_features = _path_features / _path_features.norm(dim=-1, keepdim=True)

                        path_features += _path_features
                    sw_cnt += 1
                
            path_features = path_features /sw_cnt # 调用SLIP () | biomedCLIP (512,512)
            logits_per_path=  path_features @ text_features.t()


            x_omic = x_omic.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # # encode pc
            # pc_features = utils.get_model(model).encode_pc(pc)
            # pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)


            # encode geneomic
            
            omic_features = utils.get_model(model).encode_omic(x_omic)
            omic_features = omic_features / omic_features.norm(dim=-1, keepdim=True)
            logits_per_omic=  omic_features @ text_features.t()
            

            
            # logits_per_omic.mean()    -0.0023   -0.0023  0.0183  0.0330 0.0431
            # logits_per_path.mean()    0.1201   0.1197 0.1201  0.1199 0.1197
            # target.mean()     282  282 282  282 282

            # path_features.sum()  296.0240   284.3953  
            # omic_features.sum()  1202.4694   1160.4327 
            # text_features.sum()  4.0695  4.0695  

            # x_path.sum()  18715120   19095696  19076868
            # x_omic.sum()  -642.9651  -642.9651  -642.9651  

            # x_path


            # logits_per_omic.mean()   -7.4722
            # logits_per_path.sum()   369.9125
            # path_features.sum()  296.0240
            # omic_features.sum()  1202.4694
            # text_features.sum()  4.0695
            # x_path.sum() 79574928  
            # x_omic.sum() -3282.3945
            # temp_record   tensor(7.1498e+08)




            # measure accuracy and record loss
            # (acc1, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 5))
            # # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            # acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            # top1.update(acc1.item(), pc.size(0))
            # top5.update(acc5.item(), pc.size(0))

            # TODO: 增加图像 / 多模态的测试结果...............................................
            # 五类以内, 所以acc5 不work
            # (acc1, acc5), correct = accuracy(logits_per_omic, target, topk=(1, 5))
            # acc1, acc5 = utils.scaled_all_reduce([acc1, acc5]).....................................................................................................................................................
            # top1.update(acc1.item(), x_omic.size(0))................
            # top5.update(acc5.item(), x_omic.size(0)).

            
            omic_acc1, omic_correct = accuracy(logits_per_omic, target)
            omic_top1.update(omic_acc1[0].item(), x_omic.size(0))
            omic_rocauc.update(roc_auc_score(np.eye(3)[target.cpu().numpy()], F.softmax(logits_per_omic,-1).cpu().numpy(), average ='micro')*100)
            omic_ap.update(average_precision_score(np.eye(3)[target.cpu().numpy()], F.softmax(logits_per_omic,-1).cpu().numpy(), average='micro')*100)
            


            path_acc1, path_correct = accuracy(logits_per_path, target) # 加不加softmax都一样
            path_top1.update(path_acc1[0].item(), x_path.size(0))

            path_rocauc.update(roc_auc_score(np.eye(3)[target.cpu().numpy()], F.softmax(logits_per_path,-1).cpu().numpy(), average ='micro')*100) # 加完softmax会发生一定下降 (规律不明..)

            path_ap.update(average_precision_score(np.eye(3)[target.cpu().numpy()], F.softmax(logits_per_path,-1).cpu().numpy(), average='micro')*100) # 加完softmax会发生一定下降 (规律不明..)

            
            prob_mm = (F.softmax(logits_per_omic,-1) + F.softmax(logits_per_path, -1)) / 2
            mm_acc1, mm_correct = accuracy(prob_mm, target)
            mm_top1.update(mm_acc1[0].item(), x_omic.size(0))
            mm_rocauc.update(roc_auc_score(np.eye(3)[target.cpu().numpy()], prob_mm.cpu().numpy(), average ='micro')*100)
            mm_ap.update(average_precision_score(np.eye(3)[target.cpu().numpy()], prob_mm.cpu().numpy(), average='micro')*100)
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            
            if logits_per_omic.shape[0] > 1:
                omic_top1_accurate = omic_correct[:1].squeeze()
                path_top1_accurate = path_correct[:1].squeeze()
                mm_top1_accurate = mm_correct[:1].squeeze()
            else:
                omic_top1_accurate = omic_correct[:1][0]
                path_top1_accurate = path_correct[:1][0]
                mm_top1_accurate = mm_correct[:1][0]

            # top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()

            for idx, name in enumerate(target_name):
                if omic_top1_accurate[idx].item():
                    omic_per_class_correct_top1[name] += 1
                if path_top1_accurate[idx].item():
                    path_per_class_correct_top1[name] += 1
                if mm_top1_accurate[idx].item():
                    mm_per_class_correct_top1[name] += 1
                # if top5_accurate[idx].item():
                #     per_class_correct_top5[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)
            
        omic_top1_accuracy_per_class = {}
        path_top1_accuracy_per_class = {}
        mm_top1_accuracy_per_class = {}
        # top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            omic_top1_accuracy_per_class[name] = omic_per_class_correct_top1[name] / per_class_stats[name]
            path_top1_accuracy_per_class[name] = path_per_class_correct_top1[name] / per_class_stats[name]
            mm_top1_accuracy_per_class[name] = mm_per_class_correct_top1[name] / per_class_stats[name]
            # top5_accuracy_per_class[name] = per_class_correct_top5[name] / per_class_stats[name]

        omic_top1_accuracy_per_class = collections.OrderedDict(omic_top1_accuracy_per_class)
        path_top1_accuracy_per_class = collections.OrderedDict(path_top1_accuracy_per_class)
        mm_top1_accuracy_per_class = collections.OrderedDict(mm_top1_accuracy_per_class)
        # top5_accuracy_per_class = collections.OrderedDict(top5_accuracy_per_class)
        print('[omic]')
        print(','.join(omic_top1_accuracy_per_class.keys()))
        print(','.join([str(value) for value in omic_top1_accuracy_per_class.values()]))

        print('[path]')
        print(','.join(path_top1_accuracy_per_class.keys()))
        print(','.join([str(value) for value in path_top1_accuracy_per_class.values()]))

        print('[mm]')
        print(','.join(mm_top1_accuracy_per_class.keys()))
        print(','.join([str(value) for value in mm_top1_accuracy_per_class.values()]))

        # print(','.join([str(value) for value in top5_accuracy_per_class.values()]))

    progress.synchronize()
    # print('0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    # return {'acc1': top1.avg, 'acc5': top5.avg}

    print(f'0-shot *[omic] Acc@1 {omic_top1.avg:.3f} | AP: {omic_ap.avg:.3f} | ROCAUC: {omic_rocauc.avg:.3f}')
    print(f'0-shot *[path] Acc@1 {path_top1.avg:.3f} | AP: {path_ap.avg:.3f} | ROCAUC: {path_rocauc.avg:.3f}')
    print(f'0-shot *[mm] Acc@1 {mm_top1.avg:.3f} | AP: {mm_ap.avg:.3f} | ROCAUC: {mm_rocauc.avg:.3f}')
    return{
            'omic': {'acc1': omic_top1.avg, 'ap': omic_ap.avg, 'rocauc': omic_rocauc.avg}, 'path': {'acc1': path_top1.avg, 'ap': path_ap.avg, 'rocauc': path_rocauc.avg}, 'mm': {'acc1': mm_top1.avg, 'ap': mm_ap.avg, 'rocauc': mm_rocauc.avg}
        }

'''
ZERO-SHOT classification
'''
def test_zeroshot_pathomic(args):
    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    try: # enter here
        model = getattr(models, old_args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    except:
        model = getattr(models, args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))

    # tokenizer = SimpleTokenizer()

    if 'biomed' in args.model.lower():

        tokenizer =  open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224') 
        biomedclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    elif 'mizero' in args.model.lower():
        args.model_checkpoint = "/data/cxli/code/MI-Zero/src/checkpoint/ctranspath_448_bioclinicalbert/checkpoints/epoch_50.pt"
        args.model_name = args.model_checkpoint.split('/')[-3]
        tokenizer = load_pretrained_tokenizer(args.model_name)
        # pass
    else:
        tokenizer = SimpleTokenizer()


    # test_dataset = get_dataset(None, tokenizer, args, 'val')
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False

    if not os.path.exists(args.checkpoints_dir): os.makedirs(args.checkpoints_dir)
    if not os.path.exists(os.path.join(args.checkpoints_dir, args.exp_name)): os.makedirs(os.path.join(args.checkpoints_dir, args.exp_name))
    if not os.path.exists(os.path.join(args.checkpoints_dir, args.exp_name, args.model_name)): os.makedirs(os.path.join(args.checkpoints_dir, args.exp_name, args.model_name))

    ### Initiate Data
    ignore_missing_histype = 1 if 'grad' in args.task else 0  # opt.task = 'grad'
    ignore_missing_moltype = 1 if 'omic' in args.mode else 0  # opt.mode = 'pathomic'

    use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if args.use_vgg_features else ('_', 'all_st')
    use_rnaseq = '_rnaseq' if args.use_rnaseq else ''

    # data_cv_path = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
    data_cv_path = '%s/splits_5cv_2022/gbmlgg5cv_%s_%d_%d_%d%s.pkl' % (args.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, args.use_vgg_features, use_rnaseq)
    print("Loading %s" % data_cv_path)
    # './data/TCGA_GBMLGG/splits_5cv_2022/gbmlgg5cv_all_st_1_1_0.pkl'
    data_cv = pickle.load(open(data_cv_path, 'rb'))
    data_cv_splits = data_cv['cv_splits']

    ### 读取裁剪之后的每张ROI对应的9个patches.
    roi_dir = 'all_st_patches_512'
    # data_cv_path_patches = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
    data_cv_path_patches = '%s/splits_5cv_2022/gbmlgg5cv_%s_%d_%d_%d%s.pkl' % (args.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, args.use_vgg_features, use_rnaseq)
    # './data/TCGA_GBMLGG/splits_5cv_2022/gbmlgg5cv_all_st_patches_512_1_1_0.pkl'
    # 512, 1, 1, 0
    print("Loading %s" % data_cv_path_patches)
    data_cv_patches = pickle.load(open(data_cv_path_patches, 'rb'))
    data_cv_splits_patches = data_cv_patches['cv_splits']
    # 每个split的训练集和测试集

    k = 1
    data = data_cv_splits[1] # 先只取单折, 进行debug调通
    
    args.tokenizer = tokenizer

    train_dataset, test_dataset, n_data = pathomic_dataset(args, data) 
    # len(train_dataset) = 1072, len(test_dataset) = 253
    data_patches = data_cv_splits_patches[k]
    test_patches_dataset = pathomic_patches_dataset(args, data_patches)
    # len(test_patches_dataset) = 

    # val_dataset = test_patches_dataset # 在patches上测试
    if args.test_mode == 'full':
        val_dataset = test_dataset
    elif args.test_mode == 'patch' or args.test_mode == 'patches':
        val_dataset = test_patches_dataset
    else:
        raise ValueError('Invalid test mode')


    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_bz, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )
    results = test_zeroshot_pathomic_core(test_loader, model, tokenizer, args)

    return results


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def tokenize(tokenizer, texts):
    tokens = tokenizer.batch_encode_plus(texts, 
                                         max_length = 64,
                                         add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                         return_token_type_ids=False,
                                         truncation = True,
                                         padding = 'max_length',
                                         return_attention_mask=True)
    return tokens['input_ids'], tokens['attention_mask']

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


def compute_accuracy(preds, labels, probs_all=None, grad_acc_test=None):
    """
    Compute the grading accuracy and return the predicted probs.
    """
    grade_pred = preds.argmax(dim=1, keepdim=True)
    grad_acc_test += grade_pred.eq(labels.view_as(grade_pred)).sum().item()
    probs_np = preds.detach().cpu().numpy()
    probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)

    return grad_acc_test, probs_all

def load_pretrained_tokenizer(model_name):
    if 'clinicalbert' in model_name:
        model_name = 'emilyalsentzer/Bio_ClinicalBERT'
        tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
    elif 'pubmed' in model_name:
        model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
    else:
        raise NotImplementedError
    
    return tokenizer



if __name__ == '__main__':
    parser = argparse.ArgumentParser('ULIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
