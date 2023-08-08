
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

>>0803

CUDA_VISIBLE_DEVICES=0 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/0803/gene_GBMLGG_fulldata/vis_adapter --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence 

CUDA_VISIBLE_DEVICES=1 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/0803/gene_GBMLGG_fulldata/vis_adapter_fixgene --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence --fix_gene 

CUDA_VISIBLE_DEVICES=2 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/0803/gene_GBMLGG_fulldata/vis_adapter_wogene --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence --wo_gene 


CUDA_VISIBLE_DEVICES=3 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/0803/gene_GBMLGG_0.5data/vis_adapter --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence --used_train_data_ratio 0.5

CUDA_VISIBLE_DEVICES=4 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/0803/gene_GBMLGG_0.5data/vis_adapter_fixgene --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence --fix_gene --used_train_data_ratio 0.5

CUDA_VISIBLE_DEVICES=5 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/0803/gene_GBMLGG_0.5data/vis_adapter_wogene --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence --wo_gene --used_train_data_ratio 0.5



CUDA_VISIBLE_DEVICES=3 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/0803/gene_GBMLGG_0.5data_label_and_pair/vis_adapter --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence


CUDA_VISIBLE_DEVICES=4 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/0803/gene_GBMLGG_0.5data_label_and_pair/vis_adapter_wogene --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence --wo_gene 




CUDA_VISIBLE_DEVICES=0 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/0803/gene_GBMLGG_0.5data/vis_adapter --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence --used_train_data_ratio 0.1

CUDA_VISIBLE_DEVICES=1 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/0803/gene_GBMLGG_0.5data/vis_adapter_fixgene --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence --fix_gene --used_train_data_ratio 0.1

CUDA_VISIBLE_DEVICES=2 python main_omic.py --model ULIP_GENE_SNN --lr 1e-3 --output_dir ./outputs/0803/gene_GBMLGG_0.5data/vis_adapter_wogene --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence --wo_gene --used_train_data_ratio 0.1

