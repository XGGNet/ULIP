bash ./scripts/test_pointbert.sh /data/cxli/data/3d_point_cloud/initialize_models/point_bert_pretrained.pt


bash ./scripts/test_pointmlp.sh /data/cxli/data/3d_point_cloud/initialize_models/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointmlp.pt



CUDA_VISIBLE_DEVICES=0 python main.py --model ULIP_PN_MLP --npoints 8192 --lr 1e-3 --output-dir ./outputs/reproduce_pointmlp_8kpts


CUDA_VISIBLE_DEVICES=0 python main.py --model ULIP_GENE_SNN --lr 1e-3 --output-dir ./outputs/gene_GBMLGG --pre_train_dataset_name modelnet40 --pretrain_dataset_prompt modelnet40_64


CUDA_VISIBLE_DEVICES=0 python main.py --model ULIP_GENE_SNN --lr 1e-3 --output-dir ./outputs/gene_GBMLGG




CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_SNN_QuiltCLIP --lr 0.001 --output_dir ./outputs/MedVLM_gene_GBMLGG/text_tune/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt

CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm.py --model ULIP_GENE_SNN_QuiltCLIP --lr 0.001 --output_dir ./outputs/MedVLM_gene_GBMLGG/text_tune/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual none --normalization data --text_mode description --use_text_prompt --exp wo_visual_tune



CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm.py --model ULIP_GENE_SNN_QuiltCLIP --lr 0.001 --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual none --normalization data --text_mode description --exp vis_cls_head_baseline --wandb --epochs 100 --only_vis_cls_head

CUDA_VISIBLE_DEVICES=2 python main_omic_medvlm.py --model ULIP_GENE_SNN_QuiltCLIP --lr 0.001 --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual none --normalization data --text_mode description --exp vis_cls_head_baseline --wandb --epochs 100 --only_vis_cls_head --lr 0.005

UDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_SNN_QuiltCLIP --lr 0.001 --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual none --normalization data --text_mode description --exp vis_cls_head_baseline --wandb --epochs 100 --only_vis_cls_head --lr 0.01