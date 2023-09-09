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



CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_SNN_QuiltCLIP --lr 0.001 --output_dir ./outputs/MedVLM_gene_GBMLGG/text_tune/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --wandb --lr 0.005 --use_max_lr


CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_SNN_QuiltCLIP --lr 0.001 --output_dir ./outputs/MedVLM_gene_GBMLGG/text_tune/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence --use_text_prompt --wandb --lr 0.005 --use_max_lr

CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_SNN_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode sentence --use_text_prompt --wandb --lr 0.001 --use_max_lr --exp texttune


CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.005 --use_max_lr --exp text-tune_visual-adapter_geneformer-cont --wandb 



CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --use_max_lr --exp text-tune_visual-adapter_geneformer-cont_lr0.001 --wandb 




CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm dnabert --lr 0.005 --use_max_lr --exp text-tune_visual-adapter_danbert-cont --wandb


CUDA_VISIBLE_DEVICES=2 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm dnabert --lr 0.001 --use_max_lr --exp text-tune_visual-adapter_danbert-cont_lr0.001 --wandb 


CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm gpn --lr 0.005 --use_max_lr --exp text-tune_visual-adapter_gpn-cont_lr0.005 --wandb 


CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm gpn --lr 0.001 --use_max_lr --exp text-tune_visual-adapter_gpn-cont_lr0.001 --wandb 


'''
采用分层lr, 对 visual adapter & cls visual head 采用更高lr; 对text prompt, gene converter 采用正常lr
'''

# 探究pre-trained weights的意义...

CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10 --wandb 




# 用了pt_weights的SNN
CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm snn --pt_snn --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_pt-snn_-cont_lr-div-r10 --wandb 

# 没用pt_weights的SNN
CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm snn --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_van-snn_-cont_lr-div-r10 --wandb 


0829
似乎 用gene_former 做对比监督的效果最好..
后续操作方向; 
- 各个loss的权重调参;
- 是否需要 gene_cls_loss?


CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-1-0 --w_image_text 1 --w_image_omic 1 --w_omic_text 0 --wandb 

CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-1-0.5  --w_image_text 1 --w_image_omic 1 --w_omic_text 0.5 --wandb 

CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5  --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb 

## 1 0.5 0.5 效果好

K-shot

CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_k=1-shot --k_shot 1 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb 

CUDA_VISIBLE_DEVICES=2 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_k=2-shot --k_shot 2 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb 

CUDA_VISIBLE_DEVICES=3 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_k=4-shot --k_shot 4 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb 

CUDA_VISIBLE_DEVICES=4 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_k=8-shot --k_shot 8 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb 

CUDA_VISIBLE_DEVICES=5 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_k=16-shot --k_shot 16 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb 


only_vis_cls_head

CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_k=1-shot_only-vis-head --k_shot 1 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb --only_vis_cls_head 

CUDA_VISIBLE_DEVICES=2 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_k=2-shot_only-vis-head --k_shot 2 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb --only_vis_cls_head 

CUDA_VISIBLE_DEVICES=3 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_k=4-shot_only-vis-head --k_shot 4 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb --only_vis_cls_head 

CUDA_VISIBLE_DEVICES=4 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_k=8-shot_only-vis-head --k_shot 8 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb --only_vis_cls_head 

CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_k=16-shot_only-vis-head --k_shot 16 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb --only_vis_cls_head 


base2new
CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_base2new_cls=0 --base2new_class 0 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb 

CUDA_VISIBLE_DEVICES=2 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_base2new_cls=1 --base2new_class 1 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb 

CUDA_VISIBLE_DEVICES=3 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5_base2new_cls=2 --base2new_class 2 --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb 



### 0905

# th200 baseline
CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm.py --model ULIP_GENE_LM_QuiltCLIP --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp th200~text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5  --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --wandb 

# graph, very coarse version, with only node cls loss, and we infer without using graph (that means that the graph module is just used to affect the non-graph module in testing)
CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp th200~tr-gh-case-cls_te-nogh_text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5-1  --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --w_graph_cls 1 --wandb 

# repeat to watch graph_cls_loss
CUDA_VISIBLE_DEVICES=2 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp th200~tr-gh-case-cls_te-nogh_text-tune_visual-adapter_geneformer-cont_lr-div-r10_loss-w-1-0.5-0.5-1  --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --w_graph_cls 1 --wandb 

# 提高lr for gnn, as i see the slow growth of acc for graph_cls_acc
CUDA_VISIBLE_DEVICES=3 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp th200~tr-gh-case-cls_te-nogh_text-tune_visual-adapter_geneformer-cont_lr-div-r10-h4gh_loss-w-1-0.5-0.5-1  --w_image_text 1 --w_image_omic 0.5 --w_omic_text 0.5 --w_graph_cls 1 --wandb



CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp th200~tr-gh-case-cls_te-nogh_text-tune_visual-adapter_geneformer-cont_lr-div-r10-h4gh_loss-w-0-0-0-1  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb



CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp1~tr-gh-case-cls_te-nogh_text-tune_visual-adapter_geneformer-cont_lr-div-r10-h4gh_loss-w-0-0-0-1  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb


CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp1_fix-infer-bug  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb


CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp2  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb



CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp_res_clshd_fasttest  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb --fast_test


CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp_res_clshd_fasttest  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb --fast_test

CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp_res_clshd_fasttest_only_image_node  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb --fast_test


CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp_res_clshd_fasttest_only_image_node  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb --fast_test

CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp_res_clshd_fasttest_only_image_node  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb --fast_test



CUDA_VISIBLE_DEVICES=1 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp_res_clshd_fasttest_only_image_node  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb --fast_test


CUDA_VISIBLE_DEVICES=0 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp_only-image-node_heat4-v1_fasttest_dp0.0  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb --fast_test --workers 4


CUDA_VISIBLE_DEVICES=2 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp_only-image-node_heat4-v2_fasttest_dp0.0  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb --fast_test --workers 4


CUDA_VISIBLE_DEVICES=3 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp_all-node_heat4-v1_fasttest_dp0.0  --w_image_text 0 --w_image_omic 0 --w_omic_text 0 --w_graph_cls 1 --wandb --fast_test --workers 4

CUDA_VISIBLE_DEVICES=4 python main_omic_medvlm_graph.py --model ULIP_GENE_LM_QuiltCLIP_Graph --output_dir ./outputs/MedVLM_gene_GBMLGG/ --input_size_path 224 --train_bz 1072 --test_bz 512 --test_mode patch --tune_visual adapter --normalization data --text_mode description --use_text_prompt --gene_lm geneformer --lr 0.001 --high_lr_ratio 10 --exp temp_all-node_heat4-v1_fasttest_dp0.0_add_previous_loss  --w_image_text 1 --w_image_omic 1 --w_omic_text 1 --w_graph_cls 1 --wandb --fast_test --workers 4