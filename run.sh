bash ./scripts/test_pointbert.sh /data/cxli/data/3d_point_cloud/initialize_models/point_bert_pretrained.pt


bash ./scripts/test_pointmlp.sh /data/cxli/data/3d_point_cloud/initialize_models/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointmlp.pt



CUDA_VISIBLE_DEVICES=0 python main.py --model ULIP_PN_MLP --npoints 8192 --lr 1e-3 --output-dir ./outputs/reproduce_pointmlp_8kpts



CUDA_VISIBLE_DEVICES=0 python main.py --model ULIP_GENE_SNN --lr 1e-3 --output-dir ./outputs/gene_GBMLGG