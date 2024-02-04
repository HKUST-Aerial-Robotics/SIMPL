CUDA_VISIBLE_DEVICES=0 python evaluation.py \
  --features_dir data_argo/features/ \
  --train_batch_size 16 \
  --val_batch_size 16 \
  --use_cuda \
  --adv_cfg_path config.simpl_cfg \
  --model_path saved_models/simpl_av1_ckpt.tar