CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node 4 evaluation_ddp.py \
  --features_dir data_av2/features/ \
  --train_batch_size 16 \
  --val_batch_size 16 \
  --use_cuda \
  --adv_cfg_path config.simpl_av2_cfg \
  --model_path saved_models/simpl_av2_bezier_ckpt.tar