CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node 1 train_ddp.py \
  --features_dir data_argo/features/ \
  --train_batch_size 4 \
  --val_batch_size 4 \
  --val_interval 2 \
  --train_epoches 50 \
  --data_aug \
  --use_cuda \
  --logger_writer \
  --adv_cfg_path config.simpl_cfg