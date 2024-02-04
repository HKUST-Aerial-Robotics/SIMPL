CUDA_VISIBLE_DEVICES=0 python visualize.py \
  --features_dir data_argo/features/ \
  --mode val \
  --use_cuda \
  --model_path saved_models/simpl_av1_ckpt.tar \
  --adv_cfg_path config.simpl_cfg \
  --visualizer simpl.av1_visualizer:Visualizer \
  --seq_id -1