echo "-- Processing AV2 val set..."
python data_av2/run_preprocess.py --mode val \
  --data_dir ~/data/dataset/argoverse2/val/ \
  --save_dir data_av2/features/ \
  --small
# --debug --viz

echo "-- Processing AV2 train set..."
python data_av2/run_preprocess.py --mode train \
  --data_dir ~/data/dataset/argoverse2/train/ \
  --save_dir data_av2/features/ \
  --small

# echo "-- Processing AV2 test set..."
# python data_av2/run_preprocess.py --mode test \
#   --data_dir ~/data/dataset/argoverse2/test/ \
#   --save_dir data_av2/features/ \
#   --small