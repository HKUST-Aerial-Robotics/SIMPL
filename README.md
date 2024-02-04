# SIMPL: A Simple and Efficient Multi-agent Motion Prediction Baseline for Autonomous Driving

## Introduction
This is the project page of the paper

* Lu Zhang, Peiliang Li, Sikang Liu, and Shaojie Shen, "SIMPL: A Simple and Efficient Multi-agent Motion Prediction Baseline for Autonomous Driving", 2023. (Corresponding author: Lu ZHANG, lzhangbz@connect.ust.hk)

* Notice: The code will be released after the publishing of this paper.

**Preprint:** Comming soon~

## Qualitative Results

* On Argoverse 1 motion forecasting dataset
<p align="center">
  <img src="files/av1-s1.png" width = "200"/>
  <img src="files/av1-s2.png" width = "200"/>
  <img src="files/av1-s3.png" width = "200"/>
  <img src="files/av1-s4.png" width = "200"/>
</p>

* On Argoverse 2 motion forecasting dataset
<p align="center">
  <img src="files/av2-s1.png" width = "200"/>
  <img src="files/av2-s2.png" width = "200"/>
  <img src="files/av2-s3.png" width = "200"/>
  <img src="files/av2-s4.png" width = "200"/>
</p>

----

## Gettting Started

### Install dependencies
- Create a new conda env
```
conda create --name simpl python=3.8
conda activate simpl
```

- Install PyTorch according to your CUDA version. We recommend CUDA >= 11.1, PyTorch >= 1.8.0.
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
```

- Install Argoverse 1 & 2 APIs, please follow this [page](https://github.com/argoai/argoverse-api) and [page](https://github.com/argoverse/av2-api).


- Install other dependencies
```
pip install scikit-image IPython tqdm ipdb tensorboard
```

### Play with pretrained models (Argoverse 1)
Generate a subset of the dataset for testing using the script. It will generate 1k samples to `data_argo/features/`:
```
sh scripts/argo_preproc_small.sh
```
The dataset directory should be organized as follows:
```
data_argo
├── features
│   ├── train
│   │   ├── 100001.pkl
│   │   ├── 100144.pkl
│   │   ├── 100189.pkl
...
│   └── val
│       ├── 10018.pkl
│       ├── 10080.pkl
│       ├── 10164.pkl
...
```

The pre-trained weights are located at `saved_models/`. Use the script below to visualize prediction results:
```
sh scripts/simpl_av1_vis.sh
```

Since we store each sequence as a single file, the system may raise error `OSError: [Erron 24] Too many open files` during evaluation and training. You may use the command below to solve this issue:
```
ulimit -SHn 51200
ulimit -s unlimited
```

To evaluate the trained models:
```
sh scripts/simpl_av1_eval.sh
```

## Todo List
- [ ] Release code for Argoverse 2 dataset
- [ ] Release training and evaluation scripts
- [x] First release

## Acknowledgement
We would like to express sincere thanks to the authors of the following packages and tools:
- [LaneGCN](https://github.com/uber-research/LaneGCN)
- [HiVT](https://github.com/ZikangZhou/HiVT)

## License
This repository is licensed under [MIT license](https://github.com/HKUST-Aerial-Robotics/SIMPL/blob/main/LICENSE).