# Wholly-WOOD
## Introduction
We develop **Wholly-WOOD** (**Wholly** Leveraging Diversified-quality Labels for **W**eakly-supervised **O**riented **O**bject **D**etection), a weakly-supervised OOD framework, capable of wholly leveraging various labeling forms (Points, HBoxes, RBoxes, and their combination) in a unified fashion. By only using HBox for training, our Wholly-WOOD achieves performance very close to that of the RBox-trained counterpart on remote sensing and other areas, which significantly reduces the tedious efforts on labor-intensive annotation for oriented objects.

This project is the [Jittor](https://github.com/Jittor/jittor) implementation of Wholly-WOOD. The code works with **Jittor 1.3.8.5**. It is modified from [JDet](https://github.com/Jittor/JDet), which is an object detection benchmark mainly focus on oriented object detection. PyTorch version: [Wholly-WOOD (PyTorch)](https://github.com/yuyi1005/whollywood).

## Models
This repository contains the Wholly-WOOD model and our series of work on weakly-supervised OOD (i.e. H2RBox, H2RBox-v2, and Point2RBox).

### 1. Wholly-WOOD
We can train/test Wholly-WOOD model by:
```shell
python tools/run_net.py --config-file=configs/whollywood/whollywood_obb_r50_adamw_fpn_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/whollywood/whollywood_obb_r50_adamw_fpn_1x_dota.py --task=test
```

### 2. H2RBox
We can train/test H2RBox model by:
```shell
python tools/run_net.py --config-file=configs/whollywood/h2rbox_obb_r50_adamw_fpn_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/whollywood/h2rbox_obb_r50_adamw_fpn_1x_dota.py --task=test
```

### 3. H2RBox-v2
We can train/test H2RBox-v2 model by:
```shell
python tools/run_net.py --config-file=configs/whollywood/h2rbox_v2p_obb_r50_adamw_fpn_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/whollywood/h2rbox_v2p_obb_r50_adamw_fpn_1x_dota.py --task=test
```

### 4. Point2RBox
We can train/test Point2RBox model by:
```shell
python tools/run_net.py --config-file=configs/whollywood/point2rbox_obb_r50_adamw_fpn_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/whollywood/point2rbox_obb_r50_adamw_fpn_1x_dota.py --task=test
```

## Installation
Recommended environments:

* System: **Linux** (e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python == 3.10
* Jittor == 1.3.8.5
* CPU Compiler: g++ == 11.4.0
* GPU Library: cuda == 12.3 & cudnn == 8.9.7.29

**Step 1: Install requirements**
```shell
git clone https://github.com/yuyi1005/whollywood-jittor
cd whollywood-jittor
python -m pip install -r requirements.txt
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

**Step 2: Install Wholly-WOOD**
 
```shell
cd whollywood-jittor
# suggest this 
python setup.py develop
# or
python setup.py install
```
If you don't have permission for install, please add ```--user```.

## Datasets
The following datasets are supported in JDet, please check the corresponding document before use. 

DOTA1.0/DOTA1.5/DOTA2.0 Dataset: [dota.md](docs/dota.md).

FAIR Dataset: [fair.md](docs/fair.md)

SSDD/SSDD+: [ssdd.md](docs/ssdd.md)

You can also build your own dataset by convert your datas to DOTA format.

## Visualization
You can test and visualize results on your own image sets by:
```shell
python tools/run_net.py --config-file=configs/whollywood/whollywood_obb_r50_adamw_fpn_1x_dota.py --task=vis_test
```
You can choose the visualization style you prefer, for more details about visualization, please refer to [visualization.md](docs/visualization.md).
<img src="https://github.com/Jittor/JDet/blob/visualization/docs/images/vis2.jpg?raw=true" alt="Visualization" width="800"/>

## Citation
```
@article{yu2025whollywood,
  title={Wholly-WOOD: Wholly Leveraging Diversified-quality Labels for Weakly-supervised Oriented Object Detection}, 
  author={Yi Yu and Xue Yang and Yansheng Li and Zhenjun Han and Feipeng Da and Junchi Yan},
  year={2025},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
}
```
