# Wholly-WOOD
## Introduction
This project is the [Jittor](https://github.com/Jittor/jittor) implementation of **Wholly-WOOD** (**Wholly** Leveraging Diversified-quality Labels for **W**eakly-supervised **O**riented **O**bject **D**etection). The code works with **Jittor 1.3.8.5**. It is modified from [JDet](https://github.com/Jittor/JDet), which is an object detection benchmark mainly focus on oriented object detection.

## Install
Environment requirements:

* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version >= 3.7
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
* GPU compiler (optional)
    * nvcc (>=10.0 for g++ or >=10.2 for clang)
* GPU library: cudnn-dev (recommend tar file installation, [reference link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar))

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
If you don't have permission for install,please add ```--user```.

Or use ```PYTHONPATH```: 
You can add ```export PYTHONPATH=$PYTHONPATH:{you_own_path}/JDet/python``` into ```.bashrc```, and run
```shell
source .bashrc
```

## Datasets
The following datasets are supported in JDet, please check the corresponding document before use. 

DOTA1.0/DOTA1.5/DOTA2.0 Dataset: [dota.md](docs/dota.md).

FAIR Dataset: [fair.md](docs/fair.md)

SSDD/SSDD+: [ssdd.md](docs/ssdd.md)

You can also build your own dataset by convert your datas to DOTA format.

## Modles
This repository contains the Wholly-WOOD model and our series work on weakly-supervised OOD (i.e. H2RBox, H2RBox-v2, and Point2RBox).

### Wholly-WOOD
We can train/test Wholly-WOOD model by:
```shell
python tools/run_net.py --config-file=configs/whollywood/whollywood_obb_r50_adamw_fpn_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/whollywood/whollywood_obb_r50_adamw_fpn_1x_dota.py --task=test
```

### H2RBox
We can train/test H2RBox model by:
```shell
python tools/run_net.py --config-file=configs/whollywood/h2rbox_obb_r50_adamw_fpn_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/whollywood/h2rbox_obb_r50_adamw_fpn_1x_dota.py --task=test
```

### H2RBox-v2
We can train/test H2RBox-v2 model by:
```shell
python tools/run_net.py --config-file=configs/whollywood/h2rbox_v2p_obb_r50_adamw_fpn_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/whollywood/h2rbox_v2p_obb_r50_adamw_fpn_1x_dota.py --task=test
```

### Point2RBox
We can train/test Point2RBox model by:
```shell
python tools/run_net.py --config-file=configs/whollywood/point2rbox_obb_r50_adamw_fpn_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/whollywood/point2rbox_obb_r50_adamw_fpn_1x_dota.py --task=test
```

## Visualization
You can test and visualize results on your own image sets by:
```shell
python tools/run_net.py --config-file=configs/whollywood/whollywood_obb_r50_adamw_fpn_1x_dota.py --task=vis_test
```
You can choose the visualization style you prefer, for more details about visualization, please refer to [visualization.md](docs/visualization.md).
<img src="https://github.com/Jittor/JDet/blob/visualization/docs/images/vis2.jpg?raw=true" alt="Visualization" width="800"/>

## Citation
Coming soon.
