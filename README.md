# MonoSKD

<p align="center"> <img src='img/MonoSKD.png' align="center" height="350px"> </p>


## Abstract

Monocular 3D object detection is an inherently ill-posed problem, as it is challenging to predict accurate 3D localization from a single image. Existing monocular 3D detection knowledge distillation methods usually project the LiDAR onto the image plane and train the teacher network accordingly. Transferring LiDAR-based model knowledge to RGB-based models is more complex, so a general distillation strategy is needed. To alleviate cross-modal problem, we propose **MonoSKD**, a novel **K**nowledge **D**istillation framework for **Mono**cular 3D detection based on **S**pearman correlation coefficient, to learn the relative correlation between cross-modal features. Considering the large gap between these features, strict alignment of features may mislead the training, so we propose a looser Spearman loss. Furthermore, by selecting appropriate distillation locations and removing redundant modules, our scheme saves more GPU resources and trains faster than existing methods. Extensive experiments are performed to verify the effectiveness of our framework on the challenging KITTI 3D object detection benchmark. Our method achieves state-of-the-art performance until submission with no additional inference computational cost. Our code will be made public once accepted.

## Overview

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Pretrained Model](#pretrained-model)

## Installation

### Installation Steps

a. Clone this repository.

b. Install the dependent libraries as follows:

* Install the dependent python libraries: 
  
  ```shell
  pip install torch==1.12.0 torchvision==0.13.0 pyyaml scikit-image opencv-python numba tqdm torchsort
  ```

* We test this repository on Nvidia 3090 GPUs and Ubuntu 18.04. You can also follow the install instructions in [GUPNet](https://github.com/SuperMHP/GUPNet) (This respository is based on it) to perform experiments with lower PyTorch/GPU versions.

## Getting Started

### Dataset Preparation

* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows:

```
this repo
├── data
│   │── KITTI3D
|   │   │── training
|   │   │   ├──calib & label_2 & image_2 & depth_dense
|   │   │── testing
|   │   │   ├──calib & image_2
├── config
├── ...
```

* You can also choose to link your KITTI dataset path by
  
  ```
  KITTI_DATA_PATH=~/data/kitti_object
  ln -s $KITTI_DATA_PATH ./data/KITTI3D
  ```

* To ease the usage,  the pre-generated dense depth files at: [Google Drive](https://drive.google.com/file/d/1mlHtG8ZXLfjm0lSpUOXHulGF9fsthRtM/view?usp=sharing) 

### Training & Testing

#### Test and evaluate the pretrained models

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config config/monoskd.yaml -e   
```

#### Train a model

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_val.py --config configs/monoskd.yaml
```

## Pretrained Model

To ease the usage, we will provide the pre-trained model upon accepted.