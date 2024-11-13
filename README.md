# RGM-YOLO

## Introduction
Underwater object detection remains a challenging task due to the presence of noise, lighting variations, and occlusions in underwater images. To address these challenges, we proposes an improved underwater object detection model based on YOLOv9, integrating advanced attention mechanisms and a dilated large kernel algorithm. Specifically, the model incorporates a Residual Attention Block (RAB) to enhance local feature extraction and denoising capabilities. Additionally, a content-guided hybrid multi-attention fusion module is designed to improve contextual awareness and target focus. Finally, a dilated large kernel network, GLSKNet, is employed to dynamically adjust the receptive field, making the model more suitable for detecting underwater objects of varying sizes, particularly small and blurred targets.

Experimental results on the RUOD and DUO datasets demonstrate that our model outperforms several state-of-the-art models, achieving impressive mAPs (mean Average Precision) of 88.8\% and 89.7\% on the RUOD and DUO datasets, respectively. These findings underscore the effectiveness of attention mechanisms and the dilated large kernel algorithm in enhancing underwater object detection performance. 

![image](https://github.com/down-with-me/RGM-YOLO/blob/main/The%20flowchart%20of%20RGM-YOLO..jpg)

## Requirements
### Dependencies
* Python>=3.10.14
* pytorch>=2.0.1
* matplotlib>=3.2.2
* numpy>=1.18.5
* opencv-python>=4.1.1
* Pillow>=7.1.2
* PyYAML>=5.3.1
* requests>=2.23.0
* scipy>=1.4.1
* thop>=0.1.1
* torch>=1.7.0
* torchvision>=0.8.1
* tqdm>=4.64.0
* pandas>=1.1.4
* seaborn>=0.11.0
* albumentations>=1.0.3
* pycocotools>=2.0


### It was ran in following environment:
* NVIDIA 3090 GPU
* Intel(R) Core(TM) i7-14700K
* Ubuntu 22.04.4 LTS

### Easily install the environment with the following command:
```
pip install -r requirements.txt
```

## Preparing Data
1. To build dataset, you'll also need following datasets.
* [RUOD](https://github.com/dlut-dimt/RUOD)
* [DUO](https://osf.io/4bja7/)


2. Structure of the generated data should be：
```
├── datasets
    ├──Images
    │  ├── train
    │  │   ├── 000001.jpg
    │  │   ├── 000002.jpg
    │  │   └── ...
    │  ├── val
    │  │   ├── 000001.jpg
    │  │   ├── 000002.jpg
    │  │   └── ...
    │  ├── test
    │  │   ├── 000001.jpg
    │  │   ├── 000002.jpg
    │  │   └── ...
    ├──Labels
    │  ├── train
    │  │   ├── 000001.jpg
    │  │   ├── 000002.jpg
    │  │   └── ...
    │  │
    │  ├── val
    │  │   ├── 000001.jpg
    │  │   ├── 000002.jpg
    │  │   └── ...
    │  └── test
    │      ├── 000001.jpg
    │      ├── 000002.jpg
    │      └── ...
    │
    └──data.yaml
```

### data.yaml can be written as follows:
```
# Path
train: ./datasets/images/train  
val: ./datasets/images/val 
test: ./datasets/images/test 

# Classes
names:
  0: holothurian
  1: echinus
  2: scallop
  3: starfish
  4: fish
  5: corals
  6: diver
  7: cuttlefish
  8: turtle
  9: jellyfish
```

## Pre-trained Model
You can get the pre-trained model based on the YOLOv9 from <a  href="https://github.com/WongKinYiu/yolov9">here</a>.


## Getting Started:
### Usage

* Training

```
python train_dual.py --workers 8 --device 0 --batch 8 --data data/pp_fall/voc.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15 --evolve
```

* Validation
```
python val_dual.py --data data/pp_fall/voc.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights './yolov9-c.pt' --save-json --name yolov9_c_640_val
```

* Evaluation
```
python detect_dual.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './yolov9-c-converted.pt' --name yolov9_c_c_640_detect
```


