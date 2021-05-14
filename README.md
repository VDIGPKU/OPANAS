## OPANAS: One-Shot Path Aggregation Network Architecture Search for Object Detection

Contact us with tingtingliang@pku.edu.cn, wyt@pku.edu.cn.


## Introduction

This project provides an implementation for our CVPR2021 paper "[OPANAS: One-Shot Path Aggregation Network Architecture Search for Object Detection](https://arxiv.org/abs/2103.04507)" on PyTorch.
The search code is coming soon.

## Citation

If you use our code/model/data, please cite our paper:
https://arxiv.org/abs/2103.04507
## License

**The project is only free for academic research purposes, but needs authorization for commerce. For commerce permission, please contact wyt@pku.edu.cn.**



## Models
Results on COCO.
Note that our Faster R-CNN uses smooth L1 loss following [the original paper](https://arxiv.org/abs/1506.01497).

| Method | Backbone | Lr schd | box AP (val)| box AP (test-dev)| Download |
| :----: | :------: | :-----: | :---------: | :--------------: | :------: |
| [Faster R-CNN](configs/opanas/faster_rcnn_r50_opa_fpn_112_sml1_coco.py) | R-50 |  1x  | 39.6 | 40.1| [model](https://drive.google.com/file/d/13PN01e30fbVDW218iFVVMdL56dKK3Da5/view?usp=sharing)  |
| [Cascade R-CNN](configs/opanas/cascade_rcnn_2r101_dcn_opa_fpn_160_2x_ms_coco.py) | R2-101 |  2x  | 51.8| 52.2| [model](https://drive.google.com/file/d/1DAXTFxgujajVTUzh9fkUjHxo672YCnz0/view?usp=sharing)  |

## Installation
Please refer to [install.md](docs/install.md) for installation and dataset preparation.
You need to install mmdetection (version 2.4.0 with mmcv 1.1.6) firstly.  More guidance can be found from [mmdeteion](https://github.com/open-mmlab/mmdetection).


## Getting Started
Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMDetection.
We use 8 GPUs (32GB V100) to train our detector, you can adjust the batch size in configs by yourselves.
* Train & Test
```shell

# Train
./tools/dist_train.sh configs/opanas/faster_rcnn_r50_opa_fpn_112_sml1_coco.py 8
./tools/dist_train.sh configs/opanas/cascade_rcnn_2r101_dcn_opa_fpn_160_2x_ms_coco.py 8

# Test
./tools/dist_test.sh configs/opanas/faster_rcnn_r50_opa_fpn_112_sml1_coco.py /path/to/your/save_dir/faster_opa_396.pth 8 --eval bbox
./tools/dist_test.sh configs/opanas/cascade_rcnn_2r101_dcn_opa_fpn_160_2x_ms_coco.py /path/to/your/save_dir/cascade_opa_522.pth 8 --eval bbox
```
    


## Acknowledgement

This repo is developed based on [mmdeteion](https://github.com/open-mmlab/mmdetection) and [SEPC](https://github.com/jshilong/SEPC). Please check mmdetection for more details and features.


