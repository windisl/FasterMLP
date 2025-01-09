## 1. Dependency Setup
After setting up dependency for Image Classification, install the following packages
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmdet==2.25.0

## 2. Data preparation
Prepare COCO according to the guidelines in MMDetection v2.25.0.

## 3. Training
To train FasterMLP-S + Mask R-CNN on COCO train2017 on a single node with 8 GPUs for 12 epochs, run

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./dist_train.sh \
configs/fastermlp/mask_rcnn_fastermlp_s_fpn_1x_coco.py 8 \
--work-dir work_dirs/mask_rcnn_fastermlp_s_fpn_1x_coco/$(date +'%Y%m%d_%H%M%S') \
--cfg-options model.pretrained=../model_ckpt/fastermlp-epoch=283-val_acc1=72.9740.pth 
