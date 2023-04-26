#!/usr/bin/env bash
EXP_NAME=$1   # 给本次运行起个名，然后会在checkpoints目录下新建一个以EXP_NAME的文件夹，
IMAGENET_PRETRAIN=data/voc/  #预训练模型的权重路径
SAVE_DIR=checkpoints/VOC/${EXP_NAME}

for repeat_id in 0 1 2 3 4 5 6 7 8 9
do
  python3 -m Faster_Rcnn.train_net --num-gpus 4 --config-file configs/voc/faster_rcnn_R_50_FPN.yaml  \
    --opts OUTPUT_DIR ${SAVE_DIR}/faster_rcnn_R_50_FPN_repeat${repeat_id}
done