#!/bin/bash
GPU=0

export PYTHONUNBUFFERED=1
IDENTIFIER='pt_seg'
LOGDIR='./log/'
CONFIG='../configs/config_3D_managerpretraining.yaml'
for SHAPE in 'gt_armadillo.xyz' #'gt_dragon.xyz' 'gt_lucy.xyz'  'gt_thai.xyz'
do
  IDENTIFIER_S=${IDENTIFIER}'_'${SHAPE}
  CUDA_VISIBLE_DEVICES=0 python3 train_3d_shape.py --logdir $LOGDIR --config $CONFIG --shape_id $SHAPE --gpu $GPU --identifier $IDENTIFIER_S
done