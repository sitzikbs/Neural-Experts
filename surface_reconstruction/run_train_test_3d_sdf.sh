#!/bin/bash
GPU=0

export PYTHONUNBUFFERED=1
IDENTIFIER='my_3dsdf_recon_experiment'
LOGDIR='./log/'
CONFIG='../configs/config_3D.yaml'
for SHAPE in 'gt_armadillo.xyz' 'gt_dragon.xyz' 'gt_lucy.xyz'  'gt_thai.xyz'
do
  IDENTIFIER_S=${IDENTIFIER}'_'${SHAPE}
  CUDA_VISIBLE_DEVICES=0 python3 train_3d_shape.py --logdir $LOGDIR --config $CONFIG --shape_id $SHAPE --gpu $GPU --identifier $IDENTIFIER_S
done