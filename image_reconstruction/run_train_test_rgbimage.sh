#!/bin/bash
GPU=0

export PYTHONUNBUFFERED=1

IDENTIFIER='my_rgb_recon_experiment'
LOGDIR='./log/'
CONFIG='../configs/config_RGB.yaml'


for IMAGE in 'astronaut.png'
do
  IDENTIFIER_S=${IDENTIFIER}'_'${IMAGE}
  python3 train_rgbimage.py --logdir $LOGDIR --config $CONFIG --image_id $IMAGE --gpu $GPU --identifier $IDENTIFIER_S
  python3 test_rgbimage.py --logdir $LOGDIR$IDENTIFIER_S --gpu $GPU --config $LOGDIR$IDENTIFIER_S'/config.yaml' --image_id $IMAGE

done