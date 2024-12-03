#!/bin/bash
GPU=0

LOGDIR='./log/'
#SEREIS='ablation_arch'
#SEREIS='ablation_pretraining'
SEREIS='rgb_sota'
#SEREIS='ablation_conditioning'
#SEREIS='ablation_bandlimit'
CONFIG_DIR='../configs/configs4paper/'${SEREIS}'/'
# create a dir named done to store the config files that have been run, if already exists skip
mkdir -p $CONFIG_DIR/done

for CONFIG in $CONFIG_DIR/*.yaml
do
  # Get the filename without the .yaml extension
  IDENTIFIER_S=$(basename $CONFIG .yaml)

  for i in {1..24} # 'sample_img_camera.png' | 'astronaut.png' | 'camera.png' | 'img_1200x1200_3x16bit_B01C00_RGB_baloons.png' |
  do
    padded_i=$(printf "%02d" $i)
    IMAGE='kodim'$padded_i'.png'
    CUTTENR_IDENTIFIER_S=${SEREIS}'/'${IDENTIFIER_S}'/'${IMAGE}
    python3 train_rgbimage.py --logdir $LOGDIR --config $CONFIG --image_id $IMAGE --gpu $GPU --identifier $CUTTENR_IDENTIFIER_S
    python3 test_rgbimage.py --logdir $LOGDIR$CUTTENR_IDENTIFIER_S --gpu $GPU --config $LOGDIR$CUTTENR_IDENTIFIER_S'/config.yaml' --image_id $IMAGE
#    python3 process_for_paper.py  --selected_examples $IMAGE
  done
  # move the done config file to the done directory if both scripts exited successfully
#  mv $CONFIG $CONFIG_DIR/done
done