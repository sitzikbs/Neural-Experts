#!/bin/bash
GPU=0

export PYTHONUNBUFFERED=1


IDENTIFIER='sota_audio/Our_SIREN_MoE'
LOGDIR='./log/'
CONFIG='../configs/config_audio.yaml'

for AUDIOFILE in 'gt_bach.wav'
do
  IDENTIFIER_S=${IDENTIFIER}'_'${AUDIOFILE}
  python3 train_audio.py --logdir $LOGDIR --config $CONFIG --audiofile_id $AUDIOFILE --gpu $GPU --identifier $IDENTIFIER_S
  python3 test_audio.py --logdir $LOGDIR$IDENTIFIER_S --gpu $GPU --config $LOGDIR$IDENTIFIER_S'/config.yaml' --audiofile_id $AUDIOFILE
done