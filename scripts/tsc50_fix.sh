#!/bin/bash
PY=~/anaconda3/envs/spectrum_seq/bin/python
DATA=/home/ubuntu/spectrum_pilot/data
CLIP=/home/ubuntu/spectrum_pilot/ViT-B-16.pt
OUT=/home/ubuntu/spectrum_pilot/results
cd /home/ubuntu/spectrum_pilot/code
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m spectrum_seq.train_v2 \
  --data $DATA --clip $CLIP --out $OUT --shots 50 --epochs 300 \
  --fuse prior --base_ids 0,1,2,3,5 --balanced --trans_w 1 --n_trans 120 \
  > $OUT/innovation/tsc50_beta1.log 2>&1
echo DONE > /tmp/tsc50.flag
