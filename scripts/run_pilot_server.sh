#!/bin/bash
# Pilot v1 full pipeline on the 3090Ti server
set -e
PY=~/anaconda3/envs/spectrum_seq/bin/python
DATA=/home/ubuntu/spectrum_pilot/data
CLIP=/home/ubuntu/spectrum_pilot/ViT-B-16.pt
OUT=/home/ubuntu/spectrum_pilot/results
cd "$(dirname "$0")/.."
$PY -m spectrum_seq.train_pilot --data $DATA --clip $CLIP --out $OUT --epochs 300 --shots 5
$PY -m spectrum_seq.rgb_baseline --scene paviau --data $DATA --clip $CLIP --out $OUT
$PY -m spectrum_seq.rgb_baseline --scene indianpines --data $DATA --clip $CLIP --out $OUT
$PY -m spectrum_seq.zero_shot_ip --data $DATA --clip $CLIP --ckpt $OUT/adapter_paviau_s5_seed0_ctx1.pt --out $OUT
