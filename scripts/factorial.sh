#!/bin/bash
PY=~/anaconda3/envs/spectrum_seq/bin/python
DATA=/home/ubuntu/spectrum_pilot/data
CLIP=/home/ubuntu/spectrum_pilot/ViT-B-16.pt
OUT=/home/ubuntu/spectrum_pilot/results
cd /home/ubuntu/spectrum_pilot/code
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m spectrum_seq.train_v2 \
  --data $DATA --clip $CLIP --out $OUT --shots 5 --epochs 300 \
  --fuse injection --enc mlp > $OUT/innovation/abl_wl50mlp.log 2>&1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m spectrum_seq.train_v2 \
  --data $DATA --clip $CLIP --out $OUT --fuse injection --enc mlp \
  --eval_scene indianpines --eval_only_ckpt $OUT/adapter_v2_paviau_s5_inj_mlp_seed0.pt \
  > $OUT/innovation/zs_wl50mlp.log 2>&1
echo DONE > /tmp/factorial.flag
