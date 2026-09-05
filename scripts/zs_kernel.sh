#!/bin/bash
PY=~/anaconda3/envs/spectrum_seq/bin/python
DATA=/home/ubuntu/spectrum_pilot/data
CLIP=/home/ubuntu/spectrum_pilot/ViT-B-16.pt
OUT=/home/ubuntu/spectrum_pilot/results
cd /home/ubuntu/spectrum_pilot/code
for ENC in cnn mlp; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m spectrum_seq.train_v2 \
    --data $DATA --clip $CLIP --out $OUT --fuse injection --enc $ENC \
    --eval_scene indianpines --eval_only_ckpt $OUT/adapter_v2_paviau_s5_inj_${ENC}_seed0.pt \
    > $OUT/innovation/zsenc_${ENC}.log 2>&1
done
echo DONE > /tmp/zskernel.flag
