#!/bin/bash
# Innovation queue: waits for the full-sup chain, then runs the M4 tokenize
# ablation (T1-T5) and the TSC matrix automatically.
set -x
while [ ! -f /tmp/fullsup_ip.flag ]; do sleep 60; done
PY=~/anaconda3/envs/spectrum_seq/bin/python
DATA=/home/ubuntu/spectrum_pilot/data
CLIP=/home/ubuntu/spectrum_pilot/ViT-B-16.pt
OUT=/home/ubuntu/spectrum_pilot/results
mkdir -p $OUT/innovation
cd /home/ubuntu/spectrum_pilot/code

run_train () {  # args: tokenize, extra flags, logname
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m spectrum_seq.train_v2 \
    --data $DATA --clip $CLIP --out $OUT --shots 5 --epochs 300 \
    --fuse injection --tokenize $1 $2 > $OUT/innovation/$3.log 2>&1
}

# tokenize ablation (T1 = wl50 re-run for consistency)
for tok in wl50 wl25 wl100 bandindex bandeq9; do
  run_train $tok "" "abl_${tok}"
  CKPT=$OUT/adapter_v2_paviau_s5_inj_${tok}_seed0.pt
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m spectrum_seq.train_v2 \
    --data $DATA --clip $CLIP --out $OUT --fuse injection --tokenize $tok \
    --eval_scene indianpines --eval_only_ckpt $CKPT > $OUT/innovation/zs_${tok}.log 2>&1
done

# TSC matrix
for BETA in 0.5 1 2; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m spectrum_seq.train_v2 \
    --data $DATA --clip $CLIP --out $OUT --shots 5 --epochs 300 \
    --fuse prior --base_ids 0,1,2,3,5 --balanced --trans_w $BETA \
    > $OUT/innovation/tsc5_beta${BETA}.log 2>&1
done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m spectrum_seq.train_v2 \
  --data $DATA --clip $CLIP --out $OUT --shots 50 --epochs 300 \
  --fuse prior --base_ids 0,1,2,3,5 --balanced --trans_w 1 \
  > $OUT/innovation/tsc50_beta1.log 2>&1

# kernel-control ablation (fixed wl50 binning, vary the sequence kernel)
for ENC in cnn mlp; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m spectrum_seq.train_v2     --data $DATA --clip $CLIP --out $OUT --shots 5 --epochs 300     --fuse injection --enc $ENC > $OUT/innovation/enc_${ENC}.log 2>&1
done

echo DONE > /tmp/innov.flag
