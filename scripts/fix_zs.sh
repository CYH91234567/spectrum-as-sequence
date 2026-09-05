#!/bin/bash
PY=~/anaconda3/envs/spectrum_seq/bin/python
DATA=/home/ubuntu/spectrum_pilot/data
CLIP=/home/ubuntu/spectrum_pilot/ViT-B-16.pt
OUT=/home/ubuntu/spectrum_pilot/results
cd /home/ubuntu/spectrum_pilot/code
run_zs () {  # ckpt_name tokenize
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m spectrum_seq.train_v2 \
    --data $DATA --clip $CLIP --out $OUT --fuse injection --tokenize $2 \
    --eval_scene indianpines --eval_only_ckpt $OUT/$1 > $OUT/innovation/zs_$2.log 2>&1
}
run_zs adapter_v2_paviau_s5_inj_seed0.pt wl50
run_zs adapter_v2_paviau_s5_inj_wl25_seed0.pt wl25
run_zs adapter_v2_paviau_s5_inj_wl100_seed0.pt wl100
run_zs adapter_v2_paviau_s5_inj_bandeq9_seed0.pt bandeq9
run_zs adapter_v2_paviau_s5_inj_bandindex_seed0.pt bandindex
echo DONE > /tmp/zsfix.flag
