#!/bin/bash
# Review checklist queue: seeds, TSC parts, factorial cell, budget points, paired eval, spatial split
PY=~/anaconda3/envs/spectrum_seq/bin/python
DATA=/home/ubuntu/spectrum_pilot/data
CLIP=/home/ubuntu/spectrum_pilot/ViT-B-16.pt
OUT=/home/ubuntu/spectrum_pilot/results
cd /home/ubuntu/spectrum_pilot/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

tr () {  # tr <logname> <args...>
  local L="$1"; shift
  $PY -u -m spectrum_seq.train_v2 --data $DATA --clip $CLIP --out $OUT "$@" > $OUT/innovation/$L.log 2>&1
}
evalzs () {  # evalzs <logname> <ckpt> <extra args...>
  $PY -u -m spectrum_seq.train_v2 --data $DATA --clip $CLIP --out $OUT \
    --fuse injection --eval_only_ckpt $OUT/$2 "${@:3}" > $OUT/innovation/$1.log 2>&1
}

# A. multi-seed closed-set mains
for SEED in 1 2; do
  tr "seedA_pavia_s$SEED"  --shots 5 --fuse injection --seed $SEED
  tr "seedA_ip_s$SEED"     --shots 5 --fuse injection --seed $SEED --train_scene indianpines --eval_scene indianpines
done

# B. multi-seed tokenize (Pavia) + zs
for SEED in 1 2; do
  for TOK in wl100 wl25 bandeq9 bandindex; do
    tr "seedB_${TOK}_s$SEED" --shots 5 --fuse injection --tokenize $TOK --seed $SEED
    evalzs "seedBzs_${TOK}_s$SEED" adapter_v2_paviau_s5_inj_${TOK}_seed$SEED.pt --eval_scene indianpines --tokenize $TOK
  done
done

# C. multi-seed kernel (Pavia) + zs
for SEED in 1 2; do
  for ENC in cnn mlp; do
    tr "seedC_${ENC}_s$SEED" --shots 5 --fuse injection --enc $ENC --seed $SEED
    evalzs "seedCzs_${ENC}_s$SEED" adapter_v2_paviau_s5_inj_${ENC}_seed$SEED.pt --eval_scene indianpines --enc $ENC
  done
done

# D. multi-seed TSC
for SEED in 1 2; do
  for BETA in 0.5 1.0 2.0; do
    tr "seedD_tsc${BETA}_s$SEED" --shots 5 --epochs 300 --fuse prior \
      --base_ids 0,1,2,3,5 --balanced --trans_w $BETA --n_trans 120 --seed $SEED
  done
done

# E. factorial completion: bandindex x mlp
tr "factorial_bandindex_mlp" --shots 5 --fuse injection --tokenize bandindex --enc mlp
evalzs "factorialzs_bandindex_mlp" adapter_v2_paviau_s5_inj_bandindex_mlp_seed0.pt --eval_scene indianpines --tokenize bandindex --enc mlp

# F. TSC component ablation + random graph control (beta=2.0)
tr "tsablation_cons"  --shots 5 --epochs 300 --fuse prior --base_ids 0,1,2,3,5 --balanced --trans_w 2.0 --n_trans 120 --tsc_parts cons
tr "tsablation_marg"  --shots 5 --epochs 300 --fuse prior --base_ids 0,1,2,3,5 --balanced --trans_w 2.0 --n_trans 120 --tsc_parts marg
tr "tsablation_randg" --shots 5 --epochs 300 --fuse prior --base_ids 0,1,2,3,5 --balanced --trans_w 2.0 --n_trans 120 --tsc_random_graph

# G. injected budget points k in {15,50}, both scenes
for K in 15 50; do
  tr "budget_pavia_k$K" --shots $K --fuse injection
  tr "budget_ip_k$K"    --shots $K --fuse injection --train_scene indianpines --eval_scene indianpines
done

# H. injected + balanced paired eval (Tier-1 transduction disclosure)
evalzs "paired_pavia_bal" adapter_v2_paviau_s5_inj_seed0.pt --eval_scene paviau --balanced

# I. spatial checkerboard full-sup ceilings
tr "spatial_pavia" --shots 10000 --epochs 60 --fuse injection --lr 1e-3 --spatial_block 20
tr "spatial_ip"    --shots 10000 --epochs 60 --fuse injection --lr 1e-3 --spatial_block 20 --train_scene indianpines --eval_scene indianpines

echo DONE > /tmp/mega.flag
