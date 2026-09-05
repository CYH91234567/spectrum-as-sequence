# Spectrum-as-Sequence (Pilot v1 + v2)

Reprogramming a frozen CLIP (ViT-B/16) for hyperspectral imagery by treating
the band axis as a **sequence**: time-series foundation-model kernels
(PatchTST-style spectral patching + channel-independent embedding,
iTransformer-style inverted band attention) are migrated onto the spectral
axis and mapped into the frozen CLIP embedding space for pixel-level
classification with text-prompt class anchors.

This repo contains the **pilot v1** experiment of the research route:

- frozen OpenAI CLIP ViT-B/16, only 0.33M trainable adapter parameters
- 5-shot / full-supervision Pavia University training
- zero-shot transfer to Indian Pines (disjoint class sets, cross-sensor)
- RGB-direct CLIP baselines (spectral -> RGB interpolation, SPECIAL-style)

## Key findings

### Pilot v2: wavelength-space tokenization + spectral token injection (this repo's final state)

| Setting | PaviaU mIoU | IP mIoU |
|---|---|---|
| 5-shot injected adapter (45/80 labels) | **50.8%** | **50.1%** |
| RGB-direct CLIP zero-shot | 3.4% | 2.6% |

- `wavelength_patcher.py`: patches defined in physical wavelength space
  (50 nm bins, absolute-wavelength positional encoding) -> the spectral token
  sequence is sensor-aligned by construction.
- `injected_vit.py`: spectral tokens cross-attend into the frozen CLIP ViT at
  layers 3/6/9 (zero-init gamma); 7.46M trainable params; 26 s training,
  2.7 GB peak on a single 3090 Ti.
- Open-vocabulary base/novel analysis (`train_v2.py --base_ids --balanced`):
  5-shot training collapses novel classes (0.0 mIoU); three prior-preservation
  mitigations (early stop, logit fusion, KL self-distillation) only trade base
  vs novel; at 50 shots/class novel classes recover (7.7 mIoU = 2.3x the RGB
  baseline) while base reaches 59.2 (8.2x baseline). The 5-shot open-vocab
  failure mode is characterised, not hidden.

### Pilot v1 (historical, pooled-vector interface)

| Setting | PaviaU mIoU | OA |
|---|---|---|
| 5-shot spectral adapter (ours) | **33.5%** | 48.9% |
| full supervision, same adapter | 60.6% | 78.6% |
| RGB-direct CLIP zero-shot | 3.4% | 9.3% |

- Spectral-aware reprogramming beats the RGB-direct CLIP route by ~10x mIoU
  with only 45 labelled pixels: RGB pathways destroy spectral evidence.
- The pooled-vector interface caps the ceiling (60.6% mIoU full supervision):
  pilot v2 moves to wavelength-space patching + token injection into the
  frozen ViT.
- Naive band-index patching does not transfer across sensors (PaviaU ->
  Indian Pines zero-shot fails): patches must be defined in wavelength space,
  which is exactly the tokenize-mechanism question this project studies.

## Layout

```
spectrum_seq/         package
  wavelength_patcher.py  M4: wavelength-space tokenization (v2)
  injected_vit.py        M2: spectral token injection into frozen CLIP ViT (v2)
  train_v2.py            v2 training / base-novel / balancing / calibration
  data.py             PaviaU / Indian Pines loaders + normalisation
  model.py            SpectralPatcher (M1) + InvertedBandAttention (M2) + adapter
  clip_utils.py       frozen CLIP loading (local TorchScript ckpt), text prompts,
                      RGB patch encoding
  train_pilot.py      5-shot / full-supervision training + full-map metrics
  zero_shot_ip.py     PaviaU -> Indian Pines zero-shot transfer evaluation
  rgb_baseline.py     RGB-direct CLIP zero-shot baseline
results/              pilot v1 metric JSONs
```

## Reproduce

```bash
pip install -r requirements.txt
# data: PaviaU.mat + PaviaU_gt.mat + Indian_pines_corrected.mat + Indian_pines_gt.mat
#   mirrors: hf-mirror.com/datasets/danaroth/pavia, danaroth/indian_pines
# CLIP ckpt: OpenAI ViT-B/16 TorchScript from openaipublic.azureedge.net
python -m spectrum_seq.train_pilot --data <data_dir> --clip <ViT-B-16.pt> \
    --out results --epochs 300 --shots 5
python -m spectrum_seq.rgb_baseline --scene paviau --data <data_dir> --clip <ViT-B-16.pt> --out results
python -m spectrum_seq.zero_shot_ip --data <data_dir> --clip <ViT-B-16.pt> \
    --ckpt results/adapter_paviau_s5_seed0_ctx1.pt --out results
```

Single RTX 3090 Ti (24 GB): training 1 s, full pipeline < 5 min, < 1 GB VRAM.
