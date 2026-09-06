# Open-Vocabulary Hyperspectral Segmentation Benchmark Protocol (v0.1)

This document fixes the evaluation protocol used across the Spectrum-as-Sequence
project so that every number in the papers/repo is reproducible and comparable.

## Scenes

| Scene | Size | Bands | Classes | Role |
|---|---|---|---|---|
| Pavia University | 610×340 | 103 (0.43–0.86 µm) | 9 | train / eval |
| Indian Pines (corrected) | 145×145 | 200 (0.4–2.45 µm) | 16 | train / eval / zero-shot target |
| Salinas (corrected) | 512×217 | 204 (0.4–2.45 µm) | 16 | zero-shot target |

Preprocessing (unsupervised, scene-level only): per-scene band-wise
z-normalisation, then per-pixel magnitude removal (unit-RMS spectrum).

## Protocol tiers

**Tier 1 — closed-set few-shot (main table).**
`shots` labelled pixels per class (5 / 15 / 50), all classes seen.
Train on scene A, evaluate on all labelled pixels of scene A.
Metrics: mIoU, OA, per-class recall. Report peak memory + wall time.

**Tier 2 — open-vocabulary base/novel (same scene).**
Base classes: PaviaU {Asphalt, Meadows, Gravel, Trees, Bare soil}
(ids 0,1,2,3,5); novel: {Painted metal sheets, Bitumen, Self-blocking
bricks, Shadows}. Shots are sampled from base classes only.
Metrics: base mIoU, novel mIoU, harmonic-mean hMIoU.
Report configurations with and without transductive inference (below).

**Tier 3 — cross-scene zero-shot (challenge setting).**
Adapter frozen after training on scene A; class anchors are the target
scene's class names encoded by the frozen text tower. Reported honestly
with the RGB-direct baseline of the same tier (no cherry-picking).

**Transductive inference (optional, always disclosed).**
Class-marginal balancing over the evaluated pixels: iterate
`logits -= tau * log(mean softmax)`, 5 iterations, tau = 1. Applied
identically to baselines and our method. Optional per-class logit
standardisation (`--calibrate`) is reported separately.

## Full-supervision ceiling

50/50 seeded split (seed 123) of labelled pixels; train on one half,
report on the held-out half (`heldout_mIoU`). Rationale: a 7.46M
adapter memorises the train half (train mIoU ≈ 1.0), so train-set-inclusive
numbers are meaningless for high-capacity adapters.

## Baselines

1. RGB-direct CLIP zero-shot (spectral → RGB interpolation, 32×32 crops).
   Two variants exist and must be labelled: **no-transduction** (PaviaU 3.38%
   mIoU) and **with class-balanced transduction** (PaviaU 5.50%).
2. **Trivial spectral 1-NN** (zero-training; `spectrum_seq/spectral_1nn.py`):
   same normalisation and seed-0 5-shot exemplars, leave-one-out nearest
   neighbour on unit-RMS spectra. PaviaU 39.07 / IP 31.60 mIoU.
3. Pooled spectral adapter (pilot v1 architecture).
4. Injected adapter (this work).
5. Reference numbers quoted from SPECIAL / HSI-Adapter / SegEarth-OV
   (their protocols differ; quoted only for context, never mixed into tables).

## Footguns (documented so they are never repeated)

- Checkpoint/metric filenames must encode every experimental variable
  (shots, fuse mode, tokenize, base_ids, seed) — we lost two metric files
  to tag collisions.
- Always report whether evaluation pixels intersect training pixels.
- Zero-shot claims require the baseline to be evaluated under the identical
  inference budget (same transduction, same calibration).
