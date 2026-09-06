"""Trivial spectral 1-NN baseline (review round-1 M3).

Protocol identical to the adapter experiments: data.py normalisation
(band-wise z-norm + per-pixel unit-RMS), seed-0 5-shot exemplars per class
(data.py sample_shots), then every labelled pixel is classified by its
nearest exemplar. On unit-RMS spectra Euclidean and cosine ordering coincide.

Usage:  python -m spectrum_seq.spectral_1nn --data <data_dir> --scene paviau
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import load_scene, sample_shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scene", default="paviau", choices=["paviau", "indianpines", "salinas"])
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scene = load_scene(args.scene, args.data)
    pixels, labels = sample_shots(scene.gt, args.shots, seed=args.seed)
    H, W, B = scene.shape
    flat = scene.cube.reshape(-1, B)
    gt = scene.gt.reshape(-1)

    ex = flat[pixels[:, 0] * W + pixels[:, 1]]                  # (C*shots, B)
    m = gt > 0
    preds = np.empty(m.sum(), dtype=np.int64)
    query = flat[m]
    # leave-one-out: an exemplar pixel must not retrieve itself
    query_ids = np.where(m)[0]
    ex_ids = pixels[:, 0] * W + pixels[:, 1]
    # chunked nearest exemplar (Euclidean on unit-RMS == cosine ordering)
    for i in range(0, len(query), 65536):
        d = ((query[i:i + 65536, None, :] - ex[None, :, :]) ** 2).sum(-1)
        for j, qid in enumerate(query_ids[i:i + 65536]):
            same = np.where(ex_ids == qid)[0]
            if same.size:
                d[j, same] = np.inf
        preds[i:i + 65536] = labels[d.argmin(1)]
    pred_map = np.zeros(H * W, dtype=np.int64)
    pred_map[m] = preds
    pred_map = pred_map.reshape(H, W)

    g = gt.reshape(-1)[m]
    p = preds + 1
    oa = float((p == g).mean())
    ious, recalls = [], []
    for c in range(1, scene.num_classes + 1):
        tp = ((p == c) & (g == c)).sum()
        fp = ((p == c) & (g != c)).sum()
        fn = ((p != c) & (g == c)).sum()
        ious.append(float(tp / max(tp + fp + fn, 1)))
        recalls.append(float(tp / max(tp + fn, 1)))
    res = {
        "scene": args.scene, "shots": args.shots, "seed": args.seed,
        "mIoU": float(np.mean(ious)), "OA": oa, "mRecall": float(np.mean(recalls)),
        "IoU_per_class": ious, "recall_per_class": recalls,
        "note": "trivial spectral 1-NN baseline, computed by us; "
                "unit-RMS spectra make Euclidean and cosine orderings identical",
    }
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2))
    tag = f"spectral_1nn_baseline_seed{args.seed}" if args.scene == "paviau" \
        else f"spectral_1nn_baseline_{args.scene}_seed{args.seed}"
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
