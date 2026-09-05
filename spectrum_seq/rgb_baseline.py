"""RGB-direct CLIP zero-shot baseline (T1.5): spectral -> RGB interpolation,
then frozen CLIP classifies each pixel from its 32x32 RGB patch.
SPECIAL-style pseudo-label route without the spectral stage.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .clip_utils import class_text_embeddings, encode_rgb_patches, load_clip
from .data import load_scene, make_rgb_cube
from .train_pilot import metrics
from .zero_shot_ip import labelled_pixels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="paviau", choices=["paviau", "indianpines"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--win", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    scene = load_scene(args.scene, args.data)
    clip_model, tokenizer, _ = load_clip(args.clip, device)
    T = class_text_embeddings(clip_model, tokenizer, scene.class_names, device)
    rgb = make_rgb_cube(scene)
    ys, xs = labelled_pixels(scene.gt)
    emb = encode_rgb_patches(clip_model, rgb, ys, xs, win=args.win, device=device)
    pred_pix = (100.0 * emb @ T.T).argmax(-1)
    pred = np.zeros(scene.gt.shape, dtype=np.int64)
    pred[ys, xs] = pred_pix.cpu().numpy()
    res = metrics(pred, scene.gt, scene.num_classes)
    res.update({"scene": args.scene, "win": args.win})
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2))
    with open(os.path.join(args.out, f"rgb_baseline_{args.scene}.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.save(os.path.join(args.out, f"pred_map_rgb_{args.scene}.npy"), pred)


if __name__ == "__main__":
    main()
