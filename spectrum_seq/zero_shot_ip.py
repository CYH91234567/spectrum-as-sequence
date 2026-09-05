"""Zero-shot transfer evaluation: Pavia-trained adapter -> Indian Pines (S2).

The adapter is frozen; Indian Pines class names are encoded with CLIP text
and every labelled IP pixel is classified by cosine similarity.
Success criterion: class-mean recall >= 1.3 x RGB-direct baseline.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .clip_utils import class_text_embeddings, encode_rgb_patches, load_clip
from .data import load_scene, make_rgb_cube
from .model import SpectrumSeqAdapter
from .train_pilot import metrics


def labelled_pixels(gt):
    ys, xs = np.where(gt > 0)
    return ys.astype(int), xs.astype(int)


def run_adapter_zero_shot(ckpt_path, scene_ip, ctx, device="cuda", batch=8192):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    adapter = SpectrumSeqAdapter(d_model=cfg["dmodel"], patch_len=cfg["patch"],
                                 stride=cfg["stride"], layers=cfg["layers"]).to(device)
    adapter.load_state_dict(ckpt["state"])
    clip_model, tokenizer, _ = load_clip(cfg["clip"], device)
    T = class_text_embeddings(clip_model, tokenizer, scene_ip.class_names, device)
    adapter.text_emb = T

    H, W, B = scene_ip.shape
    cube = torch.from_numpy(scene_ip.cube).to(device)
    if ctx > 1:
        r = ctx // 2
        cube = F.avg_pool2d(cube.permute(2, 0, 1)[None], ctx, stride=1, padding=r,
                            count_include_pad=False)[0].permute(1, 2, 0)
    flat = cube.reshape(-1, B)
    preds = []
    adapter.eval()
    for i in range(0, len(flat), batch):
        with torch.autocast(device_type="cuda"):
            z = adapter(flat[i:i + batch].float())
        preds.append((adapter.logit_scale * z @ T.T).argmax(-1))
    return torch.cat(preds).reshape(H, W).cpu().numpy()


def run_rgb_zero_shot(scene_ip, clip_path, device="cuda", win=32):
    clip_model, tokenizer, _ = load_clip(clip_path, device)
    T = class_text_embeddings(clip_model, tokenizer, scene_ip.class_names, device)
    rgb = make_rgb_cube(scene_ip)
    ys, xs = labelled_pixels(scene_ip.gt)
    emb = encode_rgb_patches(clip_model, rgb, ys, xs, win=win, device=device)
    pred_pix = (100.0 * emb @ T.T).argmax(-1)
    pred = np.zeros(scene_ip.gt.shape, dtype=np.int64)
    pred[ys, xs] = pred_pix.cpu().numpy()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--ckpt", required=True, help="Pavia-trained adapter checkpoint")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--ctx", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    scene = load_scene("indianpines", args.data)

    pred_a = run_adapter_zero_shot(args.ckpt, scene, ctx=args.ctx, device=device)
    res_a = metrics(pred_a, scene.gt, scene.num_classes)

    pred_r = run_rgb_zero_shot(scene, args.clip, device=device)
    res_r = metrics(pred_r, scene.gt, scene.num_classes)

    out = {
        "adapter_zero_shot": res_a,
        "rgb_baseline_zero_shot": res_r,
        "ratio_mRecall": res_a["mRecall"] / max(res_r["mRecall"], 1e-9),
        "ctx": args.ctx,
    }
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=2))
    print("adapter mIoU", res_a["mIoU"], "rgb mIoU", res_r["mIoU"])
    tag = f"ip_zero_shot_ctx{args.ctx}"
    with open(os.path.join(args.out, f"zs_metrics_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.save(os.path.join(args.out, f"pred_map_{tag}.npy"), pred_a)


if __name__ == "__main__":
    main()
