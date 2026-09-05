"""Pilot training: 5-shot spectral adapter on Pavia University (T1.4).

Usage (server):
  python -m spectrum_seq.train_pilot --data /home/ubuntu/spectrum_pilot/data \
      --clip /home/ubuntu/spectrum_pilot/ViT-B-16.pt --out ../results --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .clip_utils import class_text_embeddings, load_clip
from .data import load_scene, sample_shots
from .model import SpectrumSeqAdapter


def gather_spectra(scene, pixels, ctx=1):
    """Return (N, B) spectra; ctx>1 averages a ctx x ctx neighbourhood
    (spatial context for M2, matching the inversion design)."""
    cube = scene.cube
    H, W, B = cube.shape
    if ctx == 1:
        return cube[pixels[:, 0], pixels[:, 1]]
    r = ctx // 2
    padded = np.pad(cube, ((r, r), (r, r), (0, 0)), mode="reflect")
    out = np.zeros((len(pixels), B), dtype=np.float32)
    for i, (y, x) in enumerate(pixels):
        out[i] = padded[y:y + ctx, x:x + ctx].mean((0, 1))
    return out


def predict_map(adapter, scene, ctx=1, device="cuda", batch=8192, amp=True):
    """Predict class index per pixel via cosine similarity to text embeddings
    stored in adapter.text_emb. Returns (H, W) int map (0 = unlabelled/-1)."""
    adapter.eval()
    H, W, B = scene.shape
    cube = torch.from_numpy(scene.cube).to(device)
    if ctx > 1:
        r = ctx // 2
        cube = F.avg_pool2d(cube.permute(2, 0, 1)[None], ctx, stride=1, padding=r,
                            count_include_pad=False)[0].permute(1, 2, 0)
    flat = cube.reshape(-1, B)
    preds = []
    T = adapter.text_emb
    for i in range(0, len(flat), batch):
        with torch.autocast(device_type="cuda", enabled=amp):
            z = adapter(flat[i:i + batch].float())
        preds.append((adapter.logit_scale * z @ T.T).argmax(-1))
    return torch.cat(preds).reshape(H, W).cpu().numpy()


def metrics(pred, gt, num_classes):
    """mIoU / OA / per-class recall over labelled pixels (gt>0)."""
    m = gt > 0
    p, g = pred[m] + 1, gt[m]
    oa = float((p == g).mean())
    ious, recalls = [], []
    for c in range(1, num_classes + 1):
        tp = ((p == c) & (g == c)).sum()
        fp = ((p == c) & (g != c)).sum()
        fn = ((p != c) & (g == c)).sum()
        iou = tp / max(tp + fp + fn, 1)
        rec = tp / max(tp + fn, 1)
        ious.append(float(iou))
        recalls.append(float(rec))
    return {
        "mIoU": float(np.mean(ious)),
        "OA": oa,
        "mRecall": float(np.mean(recalls)),
        "IoU_per_class": ious,
        "recall_per_class": recalls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ctx", type=int, default=1, help="train-time spatial context window")
    ap.add_argument("--patch", type=int, default=25)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--dmodel", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scene = load_scene("paviau", args.data)
    pixels, labels = sample_shots(scene.gt, args.shots, seed=args.seed)
    print(f"PaviaU: {scene.shape}, {scene.num_classes} classes, train pixels: {len(pixels)}")

    clip_model, tokenizer, _ = load_clip(args.clip, device)
    T = class_text_embeddings(clip_model, tokenizer, scene.class_names, device)

    adapter = SpectrumSeqAdapter(d_model=args.dmodel, patch_len=args.patch,
                                 stride=args.stride, layers=args.layers).to(device)
    adapter.text_emb = T

    test_mask = None
    if len(pixels) > 2048:
        # 50/50 split for the full-supervision ceiling (same protocol as v2)
        rng_s = np.random.default_rng(123)
        perm = rng_s.permutation(len(pixels))
        half = len(perm) // 2
        tr, te = perm[:half], perm[half:]
        test_mask = np.zeros(scene.gt.shape, dtype=bool)
        test_mask[pixels[te, 0], pixels[te, 1]] = True
        train_mask = np.zeros(scene.gt.shape, dtype=bool)
        train_mask[pixels[tr, 0], pixels[tr, 1]] = True
        print(f"split: train {len(tr)} px, test {len(te)} px")
        pixels, labels = pixels[tr], labels[tr]
    spec = torch.from_numpy(gather_spectra(scene, pixels, ctx=args.ctx)).to(device)
    y = torch.from_numpy(labels).to(device)

    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for ep in range(args.epochs):
        adapter.train()
        # spectral augmentation: gaussian noise + random band dropout
        noise = 0.02 * torch.randn_like(spec)
        drop = (torch.rand_like(spec) < 0.05).float()
        spec_aug = spec + noise
        spec_aug = torch.where(drop.bool(), torch.zeros_like(spec_aug), spec_aug)
        with torch.autocast(device_type="cuda"):
            z = adapter(spec_aug)
            logits = adapter.logit_scale * z @ T.T
            loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"ep {ep+1}: loss {loss.item():.4f}")

    train_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 2**30

    pred = predict_map(adapter, scene, ctx=args.ctx, device=device)
    res = metrics(pred, scene.gt, scene.num_classes)
    if test_mask is not None:
        from .train_v2 import metrics_masked
        res_test = metrics_masked(pred, scene.gt, scene.num_classes, test_mask)
        res["heldout_mIoU"] = res_test["mIoU"]
        res["heldout_OA"] = res_test["OA"]
        res["heldout_mRecall"] = res_test["mRecall"]
        print(f"held-out ceiling: mIoU {res['heldout_mIoU']:.4f} OA {res['heldout_OA']:.4f}")
    res.update({
        "seed": args.seed, "shots": args.shots, "ctx": args.ctx,
        "trainable_params_M": n_params / 1e6,
        "train_time_s": train_time,
        "peak_mem_GB": peak_mem,
    })
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2))

    tag = f"paviau_s{args.shots}_seed{args.seed}_ctx{args.ctx}"
    torch.save({"state": adapter.state_dict(), "config": vars(args), "text_emb": T.cpu()},
               os.path.join(args.out, f"adapter_{tag}.pt"))
    with open(os.path.join(args.out, f"train_metrics_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.save(os.path.join(args.out, f"pred_map_{tag}.npy"), pred)


if __name__ == "__main__":
    main()
