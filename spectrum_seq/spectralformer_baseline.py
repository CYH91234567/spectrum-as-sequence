"""Supervised spectral-transformer baseline (SpectralFormer-style, our
implementation) for the same-budget / same-split upper-bound row.

Design follows the essential components of SpectralFormer [Hong et al.,
IEEE TGRS 2021]: one token per band, group-wise (local) attention over
band tokens, and residual learning. This is NOT the official
implementation; it is a compact re-implementation run under the exact
protocol of this paper (data.py normalisation, seed-123 50/50 split or
20x20 checkerboard, identical metrics) so the comparison is
budget-matched rather than implementation-matched.

Usage:  python -m spectrum_seq.spectralformer_baseline --data <dir> \
            --scene paviau --epochs 60 --spatial_block 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import load_scene, sample_shots
from .train_v2 import metrics_masked


class SpectralTransformerBaseline(nn.Module):
    def __init__(self, num_bands: int, num_classes: int, d: int = 64,
                 layers: int = 4, group: int = 17, heads: int = 2):
        super().__init__()
        self.embed = nn.Linear(1, d)
        self.register_buffer("pe", self._pe(num_bands, d), persistent=False)
        self.groups = math.ceil(num_bands / group)
        self.group = group
        self.num_bands = num_bands
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, num_classes))

    @staticmethod
    def _pe(n, d):
        pos = torch.arange(n, dtype=torch.float32) / max(n - 1, 1)
        half = d // 2
        freq = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(1e4) / max(half - 1, 1)))
        ang = pos[:, None] * freq[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, x):  # x: (N, B)
        N, B = x.shape
        tok = self.embed(x[:, :, None]) + self.pe[:B][None]
        pad = self.groups * self.group - B
        if pad > 0:
            tok = torch.cat([tok, torch.zeros(N, pad, tok.shape[-1], device=x.device)], dim=1)
        outs = []
        for g in range(self.groups):  # group-wise (local spectral) attention
            outs.append(self.encoder(tok[:, g * self.group:(g + 1) * self.group]))
        h = torch.cat(outs, dim=1)[:, :B].mean(dim=1)
        return self.head(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scene", default="paviau", choices=["paviau", "indianpines"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--spatial_block", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scene = load_scene(args.scene, args.data)
    gt = scene.gt
    ys, xs = np.where(gt > 0)
    labels_all = gt[ys, xs] - 1
    if args.spatial_block > 0:
        blk = args.spatial_block
        test_mask = (((ys // blk) + (xs // blk)) % 2 == 0)
    else:
        rng = np.random.default_rng(123)
        pick = rng.permutation(len(ys))
        test_mask = np.zeros(len(ys), dtype=bool)
        test_mask[pick[len(pick) // 2:]] = True
    tr_idx, te_idx = np.where(~test_mask)[0], np.where(test_mask)[0]
    print(f"split: train {len(tr_idx)} px, test {len(te_idx)} px", flush=True)

    model = SpectralTransformerBaseline(scene.shape[-1], scene.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params/1e6:.2f}M", flush=True)

    spec_tr = torch.from_numpy(scene.cube[ys[tr_idx], xs[tr_idx]]).to(device)
    y_tr = torch.from_numpy(labels_all[tr_idx]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(spec_tr))
        ep_loss, nb = 0.0, 0
        for bi in range(0, len(perm), args.bs):
            idx = perm[bi:bi + args.bs]
            with torch.autocast(device_type="cuda"):
                logits = model(spec_tr[torch.from_numpy(idx).to(device)])
                loss = F.cross_entropy(logits, y_tr[torch.from_numpy(idx).to(device)])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += float(loss)
            nb += 1
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"ep {ep+1}: loss {ep_loss/nb:.4f}", flush=True)
    train_time = time.time() - t0

    model.eval()
    preds = np.zeros(len(ys), dtype=np.int64)
    with torch.no_grad():
        for bi in range(0, len(te_idx), 8192):
            idx = te_idx[bi:bi + 8192]
            spec = torch.from_numpy(scene.cube[ys[idx], xs[idx]]).to(device)
            with torch.autocast(device_type="cuda"):
                logits = model(spec)
            preds[idx] = logits.argmax(-1).cpu().numpy()

    # metrics on the labelled pixels of the held-out split
    gt_mask = np.zeros(gt.shape, dtype=bool)
    gt_mask[ys[te_idx], xs[te_idx]] = True
    pred_map = np.zeros(gt.shape, dtype=np.int64)
    pred_map[ys, xs] = preds
    res = metrics_masked(pred_map, gt, scene.num_classes, gt_mask)
    res.update({"scene": args.scene, "params_M": n_params / 1e6,
                "train_time_s": train_time, "seed": args.seed,
                "spatial_block": args.spatial_block,
                "note": "SpectralFormer-style supervised baseline (our compact "
                        "re-implementation, same budget/split as the paper)"})
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2))
    tag = f"spectralformer_baseline_{args.scene}_seed{args.seed}"
    if args.spatial_block:
        tag += f"_sp{args.spatial_block}"
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
