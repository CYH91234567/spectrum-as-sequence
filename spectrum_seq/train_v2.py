"""Pilot v2 training + evaluation (wavelength-space patching + ViT injection).

Usage (server):
  python -m spectrum_seq.train_v2 --data ... --clip ... --out ... \
      --shots 5 --epochs 300
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
from .data import load_scene, make_rgb_cube, sample_shots
from .injected_vit import InjectedCLIP
from .train_pilot import metrics
from .wavelength_patcher import build_bin_plan

# shared VNIR window; both sensors cover it
WL_MIN, WL_MAX, BIN_UM = 0.43, 0.86, 0.05
MAX_BANDS = 14          # PaviaU ~12 bands per 50nm bin at 4.2nm spacing
IMG_WIN = 32            # RGB crop window fed to CLIP


def rgb_crop_tensor(rgb_cube, ys, xs, device, half=IMG_WIN // 2):
    """(N,3,224,224) CLIP-normalised crops centred at (ys,xs)."""
    import torch as _t
    H, W, _ = rgb_cube.shape
    padded = np.pad(rgb_cube, ((half, half), (half, half), (0, 0)), mode="edge")
    crops = []
    mean = _t.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[:, None, None]
    std = _t.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[:, None, None]
    for y, x in zip(ys, xs):
        crops.append(padded[y:y + IMG_WIN, x:x + IMG_WIN])
    batch = _t.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(device)
    batch = F.interpolate(batch, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
    return (batch - mean) / std


class CropBank:
    """All IMG_WIN x IMG_WIN edge-padded crops of a scene on GPU via one unfold
    (~263MB fp16 for 42k-pixel scenes). Batch gather + bicubic resize replace
    the per-crop python loop that made full-supervision epochs ~10x too slow."""

    def __init__(self, rgb_cube, device, half=IMG_WIN // 2):
        import torch as _t
        H, W, _ = rgb_cube.shape
        padded = np.pad(rgb_cube, ((half, half), (half, half), (0, 0)), mode="edge")
        cube = _t.from_numpy(padded).permute(2, 0, 1)[None].to(device).half()  # (1,3,H+2p,W+2p)
        self.win = IMG_WIN
        # windows: (1, 3, H, W, win, win) view via unfold
        w = cube.unfold(2, IMG_WIN, 1).unfold(3, IMG_WIN, 1)[0]  # (3,H+2p-win+1,...)
        w = w[:, :H, :W]                                          # keep window starts 0..H-1
        self.bank = w.permute(1, 2, 0, 3, 4).reshape(H * W, 3, IMG_WIN, IMG_WIN)
        self.mean = _t.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = _t.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def fetch(self, ys, xs, W):
        ids = torch.as_tensor(ys, device=self.bank.device) * W + torch.as_tensor(xs, device=self.bank.device)
        batch = self.bank[ids].float()
        batch = F.interpolate(batch, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
        return (batch - self.mean) / self.std


@torch.no_grad()
def predict_full_map(model, scene, T, device="cuda", batch=384, fuse="injection", s_res=None,
                     calibrate=False, labelled_only=True, balanced=False,
                     balanced_iters=5, balanced_tau=1.0):
    """Classify every pixel: 32x32 RGB crop + pixel spectrum -> CLIP space.
    calibrate: per-class logit standardisation over evaluated pixels
    ((logits - mu_c) / sigma_c) to remove base/novel logit-scale bias."""
    model.eval()
    rgb = make_rgb_cube(scene)
    H, W, B = scene.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    if labelled_only:
        m = scene.gt > 0
        ys, xs = ys[m], xs[m]
    else:
        ys, xs = ys.reshape(-1), xs.reshape(-1)
    spec_all = torch.from_numpy(scene.cube.reshape(-1, B)).to(device)
    flat_ids = torch.from_numpy((ys * W + xs).astype(np.int64)).to(device)
    all_logits = []
    for i in range(0, len(ys), batch):
        yb, xb = ys[i:i + batch], xs[i:i + batch]
        rgb_b = rgb_crop_tensor(rgb, yb, xb, device)
        spec_b = spec_all[flat_ids[i:i + batch]]
        with torch.autocast(device_type="cuda"):
            if fuse == "prior":
                z0, z1 = model.encode_image_with_prior(rgb_b, spec_b)
                n0 = F.normalize(z0.float(), dim=-1)
                n1 = F.normalize(z1.float(), dim=-1)
                logits = 100.0 * n0 @ T.T + float(s_res.exp()) * 100.0 * n1 @ T.T
            else:
                z = model.encode_image(rgb_b, spec_b)
                logits = 100.0 * F.normalize(z.float(), dim=-1) @ T.T
        all_logits.append(logits.float().cpu())
    logits = torch.cat(all_logits, 0)
    if calibrate:
        mu = logits.mean(0, keepdim=True)
        sd = logits.std(0, keepdim=True) + 1e-6
        logits = (logits - mu) / sd
    if balanced:
        # class-marginal balancing (transductive logit adjustment): iterate
        # subtracting log of the predicted class marginal -> uniform marginal
        for _ in range(balanced_iters):
            r = logits.softmax(-1).mean(0).clamp_min(1e-8)
            logits = logits - balanced_tau * r.log()
    p = logits.argmax(-1)
    preds = np.zeros((H, W), dtype=np.int64)
    preds[ys, xs] = p.numpy()
    return preds


def metrics_masked(pred, gt, num_classes, mask):
    """metrics restricted to a boolean pixel mask (held-out split)."""
    m = (gt > 0) & mask
    p, g = pred[m] + 1, gt[m]
    oa = float((p == g).mean())
    ious, recalls = [], []
    for c in range(1, num_classes + 1):
        tp = ((p == c) & (g == c)).sum()
        fp = ((p == c) & (g != c)).sum()
        fn = ((p != c) & (g == c)).sum()
        ious.append(float(tp / max(tp + fp + fn, 1)))
        recalls.append(float(tp / max(tp + fn, 1)))
    return {"mIoU": float(np.mean(ious)), "OA": oa, "mRecall": float(np.mean(recalls)),
            "IoU_per_class": ious, "recall_per_class": recalls}


def split_metrics(res, base_ids, num_classes):
    """Add base/novel mean-IoU and harmonic-mean hMIoU to a metrics dict."""
    novel = [c for c in range(num_classes) if c not in base_ids]

    def mean_iou(idxs):
        return float(np.mean([res["IoU_per_class"][c] for c in idxs])) if idxs else 0.0

    b, n = mean_iou(base_ids), mean_iou(novel)
    res["base_mIoU"] = b
    res["novel_mIoU"] = n
    res["hMIoU"] = (2 * b * n / (b + n)) if (b + n) > 0 else 0.0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d_spec", type=int, default=128)
    ap.add_argument("--spec_layers", type=int, default=2)
    ap.add_argument("--inject_layers", default="3,6,9")
    ap.add_argument("--eval_only_ckpt", default=None)
    ap.add_argument("--eval_scene", default="paviau")
    ap.add_argument("--train_scene", default=None,
                    help="defaults to eval_scene; set e.g. indianpines for "
                         "cross-scene zero-shot evaluation")
    ap.add_argument("--base_ids", default=None,
                    help="comma-separated 0-based base-class ids; shots are "
                         "sampled only from these, novel classes held out")
    ap.add_argument("--calibrate", action="store_true",
                    help="per-class logit standardisation before argmax")
    ap.add_argument("--balanced", action="store_true",
                    help="transductive class-marginal balancing on evaluated pixels")
    ap.add_argument("--balanced_iters", type=int, default=5)
    ap.add_argument("--balanced_tau", type=float, default=1.0)
    ap.add_argument("--prior_kl", type=float, default=0.0,
                    help="weight of KL(frozen||injected) on unlabeled pixels "
                         "(prior-preserving self-distillation)")
    ap.add_argument("--n_unlab", type=int, default=1024,
                    help="unlabeled pixels sampled for the KL term")
    ap.add_argument("--enc", default="ts", choices=["ts", "cnn", "mlp"],
                    help="kernel-control ablation: sequence-modeling kernel over bin tokens")
    ap.add_argument("--tokenize", default="wl50",
                    help="M4 ablation: wl25/wl50/wl100 wavelength bins, "
                         "bandindex (P25 S5), bandeq9 (9 equal chunks)")
    ap.add_argument("--trans_w", type=float, default=0.0,
                    help="TSC: weight of spectral-neighbourhood consistency + "
                         "marginal balancing on unlabeled pixels")
    ap.add_argument("--n_trans", type=int, default=512)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--fuse", choices=["injection", "prior"], default="injection",
                    help="injection: logits from injected embedding only; "
                         "prior: logits = frozen zero-shot + s * injected "
                         "(s learnable, init 0) to preserve novel classes")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_scene_name = args.train_scene or args.eval_scene

    scene = load_scene(args.eval_scene, args.data)
    bins, centers = build_bin_plan(scene.wavelengths_um, WL_MIN, WL_MAX, BIN_UM)
    print(f"bins: {len(bins)} over [{WL_MIN},{WL_MAX}]um width {BIN_UM}")

    clip_model, tokenizer, _ = load_clip(args.clip, device)
    T = class_text_embeddings(clip_model, tokenizer, scene.class_names, device)

    # ckpt config first: band-index tokenizers must reuse the TRAIN scene's
    # chunk definition at zero-shot time (that misalignment IS the ablation)
    test_mask = None
    train_mask = None
    eval_cfg = None
    if args.eval_only_ckpt:
        eval_cfg = torch.load(args.eval_only_ckpt, map_location="cpu", weights_only=False).get("config", {})
    token_src = args.eval_scene
    if args.tokenize != "wl50" and eval_cfg is not None:
        token_src = eval_cfg.get("train_scene") or eval_cfg.get("eval_scene") or args.eval_scene
    spec_enc = None
    if args.enc != "ts":
        from .wavelength_patcher import SpecEncoderVariant
        spec_enc = SpecEncoderVariant(bins, centers, MAX_BANDS,
                                      d_model=args.d_spec, variant=args.enc)
        print(f"enc variant: {args.enc}")
    if args.tokenize != "wl50":
        from .wavelength_patcher import build_spec_encoder
        spec_enc, n_tok = build_spec_encoder(args.tokenize, load_scene(token_src, args.data),
                                             d_model=args.d_spec, layers=args.spec_layers)
        print(f"tokenize={args.tokenize} (source scene {token_src}): {n_tok} tokens")
    model = InjectedCLIP(clip_model, bins, centers, MAX_BANDS,
                         d_spec=args.d_spec, spec_layers=args.spec_layers,
                         inject_layers=tuple(int(i) for i in args.inject_layers.split(",")),
                         spec_enc=spec_enc).to(device)
    n_params = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable params: {n_params/1e6:.2f}M")

    tag = f"v2_{args.eval_scene}_s{args.shots}_seed{args.seed}"
    ckpt_path = os.path.join(args.out, f"adapter_{tag}.pt")
    s_res = torch.zeros(1, device=device) if args.fuse == "prior" else None
    if args.eval_only_ckpt:
        ckpt = torch.load(args.eval_only_ckpt, map_location=device, weights_only=False) if eval_cfg is None             else {"state": torch.load(args.eval_only_ckpt, map_location=device, weights_only=False)["state"]}
        model.load_state_dict({k: v for k, v in ckpt["state"].items()}, strict=False)
        tag = "zs_" + os.path.basename(args.eval_only_ckpt).replace("adapter_", "").replace(".pt", "") \
            + f"_to_{args.eval_scene}"
        ckpt_path = os.path.join(args.out, f"adapter_{tag}.pt")
    else:
        train_scene = load_scene(train_scene_name, args.data)
        fuse_tag = "prior" if args.fuse == "prior" else "inj"
        base_tag = "_base" + args.base_ids.replace(",", "-") if args.base_ids else ""
        tok_tag = "" if args.tokenize == "wl50" else f"_{args.tokenize}"
        tsc_tag = f"_tsc{args.trans_w}" if args.trans_w > 0 else ""
        enc_tag = "" if args.enc == "ts" else f"_{args.enc}"
        tag = f"v2_{train_scene_name}_s{args.shots}_{fuse_tag}{tok_tag}{enc_tag}{base_tag}{tsc_tag}_seed{args.seed}"
        ckpt_path = os.path.join(args.out, f"adapter_{tag}.pt")
        base_ids = [int(i) for i in args.base_ids.split(",")] if args.base_ids else None
        rgb = make_rgb_cube(train_scene)
        if base_ids:
            # base-novel protocol: sample shots from base classes only
            keep = np.isin(train_scene.gt - 1, base_ids) & (train_scene.gt > 0)
            ys_b, xs_b = np.where(keep)
            labels_b = train_scene.gt[ys_b, xs_b] - 1
            rng = np.random.default_rng(args.seed)
            chosen = []
            for c in base_ids:
                idx_c = np.where(labels_b == c)[0]
                chosen += rng.choice(idx_c, size=min(args.shots, len(idx_c)), replace=False).tolist()
            pixels = np.stack([ys_b[chosen], xs_b[chosen]], axis=1)
            labels = labels_b[chosen]
        else:
            pixels, labels = sample_shots(train_scene.gt, args.shots, seed=args.seed)
        big_mode = len(pixels) > 2048       # full supervision: batched epochs
        test_mask = None
        if big_mode:
            # 50/50 train/test split: a 7.46M adapter memorises the train set,
            # so the ceiling must be measured on held-out pixels
            rng_s = np.random.default_rng(123)
            perm = rng_s.permutation(len(pixels))
            half = len(perm) // 2
            tr, te = perm[:half], perm[half:]
            pixels_tr, labels_tr = pixels[tr], labels[tr]
            test_mask = np.zeros(train_scene.gt.shape, dtype=bool)
            test_mask[pixels[te, 0], pixels[te, 1]] = True
            train_mask = np.zeros(train_scene.gt.shape, dtype=bool)
            train_mask[pixels_tr[:, 0], pixels_tr[:, 1]] = True
            print(f"split: train {len(tr)} px, test {len(te)} px", flush=True)
            pixels, labels = pixels_tr, labels_tr
        spec_b = torch.from_numpy(train_scene.cube[pixels[:, 0], pixels[:, 1]]).to(device)
        y = torch.from_numpy(labels).to(device)
        rgb_b = None if big_mode else rgb_crop_tensor(rgb, pixels[:, 0], pixels[:, 1], device)
        # unlabeled set for prior-preserving self-distillation
        if args.prior_kl > 0:
            rng = np.random.default_rng(args.seed + 1)
            Hs, Ws, _ = scene.shape
            train_set = set(map(tuple, pixels.tolist()))
            cand = [(y, x) for y in range(Hs) for x in range(Ws)
                    if (y, x) not in train_set]
            sel = rng.choice(len(cand), size=min(args.n_unlab, len(cand)), replace=False)
            upix = np.array([cand[i] for i in sel])
            rgb_u = rgb_crop_tensor(rgb, upix[:, 0], upix[:, 1], device).half()
            spec_u = torch.from_numpy(train_scene.cube[upix[:, 0], upix[:, 1]]).to(device).half()
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                z0u = model.encode_image(rgb_u.float(), spec_u.float())
                logits0u = (100.0 * F.normalize(z0u.float(), dim=-1) @ T.T).detach()
                del z0u
                torch.cuda.empty_cache()
        # TSC: unlabeled pixels + spectral kNN graph (precomputed once)
        trans_data = None
        if args.trans_w > 0 and not big_mode:
            rng_t = np.random.default_rng(args.seed + 2)
            Hs, Ws, _ = train_scene.shape
            all_pix = [(yy, xx) for yy in range(Hs) for xx in range(Ws)]
            sel_t = rng_t.choice(len(all_pix), size=min(args.n_trans, len(all_pix)), replace=False)
            tpix = np.array([all_pix[i] for i in sel_t])
            tspec = train_scene.cube[tpix[:, 0], tpix[:, 1]]
            # cosine kNN on raw normalized spectra
            sim = tspec @ tspec.T
            np.fill_diagonal(sim, -9.0)
            knn_idx = np.argsort(-sim, axis=1)[:, :args.knn]        # (N, k)
            t_rgb = rgb_crop_tensor(rgb, tpix[:, 0], tpix[:, 1], device).half()
            t_spec = torch.from_numpy(tspec).to(device).half()
            trans_data = (t_rgb, t_spec, torch.from_numpy(knn_idx).to(device))
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.01)
        s_res = None
        if args.fuse == "prior":
            s_res = torch.nn.Parameter(torch.zeros(1, device=device))
            opt.add_param_group({"params": [s_res], "lr": args.lr * 10})
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        bs_train = 128
        crop_bank = None
        scaler = torch.cuda.amp.GradScaler() if big_mode else None
        for ep in range(args.epochs):
            model.train()
            if big_mode:
                if crop_bank is None:
                    crop_bank = CropBank(rgb, device)
                    print("crop bank ready", flush=True)
                perm = np.random.permutation(len(pixels))
                ep_loss, nb = 0.0, 0
                Hs, Ws, _ = train_scene.shape
                for bi in range(0, len(perm), bs_train):
                    idx = perm[bi:bi + bs_train]
                    rgb_bi = crop_bank.fetch(pixels[idx, 0], pixels[idx, 1], Ws)
                    spec_bi = spec_b[torch.from_numpy(idx).to(device)]
                    with torch.autocast(device_type="cuda"):
                        z = model.encode_image(rgb_bi, spec_bi)
                        logits = 100.0 * F.normalize(z.float(), dim=-1) @ T.T
                        l = F.cross_entropy(logits, y[torch.from_numpy(idx).to(device)])
                    opt.zero_grad()
                    scaler.scale(l).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    ep_loss += float(l)
                    nb += 1
                sched.step()
                if (ep + 1) % 10 == 0 or ep == 0:
                    print(f"ep {ep+1}: loss {ep_loss/nb:.4f}", flush=True)
                continue
            noise = 0.02 * torch.randn_like(spec_b)
            with torch.autocast(device_type="cuda"):
                if args.fuse == "prior":
                    z0, z1 = model.encode_image_with_prior(rgb_b, spec_b + noise)
                    n0 = F.normalize(z0.float(), dim=-1)
                    n1 = F.normalize(z1.float(), dim=-1)
                    logits = 100.0 * n0 @ T.T + s_res.exp() * 100.0 * n1 @ T.T
                else:
                    z = model.encode_image(rgb_b, spec_b + noise)
                    logits = 100.0 * F.normalize(z.float(), dim=-1) @ T.T
                loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            kl_val = 0.0
            if args.prior_kl > 0:
                # per-chunk backward: accumulating graphs across chunks OOMs
                n_chunks = (spec_u.shape[0] + 127) // 128
                for j in range(0, spec_u.shape[0], 128):
                    with torch.autocast(device_type="cuda"):
                        z1u = model.encode_image(rgb_u[j:j + 128].float(), spec_u[j:j + 128].float())
                        logits1u = 100.0 * F.normalize(z1u.float(), dim=-1) @ T.T
                    kl_c = F.kl_div(F.log_softmax(logits1u, -1),
                                    F.softmax(logits0u[j:j + 128], -1),
                                    reduction="batchmean")
                    (args.prior_kl * kl_c / n_chunks).backward()
                    kl_val += float(kl_c)
            if trans_data is not None:
                # TSC: spectral-neighbourhood consistency + marginal balancing.
                # Neighbour graph lives on the spectral manifold (not RGB/space).
                t_rgb, t_spec, knn_idx = trans_data
                # per-chunk forward+backward: retaining all chunk graphs OOMs
                n_chunks = (t_spec.shape[0] + 127) // 128
                tsc_val = 0.0
                for cj in range(0, t_spec.shape[0], 128):
                    with torch.autocast(device_type="cuda"):
                        zt = model.encode_image(t_rgb[cj:cj + 128].float(), t_spec[cj:cj + 128].float())
                    t_logits = F.normalize(zt.float(), dim=-1) @ T.T
                    p_all = F.softmax(t_logits, -1)
                    # KL(mean(p) || uniform) = log C + sum m_c log m_c  (>= 0)
                    m = p_all.mean(0)
                    l_marg = (m * m.clamp_min(1e-8).log()).sum() +                         torch.log(torch.tensor(float(p_all.shape[1]), device=t_logits.device))
                    # JS(p_i, p_j) over spectral kNN pairs
                    pj = p_all[knn_idx[cj:cj + 128]]                   # (n, k, C)
                    log_m = ((p_all.unsqueeze(1) + pj) / 2).clamp_min(1e-8).log()
                    l_cons = 0.5 * F.kl_div(log_m, p_all.unsqueeze(1).expand_as(pj),
                                            reduction="batchmean") +                         0.5 * F.kl_div(log_m, pj, reduction="batchmean")
                    l_tsc = (l_marg + l_cons) / n_chunks
                    (args.trans_w * l_tsc).backward()
                    tsc_val += float(l_tsc)
                loss = loss.detach() + args.trans_w * tsc_val
            opt.step()
            sched.step()
            if args.prior_kl > 0:
                loss = loss.detach() + args.prior_kl * kl_val / max(n_chunks, 1)
            if (ep + 1) % 100 == 0 or ep == 0:
                print(f"ep {ep+1}: loss {loss.item():.4f}")
        train_time = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        torch.save({"state": model.state_dict(), "config": vars(args)}, ckpt_path)
        print(f"trained in {train_time:.1f}s, peak {peak:.2f}GB")

    # full-map evaluation
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    pred = predict_full_map(model, scene, T, device=device, fuse=args.fuse, s_res=s_res,
                            calibrate=args.calibrate, balanced=args.balanced,
                            balanced_iters=args.balanced_iters, balanced_tau=args.balanced_tau)
    eval_time = time.time() - t1
    res = metrics(pred, scene.gt, scene.num_classes)
    if test_mask is not None:
        res_test = metrics_masked(pred, scene.gt, scene.num_classes, test_mask)
        res_train = metrics_masked(pred, scene.gt, scene.num_classes, train_mask)
        res["heldout_mIoU"] = res_test["mIoU"]
        res["heldout_OA"] = res_test["OA"]
        res["heldout_mRecall"] = res_test["mRecall"]
        res["train_mIoU"] = res_train["mIoU"]
        res["heldout_IoU_per_class"] = res_test["IoU_per_class"]
        print(f"held-out ceiling: mIoU {res['heldout_mIoU']:.4f} OA {res['heldout_OA']:.4f}", flush=True)
    if args.base_ids:
        res = split_metrics(res, [int(i) for i in args.base_ids.split(",")], scene.num_classes)
        res["s_residual"] = float(s_res.exp()) if s_res is not None else None
    res.update({"tag": tag, "eval_time_s": eval_time,
                "eval_peak_GB": torch.cuda.max_memory_allocated() / 2**30,
                "bins": len(bins), "bin_um": BIN_UM,
                "tokenize": args.tokenize, "trans_w": args.trans_w,
                "knn": args.knn, "fuse": args.fuse, "enc": args.enc})
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2))
    with open(os.path.join(args.out, f"train_metrics_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.save(os.path.join(args.out, f"pred_map_{tag}.npy"), pred)


if __name__ == "__main__":
    main()
