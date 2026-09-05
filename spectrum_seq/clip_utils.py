"""Frozen CLIP utilities (local checkpoint path; no network needed)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_clip(ckpt_path: str, device: str = "cuda"):
    """Load OpenAI CLIP ViT-B/16 from a local TorchScript checkpoint.

    open_clip.load_openai_model handles the TorchScript archive (torch>=2.6
    changed torch.load defaults, so create_model_and_transforms fails on it).
    """
    import open_clip
    model = open_clip.load_openai_model(ckpt_path, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    return model, tokenizer, None


@torch.no_grad()
def class_text_embeddings(clip_model, tokenize, names, device,
                          template="a remote sensing photo of {}.", batch=32):
    """Encode class-name prompts with frozen CLIP text tower -> (C, D) unit rows."""
    embs = []
    for i in range(0, len(names), batch):
        chunk = [template.format(n) for n in names[i:i + batch]]
        toks = tokenize(chunk).to(device)
        e = clip_model.encode_text(toks)
        embs.append(F.normalize(e.float(), dim=-1))
    return torch.cat(embs, 0)


@torch.no_grad()
def encode_rgb_patches(clip_model, rgb_cube, ys, xs, win=32, device="cuda", batch=1024, amp=True):
    """Encode win x win RGB patches centred at (ys, xs) through frozen CLIP.

    rgb_cube: (H, W, 3) float in [0, 1]. Mirrors the standard CLIP preprocess
    (bicubic resize to 224 + CLIP normalisation) directly in torch.
    Returns unit-norm embeddings (N, D).
    """
    H, W, _ = rgb_cube.shape
    pad = win // 2
    padded = np.pad(rgb_cube, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    cube_t = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H+2p,W+2p)
    mean = torch.tensor(CLIP_MEAN, device=device)[:, None, None]
    std = torch.tensor(CLIP_STD, device=device)[:, None, None]
    out = []
    for i in range(0, len(ys), batch):
        crops = []
        for y, x in zip(ys[i:i + batch], xs[i:i + batch]):
            crops.append(cube_t[:, :, y:y + win, x:x + win])
        inp = torch.cat(crops, 0)                                   # (b,3,win,win)
        inp = F.interpolate(inp, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
        inp = (inp - mean) / std
        with torch.autocast(device_type="cuda", enabled=amp):
            e = clip_model.encode_image(inp)
        out.append(F.normalize(e.float(), dim=-1))
    return torch.cat(out, 0)
