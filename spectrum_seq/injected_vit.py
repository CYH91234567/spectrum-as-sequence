"""Pilot v2: spectral token injection into the frozen CLIP ViT.

The wavelength-aligned spectral tokens (WavelengthPatcher -> SpecEncoder)
cross-attend INTO the frozen ViT at selected depths; the ViT then pools its
CLS token against text prompts as usual. Injection starts at zero gain
(gamma=0) so the model is exactly frozen CLIP at initialisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .wavelength_patcher import SpecEncoder


class SpecInjection(nn.Module):
    def __init__(self, d: int = 768, heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, s):
        a, _ = self.attn(self.norm_q(x), self.norm_kv(s), self.norm_kv(s), need_weights=False)
        return x + self.gamma * a


class InjectedCLIP(nn.Module):
    """Frozen CLIP ViT-B/16 + trainable spectral side-channel.

    rgb  : (N, 3, 224, 224) preprocessed image crops
    spec : (N, B) normalised spectra
    ->   (N, 512) CLIP embedding space (unit-normalise outside)
    """

    def __init__(self, clip_model, bins, centers, max_bands,
                 d_spec=128, spec_layers=2, inject_layers=(3, 6, 9), spec_enc=None):
        super().__init__()
        self.v = clip_model.visual
        for p in self.v.parameters():
            p.requires_grad_(False)
        # spec_enc override supports the M4 tokenize ablation (BandIndexPatcher etc.)
        self.spec_enc = spec_enc if spec_enc is not None else SpecEncoder(
            bins, centers, max_bands, d_model=d_spec, layers=spec_layers)
        self.inject_layers = tuple(inject_layers)
        self.inject = nn.ModuleList([SpecInjection(768) for _ in self.inject_layers])
        self.text_tower = clip_model.transformer  # unused here; text encoded separately
        self.proj = self.v.proj

    def encode_image(self, rgb, spec):
        v = self.v
        x = v._embeds(rgb)                       # (N, 197, 768)
        st = self.spec_enc(spec)                 # (N, T, 768)
        for i, r in enumerate(v.transformer.resblocks):
            x = r(x)
            if i in self.inject_layers:
                x = self.inject[self.inject_layers.index(i)](x, st)
        pooled, _ = v._pool(x)
        return pooled @ v.proj

    def encode_image_with_prior(self, rgb, spec):
        """Prior-preserving pair: z0 = frozen CLIP (no injection), z1 = injected.
        Caller fuses logits as 100*n0@T + s * 100*n1@T with learnable s >= 0."""
        v = self.v
        with torch.no_grad():
            x0 = v._embeds(rgb)
            for r in v.transformer.resblocks:
                x0 = r(x0)
            p0, _ = v._pool(x0)
            z0 = p0 @ v.proj
        x = v._embeds(rgb)
        st = self.spec_enc(spec)
        for i, r in enumerate(v.transformer.resblocks):
            x = r(x)
            if i in self.inject_layers:
                x = self.inject[self.inject_layers.index(i)](x, st)
        p1, _ = v._pool(x)
        z1 = p1 @ v.proj
        return z0, z1

    def trainable_parameters(self):
        return list(self.spec_enc.parameters()) + list(self.inject.parameters())
