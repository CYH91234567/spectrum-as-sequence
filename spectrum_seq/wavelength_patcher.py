"""M4 wavelength-space tokenization (pilot v2 core).

Patches are defined in PHYSICAL wavelength space (fixed-width bins centred on
absolute wavelengths), not in band-index space. Every sensor yields the same
token sequence over the shared VNIR window [wl_min, wl_max], so the token
axis is sensor-aligned by construction -- this is the mechanism that failed
in pilot v1 when patching by band index.

Each bin: (bands falling inside, zero-padded to `max_bands`, plus a per-band
validity mask). A shared channel-independent linear embeds each bin; the
positional encoding encodes the ABSOLUTE bin-centre wavelength (micrometres),
so tokens carry their physical identity across sensors.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_bin_plan(wavelengths_um, wl_min, wl_max, bin_width_um):
    """Group band indices into wavelength bins.

    Returns (bins, centers): bins[i] = list of band indices; centers[i] in um.
    """
    bins, centers = [], []
    n = int(math.ceil((wl_max - wl_min) / bin_width_um))
    for i in range(n):
        lo = wl_min + i * bin_width_um
        hi = lo + bin_width_um
        idx = [j for j, w in enumerate(wavelengths_um) if lo <= w < hi]
        if idx:
            bins.append(idx)
            centers.append(lo + bin_width_um / 2)
    return bins, centers


class WavelengthPatcher(nn.Module):
    """Spectrum (N, B) + bin plan -> aligned spectral tokens (N, T, d)."""

    def __init__(self, bins, centers, max_bands: int, d_model: int = 128):
        super().__init__()
        self.bins = bins
        self.centers = centers
        self.max_bands = max_bands
        self.d = d_model
        # channel-independent embedding: band values + validity mask channel
        self.embed = nn.Linear(2 * max_bands, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.register_buffer("pe", self._abs_wl_pe(centers, d_model), persistent=False)

    @staticmethod
    def _abs_wl_pe(centers_um, d):
        """Sinusoidal PE over absolute wavelength (um), scaled so 0.4-2.5um
        spans a comfortable phase range."""
        half = d // 2
        freq = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(1000.0) / max(half - 1, 1)))
        ang = torch.tensor(centers_um, dtype=torch.float32)[:, None] * freq[None, :] * 10.0
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (T, d)

    def forward(self, x):
        N, B = x.shape
        T = len(self.bins)
        vals = x.new_zeros(N, T, self.max_bands)
        mask = x.new_zeros(N, T, self.max_bands)
        for t, idx in enumerate(self.bins):
            k = min(len(idx), self.max_bands)
            vals[:, t, :k] = x[:, idx[:k]]
            mask[:, t, :k] = 1.0
        tok = self.ln(self.embed(torch.cat([vals, mask], dim=-1)))
        return tok + self.pe[None, :, :]


class SpecEncoder(nn.Module):
    """M1+M2 over wavelength-aligned tokens: channel-independent embedding,
    then a small transformer over the (sensor-aligned) token sequence."""

    def __init__(self, bins, centers, max_bands, d_model=128, layers=2, heads=4, ff=256):
        super().__init__()
        self.patcher = WavelengthPatcher(bins, centers, max_bands, d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d_model, 768)  # project to CLIP ViT-B/16 width

    def forward(self, x):
        return self.out(self.encoder(self.patcher(x)))  # (N, T, 768)


class BandIndexPatcher(nn.Module):
    """Ablation T4/T5: tokenize by band index (equal-band-count patches), i.e.
    v1-style patching inside the injected architecture. Physical axis plays no
    role; positional encoding uses relative token position. When n_tokens is
    given, bands are split into that many contiguous equal chunks (T5)."""

    def __init__(self, num_bands: int, patch_len: int = 25, stride: int = 5,
                 n_tokens: int = None, d_model: int = 128):
        super().__init__()
        self.d = d_model
        self.max_bands = num_bands
        if n_tokens is None:                       # T4: overlapping patches
            self.P, self.S = patch_len, stride
            self.T = (num_bands - patch_len) // stride + 1
            self.chunks = [list(range(i * stride, i * stride + patch_len))
                           for i in range(self.T)]
        else:                                      # T5: n_tokens equal chunks
            self.P = self.S = None
            self.T = n_tokens
            k = num_bands // n_tokens
            self.chunks = [list(range(i * k, (i + 1) * k)) for i in range(n_tokens)]
        self.max_chunk = max(len(c) for c in self.chunks)
        self.embed = nn.Linear(2 * self.max_chunk, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.register_buffer("pe", self._rel_pe(self.T, d_model), persistent=False)

    @staticmethod
    def _rel_pe(T, d):
        half = d // 2
        pos = torch.arange(T, dtype=torch.float32) / max(T - 1, 1)
        freq = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1)))
        ang = pos[:, None] * freq[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, x):
        N, B = x.shape
        vals = x.new_zeros(N, self.T, self.max_chunk)
        mask = x.new_zeros(N, self.T, self.max_chunk)
        for t, idx in enumerate(self.chunks):
            k = len(idx)
            vals[:, t, :k] = x[:, idx]
            mask[:, t, :k] = 1.0
        tok = self.ln(self.embed(torch.cat([vals, mask], dim=-1)))
        return tok + self.pe[None, :, :]


def build_spec_encoder(tokenize: str, scene, d_model: int = 128, layers: int = 2):
    """Factory for the M4 tokenize ablation: returns encoder producing
    (N, T, 768) tokens from (N, B) spectra."""
    from .wavelength_patcher import SpecEncoder  # local import guard

    if tokenize.startswith("wl"):
        width = float(tokenize[2:]) / 1000.0     # 'wl50' -> 0.05um
        bins, centers = build_bin_plan(scene.wavelengths_um, 0.43, 0.86, width)
        max_bands = max(len(b) for b in bins) + 2
        return SpecEncoder(bins, centers, max_bands, d_model=d_model, layers=layers), len(bins)
    elif tokenize == "bandindex":
        enc = SpecEncoderFromPatcher(BandIndexPatcher(scene.cube.shape[-1]), d_model, layers)
        return enc, BandIndexPatcher(scene.cube.shape[-1]).T
    elif tokenize.startswith("bandeq"):
        n_tok = int(tokenize[6:])
        enc = SpecEncoderFromPatcher(BandIndexPatcher(scene.cube.shape[-1], n_tokens=n_tok), d_model, layers)
        return enc, n_tok
    raise ValueError(tokenize)


class SpecEncoderFromPatcher(nn.Module):
    """Same head as SpecEncoder but around a BandIndexPatcher."""

    def __init__(self, patcher, d_model: int = 128, layers: int = 2):
        super().__init__()
        self.patcher = patcher
        enc = nn.TransformerEncoderLayer(d_model, 4, 256, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d_model, 768)

    def forward(self, x):
        return self.out(self.encoder(self.patcher(x)))


class SpecEncoderVariant(nn.Module):
    """Kernel-control ablation (T3.3): same wavelength binning, same interface,
    three sequence-modeling kernels:
      ts  : shared embed + TransformerEncoder over bin tokens (default)
      cnn : shared embed + temporal Conv1d over bin tokens
      mlp : shared embed only (no cross-token interaction)
    """

    def __init__(self, bins, centers, max_bands, d_model=128, variant="ts"):
        super().__init__()
        self.patcher = WavelengthPatcher(bins, centers, max_bands, d_model)
        self.variant = variant
        if variant == "ts":
            enc = nn.TransformerEncoderLayer(d_model, 4, 256, dropout=0.1, batch_first=True)
            self.body = nn.TransformerEncoder(enc, 2)
        elif variant == "cnn":
            self.body = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1))
        elif variant == "mlp":
            self.body = nn.Identity()
        self.out = nn.Linear(d_model, 768)

    def forward(self, x):
        tok = self.patcher(x)
        if self.variant == "cnn":
            tok = self.body(tok.transpose(1, 2)).transpose(1, 2)
        else:
            tok = self.body(tok)
        return self.out(tok)
