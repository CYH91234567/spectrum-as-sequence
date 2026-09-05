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
