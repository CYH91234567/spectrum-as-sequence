"""Spectrum-as-Sequence pilot model.

M1 SpectralPatcher      : band-axis patching + channel-independent embedding
                          (kernel migrated from PatchTST)
M2 InvertedBandAttention: band-patch tokens carry a spatial-context summary of the
                          whole spectrum (kernel migrated from iTransformer's
                          variable-token inversion); 2-layer transformer encoder
Head                    : linear projection into the frozen CLIP image-embedding
                          space (ViT-B/16, dim 512), cosine similarity to text
                          prompts for pixel classification.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SpectralPatcher(nn.Module):
    """M1: split a B-band spectrum into overlapping patches of length P
    (stride S), embed each patch with a shared linear layer (channel
    independence). Positional encoding encodes the patch's *relative*
    wavelength position in [0, 1] so the adapter transfers across sensors
    with different band counts/ranges."""

    def __init__(self, patch_len: int = 25, stride: int = 5, d_model: int = 128):
        super().__init__()
        self.P, self.S, self.d = patch_len, stride, d_model
        self.embed = nn.Linear(patch_len, d_model)
        self.ln = nn.LayerNorm(d_model)

    def num_tokens(self, B: int) -> int:
        return math.ceil(max(B - self.P, 0) / self.S) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, B) -> tokens (N, L, d). Pads B with reflect to fit patches."""
        N, B = x.shape
        L = self.num_tokens(B)
        need = self.P + self.S * (L - 1)
        if need > B:
            # pad the BAND axis (last dim of (N,1,B)) by replicating the last band
            x = torch.nn.functional.pad(x[:, None, :], (0, need - B, 0, 0), mode="replicate")[:, 0]
        # unfold into patches
        idx = torch.arange(self.P, device=x.device)[None, :] + self.S * torch.arange(L, device=x.device)[:, None]
        patches = x[:, idx]                      # (N, L, P)
        tok = self.ln(self.embed(patches))       # (N, L, d)
        # relative-wavelength sinusoidal positional encoding
        pos = (self.S * torch.arange(L, device=x.device, dtype=torch.float32)
               + self.P / 2) / max(need, B)      # (L,) in [0, 1]
        pe = self._sinpe(pos, self.d)            # (L, d)
        return tok + pe[None]

    @staticmethod
    def _sinpe(pos: torch.Tensor, d: int) -> torch.Tensor:
        half = d // 2
        freq = torch.exp(torch.arange(half, device=pos.device, dtype=torch.float32)
                         * (-math.log(10000.0) / max(half - 1, 1)))
        ang = pos[:, None] * freq[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class InvertedBandAttention(nn.Module):
    """M2: every band-patch token receives a projection of the *whole-spectrum*
    spatial context (inversion of the time-series variable-token idea), then a
    small transformer encoder models band-patch interactions."""

    def __init__(self, d_model: int = 128, nhead: int = 4, ff: int = 256, layers: int = 2, ctx_len: int = 3):
        super().__init__()
        self.ctx_len = ctx_len
        self.ctx = nn.Linear(1, d_model)  # per-band context scalar -> d (applied to pooled spectrum)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)

    def forward(self, tok: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        """tok: (N, L, d); spectrum: (N, B) raw (normalised) spectrum for context."""
        # spatial-context summary: simple band statistics of the pixel itself +
        # its 3x3 neighbourhood is folded in upstream; here the global spectral
        # context is broadcast to every token (band axis -> token axis inversion)
        ctx = self.ctx(spectrum[:, :, None].mean(dim=1, keepdim=True))  # (N, 1, d)
        return self.encoder(tok + ctx)


class SpectrumSeqAdapter(nn.Module):
    def __init__(self, d_model: int = 128, patch_len: int = 25, stride: int = 5,
                 layers: int = 2, clip_dim: int = 512, ctx_len: int = 3):
        super().__init__()
        self.patcher = SpectralPatcher(patch_len, stride, d_model)
        self.backbone = InvertedBandAttention(d_model, layers=layers, ctx_len=ctx_len)
        self.head = nn.Linear(d_model, clip_dim)
        self.logit_scale = 100.0

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """spectrum: (N, B) -> unit-norm CLIP-space embedding (N, 512)."""
        tok = self.patcher(spectrum)
        h = self.backbone(tok, spectrum).mean(dim=1)
        z = self.head(h)
        return torch.nn.functional.normalize(z, dim=-1)
