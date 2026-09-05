"""Hyperspectral dataset loaders for the Spectrum-as-Sequence pilot.

Pavia University  : (610, 340, 103), 9 classes  (0.43-0.86 um, ~4.2 nm spacing)
Indian Pines (corr): (145, 145, 200), 16 classes (0.4-2.45 um, ~10 nm spacing)

Preprocessing (unsupervised, scene-level):
  1. per-scene band-wise z-normalisation
  2. per-pixel scale normalisation (remove illumination magnitude, keep spectral shape)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.io as sio


@dataclass
class HSIScene:
    cube: np.ndarray          # (H, W, B) float32, normalised
    gt: np.ndarray            # (H, W) int, 0 = unlabeled
    class_names: list
    wavelengths_um: np.ndarray  # (B,) approximate band centre wavelengths

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def shape(self):
        return self.cube.shape


PAVIAU_BANDS_UM = np.linspace(0.43, 0.86, 103)
# AVIRIS (Indian Pines / Salinas): 224 nominal bands, 20 water-absorption
# bands removed -> 200 (IP) / 204 (Salinas) bands, ~10 nm VNIR spacing.
AVIRIS_BANDS_UM = np.concatenate([
    np.arange(0.4005, 1.33, 0.01),          # 94 bands 0.40-1.32
    np.arange(1.43, 1.81, 0.01),            # 39 bands 1.43-1.80
    np.arange(1.96, 2.39, 0.01),            # 44+ bands up to 2.36
])
IP_BANDS_UM = AVIRIS_BANDS_UM[:200]
SALINAS_BANDS_UM = AVIRIS_BANDS_UM[:204]


PAVIAU_CLASSES = [
    "Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets",
    "Bare soil", "Bitumen", "Self-blocking bricks", "Shadows",
]
IP_CLASSES = [
    "Alfalfa", "Corn notill", "Corn mintill", "Corn", "Grass pasture",
    "Grass trees", "Grass pasture mowed", "Hay windrowed", "Oats",
    "Soybean notill", "Soybean mintill", "Soybean clean", "Wheat",
    "Woods", "Buildings grass trees drives", "Stone steel towers",
]
SALINAS_CLASSES = [
    "Broccoli green weeds 1", "Broccoli green weeds 2", "Fallow",
    "Fallow rough plow", "Fallow smooth", "Stubble", "Celery",
    "Grapes untrained", "Soil vineyard develop",
    "Corn senesced green weeds", "Lettuce romaine 4wk",
    "Lettuce romaine 5wk", "Lettuce romaine 6wk", "Lettuce romaine 7wk",
    "Vinyard untrained", "Vinyard vertical trellis",
]

_RGB_BANDS = {
    # scene key -> (blue, green, red) band indices approximating 470/550/640 nm
    "paviau": (9, 29, 52),
    "indianpines": (10, 15, 25),
    "salinas": (10, 15, 25),
}


def _norm_cube(cube: np.ndarray) -> np.ndarray:
    cube = cube.astype(np.float32)
    mu = cube.reshape(-1, cube.shape[-1]).mean(0, keepdims=True)
    sd = cube.reshape(-1, cube.shape[-1]).std(0, keepdims=True) + 1e-6
    cube = (cube - mu) / sd
    # per-pixel magnitude removal, keep spectral shape
    scale = np.linalg.norm(cube, axis=-1, keepdims=True) / np.float32(np.sqrt(cube.shape[-1]))
    return (cube / (scale + np.float32(1e-6))).astype(np.float32, copy=False)


def load_scene(name: str, data_dir: str) -> HSIScene:
    name = name.lower()
    if name in ("paviau", "pavia", "pavia_university"):
        cube = sio.loadmat(os.path.join(data_dir, "PaviaU.mat"))["paviaU"]
        gt = sio.loadmat(os.path.join(data_dir, "PaviaU_gt.mat"))["paviaU_gt"]
        scene = HSIScene(_norm_cube(cube), gt.astype(np.int64), PAVIAU_CLASSES, PAVIAU_BANDS_UM)
        scene.key = "paviau"
    elif name in ("indianpines", "ip", "indian_pines"):
        cube = sio.loadmat(os.path.join(data_dir, "Indian_pines_corrected.mat"))["indian_pines_corrected"]
        gt = sio.loadmat(os.path.join(data_dir, "Indian_pines_gt.mat"))["indian_pines_gt"]
        scene = HSIScene(_norm_cube(cube), gt.astype(np.int64), IP_CLASSES, IP_BANDS_UM)
        scene.key = "indianpines"
    elif name in ("salinas", "salinas_corrected"):
        cube = sio.loadmat(os.path.join(data_dir, "Salinas_corrected.mat"))["salinas_corrected"]
        gt = sio.loadmat(os.path.join(data_dir, "Salinas_gt.mat"))["salinas_gt"]
        scene = HSIScene(_norm_cube(cube), gt.astype(np.int64), SALINAS_CLASSES, SALINAS_BANDS_UM)
        scene.key = "salinas"
    else:
        raise ValueError(f"unknown scene {name}")
    return scene


def sample_shots(gt: np.ndarray, shots: int, seed: int = 0):
    """Return (pixels, labels): exactly `shots` labelled pixels per class."""
    rng = np.random.default_rng(seed)
    pixels, labels = [], []
    for c in range(1, gt.max() + 1):
        ys, xs = np.where(gt == c)
        idx = rng.choice(len(ys), size=min(shots, len(ys)), replace=False)
        for i in idx:
            pixels.append((int(ys[i]), int(xs[i])))
            labels.append(c - 1)
    return np.array(pixels), np.array(labels, dtype=np.int64)


def rgb_bands(scene: HSIScene):
    return _RGB_BANDS[scene.key]


def make_rgb_cube(scene: HSIScene) -> np.ndarray:
    """(H, W, 3) uint8-style float cube from approximate RGB bands."""
    b, g, r = rgb_bands(scene)
    rgb = scene.cube[..., [b, g, r]]
    lo = rgb.min()
    hi = rgb.max()
    return ((rgb - lo) / (hi - lo + 1e-9)).astype(np.float32)
