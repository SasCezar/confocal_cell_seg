from __future__ import annotations
import logging
from typing import Dict, Tuple, Literal, Any
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import ball, closing, h_maxima, opening, remove_small_objects
from skimage.segmentation import watershed, relabel_sequential

from skimage.measure import regionprops
from ccseg.io.base import Metadata
from ccseg.settings import SegmentationCfg
from skimage.morphology import local_maxima

Kind = Literal["image", "labels", "points", "auto"]


def centroids_from_labels(
    labels: np.ndarray,
    spacing_um: Tuple[float, float, float],
) -> np.ndarray:
    dz, dy, dx = spacing_um
    rows = []
    for p in regionprops(labels):
        zc, yc, xc = p.centroid
        rows.append(
            {
                "label": int(p.label),
                "z_vox": float(zc),
                "y_vox": float(yc),
                "x_vox": float(xc),
                "z_um": float(zc * dz),
                "y_um": float(yc * dy),
                "x_um": float(xc * dx),
                "volume_vox": int(p.area),
            }
        )
    df = pd.DataFrame(rows)
    pts = df[["z_vox", "y_vox", "x_vox"]].to_numpy(dtype=np.float32) if len(df) else np.empty((0, 3))
    return df, pts


def preprocess_volume(volume: npt.NDArray) -> npt.NDArray:
    return np.asarray(volume, dtype=np.float32)


def binarize_otsu(vol: npt.NDArray) -> tuple[npt.NDArray[np.bool_], float]:
    th = float(threshold_otsu(vol))
    return (vol > th), th


def apply_morphology(
    bw: npt.NDArray[np.bool_],
    open_radius: int | None,
    close_radius: int | None,
) -> npt.NDArray[np.bool_]:
    if open_radius:
        bw = opening(bw, ball(open_radius))
    if close_radius:
        bw = closing(bw, ball(close_radius))
    return bw


def remove_small(bw: npt.NDArray[np.bool_], min_voxels: int) -> npt.NDArray[np.bool_]:
    if min_voxels > 1:
        bw = remove_small_objects(bw, min_voxels)
    return bw


def compute_distance(bw: npt.NDArray[np.bool_], spacing: Tuple[float, float, float]) -> npt.NDArray:
    dz, dy, dx = spacing
    return ndi.distance_transform_edt(bw, sampling=(dz, dy, dx)).astype(np.float16, copy=False)


def seed_markers(
    bw: npt.NDArray[np.bool_],
    distance: npt.NDArray,
    cfg,  # MarkersCfg with .h
    spacing: tuple[float, float, float],
) -> np.ndarray:
    dz, dy, dx = spacing
    mu_per_vox = (float(dz) + float(dy) + float(dx)) / 3.0

    # 1) Work in float to avoid unsigned underflow / integer no-ops
    img = np.asarray(distance, dtype=np.float32)

    # 2) Clean numerics
    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # 3) Set background to a value strictly below any foreground
    #    so reconstruction doesn’t get confused at mask borders
    h_val = float(cfg.h) * mu_per_vox
    h_val = max(h_val, np.finfo(np.float32).eps)
    bg_fill = (img.min() - 2.0 * h_val) if img.size else -1.0
    if bw is not None:
        img = np.where(bw, img, bg_fill)

    # 4) Try h-maxima; if reconstruction fails for any reason, fall back
    try:
        seeds = h_maxima(img, h=h_val)
    except ValueError:
        # Fallback: purely morphological local maxima inside mask
        logging.warning("Using Local Maxima")
        seeds = local_maxima(img)

    seeds &= bw
    markers, _ = ndi.label(seeds, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return markers


def run_watershed(
    distance: npt.NDArray,
    markers: np.ndarray,
    mask: npt.NDArray[np.bool_],
    compactness: float,
) -> np.ndarray:
    return watershed(-distance, markers=markers, mask=mask, compactness=float(compactness))


def clean_and_relabel(labels: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels > 1:
        sizes = np.bincount(labels.ravel())
        if sizes.size:
            small = np.where(sizes < min_voxels)[0]
            small = small[small != 0]
            if small.size:
                labels[np.isin(labels, small)] = 0
    labels, _, _ = relabel_sequential(labels)
    return labels


def _density_from_mask(bw: np.ndarray, spacing: tuple, sigma_um: float = 2.5, core_alpha: float = 1.5) -> np.ndarray:
    dz, dy, dx = spacing
    d_in = ndi.distance_transform_edt(bw, sampling=(dz, dy, dx)).astype(np.float32)
    mass = (d_in**core_alpha).astype(np.float32) if core_alpha > 0 else bw.astype(np.float32)
    sig = (sigma_um / dz, sigma_um / dy, sigma_um / dx)
    density = ndi.gaussian_filter(mass, sigma=sig).astype(np.float32)
    norm = ndi.gaussian_filter(np.ones_like(mass, dtype=np.float32), sigma=sig)
    eps = np.finfo(np.float32).eps
    density /= norm + eps
    m = float(density.max())
    if m > 0:
        density /= m
    return density


def segment_3d(
    channels: list[np.ndarray],
    metadata: Metadata,
    cfg: SegmentationCfg,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (bw, labels, stages)
    stages: dict[str, Union[np.ndarray, Dict[str, Any]]]
      If value is a dict, it must have: {"kind": Kind, "data": ndarray, "kwargs": dict}
    """
    stages: Dict[str, Any] = {}

    def collect(name: str, data: np.ndarray, *, kind: Kind = "auto", **kwargs) -> None:
        stages[name] = {"kind": kind, "data": data, "kwargs": kwargs}

    # Convenience wrappers to minimize kwargs repetition
    def collect_img(name: str, data: np.ndarray, **kwargs) -> None:
        collect(name, data, kind="image", **kwargs)

    def collect_labels(name: str, data: np.ndarray, **kwargs) -> None:
        collect(name, data.astype(np.int32, copy=False), kind="labels", **kwargs)

    def collect_points(name: str, pts: np.ndarray, **kwargs) -> None:
        collect(name, pts.astype(np.float32, copy=False), kind="points", **kwargs)

    # ---------------- pipeline ----------------
    volume = channels[0]
    vol = preprocess_volume(volume)
    collect_img("volume", vol, colormap="gray", opacity=0.85)

    bw, th = binarize_otsu(vol)
    collect_img("mask_otsu", bw, colormap="gray")

    bw_morph = apply_morphology(bw, open_radius=cfg.morphology.open_radius, close_radius=cfg.morphology.close_radius)
    collect_img("mask_morph", bw_morph, colormap="gray")

    bw_clean = remove_small(bw_morph, cfg.min_voxels)
    collect_img("mask_small_removed", bw_clean, colormap="gray")

    spacing: Tuple[float, float, float] = metadata.spacing  # (dz, dy, dx)
    distance = compute_distance(bw_clean, spacing)
    collect_img("distance_um", distance, colormap="magma", opacity=0.7)

    markers = seed_markers(bw_clean, distance, cfg.markers, spacing)

    labels_raw = run_watershed(
        distance=distance, markers=markers, mask=bw_clean, compactness=cfg.watershed.compactness
    )
    collect_labels("labels_raw", labels_raw, blending="translucent")

    labels = clean_and_relabel(labels_raw, cfg.min_voxels)
    collect_labels("labels", labels, blending="translucent")

    df, pts = centroids_from_labels(labels, metadata.spacing)
    collect_points("markers", pts)

    density = _density_from_mask(bw_clean, metadata.spacing)
    collect_img("density_prob", density, colormap="magma")

    return df, stages
