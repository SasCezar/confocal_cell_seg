from __future__ import annotations
from typing import Iterable, Optional, Dict, Any, Tuple
import numpy as np
import napari


def _is_binary_like(arr: np.ndarray) -> bool:
    if arr.dtype == np.bool_:
        return True
    if np.issubdtype(arr.dtype, np.integer):
        # fast binary check: only 0/1 present
        amin = int(np.nanmin(arr))
        amax = int(np.nanmax(arr))
        return 0 <= amin and amax <= 1
    return False


def _safe_contrast_limits(arr: np.ndarray) -> Tuple[float, float]:
    """Return strictly increasing (lo, hi) suitable for napari."""
    if arr.size == 0:
        return (0.0, 1.0)

    # work with finite values only
    a = np.asarray(arr)
    finite = np.isfinite(a)
    if not finite.any():
        return (0.0, 1.0)

    a = a[finite]

    # robust to outliers; ignore NaNs
    v1, v99 = np.nanpercentile(a, (1, 99))

    # fallbacks if percentiles are degenerate
    if not np.isfinite(v1) or not np.isfinite(v99):
        v1, v99 = float(np.nanmin(a)), float(np.nanmax(a))

    if v1 == v99:
        # binary/constant images: force a visible window
        if _is_binary_like(arr):
            return (0.0, 1.0)
        # otherwise add a small epsilon depending on dtype
        if np.issubdtype(arr.dtype, np.integer):
            eps = 1.0  # one count
        else:
            # small relative epsilon for floats
            base = abs(v1) if v1 != 0.0 else 1.0
            eps = max(1e-6, 1e-3 * base)
        return (float(v1), float(v1 + eps))

    return (float(v1), float(v99))


def _default_img_kwargs(arr: np.ndarray, render_3d: bool, volume_mode: str) -> dict:
    lo, hi = _safe_contrast_limits(np.asarray(arr))
    kw = {"blending": "translucent", "contrast_limits": (lo, hi)}
    if render_3d:
        kw["rendering"] = volume_mode
        kw["depiction"] = "volume"
    else:
        kw["rendering"] = "mip"
    return kw


def _infer_from_array(name: str, arr: np.ndarray) -> Tuple[str, np.ndarray, Dict[str, Any]]:
    nm = name.lower()
    if arr.ndim == 2 and arr.shape[1] in (2, 3) and np.issubdtype(arr.dtype, np.number):
        return "points", arr.astype(np.float32, copy=False), {"face_color": "yellow"}
    if arr.dtype == np.bool_:
        # keep as image so it overlays nicely; colormap gray
        return "image", arr.astype(np.uint8, copy=False), {"colormap": "gray", "opacity": 0.6}
    if np.issubdtype(arr.dtype, np.integer):
        return "labels", arr.astype(np.int32, copy=False), {"blending": "translucent"}
    # float
    if any(k in nm for k in ("distance", "prob", "density")):
        return "image", arr, {"colormap": "magma", "opacity": 0.7}
    return "image", arr, {"colormap": "gray", "opacity": 0.85}


def visualize(
    stages: Dict[str, Any],
    title: str = "Segmentation debug",
    keys: Optional[Iterable[str]] = None,
    render_3d: bool = True,
    volume_mode: str = "attenuated_mip",
    points_size: int = 8,
):
    # preserve insertion order
    items = list(stages.items()) if keys is None else [(k, stages[k]) for k in keys if k in stages]

    v = napari.Viewer(title=title, ndisplay=3 if render_3d else 2)

    for name, obj in items:
        if isinstance(obj, dict) and "kind" in obj and "data" in obj:
            kind = obj["kind"]
            data = obj["data"]
            kwargs = dict(obj.get("kwargs") or {})
        else:
            kind, data, kwargs = _infer_from_array(name, np.asarray(obj))

        if kind == "image":
            base = _default_img_kwargs(np.asarray(data), render_3d, volume_mode)
            for kk, vv in base.items():
                kwargs.setdefault(kk, vv)

            # optional: hide truly empty images to reduce clutter
            try:
                if np.nanmax(data) == 0:
                    kwargs.setdefault("visible", False)
            except Exception:
                pass

            v.add_image(data, name=name, **kwargs)

        elif kind == "labels":
            kwargs.setdefault("blending", "translucent")
            v.add_labels(data, name=name, **kwargs)

        elif kind == "points":
            kwargs.setdefault("size", points_size)
            kwargs.setdefault("face_color", "yellow")
            v.add_points(data, name=name, **kwargs)

    napari.run()
