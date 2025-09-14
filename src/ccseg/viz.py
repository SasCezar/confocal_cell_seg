# ccseg/view/napari_view.py
from __future__ import annotations
from typing import Iterable, Optional, Dict, Any, Tuple
import numpy as np
import napari


def _default_img_kwargs(arr: np.ndarray, render_3d: bool, volume_mode: str) -> dict:
    # avoid percentile issues for small ranges
    v1, v99 = np.percentile(arr, (1, 99)) if arr.size else (0.0, 1.0)
    kw = {"blending": "translucent", "contrast_limits": (float(v1), float(v99))}
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
            v.add_image(data, name=name, **kwargs)
        elif kind == "labels":
            kwargs.setdefault("blending", "translucent")
            v.add_labels(data, name=name, **kwargs)
        elif kind == "points":
            kwargs.setdefault("size", points_size)
            kwargs.setdefault("face_color", "yellow")
            v.add_points(data, name=name, **kwargs)

    napari.run()
