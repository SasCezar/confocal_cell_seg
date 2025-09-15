from __future__ import annotations
import logging
from typing import Iterable, Optional, Dict, Any
import numpy as np
import tifffile
from pathlib import Path
from ccseg.viz import _infer_from_array


def dump(stages: Dict[str, Any], keys: Optional[Iterable[str]], image_name: str, out_dir: Path | str, spacing):
    items = list(stages.items()) if keys is None else [(k, stages[k]) for k in keys if k in stages]

    out_dir = Path(out_dir)
    for name, obj in items:
        if isinstance(obj, dict) and "kind" in obj and "data" in obj:
            kind = obj["kind"]
            data = obj["data"]
            kwargs = dict(obj.get("kwargs") or {})
        else:
            kind, data, kwargs = _infer_from_array(name, np.asarray(obj))

        data = data.astype(np.float32)

        out_path = out_dir / name / f"{image_name}.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving layer {name} - Kind {kind} - data shape {data.shape}")
        dz, dy, dx = spacing
        ij_meta = {"axes": "ZYX", "unit": "micron", "spacing": dz}
        xres = 1.0 / dx  # px/µm
        yres = 1.0 / dy
        tifffile.imwrite(
            out_path,
            data,
            imagej=True,
            metadata=ij_meta,
            photometric="minisblack",
            compression="zlib",
            resolution=(xres, yres),
            resolutionunit="NONE",  # means "unit" above defines scale
        )
