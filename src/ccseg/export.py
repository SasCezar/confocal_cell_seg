from __future__ import annotations
from typing import Iterable, Optional, Dict, Any
import tifffile
from pathlib import Path


def dump(stages: Dict[str, Any], keys: Optional[Iterable[str]], image_name: str, out_dir: Path | str):
    items = list(stages.items()) if keys is None else [(k, stages[k]) for k in keys if k in stages]

    out_dir = Path(out_dir)
    for name, obj in items:
        out_path = out_dir / name / f"{image_name}.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out_path, obj["data"])
