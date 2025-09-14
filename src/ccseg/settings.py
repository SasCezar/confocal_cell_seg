# src/ccseg/settings.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator
import yaml


class DataCfg(BaseModel):
    input_dir: Path = Field(default=Path("./data/"))
    output_dir: Path = Field(default=Path("/data/out/"))


class MorphologyCfg(BaseModel):
    """Structuring-element radii for 3D opening/closing (in voxels)."""

    open_radius: Optional[int] = Field(default=None, ge=0)
    close_radius: Optional[int] = Field(default=None, ge=0)


class MarkersCfg(BaseModel):
    """
    h-maxima sensitivity. The effective threshold is h * mean(voxel_size_um),
    so keep this scale-agnostic (≈ 0.5–2.0 is typical).
    """

    h: float = Field(default=1.0, ge=0.0)


class WatershedCfg(BaseModel):
    """Watershed post-splitting parameters."""

    compactness: float = Field(default=0.0, ge=0.0)


class SegmentationCfg(BaseModel):
    """
    Full segmentation configuration.

    Maps to your pipeline as:
      - morphology.open_radius / close_radius -> apply_morphology(...)
      - min_voxels -> remove_small(...), clean_and_relabel(...)
      - markers.h -> seed_markers(...)
      - watershed.compactness -> run_watershed(...)
    """

    min_voxels: int = Field(default=64, ge=1)
    morphology: MorphologyCfg = MorphologyCfg()
    markers: MarkersCfg = MarkersCfg()
    watershed: WatershedCfg = WatershedCfg()

    # --- light sanity checks ---

    @field_validator("morphology")
    @classmethod
    def _none_to_valid(cls, v: MorphologyCfg) -> MorphologyCfg:
        # Treat 0 like "no-op" to match your helpers’ truthy checks.
        if v.open_radius == 0:
            v.open_radius = None
        if v.close_radius == 0:
            v.close_radius = None
        return v


class RuntimeCfg(BaseModel):
    log_level: str = "INFO"


class VisualizationCfg(BaseModel):
    enable: bool = Field(default=True)
    layers: list = Field(default=[])


class ExportCfg(BaseModel):
    enable: bool = Field(default=True)
    layers: list = Field(default=[])


class ConvertCfg(BaseModel):
    lif_dir: Path = Field(default=Path("./data/"))
    tiff_out: Path = Field(default=Path("/data/out/"))


class Settings(BaseModel):
    segmentation: SegmentationCfg = SegmentationCfg()
    runtime: RuntimeCfg = RuntimeCfg()
    data: DataCfg = DataCfg()
    visualization: VisualizationCfg = VisualizationCfg()
    export: ExportCfg = ExportCfg()
    convert: ConvertCfg = ConvertCfg()

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
