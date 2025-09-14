from __future__ import annotations

from typing import Tuple
from pathlib import Path

import numpy as np
from readlif.reader import LifFile, LifImage

from ccseg.io.base import ImageReader, Metadata


class LifImageReader(ImageReader):
    """
    Read a Leica .lif series as a (Z, Y, X) numpy array and basic metadata.
    Keeps the same (img, meta) contract as TiffImageReader.
    """

    def read(self, filepath: str | Path, *, series: int = 1):
        lif = LifFile(str(filepath))

        # Get series
        img = self._get_series(lif, series)
        name = f"{Path(filepath).stem}_{img.name}"
        # Spacing (µm)
        dz, dy, dx = self._spacing_um_from_img(img)

        nc = 2

        channels: list[np.ndarray] = []
        for c in range(nc):
            # Volume (Z, Y, X)
            volume = self._load_stack_zyx(img, c=c)
            channels.append(volume)

        meta = Metadata(dx=dx, dy=dy, dz=dz, channels=nc, name=name, slices=channels[0].shape[0])
        return channels, meta

    # -------------------- helpers --------------------

    def _get_series(self, lif: LifFile, series: int) -> LifImage:
        # Prefer direct indexing if available, else fall back to iteration
        try:
            return lif.get_image(series)
        except Exception:
            for idx, img in enumerate(lif.get_iter_image()):
                if idx == series:
                    return img
        raise IndexError(f"Series {series} not found in LIF file.")

    def _spacing_um_from_img(self, img: LifImage) -> Tuple:
        """
        Return (dz, dy, dx) in µm. readlif's `img.scale` is in px/µm for (x, y, z, t).
        We invert safely and return None if missing.
        """
        # img.scale -> (sx_px_per_um, sy_px_per_um, sz_px_per_um, st_frames_per_sec_or_None)
        sx, sy, sz, _ = getattr(img, "scale", (1, 1, 1, None))

        def inv_or_none(v) -> float:
            try:
                v = float(v)
                return 1.0 / v if v > 0 else 1
            except Exception:
                return 1

        dx_um = inv_or_none(sx)
        dy_um = inv_or_none(sy)
        dz_um = inv_or_none(sz)
        return (dz_um, dy_um, dx_um)

    def _load_stack_zyx(self, img: LifImage, *, c: int) -> np.ndarray:
        """
        Return a (Z, Y, X) stack for a given time & channel.
        Clamps t/c inside valid ranges. Uses mode m=0 (first acquisition mode).
        """
        z_dim = img.dims.z

        if z_dim < 1:
            raise ValueError("This series has no Z planes.")

        planes = [np.asarray(pil_im) for pil_im in img.get_iter_z(t=0, c=c, m=0)]

        return np.stack(planes, axis=0)
