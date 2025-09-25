from __future__ import annotations

from typing import Optional
import xml.etree.ElementTree as ET

from tifffile import TiffFile
from ccseg.io.base import ImageReader, Metadata
import numpy as np


class TiffImageReader(ImageReader):
    def read(self, filepath):
        with TiffFile(filepath) as tif:
            s = tif.series[0]
            axes = s.axes  # expected among: 'ZCYX', 'CYX', 'ZYX', 'YX', 'TZCYX', ...
            arr = s.asarray()

            # Require a real Z-stack
            if "Z" not in axes:
                return [], {'axes': axes}
            zsize = arr.shape[axes.index("Z")]
            if zsize <= 1:
                return [], {"axes": axes}

            # Bring to (C, Z, Y, X); synthesize C=1 if missing
            order_axes = [a for a in "CZYX" if a in axes]
            if not {"Y", "X"}.issubset(set(order_axes)):
                return [], {"axes": axes}  # not a planar image

            arr = np.transpose(arr, [axes.index(a) for a in order_axes])
            # Now arr has axes == order_axes
            if "C" not in order_axes:
                arr = arr[np.newaxis, ...]  # add C=1 at front
                order_axes = ["C"] + order_axes

            # Ensure final layout is exactly (C,Z,Y,X)
            # (Z is guaranteed to exist; we already checked)
            want = ["C", "Z", "Y", "X"]
            if order_axes != want:
                # Reorder if needed (usually it's already correct)
                idx = [order_axes.index(a) for a in want]
                arr = np.transpose(arr, idx)

            c, z, y, x = arr.shape
            dz, dy, dx = self._voxel_size_um(tif)
            meta = Metadata(dx=dx, dy=dy, dz=abs(dz), slices=z, channels=c, name="")

            # Return list of C arrays, each (Z, Y, X)
            channels = [arr[i] for i in range(c)]
            return channels, meta

    # voxel size helpers
    def _voxel_size_um(self, tif: TiffFile):
        """Return (dz, dy, dx) in microns if available; None for missing axes."""
        # 1) OME-TIFF (gold standard)
        ome = self._from_ome(tif.ome_metadata) if tif.ome_metadata else None
        if ome:
            return ome

        # 2) ImageJ TIFF + resolution tags
        ij = self._from_imagej(tif)
        if ij:
            return ij

        return (1, 1, 1)

    def _from_ome(self, ome_xml: str) -> tuple:
        """Parse OME-XML PhysicalSize* (convert to µm)."""
        try:
            root = ET.fromstring(ome_xml)
            pixels = root.find(".//{*}Pixels")
            if pixels is None:
                return (1, 1, 1)

            def to_um(val: Optional[str], unit: Optional[str]) -> Optional[float]:
                if val is None:
                    return 1
                u = (unit or "").lower()
                v = float(val)
                if u in ("µm", "um", "micrometer", "micrometre", "micron", "microns"):
                    return v
                if u in ("nm", "nanometer", "nanometre"):
                    return v / 1000.0
                if u in ("mm", "millimeter", "millimetre"):
                    return v * 1000.0
                if u in ("cm", "centimeter", "centimetre"):
                    return v * 10000.0
                if u in ("m", "meter", "metre"):
                    return v * 1_000_000.0
                return 1

            dz = to_um(pixels.get("PhysicalSizeZ"), pixels.get("PhysicalSizeZUnit"))
            dy = to_um(pixels.get("PhysicalSizeY"), pixels.get("PhysicalSizeYUnit"))
            dx = to_um(pixels.get("PhysicalSizeX"), pixels.get("PhysicalSizeXUnit"))
            if any(v is not None for v in (dz, dy, dx)):
                return (dz, dy, dx)
        except Exception:
            pass
        return (1, 1, 1)

    def _from_imagej(self, tif: TiffFile):
        """
        ImageJ TIFF: dz from 'spacing' (in 'unit'), XY from X/YResolution + ResolutionUnit.
        If ResolutionUnit==1 (none), fall back to ImageJ unit='micron' meaning ppu = px/µm.
        """
        page0 = tif.pages[0]
        ij = tif.imagej_metadata or {}
        unit = (ij.get("unit") or "").strip().lower()

        # dz from ImageJ spacing
        dz = ij.get("spacing")
        dz = float(dz) if dz is not None else 1

        # XY from resolution tags
        xres = page0.tags.get("XResolution")
        yres = page0.tags.get("YResolution")
        res_unit_tag = page0.tags.get("ResolutionUnit")
        res_unit = int(res_unit_tag.value) if res_unit_tag else 1  # 1=None, 2=Inch, 3=Centimeter

        def ratio_to_float(v) -> Optional[float]:
            if not v:
                return None
            try:
                # tuples are (num, den)
                if isinstance(v, tuple) and len(v) == 2 and v[1] != 0:
                    return v[0] / v[1]
                return float(v)
            except Exception:
                return 1

        x_ppu = ratio_to_float(xres.value) if xres else None  # pixels per unit
        y_ppu = ratio_to_float(yres.value) if yres else None

        def um_per_px(ppu: Optional[float]) -> Optional[float]:
            if not ppu or ppu <= 0:
                return None
            if res_unit == 2:  # inch
                return 25400.0 / ppu
            if res_unit == 3:  # centimeter
                return 10000.0 / ppu
            # res_unit == 1 (no absolute): ImageJ unit often 'micron' => ppu = px/µm
            if unit in ("µm", "um", "micrometer", "micrometre", "micron", "microns"):
                return 1.0 / ppu
            return 1

        dx = um_per_px(x_ppu)
        dy = um_per_px(y_ppu)

        if any(v is not None for v in (dz, dy, dx)):
            return (dz, dy, dx)
        return (1, 1, 1)
