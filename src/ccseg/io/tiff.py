from __future__ import annotations

from typing import Optional
import xml.etree.ElementTree as ET

from tifffile import TiffFile
from ccseg.io.base import ImageReader, Metadata


class TiffImageReader(ImageReader):
    def read(self, filepath):
        with TiffFile(filepath) as tif:
            arr = tif.asarray()  # raw array (z, c, y, x)
            if len(arr.shape) < 4:
                return [], {}
            arr = arr.transpose(1, 0, 2, 3)
            dz, dy, dx = self._voxel_size_um(tif)
            c = arr.shape[0]
            z = arr.shape[1]
            meta = Metadata(dx=dx, dy=dy, dz=dz, slices=z, channels=c, name="")

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
