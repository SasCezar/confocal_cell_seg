import numpy as np
from tifffile import imwrite

from readlif.reader import LifImage


def _need_bigtiff(shape, dtype=np.uint16):
    # rough size check: T*Z*C*Y*X*bytes_per_pixel
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    return nbytes >= (2**32 - 1)  # ~4 GiB


def lif_image_to_imagej_tiff_raw(img: LifImage, out_path, channel_names=None, compression="zlib"):
    T, Z, C = int(img.dims.t), int(img.dims.z), int(img.channels)
    Y, X = int(img.dims_n[2]), int(img.dims_n[1])

    bit_depth = max(img.bit_depth) if isinstance(img.bit_depth, tuple) else int(img.bit_depth)
    dtype = np.uint8 if bit_depth <= 8 else np.uint16

    data = np.empty((T, Z, C, Y, X), dtype=dtype)
    for t in range(T):
        for z in range(Z):
            for c in range(C):
                a = np.asarray(img.get_frame(z=z, t=t, c=c), dtype=dtype)
                data[t, z, c] = a

    sx_ppu, sy_ppu, sz_ppu, _ = img.scale
    dz_um = (1.0 / sz_ppu) if sz_ppu else 1.0

    ij_meta = {
        "axes": "TZCYX",
        "unit": "micron",
        "spacing": dz_um,  # µm
        "Labels": channel_names or [f"Ch{i}" for i in range(C)],
    }

    imwrite(
        str(out_path),
        data,
        imagej=True,
        metadata=ij_meta,
        resolution=(float(sx_ppu) if sx_ppu else 1.0, float(sy_ppu) if sy_ppu else 1.0),  # px/µm as-is
        resolutionunit="NONE",
        photometric="minisblack",
        compression=compression,  # helps stay <4 GiB
        bigtiff=_need_bigtiff((T, Z, C, Y, X)),  # False if small enough
    )
