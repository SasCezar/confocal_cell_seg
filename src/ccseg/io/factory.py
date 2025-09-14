from pathlib import Path
from .lif import LifImageReader
from .tiff import TiffImageReader
from .base import ImageReader


class ImageReaderFactory:
    """Factory to return appropriate reader based on file extension."""

    _readers = {
        ".lif": LifImageReader,
        ".tif": TiffImageReader,
        ".tiff": TiffImageReader,
    }

    @classmethod
    def get_reader(cls, filepath: str | Path) -> ImageReader:
        path = Path(filepath)
        ext = path.suffix.lower()
        if ext not in cls._readers:
            raise ValueError(f"Unsupported file extension: {ext}")
        return cls._readers[ext]()
