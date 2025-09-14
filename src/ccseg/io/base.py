from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Metadata:
    dx: float
    dy: float
    dz: float
    channels: int
    slices: int
    name: Optional[str]

    @property
    def spacing(self):
        return (self.dz, self.dx, self.dy)


class ImageReader(ABC):
    @abstractmethod
    def read(self, filepath) -> tuple[list[np.ndarray], Metadata]:
        """Read an image from the given filepath."""
        pass
