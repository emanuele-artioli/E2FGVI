"""Model definitions for E2FGVI."""

from .e2fgvi import InpaintGenerator as E2FGVI
from .e2fgvi_hq import InpaintGenerator as E2FGVIHQ

__all__ = ["E2FGVI", "E2FGVIHQ"]
