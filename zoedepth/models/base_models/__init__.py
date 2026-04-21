"""
Base models package.
"""

from zoedepth.models.base_models.midas import MidasDecoder
from zoedepth.models.base_models.depth_anything import DepthAnythingEncoder

__all__ = ['MidasDecoder', 'DepthAnythingEncoder']