"""
ZoeDepth: Zero-shot Metric Depth Estimation
Embedded version for trial repo - no external pip install required.
"""

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

__version__ = "0.1.0"

__all__ = ['build_model', 'get_config']