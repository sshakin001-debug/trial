"""
Depth Anything encoder adapter for ZoeDepth framework.
Wraps the DINOv2-based Depth Anything encoder into the ZoeDepth pipeline.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

from zoedepth.models.base_models.dpt_dinov2.dpt import DepthAnythingDepthModel


class DepthAnythingEncoder(nn.Module):
    """
    Depth Anything encoder using DINOv2 + DPT decoder.
    This replaces the MiDaS encoder in standard ZoeDepth.
    """
    
    def __init__(self, model_type="vitl", **kwargs):
        super().__init__()
        self.model_type = model_type
        self.encoder = DepthAnythingDepthModel(model_type=model_type, **kwargs)
        
    def forward(self, x):
        return self.encoder(x)
    
    @property
    def embed_dim(self):
        """Return embedding dimension for DINOv2 variant."""
        dims = {
            "vits": 384,
            "vitb": 768,
            "vitl": 1024,
            "vitg": 1536,
        }
        return dims.get(self.model_type, 1024)


def build_depth_anything(model_type="vitl", pretrained_resources=None, **kwargs):
    """
    Factory function matching ZoeDepth builder pattern.
    
    Args:
        model_type: ViT model size ('vits', 'vitb', 'vitl', 'vitg')
        pretrained_resources: Checkpoint path or URL
        **kwargs: Additional arguments
        
    Returns:
        DepthAnythingEncoder model
    """
    model = DepthAnythingEncoder(model_type=model_type, **kwargs)
    
    if pretrained_resources:
        from zoedepth.models.model_io import load_local_checkpoint
        model = load_local_checkpoint(pretrained_resources, model)
    
    return model