"""
Model builder for ZoeDepth.
Handles local checkpoint loading and model construction.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union
import sys

from zoedepth.utils.config import get_config
from zoedepth.models.depth_model import DepthModel


def load_checkpoint(checkpoint_path: str, model: nn.Module, strict: bool = False) -> nn.Module:
    """Load checkpoint from local path."""
    if checkpoint_path.startswith('local::'):
        checkpoint_path = checkpoint_path[7:]
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict, strict=False)
    return model


def build_model(config, device: str = 'cuda') -> DepthModel:
    """
    Build ZoeDepth model based on configuration.
    
    Args:
        config: Configuration object from get_config()
        device: Device to load model on
        
    Returns:
        DepthModel instance
    """
    model_name = config.model.get('name', 'ZoeDepth')
    version = config.model.get('version_name', 'v1')
    use_depth_anything = config.model.get('use_depth_anything', False)
    
    if use_depth_anything:
        from zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepthAnythingV1
        model = ZoeDepthAnythingV1(config)
    elif 'nk' in version.lower():
        from zoedepth.models.zoedepth_nk.zoedepth_nk_v1 import ZoeDepthNKV1
        model = ZoeDepthNKV1(config)
    else:
        from zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepthV1
        model = ZoeDepthV1(config)
    
    pretrained_resource = getattr(config, 'pretrained_resource', None)
    if pretrained_resource:
        model = load_checkpoint(pretrained_resource, model)
    
    model.to(device)
    model.eval()
    return model
