"""
Checkpoint loading utilities.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
import re


def load_checkpoint(checkpoint_path: str, map_location: str = 'cpu', strict: bool = True) -> Dict[str, Any]:
    """
    Load checkpoint from path.
    
    Args:
        checkpoint_path: Path to checkpoint (supports 'local::' prefix)
        map_location: Device to map tensors to
        strict: Whether to strictly enforce state dict matching
        
    Returns:
        State dictionary
    """
    if checkpoint_path.startswith('local::'):
        checkpoint_path = checkpoint_path[7:]
    
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint


def load_local_checkpoint(checkpoint_path: str, model: nn.Module, 
                        strict: bool = True, key: str = 'model') -> nn.Module:
    """
    Load weights into model from local checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        strict: Whether to strictly enforce matching
        key: Key in checkpoint dict containing state_dict
        
    Returns:
        Model with loaded weights
    """
    checkpoint = load_checkpoint(checkpoint_path)
    
    if key in checkpoint:
        state_dict = checkpoint[key]
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
    return model