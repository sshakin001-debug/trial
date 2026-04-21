"""
Configuration loading utilities for ZoeDepth.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from easydict import EasyDict


def get_config(model_name: str = "zoedepth", mode: str = "infer", dataset: str = "kitti") -> EasyDict:
    """
    Load configuration for ZoeDepth model.
    
    Args:
        model_name: Name of the model configuration to load
        mode: Mode ('train', 'infer', 'eval')
        dataset: Dataset name ('kitti', 'nyu')
        
    Returns:
        Configuration object
    """
    config_dir = Path(__file__).parent.parent / "models" / "zoedepth"
    
    config_files = {
        "kitti": config_dir / "config_zoedepth_kitti.json",
        "nyu": config_dir / "config_zoedepth.json",
        "default": config_dir / "config_zoedepth.json"
    }
    
    config_file = config_files.get(dataset.lower(), config_files["default"])
    
    if not config_file.exists():
        config_data = {
            "model": {
                "name": model_name,
                "version_name": "v1",
                "n_bins": 64,
                "bin_embedding_dim": 128,
                "n_attractors": [16, 8, 4, 1],
                "attractor_alpha": 1000,
                "attractor_gamma": 2,
                "min_temp": 0.1,
                "max_temp": 80.0 if dataset.lower() == "kitti" else 10.0,
                "use_depth_anything": True,
                "depth_anything_model_type": "vitl"
            },
            mode: {
                "pretrained_resource": None
            }
        }
        return EasyDict(config_data)
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    return EasyDict(config_data)


def save_config(config: EasyDict, path: Path) -> None:
    """
    Save configuration to file.
    """
    with open(path, 'w') as f:
        json.dump(dict(config), f, indent=2)