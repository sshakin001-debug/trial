Option A: Embed zoedepth/ directly in trial repo (like image-to-pcd does)
Structure:
plain
Copy
trial-repo/
├── zoedepth/                    ← Copy entire zoedepth/ from image-to-pcd
│   ├── models/
│   ├── utils/
│   └── ...
├── dataset_creation/
│   └── inference/
│       ├── depth_estimator.py   ← Modified to find local zoedepth
│       └── ...
Then fix depth_estimator.py path resolution:
Python
Copy
# In depth_estimator.py — replace the try/except block with this:

import sys
from pathlib import Path

# Add repo root to path so 'import zoedepth' works from any subdirectory
_REPO_ROOT = Path(__file__).parent.parent.parent  # dataset_creation/inference/ → repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config





Recommended: Option A with Robust Path Handling
Here's the corrected depth_estimator.py for your trial repo:
Python
Copy
"""
Standalone metric depth estimator using embedded ZoeDepth.
No external dependencies — zoedepth is vendored in repo root.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms
import sys

# ── CRITICAL: Add repo root to Python path ──────────────────────────
# This ensures 'import zoedepth' finds the embedded package
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# Goes: dataset_creation/inference/file.py → dataset_creation/ → repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    print(f"[DepthEstimator] Added to sys.path: {_REPO_ROOT}")

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# ── Rest of your class unchanged ────────────────────────────────────
class MetricDepthEstimator:
    def __init__(self, checkpoint_path: str, dataset: str = 'kitti', 
                 device: str = 'cuda', calibration_path: Optional[str] = None):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.dataset = dataset.lower()
        self.calibration = self._load_calibration(calibration_path) if calibration_path else None
        
        self.model = self._build_zoedepth(checkpoint_path)
        print(f"[DepthEstimator] Dataset: {dataset}, Device: {self.device}")
    
    def _build_zoedepth(self, checkpoint_path: str):
        config = get_config("zoedepth", "eval", self.dataset)
        if not checkpoint_path.startswith(('local::', 'url::')):
            checkpoint_path = f"local::{checkpoint_path}"
        config.pretrained_resource = checkpoint_path
        
        model = build_model(config)
        model.to(self.device)
        model.eval()
        return model
    
    # ... rest of methods unchanged ...