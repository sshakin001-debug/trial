"""
Standalone metric depth estimator using embedded ZoeDepth+DepthAnything.
No external pip dependency required - all code is vendored in repo.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    print(f"[DepthEstimator] Added to sys.path: {_REPO_ROOT}")

_TORCHHUB_ROOT = _REPO_ROOT / "torchhub"
if str(_TORCHHUB_ROOT) not in sys.path:
    sys.path.insert(0, str(_TORCHHUB_ROOT))

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


class MetricDepthEstimator:
    """
    Metric depth estimation for outdoor/indoor scenes.
    Uses embedded ZoeDepth + DepthAnything backbone.
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 dataset: str = 'kitti',
                 device: str = 'cuda',
                 calibration_path: Optional[str] = None):
        """
        Args:
            checkpoint_path: Path to .pt checkpoint file
            dataset: 'kitti' (outdoor, 0-80m) or 'nyu' (indoor, 0-10m)
            device: 'cuda' or 'cpu'
            calibration_path: Path to .npz with Camera_matrix, distCoeff
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.dataset = dataset.lower()
        
        self.calibration = self._load_calibration(calibration_path) if calibration_path else None
        
        self.model = self._build_model(checkpoint_path)
        
        print(f"[DepthEstimator] Dataset: {dataset}, Device: {self.device}")
        if self.calibration:
            print(f"  Calibration: fx={self.calibration['fx']:.1f}, "
                  f"fy={self.calibration['fy']:.1f}, "
                  f"cx={self.calibration['cx']:.1f}, "
                  f"cy={self.calibration['cy']:.1f}")
    
    def _build_model(self, checkpoint_path: str):
        """Build ZoeDepth model with DepthAnything backbone."""
        config = get_config("zoedepth", "eval", self.dataset)
        
        if not checkpoint_path.startswith(('local::', 'url::')):
            checkpoint_path = f"local::{checkpoint_path}"
        
        config.pretrained_resource = checkpoint_path
        
        # build_model does not take device — move to device after
        self.model = build_model(config)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_calibration(self, path: str) -> Dict[str, Any]:
        """Load calibration NPZ."""
        data = np.load(path)
        K = data['Camera_matrix']
        
        return {
            'camera_matrix': K,
            'fx': float(K[0, 0]),
            'fy': float(K[1, 1]),
            'cx': float(K[0, 2]),
            'cy': float(K[1, 2]),
            'dist_coeffs': data.get('distCoeff', np.zeros(5)),
        }
    
    def estimate(self, image: np.ndarray, original_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Estimate metric depth map.
        
        Args:
            image: BGR image (H, W, 3)
            original_size: (W, H) to resize output to
            
        Returns:
            depth_map: (H, W) float32 array in meters
        """
        if original_size is None:
            original_size = (image.shape[1], image.shape[0])
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = transforms.ToTensor()(pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred = self.model(tensor, dataset=self.dataset)
        
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        
        depth = pred.squeeze().cpu().numpy()
        
        if depth.shape != (original_size[1], original_size[0]):
            depth_pil = Image.fromarray(depth)
            depth = np.array(depth_pil.resize(original_size, Image.NEAREST))
        
        return depth.astype(np.float32)
    
    def backproject(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Backproject depth map to 3D point cloud using camera intrinsics.
        """
        if self.calibration is None:
            raise ValueError("Camera calibration required for backprojection")
        
        h, w = depth_map.shape
        fx, fy = self.calibration['fx'], self.calibration['fy']
        cx, cy = self.calibration['cx'], self.calibration['cy']
        
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        Z = depth_map
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        
        valid = (Z.reshape(-1) > 0.1) & np.isfinite(Z.reshape(-1))
        return points[valid]
    
    def pixel_to_3d(self, u: float, v: float, depth: float) -> Tuple[float, float, float]:
        """Convert single pixel + depth to 3D camera coordinates."""
        if self.calibration is None:
            raise ValueError("Calibration required")
        
        fx, fy = self.calibration['fx'], self.calibration['fy']
        cx, cy = self.calibration['cx'], self.calibration['cy']
        
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        return X, Y, Z
