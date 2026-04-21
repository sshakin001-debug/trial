"""
Depth inference using embedded ZoeDepth + Depth Anything.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms
import cv2

_ZOEDEPTH_ROOT = Path(__file__).parent.parent.parent / "zoedepth"
if str(_ZOEDEPTH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZOEDEPTH_ROOT))

_TORCHHUB_ROOT = Path(__file__).parent.parent.parent / "torchhub"
if str(_TORCHHUB_ROOT) not in sys.path:
    sys.path.insert(0, str(_TORCHHUB_ROOT))

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


class DepthInferencer:
    """Depth inference using embedded ZoeDepth for metric depth estimation"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', dataset: str = 'kitti',
                 calibration_path: Optional[str] = None):
        """
        Args:
            checkpoint_path: Path to the depth model checkpoint (.pt file)
            device: 'cuda' or 'cpu'
            dataset: 'kitti' for outdoor (0-80m), 'nyu' for indoor (0-10m)
            calibration_path: Optional path to calibration NPZ
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.dataset = dataset.lower()
        
        self.calibration = None
        if calibration_path:
            self.calibration = self._load_calibration(calibration_path)
        
        config = get_config("zoedepth", "eval", self.dataset)
        
        if not checkpoint_path.startswith(('local::', 'url::')):
            checkpoint_path = f"local::{checkpoint_path}"
        
        config.pretrained_resource = checkpoint_path
        
        self.model = build_model(config, device=self.device)
        self.model.eval()
        
        print(f"[DepthInferencer] Loaded depth model from {checkpoint_path}")
        print(f"Dataset: {self.dataset}, Device: {self.device}")
    
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
    
    def infer_depth(self, image: np.ndarray, original_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Infer metric depth map for a single image
        
        Args:
            image: BGR or RGB image as numpy array (H, W, 3)
            original_size: Original (width, height) - if None, uses image shape
            
        Returns:
            depth_map: numpy float32 array of shape (H, W) with depth in meters
        """
        if original_size is None:
            original_size = (image.shape[1], image.shape[0])
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        image_tensor = transforms.ToTensor()(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred = self.model(image_tensor, dataset=self.dataset)
        
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        
        depth = pred.squeeze().detach().cpu().numpy()
        
        h, w = depth.shape
        orig_w, orig_h = original_size
        
        if h != orig_h or w != orig_w:
            depth_pil = Image.fromarray(depth)
            depth = np.array(depth_pil.resize((orig_w, orig_h), Image.NEAREST)).astype(np.float32)
        
        return depth.astype(np.float32)
    
    def backproject(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert depth map to point cloud using calibration."""
        if self.calibration is None:
            raise ValueError("Calibration required for backprojection")
        
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
    
    def process_frames(self, frames_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Process all frames in a directory"""
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        depth_maps = {}
        
        frame_paths = sorted(frames_dir.glob('*.jpg')) + \
                      sorted(frames_dir.glob('*.png')) + \
                      sorted(frames_dir.glob('*.jpeg'))
        
        for frame_path in frame_paths:
            print(f"Processing depth for {frame_path.name}...")
            
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            
            original_size = (image.shape[1], image.shape[0])
            depth_map = self.infer_depth(image, original_size)
            
            depth_path = output_dir / f"{frame_path.stem}_depth.npy"
            np.save(str(depth_path), depth_map)
            
            depth_maps[frame_path.name] = {
                'frame': frame_path.name,
                'depth_map_path': str(depth_path),
                'depth_min': float(depth_map.min()),
                'depth_max': float(depth_map.max()),
                'depth_mean': float(depth_map.mean())
            }
        
        return depth_maps