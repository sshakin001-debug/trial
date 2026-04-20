import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms
import cv2

class DepthInferencer:
    """Depth inference using Depth Anything / ZoeDepth for metric depth estimation"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', dataset: str = 'kitti'):
        """
        Args:
            checkpoint_path: Path to the depth model checkpoint (.pt file)
            device: 'cuda' or 'cpu'
            dataset: 'kitti' for outdoor (0-80m), 'nyu' for indoor (0-10m)
        """
        self.device = device
        self.dataset = dataset
        
        # Add the image-to-pcd path to sys.path if needed
        image_to_pcd_path = Path("D:/Point_cloud/image-to-pcd")
        if str(image_to_pcd_path) not in sys.path:
            sys.path.insert(0, str(image_to_pcd_path))
        
        # Import zoedepth components
        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config
        
        # Build model configuration
        config = get_config("zoedepth", "eval", dataset)
        config.pretrained_resource = f'local::{checkpoint_path}'
        
        # Build and load the model
        self.model = build_model(config)
        self.model.to(device)
        self.model.eval()
        
        print(f"Loaded depth model from {checkpoint_path}")
        print(f"Dataset: {dataset}")
    
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
            original_size = (image.shape[1], image.shape[0])  # (W, H)
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if BGR (OpenCV default) or RGB
            # Assume BGR since we use OpenCV elsewhere in the project
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Convert to tensor
        image_tensor = transforms.ToTensor()(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            pred = self.model(image_tensor, dataset=self.dataset)
        
        # Handle different output formats
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        
        # Squeeze to get numpy array
        pred = pred.squeeze().detach().cpu().numpy()
        
        # Resize to original image dimensions
        h, w = pred.shape
        orig_w, orig_h = original_size
        
        # Only resize if dimensions differ
        if h != orig_h or w != orig_w:
            pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            pred_resized = pred
        
        return pred_resized.astype(np.float32)
    
    def process_frames(self, frames_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Process all frames in a directory"""
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        depth_maps = {}
        
        # Process each frame
        frame_paths = sorted(frames_dir.glob('*.jpg')) + \
                      sorted(frames_dir.glob('*.png')) + \
                      sorted(frames_dir.glob('*.jpeg'))
        
        for frame_path in frame_paths:
            print(f"Processing depth for {frame_path.name}...")
            
            # Read image
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            
            # Get original size before inference
            original_size = (image.shape[1], image.shape[0])
            
            # Infer depth
            depth_map = self.infer_depth(image, original_size)
            
            # Save depth map
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