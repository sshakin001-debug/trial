The build_model function in the image-to-pcd zoedepth doesn't accept a device argument. Fix depth_estimator.py:
pythondepth_estimator_code = '''import torch
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

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


class MetricDepthEstimator:
    def __init__(self, checkpoint_path, dataset="kitti", device="cuda", calibration_path=None):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dataset = dataset.lower()
        self.calibration = self._load_calibration(calibration_path) if calibration_path else None

        print(f"[DepthEstimator] Building ZoeDepth model...")
        config = get_config("zoedepth", "eval", self.dataset)

        if not checkpoint_path.startswith(("local::", "url::")):
            checkpoint_path = f"local::{checkpoint_path}"
        config.pretrained_resource = checkpoint_path

        # build_model does not take device — move to device after
        self.model = build_model(config)
        self.model.to(self.device)
        self.model.eval()

        print(f"[DepthEstimator] Dataset: {dataset}, Device: {self.device}")
        if self.calibration:
            print(f"  Calibration: fx={self.calibration[\'fx\']:.1f}, "
                  f"fy={self.calibration[\'fy\']:.1f}, "
                  f"cx={self.calibration[\'cx\']:.1f}, "
                  f"cy={self.calibration[\'cy\']:.1f}")

    def _load_calibration(self, path):
        data = np.load(path)
        K = data["Camera_matrix"]
        return {
            "camera_matrix": K,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "dist_coeffs": data.get("distCoeff", np.zeros(5)),
        }

    def estimate(self, image, original_size=None):
        if original_size is None:
            original_size = (image.shape[1], image.shape[0])

        rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil    = Image.fromarray(rgb)
        tensor = transforms.ToTensor()(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(tensor, dataset=self.dataset)

        if isinstance(pred, dict):
            pred = pred.get("metric_depth", pred.get("out"))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]

        depth = pred.squeeze().detach().cpu().numpy()

        orig_w, orig_h = original_size
        if depth.shape != (orig_h, orig_w):
            depth = np.array(Image.fromarray(depth).resize((orig_w, orig_h), Image.NEAREST))

        return depth.astype(np.float32)

    def backproject(self, depth_map):
        if self.calibration is None:
            raise ValueError("Calibration required")
        h, w   = depth_map.shape
        fx, fy = self.calibration["fx"], self.calibration["fy"]
        cx, cy = self.calibration["cx"], self.calibration["cy"]
        u, v   = np.meshgrid(np.arange(w), np.arange(h))
        Z = depth_map
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        valid  = (Z.reshape(-1) > 0.1) & np.isfinite(Z.reshape(-1))
        return points[valid]

    def pixel_to_3d(self, u, v, depth):
        if self.calibration is None:
            raise ValueError("Calibration required")
        fx, fy = self.calibration["fx"], self.calibration["fy"]
        cx, cy = self.calibration["cx"], self.calibration["cy"]
        return (u - cx) * depth / fx, (v - cy) * depth / fy, depth
'''

with open('/content/trial/dataset_creation/inference/depth_estimator.py', 'w') as f:
    f.write(depth_estimator_code)
print("✓ depth_estimator.py fixed")

import sys
to_remove = [k for k in sys.modules if any(x in k for x in
             ['zoedepth', 'batch_processor', 'depth_estimator',
              'lateral_fusion', 'segformer_inference', 'yolo_inference'])]
for k in to_remove:
    del sys.modules[k]
print(f"✓ Cleared {len(to_remove)} modules — re-run pipeline now")


changes in batch processor-

We need to pass them through. Here's the complete fix:
pythonwith open('/content/trial/dataset_creation/inference/batch_processor.py', 'r') as f:
    content = f.read()

# Fix 1: store segformer results in process_frame output
old_return = '''        return {
            'frame_name': frame_name,
            'image_size': {'width': w, 'height': h},
            'ego_vehicle': {'''

new_return = '''        # Store masks for visualization
        self._last_segformer_results = segformer_results

        return {
            'frame_name': frame_name,
            'image_size': {'width': w, 'height': h},
            'ego_vehicle': {'''

# Fix 2: use masks in _visualize
old_viz = '''    def _visualize(self, image: np.ndarray, ann: Dict[str, Any]) -> np.ndarray:
        """Create visualization with 3D annotations."""
        vis = image.copy()
        
        # Lane markings: skip drawing extended lines entirely
        # The segformer mask overlay is already saved separately'''

new_viz = '''    def _visualize(self, image: np.ndarray, ann: Dict[str, Any]) -> np.ndarray:
        """Create visualization with 3D annotations."""
        vis = image.copy()

        # Segformer mask overlays (same as debug_seg visualization)
        seg = getattr(self, '_last_segformer_results', None)
        if seg is not None:
            pavement_overlay = np.zeros_like(vis)
            pavement_overlay[seg['pavement_mask'] > 0] = [128, 0, 128]
            vis = cv2.addWeighted(vis, 1.0, pavement_overlay, 0.4, 0)

            lane_overlay = np.zeros_like(vis)
            lane_overlay[seg['lane_mask'] > 0] = [0, 255, 255]
            vis = cv2.addWeighted(vis, 1.0, lane_overlay, 0.6, 0)

            lane_contours, _ = cv2.findContours(
                seg['lane_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, lane_contours, -1, (0, 255, 255), 2)'''

if old_return in content and old_viz in content:
    content = content.replace(old_return, new_return)
    content = content.replace(old_viz, new_viz)
    with open('/content/trial/dataset_creation/inference/batch_processor.py', 'w') as f:
        f.write(content)
    print("✓ batch_processor.py fixed")
else:
    print("❌ Text not found")
    print("old_return found:", old_return in content)
    print("old_viz found:", old_viz in content)