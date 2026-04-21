"""
Geometry utilities for depth estimation.
"""

import numpy as np
from typing import Tuple, Optional


def estimate_camera_params(image_width: int, image_height: int, 
                          focal_length: Optional[float] = None,
                          camera_model: str = 'pinhole') -> dict:
    """
    Estimate camera intrinsic parameters from image dimensions.
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        focal_length: Optional focal length override
        camera_model: Camera model type
        
    Returns:
        Dictionary with camera parameters
    """
    fx = focal_length if focal_length else image_width * 0.8
    fy = fx
    cx = image_width / 2
    cy = image_height / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return {
        'K': K,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'width': image_width,
        'height': image_height,
        'camera_model': camera_model
    }


def depth_to_point_cloud(depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert depth map to point cloud.
    
    Args:
        depth_map: (H, W) depth map in meters
        K: Camera intrinsic matrix (3, 3)
        
    Returns:
        (N, 3) point cloud
    """
    H, W = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    Z = depth_map
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    
    valid = (Z.reshape(-1) > 0.1) & np.isfinite(Z.reshape(-1))
    return points[valid]


def project_to_image(points: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to image plane.
    
    Args:
        points: (N, 3) point cloud
        K: Camera intrinsic matrix (3, 3)
        
    Returns:
        Tuple of (u, v) pixel coordinates
    """
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    
    Z = np.maximum(Z, 1e-6)
    
    u = (X * K[0, 0] / Z) + K[0, 2]
    v = (Y * K[1, 1] / Z) + K[1, 2]
    
    return u, v