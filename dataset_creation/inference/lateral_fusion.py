"""
Lateral position computation using metric depth and camera calibration.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple


class LateralPositionFusion:
    """
    Computes lateral positions of vehicles relative to lanes and ego-vehicle.
    Uses metric depth (meters) and calibrated camera intrinsics.
    """
    
    def __init__(self, calibration: Dict[str, Any]):
        """
        Args:
            calibration: Dict with fx, fy, cx, cy from camera calibration
        """
        self.fx = calibration['fx']
        self.fy = calibration['fy']
        self.cx = calibration['cx']
        self.cy = calibration['cy']
        self.camera_height_m = calibration.get('camera_height_m', 1.2)
    
    def pixel_to_3d(self, u: float, v: float, z: float) -> Tuple[float, float, float]:
        """Back-project pixel to 3D: X = (u-cx)*Z/fx, Y = (v-cy)*Z/fy."""
        X = (u - self.cx) * z / self.fx
        Y = (v - self.cy) * z / self.fy
        return X, Y, z
    
    def compute_vehicle_3d(self,
                          bbox: List[float],
                          mask: np.ndarray,
                          depth_map: np.ndarray) -> Dict[str, Any]:
        """
        Compute 3D position and lateral offset for a vehicle.
        
        Args:
            bbox: [x1, y1, x2, y2]
            mask: Binary mask (H, W)
            depth_map: Metric depth (H, W) in meters
            
        Returns:
            Dict with depth_stats, position_3d, ground_contact, lateral_offset
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_map.shape
        
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), 
                            interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)
        
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return self._empty_result()
        
        depths = depth_map[ys, xs]
        valid = depths[(depths > 0.5) & np.isfinite(depths)]
        
        if len(valid) == 0:
            return self._empty_result()
        
        median_depth = float(np.median(valid))
        min_depth = float(np.min(valid))
        max_depth = float(np.max(valid))
        std_depth = float(np.std(valid))
        
        ground_u = (x1 + x2) / 2
        ground_v = float(y2)
        
        bottom_mask = ys > (y1 + (y2 - y1) * 0.7)
        if bottom_mask.sum() > 0:
            ground_depth = float(np.median(depths[bottom_mask]))
        else:
            ground_depth = median_depth
        
        gX, gY, gZ = self.pixel_to_3d(ground_u, ground_v, ground_depth)
        
        cent_u, cent_v = np.mean(xs), np.mean(ys)
        cX, cY, cZ = self.pixel_to_3d(cent_u, cent_v, median_depth)
        
        lateral_offset = gX
        
        est_width = (x2 - x1) * ground_depth / self.fx
        est_height = (y2 - y1) * ground_depth / self.fy
        
        return {
            'depth_stats': {
                'median_depth_m': median_depth,
                'min_depth_m': min_depth,
                'max_depth_m': max_depth,
                'std_depth_m': std_depth,
                'pixel_count': int(len(valid)),
                'reliable': len(valid) >= 50
            },
            'position_3d': {
                'lateral_x_m': float(cX),
                'vertical_y_m': float(cY),
                'longitudinal_z_m': float(cZ)
            },
            'ground_contact': {
                'lateral_x_m': float(gX),
                'vertical_y_m': float(gY),
                'longitudinal_z_m': float(gZ)
            },
            'lateral_offset_from_ego_m': float(lateral_offset),
            'estimated_dimensions_m': {
                'width': float(est_width),
                'height': float(est_height)
            }
        }
    
    def compute_lane_3d(self,
                       start_pixel: List[int],
                       end_pixel: List[int],
                       depth_map: np.ndarray) -> Dict[str, Any]:
        """Compute 3D coordinates for a lane marking segment."""
        h, w = depth_map.shape
        u1, v1 = start_pixel
        u2, v2 = end_pixel
        
        u1 = max(0, min(w-1, u1)); v1 = max(0, min(h-1, v1))
        u2 = max(0, min(w-1, u2)); v2 = max(0, min(h-1, v2))
        
        z1 = depth_map[v1, u1]
        z2 = depth_map[v2, u2]
        
        x1, y1, z1_3d = self.pixel_to_3d(u1, v1, z1)
        x2, y2, z2_3d = self.pixel_to_3d(u2, v2, z2)
        
        dx = x2 - x1
        dz = z2_3d - z1_3d
        length = np.sqrt(dx**2 + dz**2)
        angle = float(np.degrees(np.arctan2(dx, dz)))
        
        return {
            'start_3d': {'x_m': float(x1), 'y_m': float(y1), 'z_m': float(z1_3d)},
            'end_3d': {'x_m': float(x2), 'y_m': float(y2), 'z_m': float(z2_3d)},
            'length_m': float(length),
            'angle_deg': angle,
            'lateral_offset_start_m': float(x1),
            'lateral_offset_end_m': float(x2)
        }
    
    def assign_lane(self,
                    vehicle_3d: Dict[str, Any],
                    lane_markings_3d: List[Dict[str, Any]],
                    lane_width_m: float = 3.5) -> Dict[str, Any]:
        """
        Determine which lane a vehicle is in.
        
        Returns:
            Dict with lane_assignment, distances to left/right markings
        """
        if not lane_markings_3d:
            return {'lane_assignment': 'unknown', 'distance_to_left_m': None, 
                    'distance_to_right_m': None}
        
        v_x = vehicle_3d['ground_contact']['lateral_x_m']
        v_z = vehicle_3d['ground_contact']['longitudinal_z_m']
        
        nearby = []
        for m in lane_markings_3d:
            mz = (m['start_3d']['z_m'] + m['end_3d']['z_m']) / 2
            if abs(mz - v_z) < 15.0:
                z_start, z_end = m['start_3d']['z_m'], m['end_3d']['z_m']
                x_start, x_end = m['start_3d']['x_m'], m['end_3d']['x_m']
                
                if abs(z_end - z_start) > 0.1:
                    t = (v_z - z_start) / (z_end - z_start)
                    t = np.clip(t, 0, 1)
                    mx = x_start + t * (x_end - x_start)
                else:
                    mx = (x_start + x_end) / 2
                
                nearby.append({'x_m': mx, 'z_m': mz})
        
        if not nearby:
            return {'lane_assignment': 'unknown', 'distance_to_left_m': None,
                    'distance_to_right_m': None}
        
        nearby.sort(key=lambda m: m['x_m'])
        
        distances = [m['x_m'] - v_x for m in nearby]
        
        left_dists = [d for d in distances if d < 0]
        right_dists = [d for d in distances if d >= 0]
        
        dist_left = abs(max(left_dists)) if left_dists else None
        dist_right = min(right_dists) if right_dists else None
        
        if dist_left is not None and dist_right is not None:
            if dist_left < lane_width_m and dist_right < lane_width_m:
                assignment = 'same'
            elif dist_left >= lane_width_m:
                assignment = 'left'
            elif dist_right >= lane_width_m:
                assignment = 'right'
            else:
                assignment = 'same'
        elif dist_right is not None and dist_right < lane_width_m * 0.8:
            assignment = 'same'
        elif dist_left is not None and dist_left < lane_width_m * 0.8:
            assignment = 'same'
        else:
            assignment = 'unknown'
        
        return {
            'lane_assignment': assignment,
            'distance_to_left_marking_m': dist_left,
            'distance_to_right_marking_m': dist_right
        }
    
    def compute_ego_lane(self, lane_markings_3d: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate ego vehicle's current lane state."""
        if len(lane_markings_3d) < 2:
            return {'current_lane': 'unknown', 'lateral_offset_m': 0.0, 
                    'lane_width_m': 3.5, 'confidence': 'low'}
        
        near = []
        for m in lane_markings_3d:
            z = m['start_3d']['z_m']
            if 5.0 < z < 30.0:
                near.append({
                    'x_m': m['start_3d']['x_m'],
                    'z_m': z,
                    'marking': m
                })
        
        if len(near) < 2:
            return {'current_lane': 'unknown', 'lateral_offset_m': 0.0,
                    'lane_width_m': 3.5, 'confidence': 'low'}
        
        near.sort(key=lambda m: m['x_m'])
        
        for i in range(len(near) - 1):
            x_left, x_right = near[i]['x_m'], near[i+1]['x_m']
            if x_left <= 0 <= x_right:
                width = x_right - x_left
                offset = (x_left + x_right) / 2
                return {
                    'current_lane': 'center',
                    'lateral_offset_from_center_m': float(offset),
                    'lane_width_m': float(width),
                    'left_marking_x_m': float(x_left),
                    'right_marking_x_m': float(x_right),
                    'confidence': 'high'
                }
        
        return {'current_lane': 'unknown', 'lateral_offset_m': 0.0,
                'lane_width_m': 3.5, 'confidence': 'low'}
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty vehicle result."""
        return {
            'depth_stats': {'median_depth_m': 0, 'min_depth_m': 0, 'max_depth_m': 0,
                           'std_depth_m': 0, 'pixel_count': 0, 'reliable': False},
            'position_3d': {'lateral_x_m': 0, 'vertical_y_m': 0, 'longitudinal_z_m': 0},
            'ground_contact': {'lateral_x_m': 0, 'vertical_y_m': 0, 'longitudinal_z_m': 0},
            'lateral_offset_from_ego_m': 0,
            'estimated_dimensions_m': {'width': 0, 'height': 0}
        }