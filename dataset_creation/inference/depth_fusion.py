import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import cv2

class DepthFusion:
    """Fusion of depth maps with segmentation masks to compute 3D coordinates"""
    
    def __init__(self, calibration: Dict[str, Any]):
        """
        Args:
            calibration: Dict with camera calibration containing:
                - fx, fy: focal lengths in pixels
                - cx, cy: principal point
                Or uses optimal_camera_matrix from calibration NPZ
        """
        # Extract camera intrinsics
        if 'optimal_camera_matrix' in calibration:
            cam_matrix = np.array(calibration['optimal_camera_matrix'])
            self.fx = float(cam_matrix[0, 0])
            self.fy = float(cam_matrix[1, 1])
            self.cx = float(cam_matrix[0, 2])
            self.cy = float(cam_matrix[1, 2])
        elif 'fx' in calibration and 'fy' in calibration:
            self.fx = float(calibration['fx'])
            self.fy = float(calibration['fy'])
            self.cx = float(calibration.get('cx', calibration.get('image_size', [0, 0])[0] / 2))
            self.cy = float(calibration.get('cy', calibration.get('image_size', [0, 0])[1] / 2))
        else:
            # Default values - try to get from camera_matrix
            cam_matrix = np.array(calibration.get('camera_matrix', np.eye(3)))
            self.fx = float(cam_matrix[0, 0])
            self.fy = float(cam_matrix[1, 1])
            self.cx = float(cam_matrix[0, 2])
            self.cy = float(cam_matrix[1, 2])
    
    def backproject_pixel(self, u: int, v: int, depth: float) -> Tuple[float, float, float]:
        """
        Back-project a pixel to 3D world coordinates
        
        Args:
            u: Pixel x coordinate (column)
            v: Pixel y coordinate (row)
            depth: Depth in meters
            
        Returns:
            (X, Y, Z): Tuple of (lateral, vertical, longitudinal) in meters
                - X: positive = right of camera
                - Y: positive = up
                - Z: forward distance
        """
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z
    
    def compute_vehicle_depth(self, mask: np.ndarray, depth_map: np.ndarray, 
                           box: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Compute 3D position for a vehicle from its mask
        
        Args:
            mask: Binary mask (H, W) - 1 for vehicle pixels
            depth_map: Depth map (H, W) in meters
            box: Optional [x1, y1, x2, y2] bounding box
            
        Returns:
            Dict with depth_stats, position_3d, and ground_contact
        """
        h, w = depth_map.shape
        
        # Ensure mask is same shape as depth map
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            mask = (mask > 0.5).astype(np.uint8)
        
        # Get mask pixel coordinates
        ys, xs = np.where(mask > 0)
        
        if len(xs) == 0:
            return {
                'depth_stats': {
                    'median_depth_m': 0.0,
                    'min_depth_m': 0.0,
                    'max_depth_m': 0.0,
                    'depth_std_m': 0.0,
                    'pixel_count': 0,
                    'reliable_estimate': False
                },
                'position_3d': {
                    'longitudinal_z_m': 0.0,
                    'lateral_x_m': 0.0,
                    'vertical_y_m': 0.0
                },
                'ground_contact': {
                    'longitudinal_z_m': 0.0,
                    'lateral_x_m': 0.0
                }
            }
        
        # Get depths for mask pixels
        depths = depth_map[ys, xs]
        
        # Check reliability - need at least 50 pixels
        reliable_estimate = len(depths) >= 50
        
        # Compute depth statistics
        median_depth = float(np.median(depths))
        min_depth = float(np.min(depths))
        max_depth = float(np.max(depths))
        depth_std = float(np.std(depths))
        
        # For vehicle position, use bottom 30% of mask (closest to ground)
        sorted_indices = np.argsort(ys)  # Sort by y (row), higher y = closer to bottom
        bottom_third_count = max(1, len(sorted_indices) // 3)
        bottom_indices = sorted_indices[-bottom_third_count:]
        bottom_ys = ys[bottom_indices]
        bottom_xs = xs[bottom_indices]
        bottom_depths = depths[bottom_indices]
        
        # Use median of bottom third for vehicle position
        vehicle_depth = float(np.median(bottom_depths))
        
        # Compute centroid
        centroid_u = int(np.mean(xs))
        centroid_v = int(np.mean(ys))
        
        # Compute 3D position at centroid
        X, Y, Z = self.backproject_pixel(centroid_u, centroid_v, vehicle_depth)
        
        # Ground contact: use bottom-center of bounding box
        if box is not None:
            x1, y1, x2, y2 = box
            ground_u = int((x1 + x2) / 2)
            ground_v = int(y2)
        else:
            # Use bottom-center of mask
            ground_u = int(np.mean(bottom_xs))
            ground_v = int(np.max(bottom_ys))
        
        # Clamp to image boundaries
        ground_u = max(0, min(w - 1, ground_u))
        ground_v = max(0, min(h - 1, ground_v))
        
        ground_depth = depth_map[ground_v, ground_u]
        ground_X, ground_Y, ground_Z = self.backproject_pixel(ground_u, ground_v, ground_depth)
        
        return {
            'depth_stats': {
                'median_depth_m': median_depth,
                'min_depth_m': min_depth,
                'max_depth_m': max_depth,
                'depth_std_m': depth_std,
                'pixel_count': int(len(depths)),
                'reliable_estimate': reliable_estimate
            },
            'position_3d': {
                'longitudinal_z_m': float(Z),
                'lateral_x_m': float(X),
                'vertical_y_m': float(Y)
            },
            'ground_contact': {
                'longitudinal_z_m': float(ground_Z),
                'lateral_x_m': float(ground_X)
            }
        }
    
    def compute_lane_marking_depth(self, lane_marking: Dict[str, Any], 
                                   depth_map: np.ndarray) -> Dict[str, Any]:
        """
        Compute 3D positions for a lane marking
        
        Args:
            lane_marking: Dict with 'start' and 'end' pixel coordinates
            depth_map: Depth map (H, W) in meters
            
        Returns:
            Dict with 3D coordinates for start, end, and sampled points
        """
        start = lane_marking['start']  # [u, v]
        end = lane_marking['end']    # [u, v]
        
        h, w = depth_map.shape
        
        # Start point depth
        u1, v1 = start
        u1 = max(0, min(w - 1, u1))
        v1 = max(0, min(h - 1, v1))
        z_start = depth_map[v1, u1]
        x_start, y_start, z_start_3d = self.backproject_pixel(u1, v1, z_start)
        
        # End point depth
        u2, v2 = end
        u2 = max(0, min(w - 1, u2))
        v2 = max(0, min(h - 1, v2))
        z_end = depth_map[v2, u2]
        x_end, y_end, z_end_3d = self.backproject_pixel(u2, v2, z_end)
        
        # Sample points along the line
        sampled_3d = []
        num_samples = max(2, int(lane_marking.get('length', 0)) // 10 + 1)
        
        for i in range(num_samples):
            t = i / max(1, num_samples - 1)
            u = int(u1 + t * (u2 - u1))
            v = int(v1 + t * (v2 - v1))
            u = max(0, min(w - 1, u))
            v = max(0, min(h - 1, v))
            
            z = depth_map[v, u]
            x, y, z_3d = self.backproject_pixel(u, v, z)
            sampled_3d.append({
                'x_m': float(x),
                'z_m': float(z_3d)
            })
        
        # Nearest and farthest points
        nearest_z = min(z_start, z_end)
        farthest_z = max(z_start, z_end)
        
        return {
            'start_3d': {
                'z_m': float(z_start_3d),
                'x_m': float(x_start)
            },
            'end_3d': {
                'z_m': float(z_end_3d),
                'x_m': float(x_end)
            },
            'nearest_point_z_m': float(nearest_z),
            'farthest_point_z_m': float(farthest_z),
            'sampled_3d_points': sampled_3d
        }
    
    def compute_pavement_depth(self, pavement_mask: np.ndarray, 
                                depth_map: np.ndarray) -> Dict[str, Any]:
        """
        Compute depth profile for pavement/road surface
        
        Args:
            pavement_mask: Binary mask (H, W)
            depth_map: Depth map (H, W) in meters
            
        Returns:
            Dict with depth zones
        """
        h, w = depth_map.shape
        
        # Ensure mask is same shape
        if pavement_mask.shape != (h, w):
            pavement_mask = cv2.resize(
                pavement_mask.astype(np.uint8), (w, h), 
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        
        # Get road surface pixels
        ys, xs = np.where(pavement_mask > 0)
        
        if len(xs) == 0:
            return {
                'near_zone_depth_m': 0.0,
                'mid_zone_depth_m': 0.0,
                'far_zone_depth_m': 0.0,
                'total_pavement_pixels': 0
            }
        
        depths = depth_map[ys, xs]
        
        # Divide into zones based on Y pixel position (row)
        # Near: bottom 1/3 of the mask (closest)
        # Mid: middle 1/3
        # Far: top 1/3 (farthest)
        y_min, y_max = ys.min(), ys.max()
        y_range = y_max - y_min
        
        near_mask = ys > (y_min + y_range * 2/3)
        mid_mask = (ys > (y_min + y_range / 3)) & (ys <= (y_min + y_range * 2/3))
        far_mask = ys <= (y_min + y_range / 3)
        
        near_zone_depth = float(np.median(depths[near_mask])) if near_mask.sum() > 0 else 0.0
        mid_zone_depth = float(np.median(depths[mid_mask])) if mid_mask.sum() > 0 else 0.0
        far_zone_depth = float(np.median(depths[far_mask])) if far_mask.sum() > 0 else 0.0
        
        return {
            'near_zone_depth_m': near_zone_depth,
            'mid_zone_depth_m': mid_zone_depth,
            'far_zone_depth_m': far_zone_depth,
            'total_pavement_pixels': int(len(depths))
        }
    
    def fuse_frame(self, vehicle_masks: List[np.ndarray], vehicle_boxes: List[List[float]],
                  vehicle_classes: List[int], vehicle_class_names: List[str],
                  vehicle_confidences: List[float], vehicle_mask_paths: List[str],
                  lane_markings: List[Dict[str, Any]], 
                  pavement_mask: np.ndarray,
                  depth_map: np.ndarray) -> Dict[str, Any]:
        """
        Fuse all annotations with depth for a single frame
        
        Returns:
            Enriched annotation dict with 3D positions
        """
        h, w = depth_map.shape
        
        # Process vehicles
        vehicles = []
        for i, (mask, box, cls, cls_name, conf, mask_path) in enumerate(zip(
            vehicle_masks, vehicle_boxes, vehicle_classes, vehicle_class_names,
            vehicle_confidences, vehicle_mask_paths
        )):
            vehicle_depth = self.compute_vehicle_depth(mask, depth_map, box)
            vehicles.append({
                'box': box,
                'class_name': cls_name,
                'confidence': float(conf),
                'mask_path': mask_path,
                'depth_stats': vehicle_depth['depth_stats'],
                'position_3d': vehicle_depth['position_3d'],
                'ground_contact': vehicle_depth['ground_contact']
            })
        
        # Process lane markings
        lane_markings_3d = []
        for marking in lane_markings:
            marking_3d = self.compute_lane_marking_depth(marking, depth_map)
            marking_3d['start_pixel'] = marking['start']
            marking_3d['end_pixel'] = marking['end']
            lane_markings_3d.append(marking_3d)
        
        # Process pavement
        pavement_depth = self.compute_pavement_depth(pavement_mask, depth_map)
        
        return {
            'vehicles': vehicles,
            'lanes': {
                'lane_markings': lane_markings_3d,
                'pavement': pavement_depth
            }
        }