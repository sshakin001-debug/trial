"""
Unified batch processor: YOLO + SegFormer + Metric Depth + Lateral Fusion.
Generates JSON annotations with 3D lateral positions.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import cv2
from typing import Dict, Any

from yolo_inference import YOLOVehicleSegmenter
from segformer_inference import SegformerLaneSegmenter
from depth_estimator import MetricDepthEstimator
from lateral_fusion import LateralPositionFusion


class UnifiedBatchProcessor:
    """
    Single-pass processor integrating all models with metric depth.
    """
    
    def __init__(self,
                 yolo_model_path: str,
                 segformer_model_path: str,
                 depth_checkpoint_path: str,
                 calibration_path: str,
                 dataset: str = 'kitti',
                 device: str = 'cuda',
                 segformer_img_size: tuple = (512, 512)):
        """
        Initialize all pipeline components.
        """
        print("=" * 50)
        print("UNIFIED DATASET CREATION PIPELINE")
        print("=" * 50)
        
        print("\n[1/4] Loading YOLO vehicle segmenter...")
        self.yolo = YOLOVehicleSegmenter(yolo_model_path, device)
        
        print("\n[2/4] Loading SegFormer lane segmenter...")
        self.segformer = SegformerLaneSegmenter(segformer_model_path, device, segformer_img_size)
        
        print("\n[3/4] Loading metric depth estimator...")
        self.depth = MetricDepthEstimator(depth_checkpoint_path, dataset, device, calibration_path)
        
        print("\n[4/4] Initializing lateral fusion...")
        self.fusion = LateralPositionFusion(self.depth.calibration)
        
        print("\n" + "=" * 50)
        print("READY")
        print("=" * 50)
    
    def process_frame(self, image: np.ndarray, frame_name: str = "") -> Dict[str, Any]:
        """
        Process single frame through all pipelines.
        """
        h, w = image.shape[:2]
        original_size = (w, h)
        
        yolo_results = self.yolo.segment_frame(image)
        segformer_results = self.segformer.segment_frame(image)
        depth_map = self.depth.estimate(image, original_size)
        
        lane_markings_3d = []
        for i, marking in enumerate(segformer_results.get('lane_markings', [])):
            lane_3d = self.fusion.compute_lane_3d(
                marking['start'], marking['end'], depth_map
            )
            lane_markings_3d.append({
                'marking_id': i,
                'start_pixel': marking['start'],
                'end_pixel': marking['end'],
                **lane_3d
            })
        
        vehicles_3d = []
        for i, (mask, box, cls, cls_name, conf) in enumerate(zip(
            yolo_results.get('masks', []),
            yolo_results.get('boxes', []),
            yolo_results.get('classes', []),
            yolo_results.get('class_names', []),
            yolo_results.get('confidences', [])
        )):
            vehicle_3d = self.fusion.compute_vehicle_3d(box, mask, depth_map)
            
            lane_info = self.fusion.assign_lane(vehicle_3d, lane_markings_3d)
            
            vehicles_3d.append({
                'vehicle_id': i,
                'class_name': cls_name,
                'class_id': int(cls),
                'confidence': float(conf),
                'bbox': [float(x) for x in box],
                **vehicle_3d,
                'lane_relative': lane_info
            })
        
        ego_lane = self.fusion.compute_ego_lane(lane_markings_3d)
        
        return {
            'frame_name': frame_name,
            'image_size': {'width': w, 'height': h},
            'ego_vehicle': {
                'lane_state': ego_lane,
                'camera_height_m': self.fusion.camera_height_m
            },
            'vehicles': vehicles_3d,
            'lanes': {
                'markings_3d': lane_markings_3d,
                'count': len(lane_markings_3d)
            },
            'calibration': {
                'fx': self.depth.calibration['fx'],
                'fy': self.depth.calibration['fy'],
                'cx': self.depth.calibration['cx'],
                'cy': self.depth.calibration['cy']
            }
        }
    
    def process_dataset(self, frames_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Process entire dataset directory.
        """
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_dir = output_dir / 'annotations'
        vis_dir = output_dir / 'visualizations'
        depth_dir = output_dir / 'depth_maps'
        
        for d in [json_dir, vis_dir, depth_dir]:
            d.mkdir(exist_ok=True)
        
        frame_paths = sorted(frames_dir.glob('*.jpg')) + \
                      sorted(frames_dir.glob('*.png')) + \
                      sorted(frames_dir.glob('*.jpeg'))
        
        print(f"\nProcessing {len(frame_paths)} frames...")
        
        all_annotations = {
            'metadata': {
                'total_frames': len(frame_paths),
                'pipeline': 'unified_depth_lateral_v1',
                'depth_model': 'zoedepth_metric',
                'dataset_type': self.depth.dataset
            },
            'frames': {}
        }
        
        for frame_path in frame_paths:
            print(f"  Processing {frame_path.name}...")
            
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            
            annotation = self.process_frame(image, frame_path.name)
            
            depth_path = depth_dir / f"{frame_path.stem}_depth.npy"
            np.save(str(depth_path), self.depth.estimate(image))
            
            json_path = json_dir / f"{frame_path.stem}.json"
            with open(json_path, 'w') as f:
                json.dump(annotation, f, indent=2, cls=NumpyEncoder)
            
            vis = self._visualize(image, annotation)
            cv2.imwrite(str(vis_dir / f"{frame_path.stem}_vis.jpg"), vis)
            
            all_annotations['frames'][frame_path.name] = annotation
        
        with open(output_dir / 'dataset_annotations.json', 'w') as f:
            json.dump(all_annotations, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nComplete! Output: {output_dir}")
        return all_annotations
    
    def _visualize(self, image: np.ndarray, ann: Dict[str, Any]) -> np.ndarray:
        """Create visualization with 3D annotations."""
        vis = image.copy()
        
        for m in ann['lanes']['markings_3d']:
            s = m['start_pixel']
            e = m['end_pixel']
            cv2.line(vis, tuple(s), tuple(e), (0, 255, 255), 2)
            label = f"x={m['lateral_offset_start_m']:.1f}m"
            cv2.putText(vis, label, (s[0], s[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        for v in ann['vehicles']:
            bbox = v['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            lane = v.get('lane_relative', {}).get('lane_assignment', 'unknown')
            color = {'same': (0, 255, 0), 'left': (255, 0, 0), 
                     'right': (0, 0, 255), 'unknown': (128, 128, 128)}.get(lane, (128, 128, 128))
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            z = v['ground_contact']['longitudinal_z_m']
            x = v['lateral_offset_from_ego_m']
            text = f"{v['class_name']} Z={z:.1f}m X={x:+.1f}m [{lane}]"
            cv2.putText(vis, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        ego = ann['ego_vehicle']['lane_state']
        info = f"Ego: {ego.get('current_lane', '?')} | offset={ego.get('lateral_offset_from_center_m', 0):.2f}m"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(description='Unified Dataset Creation')
    parser.add_argument('--frames_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--yolo_model', required=True)
    parser.add_argument('--segformer_model', required=True)
    parser.add_argument('--depth_checkpoint', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--dataset', default='kitti', choices=['kitti', 'nyu'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--segformer_img_size', type=int, nargs=2, default=[512, 512])
    
    args = parser.parse_args()
    
    processor = UnifiedBatchProcessor(
        args.yolo_model, args.segformer_model, args.depth_checkpoint,
        args.calibration, args.dataset, args.device, tuple(args.segformer_img_size)
    )
    
    processor.process_dataset(Path(args.frames_dir), Path(args.output_dir))


if __name__ == '__main__':
    main()