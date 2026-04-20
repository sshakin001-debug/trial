import argparse
from pathlib import Path
import json
import numpy as np
import cv2
from yolo_inference import YOLOVehicleSegmenter
from segformer_inference import SegformerLaneSegmenter
from depth_inference import DepthInferencer
from depth_fusion import DepthFusion

class BatchProcessor:
    """"Process all frames with YOLO, SegFormer, and Depth models and combine annotations"""
    
    def __init__(self, yolo_model_path: str, segformer_model_path: str, 
                 calibration_path: str, device: str = 'cuda',
                 segformer_img_size: tuple = (512, 512),
                 depth_checkpoint_path: str = None):
        self.yolo_segmenter = YOLOVehicleSegmenter(yolo_model_path, device)
        self.segformer_segmenter = SegformerLaneSegmenter(
            segformer_model_path, device, segformer_img_size
        )
        
        # Initialize depth model if checkpoint provided
        self.depth_inferencer = None
        if depth_checkpoint_path and Path(depth_checkpoint_path).exists():
            self.depth_inferencer = DepthInferencer(
                depth_checkpoint_path, device, dataset='kitti'
            )
            print(f"Loaded depth model from {depth_checkpoint_path}")
        
        # Load calibration
        calib_data = np.load(calibration_path)
        self.calibration = {
            'camera_matrix': calib_data['camera_matrix'].tolist(),
            'distortion_coefficients': calib_data['distortion_coefficients'].tolist(),
            'optimal_camera_matrix': calib_data['optimal_camera_matrix'].tolist(),
            'image_size': calib_data['image_size'].tolist()
        }
        
        # Initialize depth fusion with camera calibration
        self.depth_fusion = DepthFusion(self.calibration)
    
    def process_dataset(self, frames_dir: Path, output_dir: Path) -> dict:
        """Process entire dataset - uses single-pass method if depth model available"""
        if self.depth_inferencer is not None:
            return self._process_dataset_single_pass(frames_dir, output_dir)
        else:
            return self._process_dataset_legacy(frames_dir, output_dir)
    
    def _process_dataset_single_pass(self, frames_dir: Path, output_dir: Path) -> dict:
        """Process each frame through all three models in a single pass"""
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        vehicles_dir = output_dir / 'vehicles'
        lanes_dir = output_dir / 'lanes'
        depth_dir = output_dir / 'depth_maps'
        
        vehicles_dir.mkdir(exist_ok=True)
        lanes_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)
        
        print("=" * 50)
        print("Processing all frames with YOLO, SegFormer, and Depth...")
        print("=" * 50)
        
        # Get all frame paths
        frame_paths = sorted(frames_dir.glob('*.jpg')) + \
                      sorted(frames_dir.glob('*.png')) + \
                      sorted(frames_dir.glob('*.jpeg'))
        
        combined_annotations = {
            'frames': {},
            'metadata': {
                'total_frames': len(frame_paths),
                'with_depth': True
            }
        }
        
        for frame_path in frame_paths:
            print(f"Processing {frame_path.name}...")
            
            # Read image
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            
            original_size = (image.shape[1], image.shape[0])
            
            # Run YOLO
            yolo_results = self.yolo_segmenter.segment_frame(image)
            
            # Run SegFormer
            segformer_results = self.segformer_segmenter.segment_frame(image)
            
            # Run Depth
            depth_map = self.depth_inferencer.infer_depth(image, original_size)
            
            # Save depth map
            depth_path = depth_dir / f"{frame_path.stem}_depth.npy"
            np.save(str(depth_path), depth_map)
            
            # Save vehicle masks
            mask_paths = []
            for i, mask in enumerate(yolo_results['masks']):
                mask_path = vehicles_dir / f"{frame_path.stem}_vehicle_{i:03d}.png"
                cv2.imwrite(str(mask_path), mask * 255)
                mask_paths.append(str(mask_path))
            
            # Save lane masks
            pavement_mask_path = lanes_dir / f"{frame_path.stem}_pavement.png"
            lane_mask_path = lanes_dir / f"{frame_path.stem}_lane.png"
            cv2.imwrite(str(pavement_mask_path), segformer_results['pavement_mask'] * 255)
            cv2.imwrite(str(lane_mask_path), segformer_results['lane_mask'] * 255)
            
            # Fuse with depth
            frame_annotations = self.depth_fusion.fuse_frame(
                vehicle_masks=yolo_results['masks'],
                vehicle_boxes=yolo_results['boxes'],
                vehicle_classes=yolo_results['classes'],
                vehicle_class_names=yolo_results['class_names'],
                vehicle_confidences=yolo_results['confidences'],
                vehicle_mask_paths=mask_paths,
                lane_markings=segformer_results['lane_markings'],
                pavement_mask=segformer_results['pavement_mask'],
                depth_map=depth_map
            )
            
            # Add depth map path
            frame_annotations['depth_map_path'] = str(depth_path)
            
            # Add lane mask paths
            frame_annotations['lane_mask_paths'] = {
                'pavement': str(pavement_mask_path),
                'lane': str(lane_mask_path)
            }
            
            combined_annotations['frames'][frame_path.name] = frame_annotations
        
        # Add calibration
        combined_annotations['calibration'] = self.calibration
        
        # Save combined annotations
        with open(output_dir / 'combined_annotations.json', 'w') as f:
            json.dump(combined_annotations, f, indent=2)
        
        print(f"\nDataset processing complete. Output saved to {output_dir}")
        return combined_annotations
    
    def _process_dataset_legacy(self, frames_dir: Path, output_dir: Path) -> dict:
        """Legacy processing - separate passes for each model"""
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 50)
        print("Processing vehicles with YOLO...")
        print("=" * 50)
        vehicle_output_dir = output_dir / 'vehicles'
        vehicle_annotations = self.yolo_segmenter.process_video_frames(
            frames_dir, vehicle_output_dir
        )
        
        print("\n" + "=" * 50)
        print("Processing lanes with SegFormer...")
        print("=" * 50)
        lane_output_dir = output_dir / 'lanes'
        lane_annotations = self.segformer_segmenter.process_video_frames(
            frames_dir, lane_output_dir
        )
        
        print("\n" + "=" * 50)
        print("Combining annotations...")
        print("=" * 50)
        combined_annotations = self._combine_annotations(
            vehicle_annotations, lane_annotations
        )
        
        # Add calibration to combined annotations
        combined_annotations['calibration'] = self.calibration
        
        # Save combined annotations
        with open(output_dir / 'combined_annotations.json', 'w') as f:
            json.dump(combined_annotations, f, indent=2)
        
        print(f"\nDataset processing complete. Output saved to {output_dir}")
        return combined_annotations
    
    def _combine_annotations(self, vehicle_anns: dict, lane_anns: dict) -> dict:
        """"Combine vehicle and lane annotations for each frame"""
        combined = {
            'frames': {},
            'metadata': {
                'total_frames': len(vehicle_anns),
                'processing_timestamp': str(Path.cwd())
            }
        }
        
        for frame_name in vehicle_anns:
            if frame_name in lane_anns:
                combined['frames'][frame_name] = {
                    'vehicles': vehicle_anns[frame_name],
                    'lanes': lane_anns[frame_name]
                }
        
        return combined

def main():
    parser = argparse.ArgumentParser(description='Batch process dataset with YOLO, SegFormer, and Depth')
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing input frames')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed dataset')
    parser.add_argument('--yolo_model', type=str, required=True,
                        help='Path to YOLO11 model checkpoint (.pt)')
    parser.add_argument('--segformer_model', type=str, required=True,
                        help='Path to SegFormer model checkpoint (.pth)')
    parser.add_argument('--calibration', type=str, required=True,
                        help='Path to camera calibration NPZ file')
    parser.add_argument('--depth_checkpoint', type=str, default=None,
                        help='Path to Depth Anything checkpoint (.pt) for 3D depth')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    parser.add_argument('--segformer_img_size', type=int, nargs=2, default=[512, 512],
                        help='Image size used for SegFormer training (height width)')
    
    args = parser.parse_args()
    
    processor = BatchProcessor(
        args.yolo_model,
        args.segformer_model,
        args.calibration,
        args.device,
        tuple(args.segformer_img_size),
        args.depth_checkpoint
    )
    
    processor.process_dataset(Path(args.frames_dir), Path(args.output_dir))

if __name__ == '__main__':
    main()
