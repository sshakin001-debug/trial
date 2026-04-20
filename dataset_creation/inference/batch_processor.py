import argparse
from pathlib import Path
import json
import numpy as np
from yolo_inference import YOLOVehicleSegmenter
from segformer_inference import SegformerLaneSegmenter

class BatchProcessor:
    """"Process all frames with both models and combine annotations"""
    
    def __init__(self, yolo_model_path: str, segformer_model_path: str, 
                 calibration_path: str, device: str = 'cuda',
                 segformer_img_size: tuple = (512, 512)):
        self.yolo_segmenter = YOLOVehicleSegmenter(yolo_model_path, device)
        self.segformer_segmenter = SegformerLaneSegmenter(
            segformer_model_path, device, segformer_img_size
        )
        
        # Load calibration
        calib_data = np.load(calibration_path)
        self.calibration = {
            'camera_matrix': calib_data['camera_matrix'].tolist(),
            'distortion_coefficients': calib_data['distortion_coefficients'].tolist(),
            'optimal_camera_matrix': calib_data['optimal_camera_matrix'].tolist(),
            'image_size': calib_data['image_size'].tolist()
        }
    
    def process_dataset(self, frames_dir: Path, output_dir: Path) -> dict:
        """"Process entire dataset"""
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
    parser = argparse.ArgumentParser(description='Batch process dataset with YOLO and SegFormer')
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
        tuple(args.segformer_img_size)
    )
    
    processor.process_dataset(Path(args.frames_dir), Path(args.output_dir))

if __name__ == '__main__':
    main()
