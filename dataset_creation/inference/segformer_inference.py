import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image

class SegformerLaneSegmenter:
    """SegFormer-based lane and pavement segmentation"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Args:
            model_path: Path to SegFormer model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        # Load model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        )
        
        # Load custom weights if provided
        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
        
        self.model.to(device)
        self.model.eval()
        
        # Initialize processor
        self.processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        )
        
        # Define class mappings (Cityscapes format)
        self.class_mapping = {
            0: 'road',      # road
            1: 'sidewalk',  # sidewalk
            2: 'building',  # building
            3: 'wall',      # wall
            4: 'fence',     # fence
            5: 'pole',      # pole
            6: 'traffic light',  # traffic light
            7: 'traffic sign',   # traffic sign
            8: 'vegetation',     # vegetation
            9: 'terrain',        # terrain
            10: 'sky',           # sky
            11: 'person',        # person
            12: 'rider',         # rider
            13: 'car',           # car
            14: 'truck',         # truck
            15: 'bus',           # bus
            16: 'train',         # train
            17: 'motorcycle',    # motorcycle
            18: 'bicycle',       # bicycle
        }
        
        # Focus classes for lane and pavement
        self.focus_classes = {
            'road': 0,
            'sidewalk': 1,
            'traffic sign': 7,
        }
    
    def segment_frame(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform lane and pavement segmentation
        
        Returns:
            Dictionary containing:
                - segmentation: Full segmentation map
                - road_mask: Binary mask for road
                - sidewalk_mask: Binary mask for sidewalk/pavement
                - lane_markings: Detected lane boundaries
                - class_distribution: Pixel count per class
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Upsample to original size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False
        )
        
        # Get segmentation map
        segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Extract specific masks
        road_mask = (segmentation == self.focus_classes['road']).astype(np.uint8)
        sidewalk_mask = (segmentation == self.focus_classes['sidewalk']).astype(np.uint8)
        
        # Detect lane markings from road mask
        lane_markings = self._detect_lane_markings(road_mask, image)
        
        # Calculate class distribution
        unique, counts = np.unique(segmentation, return_counts=True)
        class_distribution = {
            self.class_mapping.get(int(cls), f'class_{cls}'): int(count)
            for cls, count in zip(unique, counts)
        }
        
        return {
            'segmentation': segmentation,
            'road_mask': road_mask,
            'sidewalk_mask': sidewalk_mask,
            'lane_markings': lane_markings,
            'class_distribution': class_distribution
        }
    
    def _detect_lane_markings(self, road_mask: np.ndarray, image: np.ndarray) -> List[Dict]:
        """Detect lane markings using edge detection on road mask"""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        masked_gray = cv2.bitwise_and(gray, gray, mask=road_mask)
        
        # Edge detection
        edges = cv2.Canny(masked_gray, 50, 150)
        
        # Hough transform for line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=100, maxLineGap=50)
        
        lane_markings = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                lane_markings.append({
                    'start': [int(x1), int(y1)],
                    'end': [int(x2), int(y2)],
                    'length': float(np.sqrt((x2-x1)**2 + (y2-y1)**2)),
                    'angle': float(np.arctan2(y2-y1, x2-x1))
                })
        
        return lane_markings
    
    def process_video_frames(self, frames_dir: Path, output_dir: Path):
        """Process all frames in a directory"""
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        masks_dir = output_dir / 'lane_masks'
        vis_dir = output_dir / 'visualizations'
        masks_dir.mkdir(exist_ok=True)
        vis_dir.mkdir(exist_ok=True)
        
        all_annotations = {}
        
        # Process each frame
        frame_paths = sorted(frames_dir.glob('*.jpg')) + \
                      sorted(frames_dir.glob('*.png'))
        
        for frame_path in frame_paths:
            print(f"Processing {frame_path.name}...")
            
            # Read image
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            
            # Segment
            results = self.segment_frame(image)
            
            # Save masks
            road_mask_path = masks_dir / f"{frame_path.stem}_road.png"
            sidewalk_mask_path = masks_dir / f"{frame_path.stem}_sidewalk.png"
            full_seg_path = masks_dir / f"{frame_path.stem}_full_seg.png"
            
            cv2.imwrite(str(road_mask_path), results['road_mask'] * 255)
            cv2.imwrite(str(sidewalk_mask_path), results['sidewalk_mask'] * 255)
            
            # Save full segmentation as color-coded image
            seg_colored = self._colorize_segmentation(results['segmentation'])
            cv2.imwrite(str(full_seg_path), seg_colored)
            
            # Create visualization
            vis_image = self._create_visualization(image, results)
            vis_path = vis_dir / f"{frame_path.stem}_vis.jpg"
            cv2.imwrite(str(vis_path), vis_image)
            
            # Store annotations
            all_annotations[frame_path.name] = {
                'frame': frame_path.name,
                'road_mask_path': str(road_mask_path),
                'sidewalk_mask_path': str(sidewalk_mask_path),
                'full_segmentation_path': str(full_seg_path),
                'visualization_path': str(vis_path),
                'class_distribution': results['class_distribution'],
                'lane_markings': results['lane_markings']
            }
        
        # Save annotations
        with open(output_dir / 'lane_annotations.json', 'w') as f:
            json.dump(all_annotations, f, indent=2)
        
        return all_annotations
    
    def _colorize_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Convert segmentation map to color image"""
        # Cityscapes color palette
        palette = np.array([
            [128, 64, 128],   # road
            [244, 35, 232],   # sidewalk
            [70, 70, 70],     # building
            [102, 102, 156],  # wall
            [190, 153, 153],  # fence
            [153, 153, 153],  # pole
            [250, 170, 30],   # traffic light
            [220, 220, 0],    # traffic sign
            [107, 142, 35],   # vegetation
            [152, 251, 152],  # terrain
            [70, 130, 180],   # sky
            [220, 20, 60],    # person
            [255, 0, 0],      # rider
            [0, 0, 142],      # car
            [0, 0, 70],       # truck
            [0, 60, 100],     # bus
            [0, 80, 100],     # train
            [0, 0, 230],      # motorcycle
            [119, 11, 32],    # bicycle
        ])
        
        h, w = segmentation.shape
        color_seg = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(palette):
            color_seg[segmentation == class_id] = color
        
        return cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)
    
    def _create_visualization(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Create visualization of lane segmentation"""
        vis_image = image.copy()
        
        # Create overlay for road (blue tint)
        road_overlay = np.zeros_like(vis_image)
        road_overlay[results['road_mask'] > 0] = [255, 0, 0]  # Blue in BGR
        vis_image = cv2.addWeighted(vis_image, 1.0, road_overlay, 0.3, 0)
        
        # Create overlay for sidewalk (green tint)
        sidewalk_overlay = np.zeros_like(vis_image)
        sidewalk_overlay[results['sidewalk_mask'] > 0] = [0, 255, 0]  # Green in BGR
        vis_image = cv2.addWeighted(vis_image, 1.0, sidewalk_overlay, 0.3, 0)
        
        # Draw lane markings
        for marking in results['lane_markings']:
            start = tuple(marking['start'])
            end = tuple(marking['end'])
            cv2.line(vis_image, start, end, (0, 255, 255), 2)  # Yellow lines
        
        return vis_image