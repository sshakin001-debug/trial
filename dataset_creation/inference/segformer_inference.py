import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import json
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image

class SegformerLaneSegmenter:
    """"SegFormer-based lane and pavement segmentation"""
    
    def __init__(self, model_path: str, device: str = 'cuda', img_size: tuple = (512, 512)):
        """""
        Args:
            model_path: Path to SegFormer model checkpoint (.pth file)
            device: 'cuda' or 'cpu'
            img_size: Input image size used during training (height, width)
        """""
        self.device = device
        self.img_size = img_size
        self.num_classes = 3  # background, lane markings, pavement
        
        # Load model with correct architecture
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b4-finetuned-ade-512-512",
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Load trained weights
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            # Handle both full checkpoint and state_dict-only saves
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
                # Load image size from checkpoint if available
                if 'img_size' in checkpoint:
                    self.img_size = checkpoint['img_size']
                    print(f"Loaded img_size from checkpoint: {self.img_size}")
            else:
                state_dict = checkpoint
            
            # Load state dict with strict=False to handle any minor mismatches
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Missing keys (expected for fresh head): {missing[:5]}...")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        self.model.to(device)
        self.model.eval()
        
        # Initialize processor for the correct model
        self.processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b4-finetuned-ade-512-512"
        )
        
        # Class definitions for this specific model
        self.class_names = {
            0: 'background',
            1: 'lane_marking',
            2: 'pavement'
        }
        
        # Focus classes mapping
        self.focus_classes = {
            'lane_marking': 1,
            'pavement': 2,
        }
    
    def segment_frame(self, image: np.ndarray) -> Dict[str, Any]:
        """""
        Perform lane and pavement segmentation
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            Dictionary containing:
                - segmentation: Full segmentation map
                - pavement_mask: Binary mask for pavement
                - lane_mask: Binary mask for lane markings
                - lane_markings: Detected lane boundaries (from lane mask)
                - class_distribution: Pixel count per class
        """""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # (H, W)
        
        # Preprocess - resize to model's expected size
        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, num_classes, H/4, W/4]
        
        # Upsample to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )
        
        # Get segmentation map
        segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Extract specific masks
        pavement_mask = (segmentation == self.focus_classes['pavement']).astype(np.uint8)
        lane_mask = (segmentation == self.focus_classes['lane_marking']).astype(np.uint8)
        
        # Detect lane boundaries from lane mask (morphological operations)
        lane_markings = self._extract_lane_lines(lane_mask, image)
        
        # Calculate class distribution
        unique, counts = np.unique(segmentation, return_counts=True)
        class_distribution = {
            self.class_names.get(int(cls), f'class_{cls}'): int(count)
            for cls, count in zip(unique, counts)
        }
        
        return {
            'segmentation': segmentation,
            'pavement_mask': pavement_mask,
            'lane_mask': lane_mask,
            'lane_markings': lane_markings,
            'class_distribution': class_distribution
        }
    
    def _extract_lane_lines(self, lane_mask: np.ndarray, image: np.ndarray) -> List[Dict]:
        """""
        Extract lane line segments from lane mask using morphological operations
        and Hough transform
        """""
        lane_markings = []
        
        if lane_mask.sum() == 0:
            return lane_markings
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        lane_mask_clean = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        lane_mask_clean = cv2.morphologyEx(lane_mask_clean, cv2.MORPH_OPEN, kernel)
        
        # Find contours of lane markings
        contours, _ = cv2.findContours(lane_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 5:
                continue
                
            # Fit line to contour points
            [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Get line endpoints
            rows, cols = lane_mask.shape
            if abs(vy) > abs(vx):  # More vertical line
                y1 = 0
                y2 = rows - 1
                x1 = int(x0 + (y1 - y0) * vx / vy) if vy != 0 else int(x0)
                x2 = int(x0 + (y2 - y0) * vx / vy) if vy != 0 else int(x0)
            else:  # More horizontal line
                x1 = 0
                x2 = cols - 1
                y1 = int(y0 + (x1 - x0) * vy / vx) if vx != 0 else int(y0)
                y2 = int(y0 + (x2 - x0) * vy / vx) if vx != 0 else int(y0)
            
            # Clip to image boundaries
            x1 = max(0, min(cols - 1, x1))
            x2 = max(0, min(cols - 1, x2))
            y1 = max(0, min(rows - 1, y1))
            y2 = max(0, min(rows - 1, y2))
            
            if abs(x2 - x1) > 20 or abs(y2 - y1) > 20:  # Minimum line length
                lane_markings.append({
                    'start': [int(x1), int(y1)],
                    'end': [int(x2), int(y2)],
                    'length': float(np.sqrt((x2-x1)**2 + (y2-y1)**2)),
                    'angle': float(np.arctan2(y2-y1, x2-x1))
                })
        
        return lane_markings
    
    def process_video_frames(self, frames_dir: Path, output_dir: Path) -> Dict:
        """"Process all frames in a directory"""
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
                      sorted(frames_dir.glob('*.png')) + \
                      sorted(frames_dir.glob('*.jpeg'))
        
        for frame_path in frame_paths:
            print(f"Processing {frame_path.name}...")
            
            # Read image
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            
            # Segment
            results = self.segment_frame(image)
            
            # Save masks
            pavement_mask_path = masks_dir / f"{frame_path.stem}_pavement.png"
            lane_mask_path = masks_dir / f"{frame_path.stem}_lane.png"
            full_seg_path = masks_dir / f"{frame_path.stem}_full_seg.png"
            
            cv2.imwrite(str(pavement_mask_path), results['pavement_mask'] * 255)
            cv2.imwrite(str(lane_mask_path), results['lane_mask'] * 255)
            
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
                'pavement_mask_path': str(pavement_mask_path),
                'lane_mask_path': str(lane_mask_path),
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
        """"Convert segmentation map to color image"""
        # Custom palette for 3 classes: background, lane, pavement
        palette = np.array([
            [0, 0, 0],        # 0: background - black
            [255, 255, 0],    # 1: lane marking - yellow
            [128, 64, 128],   # 2: pavement - purple
        ])
        
        h, w = segmentation.shape
        color_seg = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(palette):
            color_seg[segmentation == class_id] = color
        
        return cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)
    
    def _create_visualization(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """"Create visualization of lane segmentation"""
        vis_image = image.copy()
        
        # Create overlay for pavement (purple/blue tint)
        pavement_overlay = np.zeros_like(vis_image)
        pavement_overlay[results['pavement_mask'] > 0] = [128, 0, 128]  # Purple in BGR
        vis_image = cv2.addWeighted(vis_image, 1.0, pavement_overlay, 0.4, 0)
        
        # Create overlay for lane markings (yellow)
        lane_overlay = np.zeros_like(vis_image)
        lane_overlay[results['lane_mask'] > 0] = [0, 255, 255]  # Yellow in BGR
        vis_image = cv2.addWeighted(vis_image, 1.0, lane_overlay, 0.6, 0)
        
        # Draw extracted lane lines (red for clarity)
        for marking in results['lane_markings']:
            start = tuple(marking['start'])
            end = tuple(marking['end'])
            cv2.line(vis_image, start, end, (0, 0, 255), 2)
        
        # Add legend
        cv2.putText(vis_image, "Pavement", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
        cv2.putText(vis_image, "Lane Markings", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_image, "Detected Lines", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_image
