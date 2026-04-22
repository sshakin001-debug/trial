import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
from ultralytics import YOLO


class YOLOVehicleSegmenter:
    """YOLO11-based vehicle segmentation for Bangladeshi vehicles"""

    def __init__(self, model_path: str, device: str = "cuda",
                 confidence_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.class_names = self.model.names
        print(f"Loaded YOLO model with classes: {self.class_names}")

    def segment_frame(self, image: np.ndarray) -> Dict[str, Any]:
        results = self.model(image, device=self.device, conf=self.confidence_threshold)[0]
        output = {
            "masks": [], "boxes": [], "classes": [],
            "class_names": [], "confidences": [],
        }
        if results.masks is not None:
            masks       = results.masks.data.cpu().numpy()
            boxes       = results.boxes.xyxy.cpu().numpy()
            classes     = results.boxes.cls.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            for mask, box, cls, conf in zip(masks, boxes, classes, confidences):
                cls_int = int(cls)
                output["masks"].append((mask > 0.5).astype(np.uint8))
                output["boxes"].append(box.tolist())
                output["classes"].append(cls_int)
                output["class_names"].append(self.class_names.get(cls_int, f"class_{cls_int}"))
                output["confidences"].append(float(conf))
        return output

    def process_video_frames(self, frames_dir: Path, output_dir: Path) -> Dict:
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "vehicle_masks"
        vis_dir   = output_dir / "visualizations"
        masks_dir.mkdir(exist_ok=True)
        vis_dir.mkdir(exist_ok=True)
        all_annotations = {}
        frame_paths = (sorted(frames_dir.glob("*.jpg")) +
                       sorted(frames_dir.glob("*.png")) +
                       sorted(frames_dir.glob("*.jpeg")))
        for frame_path in frame_paths:
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            results = self.segment_frame(image)
            frame_masks = []
            for i, mask in enumerate(results["masks"]):
                mask_path = masks_dir / f"{frame_path.stem}_vehicle_{i:03d}.png"
                cv2.imwrite(str(mask_path), mask * 255)
                frame_masks.append(str(mask_path))
            vis_image = self._create_visualization(image, results)
            vis_path  = vis_dir / f"{frame_path.stem}_vis.jpg"
            cv2.imwrite(str(vis_path), vis_image)
            all_annotations[frame_path.name] = {
                "frame": frame_path.name,
                "num_vehicles": len(results["masks"]),
                "boxes": results["boxes"],
                "classes": results["classes"],
                "class_names": results["class_names"],
                "confidences": results["confidences"],
                "mask_paths": frame_masks,
                "visualization_path": str(vis_path),
            }
        with open(output_dir / "vehicle_annotations.json", "w") as f:
            json.dump(all_annotations, f, indent=2)
        return all_annotations

    def _create_visualization(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        vis_image = image.copy()
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(max(len(results["masks"]), 1), 3))
        for i, (mask, box, class_name, conf) in enumerate(zip(
            results["masks"], results["boxes"],
            results["class_names"], results["confidences"]
        )):
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask > 0] = colors[i]
            vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, 0.5, 0)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), colors[i].tolist(), 2)
            cv2.putText(vis_image, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i].tolist(), 2)
        return vis_image