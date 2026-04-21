1. segformer_inference.py — _create_visualization draws extended lines (the main visual bug)
2. yolo_inference.py — confidence threshold 0.5 is too high (misses bicycle)
3. zoedepth/models/builder.py — build_model passes device to build_model but the function signature accepts it but never uses it for CPU xformers fix
4. zoedepth/models/base_models/dpt_dinov2/dpt.py — hardcoded ./torchhub path
Let me write out the exact fixes for each file so you can apply and push:

Fix 1 — dataset_creation/inference/segformer_inference.py
Replace _create_visualization method (lines 273–301):
python    def _create_visualization(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """"Create visualization of lane segmentation"""
        vis_image = image.copy()
        
        # Pavement overlay (purple)
        pavement_overlay = np.zeros_like(vis_image)
        pavement_overlay[results['pavement_mask'] > 0] = [128, 0, 128]
        vis_image = cv2.addWeighted(vis_image, 1.0, pavement_overlay, 0.4, 0)
        
        # Lane marking overlay (yellow fill)
        lane_overlay = np.zeros_like(vis_image)
        lane_overlay[results['lane_mask'] > 0] = [0, 255, 255]
        vis_image = cv2.addWeighted(vis_image, 1.0, lane_overlay, 0.6, 0)
        
        # Draw lane mask contours only — NO extended fitted lines
        lane_contours, _ = cv2.findContours(
            results['lane_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_image, lane_contours, -1, (0, 255, 255), 2)
        
        # Legend
        cv2.putText(vis_image, "Pavement", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
        cv2.putText(vis_image, "Lane Markings", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return vis_image

Fix 2 — dataset_creation/inference/yolo_inference.py
Line 13, change:
pythonconfidence_threshold: float = 0.5
to:
pythonconfidence_threshold: float = 0.25

Fix 3 — zoedepth/models/base_models/dpt_dinov2/dpt.py
The file you shared doesn't use torch.hub.load at all — it uses a custom ViTAdapter. This means the ./torchhub issue only exists in the image-to-pcd zoedepth that was swapped in earlier. Since your repo's own dpt.py is self-contained, no fix needed here — but confirm which zoedepth is active:
python!head -5 /content/trial/zoedepth/models/base_models/dpt_dinov2/dpt.py
If it shows the ViTAdapter version above, your repo's own code is being used and the torchhub issue is gone. If it shows torch.hub.load, the image-to-pcd version is still there.



Files changed:

dataset_creation/inference/depth_estimator.py — fully replaced (the version you just shared is the correct final one)
dataset_creation/inference/segformer_inference.py — _create_visualization fix
dataset_creation/inference/yolo_inference.py — confidence 0.5 → 0.25
zoedepth/ — entire folder replaced with image-to-pcd's version
zoedepth/models/builder.py — two patches: space fix in class name + strict=False