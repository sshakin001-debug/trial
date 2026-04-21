import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import glob

class CameraCalibrator:
    """Camera calibration using checkerboard patterns"""
    
    def __init__(self, checkerboard_size: Tuple[int, int] = (8, 6), 
                 square_size: float = 0.08,
                 max_image_dimension: int = 2000):
        """
        Args:
            checkerboard_size: Number of inner corners (width, height)
            square_size: Size of checkerboard squares in meters
            max_image_dimension: Maximum width/height in pixels before downscaling
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.max_image_dimension = max_image_dimension
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                    0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
    def _resize_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize image if it exceeds max dimension while maintaining aspect ratio"""
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= self.max_image_dimension:
            return image, 1.0
        
        scale = self.max_image_dimension / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized, scale
    
    def detect_checkerboard(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], 
                                                               Optional[np.ndarray]]:
        """Detect checkerboard corners in image"""
        # Resize high-resolution images before detection
        resized_image, scale = self._resize_if_needed(image)
        
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            # Scale corners back to original image coordinates
            if scale != 1.0:
                corners2 = corners2 / scale
            return True, corners2, image.shape[:2][::-1]  # Use original image size
        return False, None, image.shape[:2][::-1]
    
    def calibrate_from_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """Perform camera calibration from multiple checkerboard images"""
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        image_size = None
        
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            ret, corners, size = self.detect_checkerboard(img)
            if ret:
                objpoints.append(self.objp)
                imgpoints.append(corners)
                if image_size is None:
                    image_size = size
        
        if len(objpoints) < 3:
            raise ValueError(f"Need at least 3 valid checkerboard images, got {len(objpoints)}")
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        
        # Calculate reprojection error
        mean_error = self._calculate_reprojection_error(objpoints, imgpoints, 
                                                        rvecs, tvecs, mtx, dist)
        
        # Get optimal new camera matrix
        h, w = image_size[1], image_size[0]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        return {
            'camera_matrix': mtx,
            'distortion_coefficients': dist,
            'rotation_vectors': rvecs,
            'translation_vectors': tvecs,
            'optimal_camera_matrix': newcameramtx,
            'roi': roi,
            'image_size': image_size,
            'reprojection_error': mean_error,
            'calibration_success': ret
        }
    
    def _calculate_reprojection_error(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        """Calculate mean reprojection error"""
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        return mean_error / len(objpoints)
    
    def save_calibration(self, calibration_data: Dict[str, Any], output_path: str):
        """Save calibration data as NPZ file"""
        np.savez_compressed(
            output_path,
            camera_matrix=calibration_data['camera_matrix'],
            distortion_coefficients=calibration_data['distortion_coefficients'],
            rotation_vectors=calibration_data['rotation_vectors'],
            translation_vectors=calibration_data['translation_vectors'],
            optimal_camera_matrix=calibration_data['optimal_camera_matrix'],
            roi=calibration_data['roi'],
            image_size=calibration_data['image_size'],
            reprojection_error=calibration_data['reprojection_error']
        )
        
        # Also save as JSON for compatibility
        json_path = output_path.replace('.npz', '.json')
        json_data = {
            'camera_matrix': calibration_data['camera_matrix'].tolist(),
            'distortion_coefficients': calibration_data['distortion_coefficients'].tolist(),
            'optimal_camera_matrix': calibration_data['optimal_camera_matrix'].tolist(),
            'roi': [int(x) for x in calibration_data['roi']],
            'image_size': calibration_data['image_size'],
            'reprojection_error': float(calibration_data['reprojection_error'])
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def load_calibration(self, input_path: str) -> Dict[str, Any]:
        """Load calibration data from NPZ file"""
        data = np.load(input_path)
        return {
            'camera_matrix': data['camera_matrix'],
            'distortion_coefficients': data['distortion_coefficients'],
            'rotation_vectors': data['rotation_vectors'],
            'translation_vectors': data['translation_vectors'],
            'optimal_camera_matrix': data['optimal_camera_matrix'],
            'roi': data['roi'],
            'image_size': data['image_size'],
            'reprojection_error': data['reprojection_error']
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Camera Calibration')
    parser.add_argument('image_dir', help='Directory containing checkerboard images')
    parser.add_argument('--output', default='calibration.npz', help='Output calibration file')
    parser.add_argument('--checkerboard_size', nargs=2, type=int, default=[9,6], help='Checkerboard size (width height)')
    parser.add_argument('--square_size', type=float, default=0.025, help='Square size in meters')
    parser.add_argument('--max_dimension', type=int, default=2000, help='Max image dimension before downscaling')
    args = parser.parse_args()

    calibrator = CameraCalibrator(
        tuple(args.checkerboard_size), 
        args.square_size,
        max_image_dimension=args.max_dimension
    )
    image_paths = glob.glob(f'{args.image_dir}/*.jpg') + glob.glob(f'{args.image_dir}/*.png')
    if not image_paths:
        print("No images found")
        return
    try:
        calibration = calibrator.calibrate_from_images(image_paths)
        calibrator.save_calibration(calibration, args.output)
        print(f"Calibration saved to {args.output}")
        print(f"Reprojection error: {calibration['reprojection_error']}")
    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main()