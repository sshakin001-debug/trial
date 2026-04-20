import argparse
from pathlib import Path
from calibrate_camera import CameraCalibrator

def main():
    parser = argparse.ArgumentParser(description='Generate camera calibration NPZ files')
    parser.add_argument('--calib_images_dir', type=str, required=True,
                        help='Directory containing calibration checkerboard images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for calibration files')
    parser.add_argument('--checkerboard_width', type=int, default=9,
                        help='Number of inner corners width-wise')
    parser.add_argument('--checkerboard_height', type=int, default=6,
                        help='Number of inner corners height-wise')
    parser.add_argument('--square_size', type=float, default=0.025,
                        help='Size of checkerboard squares in meters')
    
    args = parser.parse_args()
    
    calib_dir = Path(args.calib_images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_paths = list(calib_dir.glob('*.jpg')) + \
                  list(calib_dir.glob('*.png')) + \
                  list(calib_dir.glob('*.jpeg'))
    
    # Perform calibration
    calibrator = CameraCalibrator(
        checkerboard_size=(args.checkerboard_width, args.checkerboard_height),
        square_size=args.square_size
    )
    
    calibration_data = calibrator.calibrate_from_images(image_paths)
    
    # Save calibration
    output_path = output_dir / 'camera_calibration.npz'
    calibrator.save_calibration(calibration_data, str(output_path))
    
    print(f"Calibration saved to {output_path}")
    print(f"Reprojection error: {calibration_data['reprojection_error']:.4f}")

if __name__ == '__main__':
    main()