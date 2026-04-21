"""
Argument parsing utilities.
"""

import argparse
from typing import List, Optional


def get_args(description: str = "ZoeDepth inference") -> argparse.Namespace:
    """
    Get command line arguments for ZoeDepth.
    """
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--model', type=str, default='zoedepth',
                      help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to checkpoint file')
    parser.add_argument('--dataset', type=str, default='kitti',
                      choices=['kitti', 'nyu'],
                      help='Dataset for depth estimation')
    parser.add_argument('--image', type=str,
                      help='Path to input image')
    parser.add_argument('--output', type=str, default='depth_output.png',
                      help='Output depth path')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to run on')
    parser.add_argument('--calibration', type=str,
                      help='Path to calibration file')
    
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> bool:
    """
    Validate command line arguments.
    """
    if not args.checkpoint:
        print("Error: --checkpoint is required")
        return False
    
    return True