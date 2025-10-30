#!/usr/bin/env python3
"""
ROVER Image Sequence to Video Converter

Converts ROVER dataset image sequences to video files for easier processing
with the video_analysis_experiments.py script.

Usage:
    python convert_rover_to_video.py \
        --input-dir /path/to/ROVER/campus_large_summer/realsense_D435i/rgb \
        --output-path summer_campus_d435i.mp4 \
        --fps 30
"""

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm


def convert_image_sequence_to_video(image_dir: str, output_path: str, fps: int = 30):
    """
    Convert a directory of timestamp-named PNG images to a video file.
    
    Args:
        image_dir: Directory containing timestamp.png files
        output_path: Output video file path
        fps: Frames per second for output video
    """
    image_dir = Path(image_dir)
    
    # Get all PNG files and sort by filename (which is the timestamp)
    image_files = sorted(image_dir.glob("*.png"))
    
    if not image_files:
        print(f"Error: No PNG files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"Error: Could not read {image_files[0]}")
        return
    
    height, width, _ = first_image.shape
    print(f"Image size: {width}x{height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create video writer for {output_path}")
        return
    
    # Write images to video
    for img_path in tqdm(image_files, desc="Converting to video"):
        img = cv2.imread(str(img_path))
        if img is not None:
            out.write(img)
        else:
            print(f"Warning: Could not read {img_path}")
    
    out.release()
    print(f"\n✓ Video saved to: {output_path}")
    print(f"  Duration: {len(image_files)/fps:.1f} seconds @ {fps} fps")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ROVER image sequences to video"
    )
    parser.add_argument(
        "--input-dir", 
        type=str, 
        required=True,
        help="Directory containing timestamp.png files"
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        required=True,
        help="Output video file path (e.g., output.mp4)"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30,
        help="Frames per second for output video (default: 30)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to include (for testing)"
    )
    
    args = parser.parse_args()
    
    convert_image_sequence_to_video(
        args.input_dir, 
        args.output_path, 
        args.fps
    )


if __name__ == "__main__":
    main()
