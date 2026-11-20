"""
================================================================================
BOROBUDUR DIGITAL PRESERVATION: REPRODUCIBLE CLAHE ENHANCEMENT PIPELINE
================================================================================
Paper: Standardized Protocol for Digital Preservation of Eroded Stone Reliefs:
       Integrating Spatially-Calibrated Acquisition and CLAHE-Based Enhancement
       for Machine Learning Applications
Journal: International Journal of Image, Graphics and Signal Processing (IJIGSP)
Authors: Peter Ardhianto, Cornelio Abdimash Christiono, Yonathan Purbo Santosa,
         Gustav Anandhita, Christian Moniaga, Gita Claudia
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
================================================================================
"""

import os
import sys
import argparse
import glob
import cv2
import numpy as np

def process_image(img_path, clip_limit=3.0, tile_grid_size=(32, 32)):
    """
    Applies the standardized 2-stage digital signal conditioning pipeline:
    1. 8-bit Grayscale Conversion (eliminating chromatic lichen/mineral noise).
    2. Contrast Limited Adaptive Histogram Equalization (CLAHE) with optimal parameters.
    """
    # 1. Read input image
    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        raise ValueError(f"Unable to read image at: {img_path}")
    
    # 2. Stage 1: Grayscale Conversion
    # Isolates luminance signal from biological stains (moss, lichen)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    
    # 3. Stage 2: CLAHE Enhancement
    # clipLimit = 3.0, tileGridSize = (32, 32) as determined via parametric optimization
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_img = clahe.apply(gray_img)
    
    return gray_img, enhanced_img

def batch_process(input_dir, output_dir, clip_limit=3.0, tile_grid_size=(32, 32)):
    """
    Batch processes all images (*.png, *.jpg, *.jpeg, *.tif) in input_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths = sorted(image_paths)
    
    print(f"=== Standardized CLAHE Enhancement Pipeline ===")
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Parameters:       clipLimit={clip_limit}, tileGridSize={tile_grid_size}")
    print(f"Found {len(image_paths)} images to process.\n")
    
    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        base_name, _ = os.path.splitext(filename)
        print(f"[{idx}/{len(image_paths)}] Processing {filename}...")
        
        _, enhanced = process_image(img_path, clip_limit, tile_grid_size)
        
        out_filename = f"{base_name}_clahe_enhanced.png"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, enhanced)
        
    print(f"\n[DONE] Successfully processed {len(image_paths)} images to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Standardized CLAHE Pipeline for Borobudur Relief Preservation")
    parser.add_argument("--input", "-i", type=str, default="./raw_samples", help="Input directory containing raw images")
    parser.add_argument("--output", "-o", type=str, default="./enhanced_output", help="Output directory for processed images")
    parser.add_argument("--clip_limit", "-c", type=float, default=3.0, help="CLAHE clipLimit (default: 3.0)")
    parser.add_argument("--grid_size", "-g", type=int, default=32, help="CLAHE tileGridSize N (default: 32 for 32x32)")
    
    args = parser.parse_args()
    batch_process(args.input, args.output, args.clip_limit, (args.grid_size, args.grid_size))

if __name__ == "__main__":
    main()
