"""
================================================================================
BOROBUDUR IMAGE QUALITY ASSESSMENT (IQA) METRICS COMPUTATION ENGINE
================================================================================
Paper: Standardized Protocol for Digital Preservation of Eroded Stone Reliefs
Journal: International Journal of Image, Graphics and Signal Processing (IJIGSP)
Metrics implemented:
1. Tenengrad Edge Gradient (Sobel Gx^2 + Gy^2)
2. Laplacian Structural Sharpness (Laplacian Variance)
3. Tonal Dynamic Spread (Standard Deviation sigma)
4. Root Mean Square (RMS) Contrast (Normalized)
5. Shannon Information Entropy (bits/pixel)
================================================================================
"""

import os
import sys
import argparse
import cv2
import numpy as np

def calculate_tenengrad(img_gray):
    """
    Computes Tenengrad Gradient using Sobel operators:
    Gx = Sobel_x(I, CV_64F)
    Gy = Sobel_y(I, CV_64F)
    Tenengrad = sqrt(mean(Gx^2 + Gy^2))
    """
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_sq = (gx ** 2) + (gy ** 2)
    return float(np.sqrt(np.mean(gradient_sq)))

def calculate_laplacian_sharpness(img_gray):
    """
    Computes structural sharpness via Laplacian variance:
    Var(Lap) = Var(cv2.Laplacian(I, CV_64F))
    """
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(lap.var())

def calculate_tonal_spread(img_gray):
    """
    Computes tonal dynamic spread (sample standard deviation of intensity values):
    sigma = std(I)
    """
    return float(img_gray.std())

def calculate_rms_contrast(img_gray):
    """
    Computes Root Mean Square (RMS) contrast normalized to [0, 1]:
    RMS = sqrt( (1 / (M*N)) * sum( ((I(x,y) - mean(I)) / 255.0)^2 ) )
    Equivalent to: std(I / 255.0)
    """
    norm_img = img_gray.astype(np.float64) / 255.0
    return float(norm_img.std())

def calculate_shannon_entropy(img_gray):
    """
    Computes Shannon Information Entropy in bits/pixel:
    H = - sum( p(k) * log2(p(k)) ) for all gray levels k with p(k) > 0.
    """
    hist, _ = np.histogram(img_gray.ravel(), bins=256, range=[0, 256])
    total_pixels = img_gray.size
    probabilities = hist / float(total_pixels)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)

def evaluate_image_pair(raw_path, enhanced_path):
    """
    Compares raw field capture against CLAHE enhanced version across all 5 metrics.
    """
    raw_bgr = cv2.imread(raw_path)
    enh_bgr = cv2.imread(enhanced_path)
    
    if raw_bgr is None or enh_bgr is None:
        raise ValueError("Failed to load input image pair.")
        
    raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    enh_gray = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2GRAY) if len(enh_bgr.shape) == 3 else enh_bgr
    
    # Compute metrics
    metrics = {
        "Tenengrad Edge Gradient (A.U.)": {
            "Raw": calculate_tenengrad(raw_gray),
            "Enhanced": calculate_tenengrad(enh_gray)
        },
        "Laplacian Structural Sharpness (A.U.)": {
            "Raw": calculate_laplacian_sharpness(raw_gray),
            "Enhanced": calculate_laplacian_sharpness(enh_gray)
        },
        "Tonal Dynamic Spread (sigma)": {
            "Raw": calculate_tonal_spread(raw_gray),
            "Enhanced": calculate_tonal_spread(enh_gray)
        },
        "RMS Contrast (Normalized)": {
            "Raw": calculate_rms_contrast(raw_gray),
            "Enhanced": calculate_rms_contrast(enh_gray)
        },
        "Shannon Information Entropy (bits/px)": {
            "Raw": calculate_shannon_entropy(raw_gray),
            "Enhanced": calculate_shannon_entropy(enh_gray)
        }
    }
    
    # Calculate gain
    for k, v in metrics.items():
        v["Gain_Pct"] = ((v["Enhanced"] - v["Raw"]) / v["Raw"]) * 100.0
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compute Image Quality Assessment (IQA) Metrics for Borobudur Reliefs")
    parser.add_argument("--raw", "-r", type=str, required=True, help="Path to raw image")
    parser.add_argument("--enhanced", "-e", type=str, required=True, help="Path to CLAHE-enhanced image")
    
    args = parser.parse_args()
    results = evaluate_image_pair(args.raw, args.enhanced)
    
    print("\n" + "="*80)
    print(f"{'Image Quality Metric':<40} | {'Raw':<10} | {'Enhanced':<10} | {'Gain (%)':<10}")
    print("="*80)
    for m, vals in results.items():
        print(f"{m:<40} | {vals['Raw']:<10.4f} | {vals['Enhanced']:<10.4f} | +{vals['Gain_Pct']:<9.2f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("Run with --raw <raw_image_path> --enhanced <enhanced_image_path>")
