#!/usr/bin/env python3
"""
Analyze OWL-ViT detection scores across the dataset.

This script runs OWL-ViT on a sample of images and reports:
1. Which queries get the highest scores
2. Score distribution per category
3. Recommended thresholds for each category
4. Visual examples of detections at different thresholds

Usage:
    python analyze_detections.py \
        --image-dir /Volumes/KAUSAR/rover_dataset/2024-01-13/realsense_D435i/rgb \
        --num-samples 20 \
        --output-dir detection_analysis
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from adaptive_threshold_config import CATEGORY_THRESHOLDS


def analyze_detections(
    image_dir: Path,
    queries: List[str],
    num_samples: int = 20,
    device: str = "mps"
) -> Dict:
    """
    Analyze detection scores across a sample of images.
    
    Returns:
        Dictionary with statistics per query category
    """
    print(f"Loading OWL-ViT model...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model.to(torch.device(device))
    
    # Get sample images
    image_paths = sorted([p for p in image_dir.glob("*.png") if not p.name.startswith("._")])
    if len(image_paths) > num_samples:
        # Sample evenly across the dataset
        step = len(image_paths) // num_samples
        image_paths = image_paths[::step][:num_samples]
    
    print(f"Analyzing {len(image_paths)} images with {len(queries)} queries...")
    
    # Collect scores per category
    category_scores = defaultdict(list)
    
    for img_path in image_paths:
        print(f"  Processing {img_path.name}...")
        image = Image.open(img_path).convert("RGB")
        
        inputs = processor(text=list(queries), images=image, return_tensors="pt")
        inputs = {k: v.to(torch.device(device)) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.0
        )[0]
        
        for score, label in zip(results["scores"], results["labels"]):
            label_text = queries[label.item()]
            category_scores[label_text].append(score.item())
    
    # Compute statistics
    stats = {}
    for category, scores in category_scores.items():
        if len(scores) == 0:
            continue
        stats[category] = {
            "count": len(scores),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "p95": float(np.percentile(scores, 95)),
            "p90": float(np.percentile(scores, 90)),
            "p75": float(np.percentile(scores, 75)),
            "recommended_threshold": float(np.percentile(scores, 75)),  # Use 75th percentile
        }
    
    return stats


def print_analysis(stats: Dict, output_dir: Path):
    """Print and save analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by max score descending
    sorted_categories = sorted(stats.items(), key=lambda x: x[1]["max"], reverse=True)
    
    print("\n" + "="*80)
    print("DETECTION SCORE ANALYSIS")
    print("="*80)
    print(f"{'Category':<15} {'Max':<7} {'P95':<7} {'P90':<7} {'Median':<7} {'Current':<7} {'Recommended':<7}")
    print("-"*80)
    
    threshold_updates = {}
    
    for category, stat in sorted_categories:
        current_thresh = CATEGORY_THRESHOLDS.get(category, 0.12)
        recommended = stat["p75"]  # Use 75th percentile
        
        # Flag if recommended is very different from current
        flag = ""
        if recommended > current_thresh + 0.03:
            flag = "⬆️ RAISE"
        elif recommended < current_thresh - 0.03:
            flag = "⬇️ LOWER"
        
        print(f"{category:<15} {stat['max']:.3f}   {stat['p95']:.3f}   "
              f"{stat['p90']:.3f}   {stat['median']:.3f}   "
              f"{current_thresh:.3f}   {recommended:.3f}   {flag}")
        
        threshold_updates[category] = recommended
    
    # Save to JSON
    with open(output_dir / "detection_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    with open(output_dir / "recommended_thresholds.json", "w") as f:
        json.dump(threshold_updates, f, indent=2)
    
    print(f"\n✓ Analysis saved to {output_dir}/")
    print(f"  - detection_statistics.json: Full statistics")
    print(f"  - recommended_thresholds.json: Suggested threshold values")


def main():
    parser = argparse.ArgumentParser(description="Analyze OWL-ViT detection scores")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of images to sample")
    parser.add_argument("--output-dir", type=str, default="detection_analysis", help="Output directory")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu/cuda/mps)")
    args = parser.parse_args()
    
    from cross_temporal_pipeline import DEFAULT_QUERIES
    
    stats = analyze_detections(
        image_dir=Path(args.image_dir),
        queries=DEFAULT_QUERIES,
        num_samples=args.num_samples,
        device=args.device
    )
    
    print_analysis(stats, Path(args.output_dir))


if __name__ == "__main__":
    main()
