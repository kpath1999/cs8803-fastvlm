#!/usr/bin/env python3
"""
Autumn-Winter Image Matching Script (Baseline B - Geometric Only)

This script implements a geometric-only matching approach using keypoint matching.
It leverages "Baseline B" principles:
1. Geometric Verification: ORB/SIFT keypoint matching within bounding boxes (Stage 5 only)
2. NO visual embeddings, NO semantic similarity, NO depth validation

Usage:
    # Basic usage - match autumn_0000 against first 30 winter frames
    python baseline_b_autumnwinter_match.py --limit 30
    
    # Full search with specific autumn frame and visualization
    python baseline_b_autumnwinter_match.py --autumn-idx 50 --verbose --visualize
    
    # Reverse: match winter frame against autumn dataset
    python baseline_b_autumnwinter_match.py --winter-idx 15 --verbose --visualize
    
    # Full dataset search (slow, ~20-30 minutes due to keypoint extraction)
    python baseline_b_autumnwinter_match.py --autumn-idx 0 --visualize

Design Decisions & Justifications:
----------------------------------
1. Pure Geometric Matching:
   - Decision: Use ONLY keypoint matching (ORB/SIFT) with RANSAC verification.
   - Justification: Tests whether low-level geometric features alone can handle 
     seasonal appearance changes without high-level semantic guidance.

2. Bounding Box Regions:
   - Decision: Extract and match keypoints only within detected bounding boxes.
   - Justification: Focuses computation on relevant regions and enables comparison
     with detections from other baselines.

3. RANSAC Filtering:
   - Decision: Use homography estimation with RANSAC to filter outliers.
   - Justification: Ensures geometric consistency and rejects spurious matches.

4. No Semantic Information:
   - Decision: Ignore detection labels and semantic content.
   - Justification: Isolates the contribution of pure geometry for research comparison.
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Optional
import csv
import time

# Add parent directory to path to import pipeline components
sys.path.append(str(Path(__file__).parent.parent))

try:
    from cross_temporal_pipeline import KeypointMatcher
except ImportError:
    print("Error: Could not import KeypointMatcher. Make sure cross_temporal_pipeline.py is in the parent directory.")
    sys.exit(1)


class GeometricMatcher:
    """
    Handles matching using ONLY geometric keypoint features (ORB/SIFT).
    """
    
    def __init__(self, method: str = "orb"):
        """Initialize the geometric matcher."""
        print(f"Initializing GeometricMatcher with method: {method}")
        try:
            self.keypoint_matcher = KeypointMatcher(method=method)
            self.method = method
            print(f"✓ Baseline B initialized (geometric-only)")
        except Exception as e:
            print(f"Error: Failed to initialize KeypointMatcher ({e})")
            sys.exit(1)
    
    def compute_geometric_score(
        self,
        query_img: Image.Image,
        query_det: Dict,
        candidate_img: Image.Image,
        candidate_det: Dict
    ) -> Tuple[int, float]:
        """
        Compute geometric matching score between two detections.
        
        Returns:
            (num_inliers, geometric_confidence)
        """
        try:
            num_inliers, confidence = self.keypoint_matcher.match_regions(
                img1=query_img,
                bbox1=query_det["bbox"],
                img2=candidate_img,
                bbox2=candidate_det["bbox"]
            )
            return num_inliers, confidence
        except Exception as e:
            return 0, 0.0



def load_data(json_path: str) -> Dict:
    """Load detections JSON file."""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute IoU between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)
    
    if x2_int <= x1_int or y2_int <= y1_int:
        return 0.0
    
    intersection = (x2_int - x1_int) * (y2_int - y1_int)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return float(intersection / union) if union > 0 else 0.0


def evaluate_match_quality(
    query_frame: Dict,
    match_frame: Dict,
    verbose: bool = False
) -> Dict:
    """Evaluate match quality using bounding box analysis."""
    query_dets = query_frame.get("detections", [])
    match_dets = match_frame.get("detections", [])
    
    # Label overlap
    query_labels = set(d["label"] for d in query_dets)
    match_labels = set(d["label"] for d in match_dets)
    
    label_jaccard = len(query_labels & match_labels) / len(query_labels | match_labels) \
                    if (query_labels or match_labels) else 0.0
    
    # Spatial layout similarity
    best_ious = []
    bbox_matches = []
    
    for q_det in query_dets:
        q_bbox = q_det["bbox"]
        q_label = q_det["label"]
        
        best_iou = 0.0
        best_match = None
        
        for m_det in match_dets:
            m_bbox = m_det["bbox"]
            iou = compute_iou(q_bbox, m_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_match = m_det
        
        best_ious.append(best_iou)
        
        if best_match:
            bbox_matches.append({
                "query_label": q_label,
                "match_label": best_match["label"],
                "iou": best_iou,
                "same_label": q_label == best_match["label"],
                "query_bbox": q_bbox,
                "match_bbox": best_match["bbox"]
            })
    
    # Aggregate metrics
    avg_iou = np.mean(best_ious) if best_ious else 0.0
    max_iou = max(best_ious) if best_ious else 0.0
    good_matches = sum(1 for iou in best_ious if iou > 0.5)
    match_rate = good_matches / len(query_dets) if query_dets else 0.0
    
    count_ratio = min(len(query_dets), len(match_dets)) / max(len(query_dets), len(match_dets)) \
                  if max(len(query_dets), len(match_dets)) > 0 else 0.0
    
    overall_score = (
        0.3 * label_jaccard +
        0.4 * avg_iou +
        0.2 * match_rate +
        0.1 * count_ratio
    )
    
    return {
        "label_jaccard": float(label_jaccard),
        "avg_iou": float(avg_iou),
        "max_iou": float(max_iou),
        "match_rate": float(match_rate),
        "good_matches": int(good_matches),
        "query_detection_count": len(query_dets),
        "match_detection_count": len(match_dets),
        "count_ratio": float(count_ratio),
        "overall_quality_score": float(overall_score),
        "bbox_matches": bbox_matches
    }


def find_matches(
    matcher: GeometricMatcher,
    query_frame: Dict,
    candidate_frames: List[Dict],
    top_k: int = 5,
    min_inliers: int = 4,
    verbose: bool = False
) -> List[Dict]:
    """
    Find top_k matches using ONLY geometric keypoint matching.
    
    Scoring: Based purely on geometric confidence and number of RANSAC inliers
    """
    query_path = query_frame["image_path"]
    query_dets = query_frame.get("detections", [])
    
    print(f"\nProcessing Query Frame: {query_frame['frame_id']}")
    print(f"  Path: {query_path}")
    print(f"  Detections: {len(query_dets)}")
    
    if not os.path.exists(query_path):
        print("Error: Query image not found.")
        return []
    
    query_img = Image.open(query_path).convert("RGB")
    
    # Compute geometric matches for each candidate
    print(f"\nComputing geometric matches for {len(candidate_frames)} candidate frames...")
    
    results = []
    
    iterator = tqdm(candidate_frames, desc="Geometric matching", unit="frame")
    
    for cand_frame in iterator:
        cand_path = cand_frame["image_path"]
        cand_dets = cand_frame.get("detections", [])
        
        if not os.path.exists(cand_path) or len(cand_dets) == 0 or len(query_dets) == 0:
            continue
        
        cand_img = Image.open(cand_path).convert("RGB")
        
        # Find best geometric match across all detection pairs
        best_inliers = 0
        best_confidence = 0.0
        total_inliers = 0
        match_count = 0
        
        for q_det in query_dets:
            for c_det in cand_dets:
                num_inliers, confidence = matcher.compute_geometric_score(
                    query_img, q_det,
                    cand_img, c_det
                )
                
                if num_inliers >= min_inliers:
                    match_count += 1
                    total_inliers += num_inliers
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_inliers = num_inliers
        
        # Final score is the best geometric confidence found
        final_score = best_confidence
        
        # Average inliers across valid matches
        avg_inliers = total_inliers / match_count if match_count > 0 else 0
        
        results.append({
            "frame_id": cand_frame["frame_id"],
            "image_path": cand_path,
            "geometric_confidence": best_confidence,
            "num_inliers": best_inliers,
            "avg_inliers": avg_inliers,
            "detection_matches": match_count,
            "final_score": final_score,
            "frame_data": cand_frame
        })
    
    # Sort by final score (geometric confidence)
    results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Evaluate match quality for top K
    print(f"\n{'='*60}")
    print("EVALUATING MATCH QUALITY (BBox Analysis)")
    print(f"{'='*60}")
    
    for i, result in enumerate(results[:top_k]):
        match_quality = evaluate_match_quality(
            query_frame,
            result["frame_data"],
            verbose=verbose and i < 3
        )
        result["quality_metrics"] = match_quality
        
        if not verbose or i >= 3:
            print(f"\nRank {i+1}: {result['frame_id']}")
            print(f"  Geometric Confidence: {result['geometric_confidence']:.3f}")
            print(f"  Max Inliers: {result['num_inliers']}, Avg: {result['avg_inliers']:.1f}")
            print(f"  Detection Matches: {result['detection_matches']}")
            print(f"  Quality Score: {match_quality['overall_quality_score']:.3f}")
    
    return results[:top_k]


def create_visualization(
    query_frame: Dict,
    matches: List[Dict],
    output_dir: str = "visualizations"
) -> None:
    """Create individual visualizations for each match with geometric scores."""
    os.makedirs(output_dir, exist_ok=True)
    
    query_path = query_frame["image_path"]
    query_id = query_frame["frame_id"]
    
    if not os.path.exists(query_path):
        print(f"Warning: Query image not found at {query_path}")
        return
    
    query_img = Image.open(query_path).convert("RGB")
    query_dets = query_frame.get("detections", [])
    
    for i, match in enumerate(matches[:5]):
        match_path = match["image_path"]
        
        if not os.path.exists(match_path):
            print(f"Warning: Match image not found at {match_path}")
            continue
        
        match_img = Image.open(match_path).convert("RGB")
        match_dets = match["frame_data"].get("detections", [])
        
        # Create canvas
        q_width, q_height = query_img.size
        m_width, m_height = match_img.size
        
        canvas_width = q_width + m_width + 40
        canvas_height = max(q_height, m_height) + 140
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        canvas.paste(query_img, (0, 70))
        canvas.paste(match_img, (q_width + 40, 70))
        
        draw = ImageDraw.Draw(canvas)
        
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
            title_font = ImageFont.truetype("Arial.ttf", 18)
            small_font = ImageFont.truetype("Arial.ttf", 11)
        except:
            font = ImageFont.load_default()
            title_font = font
            small_font = font
        
        # Titles
        draw.text((q_width // 2 - 60, 10), f"Query: {query_id}", fill='black', font=title_font)
        draw.text((q_width + 40 + m_width // 2 - 60, 10),
                 f"Match {i+1}: {match['frame_id']}", fill='black', font=title_font)
        
        # Geometric scores
        score_text = f"Geometric: {match['geometric_confidence']:.3f} | "
        score_text += f"Inliers: {match['num_inliers']} (Avg: {match['avg_inliers']:.1f}) | "
        score_text += f"Det Matches: {match['detection_matches']}"
        draw.text((20, 40), score_text, fill='black', font=small_font)
        
        # Match bounding boxes
        matches_found = []
        used_match_indices = set()
        
        for q_det in query_dets:
            q_label = q_det["label"]
            q_bbox = q_det["bbox"]
            
            best_iou = 0.0
            best_match_idx = None
            
            for m_idx, m_det in enumerate(match_dets):
                if m_idx in used_match_indices:
                    continue
                
                m_bbox = m_det["bbox"]
                iou = compute_iou(q_bbox, m_bbox)
                
                # Boost score if same label
                effective_iou = iou * 1.5 if q_label == m_det["label"] else iou
                
                if effective_iou > best_iou and effective_iou > 0.1:
                    best_iou = effective_iou
                    best_match_idx = m_idx
            
            if best_match_idx is not None:
                matches_found.append({
                    "query": q_det,
                    "match": match_dets[best_match_idx],
                    "iou": compute_iou(q_bbox, match_dets[best_match_idx]["bbox"]),
                    "same_label": q_label == match_dets[best_match_idx]["label"]
                })
                used_match_indices.add(best_match_idx)
        
        # Draw matched boxes with connecting lines
        for bbox_match in matches_found:
            q_det = bbox_match["query"]
            m_det = bbox_match["match"]
            iou = bbox_match["iou"]
            same_label = bbox_match["same_label"]
            
            # Color based on match quality
            if same_label and iou > 0.5:
                color = 'green'  # Good geometric match
            elif iou > 0.5:
                color = 'yellow'  # Spatial match only
            elif same_label:
                color = 'orange'  # Semantic match only
            else:
                color = 'red'  # Weak match
            
            # Draw boxes
            x1, y1, x2, y2 = q_det["bbox"]
            draw.rectangle([x1, y1 + 70, x2, y2 + 70], outline=color, width=3)
            
            x1, y1, x2, y2 = m_det["bbox"]
            draw.rectangle([x1 + q_width + 40, y1 + 70, x2 + q_width + 40, y2 + 70],
                          outline=color, width=3)
            
            # Draw connecting line
            q_center_x = (q_det["bbox"][0] + q_det["bbox"][2]) / 2
            q_center_y = (q_det["bbox"][1] + q_det["bbox"][3]) / 2 + 70
            m_center_x = (m_det["bbox"][0] + m_det["bbox"][2]) / 2 + q_width + 40
            m_center_y = (m_det["bbox"][1] + m_det["bbox"][3]) / 2 + 70
            
            draw.line([q_center_x, q_center_y, m_center_x, m_center_y],
                     fill=color, width=2)
        
        # Draw unmatched boxes
        matched_q_indices = {id(m["query"]) for m in matches_found}
        for q_det in query_dets:
            if id(q_det) not in matched_q_indices:
                x1, y1, x2, y2 = q_det["bbox"]
                draw.rectangle([x1, y1 + 70, x2, y2 + 70], outline='blue', width=2)
        
        matched_m_indices = {id(m["match"]) for m in matches_found}
        for m_det in match_dets:
            if id(m_det) not in matched_m_indices:
                x1, y1, x2, y2 = m_det["bbox"]
                draw.rectangle([x1 + q_width + 40, y1 + 70, x2 + q_width + 40, y2 + 70],
                              outline='blue', width=2)
        
        # Legend
        legend_y = canvas_height - 60
        legend_items = [
            ('green', 'Good match (label+spatial)'),
            ('yellow', 'Spatial match only'),
            ('orange', 'Semantic match only'),
            ('blue', 'Unmatched')
        ]
        
        x_offset = 20
        for color, label in legend_items:
            draw.rectangle([x_offset, legend_y, x_offset + 15, legend_y + 15],
                          outline=color, fill=color, width=2)
            draw.text((x_offset + 20, legend_y + 2), label, fill='black', font=small_font)
            x_offset += 200
        
        # Stats
        stats_y = legend_y + 25
        stats_text = f"Matched: {len(matches_found)} | Query dets: {len(query_dets)} | Match dets: {len(match_dets)}"
        draw.text((20, stats_y), stats_text, fill='black', font=small_font)
        
        # Save
        output_path = os.path.join(output_dir,
                                   f"baseline_b_match_pair_{query_id}_rank{i+1}_{match['frame_id']}.png")
        canvas.save(output_path)
        print(f"  ✓ Saved visualization: {output_path}")


def log_results(results: List[Dict], query_id: str, log_file: str):
    """Log matching results to a CSV file."""
    file_exists = os.path.exists(log_file)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            header = [
                "timestamp", "query_id", "rank", "match_id",
                "geometric_confidence", "num_inliers", "avg_inliers", "detection_matches",
                "label_jaccard", "avg_iou", "match_rate", "overall_quality_score"
            ]
            writer.writerow(header)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        for i, match in enumerate(results):
            # Get quality metrics if available
            qm = match.get("quality_metrics", {})
            
            row = [
                timestamp,
                query_id,
                i + 1,  # Rank
                match["frame_id"],
                f"{match['geometric_confidence']:.4f}",
                match['num_inliers'],
                f"{match['avg_inliers']:.2f}",
                match['detection_matches'],
                f"{qm.get('label_jaccard', 0.0):.4f}",
                f"{qm.get('avg_iou', 0.0):.4f}",
                f"{qm.get('match_rate', 0.0):.4f}",
                f"{qm.get('overall_quality_score', 0.0):.4f}"
            ]
            writer.writerow(row)
    
    print(f"  ✓ Logged results to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Baseline B: Geometric-only matching across seasons (bidirectional).")
    parser.add_argument("--detections", type=str, default="detections.json",
                       help="Path to detections.json")
    
    # Query selection
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--autumn-idx", type=int, help="Index of autumn frame to use as query")
    query_group.add_argument("--winter-idx", type=int, help="Index of winter frame to use as query")
    
    parser.add_argument("--method", type=str, default="orb", choices=["orb", "sift"],
                       help="Keypoint detection method")
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to retrieve")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of candidate frames (for testing)")
    parser.add_argument("--min-inliers", type=int, default=4,
                       help="Minimum RANSAC inliers for valid match")
    parser.add_argument("--verbose", action="store_true", help="Show detailed calculations")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--log-file", type=str, default="baseline_b_results.csv", help="Path to CSV log file")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.detections):
        print(f"Error: {args.detections} not found.")
        return
    
    if args.autumn_idx is None and args.winter_idx is None:
        print("Error: Must specify either --autumn-idx or --winter-idx")
        return
    
    # Load data
    data = load_data(args.detections)
    autumn_frames = data.get("autumn", [])
    winter_frames = data.get("winter", [])
    
    if not autumn_frames or not winter_frames:
        print("Error: Missing autumn or winter data.")
        return
    
    # Determine query and candidates
    if args.autumn_idx is not None:
        if args.autumn_idx >= len(autumn_frames):
            print(f"Error: Autumn index {args.autumn_idx} out of range.")
            return
        
        query_frame = autumn_frames[args.autumn_idx]
        candidate_frames = winter_frames
        query_season = "autumn"
        candidate_season = "winter"
    else:
        if args.winter_idx >= len(winter_frames):
            print(f"Error: Winter index {args.winter_idx} out of range.")
            return
        
        query_frame = winter_frames[args.winter_idx]
        candidate_frames = autumn_frames
        query_season = "winter"
        candidate_season = "autumn"
    
    if args.limit:
        candidate_frames = candidate_frames[:args.limit]
        print(f"Limiting search to first {args.limit} {candidate_season} frames.")
    
    # Initialize matcher
    matcher = GeometricMatcher(method=args.method)
    
    # Run matching
    top_matches = find_matches(
        matcher, query_frame, candidate_frames,
        top_k=args.top_k, min_inliers=args.min_inliers,
        verbose=args.verbose
    )
    
    # Display results
    print(f"\n{'='*60}")
    print(f"BASELINE B RESULTS FOR {query_frame['frame_id']} ({query_season.upper()})")
    print(f"Searching in {candidate_season.upper()} dataset")
    print(f"{'='*60}")
    print(f"Method: {args.method.upper()}")
    print("-" * 60)
    
    for i, match in enumerate(top_matches):
        print(f"\nRank {i+1}: {match['frame_id']}")
        print(f"  Geometric Confidence: {match['geometric_confidence']:.4f}")
        print(f"  RANSAC Inliers: {match['num_inliers']} (Avg: {match['avg_inliers']:.1f})")
        print(f"  Detection Matches: {match['detection_matches']}")
        
        if "quality_metrics" in match:
            qm = match["quality_metrics"]
            print(f"  Quality Score: {qm['overall_quality_score']:.4f} "
                  f"(IoU: {qm['avg_iou']:.3f}, Match Rate: {qm['match_rate']:.1%})")
        
        print("-" * 60)
    
    # Create visualizations
    if args.visualize:
        create_visualization(query_frame, top_matches)
    
    # Log results
    if top_matches:
        log_results(top_matches, query_frame['frame_id'], args.log_file)
    
    # Summary statistics
    if top_matches:
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        
        avg_confidence = np.mean([m["geometric_confidence"] for m in top_matches])
        avg_inliers = np.mean([m["num_inliers"] for m in top_matches])
        avg_det_matches = np.mean([m["detection_matches"] for m in top_matches])
        
        high_conf_matches = sum(1 for m in top_matches if m["geometric_confidence"] > 0.7)
        
        print(f"Average Geometric Confidence: {avg_confidence:.3f}")
        print(f"Average RANSAC Inliers: {avg_inliers:.1f}")
        print(f"Average Detection Matches: {avg_det_matches:.1f}")
        print(f"High-Confidence Matches (>0.7): {high_conf_matches}/{len(top_matches)}")


if __name__ == "__main__":
    main()
