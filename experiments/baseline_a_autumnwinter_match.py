#!/usr/bin/env python3
"""
Autumn-Winter Image Matching Script (Baseline A)

This script implements a retrieval-based approach to find the most similar winter images
for a given autumn query image. It leverages "Baseline A" principles:
1. Visual Similarity: Cosine distance on FastVLM visual embeddings.
2. Semantic Similarity: Text-based keyword matching on object descriptions (labels).

Usage:
    # Basic usage - match autumn_0000 against first 30 winter frames
    python autumn2winter_match.py --limit 30
    
    # Full search with specific autumn frame and visualization
    python autumn2winter_match.py --autumn-idx 50 --verbose --visualize
    
    # Reverse: match winter frame against autumn dataset
    python autumn2winter_match.py --winter-idx 15 --verbose --visualize
    
    # Custom parameters
    python autumn2winter_match.py --autumn-idx 100 --top-k 10 --limit 50 --verbose --visualize
    
    # Full dataset search (slow, ~10-15 minutes)
    python autumn2winter_match.py --autumn-idx 0 --visualize

Design Decisions & Justifications:
----------------------------------
1. Global Image Embeddings:
   - Decision: We use the embedding of the entire image rather than individual object crops.
   - Justification: For image-to-image retrieval, the global scene context is crucial. 
     Matching individual objects (like "tree" to "tree") is ambiguous without spatial constraints.
     Global embeddings capture the overall scene composition and atmosphere efficiently.

2. Labels as Semantic Keywords:
   - Decision: We use the pre-computed detection labels from detections.json as semantic keywords.
   - Justification: Generating dense captions for every image on-the-fly is computationally expensive.
     The detection labels ("house", "tree", "path") provide a strong, structured semantic summary 
     of the scene content, satisfying the "text-based keyword matching" requirement efficiently.

3. Hybrid Scoring Metric:
   - Decision: Final score = 0.7 * Visual_Sim + 0.3 * Semantic_Sim.
   - Justification: Visual similarity is the primary signal for "looking the same". 
     Semantic similarity acts as a regularizer to ensure we match scenes with similar content 
     (e.g., ensuring both have a "house") even if the season changes the visual appearance significantly.

4. FastVLM Integration:
   - Decision: Reuse the FastVLMAnalyzer from the pipeline.
   - Justification: Ensures consistency with the rest of the project and leverages the 
     specific visual encoder trained/finetuned for this task.
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add parent directory to path to import pipeline components
sys.path.append(str(Path(__file__).parent.parent))

try:
    from cross_temporal_pipeline import FastVLMAnalyzer
except ImportError:
    print("Error: Could not import FastVLMAnalyzer. Make sure cross_temporal_pipeline.py is in the parent directory.")
    sys.exit(1)


class ImageMatcher:
    """
    Handles the matching of images using Visual and Semantic features.
    """
    
    def __init__(self, model_path: str, device: str = "mps"):
        """
        Initialize the matcher with the FastVLM model.
        """
        print(f"Initializing ImageMatcher with model: {model_path}")
        try:
            self.analyzer = FastVLMAnalyzer(model_path=model_path, device=device)
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Failed to load model ({e}). Visual features will be random/zeros.")
            self.model_loaded = False
            self.analyzer = None
        
        # Cache for embeddings to avoid recomputation
        self.embedding_cache = {}

    def get_visual_embedding(self, image_path: str) -> np.ndarray:
        """
        Compute the global visual embedding for an image using FastVLM.
        Uses caching to avoid recomputing embeddings.
        """
        # Check cache first
        if image_path in self.embedding_cache:
            return self.embedding_cache[image_path]
        
        if not self.model_loaded:
            return np.zeros(512)  # Placeholder if model fails
            
        try:
            if not os.path.exists(image_path):
                # Try to handle potential path issues (e.g. different mount points)
                # This is a simple fallback for the user's environment
                filename = os.path.basename(image_path)
                # You might add other search paths here if needed
                return np.zeros(1) # Return invalid to skip
                
            image = Image.open(image_path).convert("RGB")
            # We use the analyzer's method to get the embedding
            # Passing the whole image gets the global scene embedding
            embedding = self.analyzer.get_vision_embedding(image)
            
            # Cache the result
            self.embedding_cache[image_path] = embedding
            return embedding
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros(1)

    def get_semantic_keywords(self, frame_data: Dict) -> Set[str]:
        """
        Extract semantic keywords (labels) from the detection data.
        """
        detections = frame_data.get("detections", [])
        # Use a set to get unique labels present in the scene
        labels = {d["label"] for d in detections}
        return labels

    def compute_visual_similarity(self, emb1: np.ndarray, emb2: np.ndarray, verbose: bool = False) -> float:
        """
        Compute Cosine Similarity between two embeddings.
        """
        if emb1.shape != emb2.shape or len(emb1.shape) == 0:
            return 0.0
            
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        dot_product = np.dot(emb1, emb2)
        similarity = float(dot_product / (norm1 * norm2))
        
        if verbose:
            print(f"\n  Cosine Similarity Calculation:")
            print(f"    Embedding 1 shape: {emb1.shape}")
            print(f"    Embedding 2 shape: {emb2.shape}")
            print(f"    Norm 1 (||e1||): {norm1:.4f}")
            print(f"    Norm 2 (||e2||): {norm2:.4f}")
            print(f"    Dot product (e1·e2): {dot_product:.4f}")
            print(f"    Cosine similarity: {dot_product:.4f} / ({norm1:.4f} * {norm2:.4f}) = {similarity:.4f}")
            print(f"    First 10 values of embedding 1: {emb1[:10]}")
            print(f"    First 10 values of embedding 2: {emb2[:10]}")
            
        return similarity

    def compute_semantic_similarity(self, keywords1: Set[str], keywords2: Set[str]) -> float:
        """
        Compute Jaccard Similarity between two sets of keywords.
        """
        if not keywords1 and not keywords2:
            return 0.0
            
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        if union == 0:
            return 0.0
            
        return float(intersection / union)


def load_data(json_path: str) -> Dict:
    """Load the detections JSON file."""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Bounding boxes are in format [x1, y1, x2, y2].
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Compute intersection
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)
    
    if x2_int <= x1_int or y2_int <= y1_int:
        return 0.0
    
    intersection = (x2_int - x1_int) * (y2_int - y1_int)
    
    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return float(intersection / union) if union > 0 else 0.0


def compute_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Compute the center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def compute_bbox_area(bbox: List[float]) -> float:
    """Compute the area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return abs(x2 - x1) * abs(y2 - y1)


def evaluate_match_quality(
    query_frame: Dict,
    match_frame: Dict,
    verbose: bool = False
) -> Dict:
    """
    Evaluate the quality of a match by comparing bounding boxes.
    
    Metrics computed:
    1. Label Overlap: Jaccard similarity of detected labels
    2. Spatial Layout Similarity: How similar are bbox positions/sizes?
    3. Best IoU Matches: For each query bbox, find best matching bbox in candidate
    4. Overall Match Score: Combined metric
    
    Args:
        query_frame: Query frame with detections
        match_frame: Matched frame with detections
        verbose: Print detailed comparison
    
    Returns:
        Dictionary with evaluation metrics
    """
    query_dets = query_frame.get("detections", [])
    match_dets = match_frame.get("detections", [])
    
    # 1. Label Overlap (Semantic Consistency)
    query_labels = set(d["label"] for d in query_dets)
    match_labels = set(d["label"] for d in match_dets)
    
    if query_labels or match_labels:
        label_jaccard = len(query_labels & match_labels) / len(query_labels | match_labels)
    else:
        label_jaccard = 0.0
    
    # 2. Spatial Layout Similarity - Best IoU matching
    # For each query detection, find the best matching detection in match frame
    best_ious = []
    same_label_ious = []
    bbox_matches = []
    
    for q_det in query_dets:
        q_bbox = q_det["bbox"]
        q_label = q_det["label"]
        
        best_iou = 0.0
        best_match = None
        best_same_label_iou = 0.0
        
        for m_det in match_dets:
            m_bbox = m_det["bbox"]
            m_label = m_det["label"]
            
            iou = compute_iou(q_bbox, m_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_match = m_det
            
            # Track IoU for same-label matches
            if q_label == m_label and iou > best_same_label_iou:
                best_same_label_iou = iou
        
        best_ious.append(best_iou)
        
        if best_same_label_iou > 0:
            same_label_ious.append(best_same_label_iou)
        
        if best_match:
            bbox_matches.append({
                "query_label": q_label,
                "match_label": best_match["label"],
                "iou": best_iou,
                "same_label": q_label == best_match["label"],
                "query_bbox": q_bbox,
                "match_bbox": best_match["bbox"]
            })
    
    # 3. Compute aggregate metrics
    avg_iou = np.mean(best_ious) if best_ious else 0.0
    avg_same_label_iou = np.mean(same_label_ious) if same_label_ious else 0.0
    max_iou = max(best_ious) if best_ious else 0.0
    
    # Count how many detections have good matches (IoU > 0.5)
    good_matches = sum(1 for iou in best_ious if iou > 0.5)
    match_rate = good_matches / len(query_dets) if query_dets else 0.0
    
    # 4. Detection count similarity
    count_diff = abs(len(query_dets) - len(match_dets))
    count_ratio = min(len(query_dets), len(match_dets)) / max(len(query_dets), len(match_dets)) if max(len(query_dets), len(match_dets)) > 0 else 0.0
    
    # 5. Overall quality score (weighted combination)
    # - Label overlap: 30%
    # - Average IoU: 40%
    # - Match rate: 20%
    # - Count similarity: 10%
    overall_score = (
        0.3 * label_jaccard +
        0.4 * avg_iou +
        0.2 * match_rate +
        0.1 * count_ratio
    )
    
    metrics = {
        "label_jaccard": float(label_jaccard),
        "avg_iou": float(avg_iou),
        "avg_same_label_iou": float(avg_same_label_iou),
        "max_iou": float(max_iou),
        "match_rate": float(match_rate),
        "good_matches": int(good_matches),
        "query_detection_count": len(query_dets),
        "match_detection_count": len(match_dets),
        "count_difference": int(count_diff),
        "count_ratio": float(count_ratio),
        "overall_quality_score": float(overall_score),
        "bbox_matches": bbox_matches
    }
    
    if verbose:
        print(f"\n  Match Quality Evaluation:")
        print(f"    Label Jaccard: {label_jaccard:.3f}")
        print(f"    Avg IoU (all): {avg_iou:.3f}")
        print(f"    Avg IoU (same label): {avg_same_label_iou:.3f}")
        print(f"    Max IoU: {max_iou:.3f}")
        print(f"    Match rate (IoU>0.5): {match_rate:.1%} ({good_matches}/{len(query_dets)})")
        print(f"    Detection counts: Query={len(query_dets)}, Match={len(match_dets)}")
        print(f"    Overall Quality Score: {overall_score:.3f}")
        
        if bbox_matches:
            print(f"\n    Top 3 BBox Matches:")
            sorted_matches = sorted(bbox_matches, key=lambda x: x["iou"], reverse=True)[:3]
            for i, m in enumerate(sorted_matches):
                same_label_str = "✓" if m["same_label"] else "✗"
                print(f"      {i+1}. {m['query_label']} → {m['match_label']} (IoU: {m['iou']:.3f}) {same_label_str}")
    
    return metrics


def find_matches(
    matcher: ImageMatcher, 
    query_frame: Dict, 
    candidate_frames: List[Dict], 
    top_k: int = 5,
    visual_weight: float = 0.7,
    verbose: bool = False
) -> Tuple[List[Dict], np.ndarray]:
    """
    Find the top_k most similar candidate frames for the query frame.
    Works bidirectionally: autumn->winter or winter->autumn.
    """
    query_path = query_frame["image_path"]
    print(f"\nProcessing Query Frame: {query_frame['frame_id']}")
    print(f"  Path: {query_path}")
    
    # 1. Get Query Features
    query_embedding = matcher.get_visual_embedding(query_path)
    if len(query_embedding) <= 1:
        print("Error: Could not extract embedding for query image. Check path.")
        return [], None
        
    query_keywords = matcher.get_semantic_keywords(query_frame)
    print(f"  Keywords: {query_keywords}")
    
    if verbose:
        print(f"\n  Query Embedding Statistics:")
        print(f"    Shape: {query_embedding.shape}")
        print(f"    Mean: {np.mean(query_embedding):.4f}")
        print(f"    Std: {np.std(query_embedding):.4f}")
        print(f"    Min: {np.min(query_embedding):.4f}")
        print(f"    Max: {np.max(query_embedding):.4f}")
        print(f"    First 20 values: {query_embedding[:20]}")
    
    # 2. Precompute all candidate embeddings (MASSIVE SPEEDUP!)
    print(f"\nPrecomputing embeddings for {len(candidate_frames)} candidate frames...")
    candidate_embeddings = []
    for cand_frame in tqdm(candidate_frames, desc="Embedding", unit="frame"):
        emb = matcher.get_visual_embedding(cand_frame["image_path"])
        candidate_embeddings.append(emb)
    
    print(f"✓ Embeddings cached. Now computing similarities...")
    
    results = []
    
    # 3. Iterate through candidates (now much faster - just similarity computation)
    iterator = tqdm(zip(candidate_frames, candidate_embeddings), 
                   total=len(candidate_frames),
                   desc="Matching", 
                   unit="frame")
    
    for idx, (cand_frame, cand_embedding) in enumerate(iterator):
        cand_path = cand_frame["image_path"]
        
        # Visual Score (embedding already precomputed)
        if len(cand_embedding) <= 1:
            continue # Skip if image load failed
        
        # For verbose mode, show details for first match or high similarity matches    
        is_verbose_candidate = verbose and (idx < 3 or True)  # Will filter later
        visual_score = matcher.compute_visual_similarity(
            query_embedding, cand_embedding, 
            verbose=False  # We'll show this later for top matches
        )
        
        # Semantic Score
        cand_keywords = matcher.get_semantic_keywords(cand_frame)
        semantic_score = matcher.compute_semantic_similarity(query_keywords, cand_keywords)
        
        # Combined Score
        # We weight visual score higher as it captures the specific viewpoint/scene
        # Semantics are a coarse filter
        final_score = (visual_weight * visual_score) + ((1 - visual_weight) * semantic_score)
        
        results.append({
            "frame_id": cand_frame["frame_id"],
            "image_path": cand_path,
            "visual_score": visual_score,
            "semantic_score": semantic_score,
            "final_score": final_score,
            "keywords": list(cand_keywords),
            "embedding": cand_embedding,
            "frame_data": cand_frame  # Store for evaluation
        })
    
    # 4. Sort and return top K
    results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # 5. Compute evaluation metrics for top matches
    print(f"\n{'='*60}")
    print("EVALUATING MATCH QUALITY (BBox Analysis)")
    print(f"{'='*60}")
    
    for i, result in enumerate(results[:top_k]):
        match_quality = evaluate_match_quality(
            query_frame, 
            result["frame_data"],
            verbose=verbose and i < 3  # Show details for top 3 if verbose
        )
        result["quality_metrics"] = match_quality
        
        if not verbose or i >= 3:
            # Show summary even in non-verbose mode
            print(f"\nRank {i+1}: {result['frame_id']}")
            print(f"  Quality Score: {match_quality['overall_quality_score']:.3f}")
            print(f"  Label Overlap: {match_quality['label_jaccard']:.3f}")
            print(f"  Avg IoU: {match_quality['avg_iou']:.3f}")
            print(f"  Match Rate: {match_quality['match_rate']:.1%}")
    
    # Show detailed calculation for top matches if verbose
    if verbose and results:
        print(f"\n{'='*60}")
        print("DETAILED SIMILARITY CALCULATIONS FOR TOP MATCHES")
        print(f"{'='*60}")
        for i, result in enumerate(results[:3]):
            print(f"\nMatch #{i+1}: {result['frame_id']}")
            matcher.compute_visual_similarity(
                query_embedding, result['embedding'], 
                verbose=True
            )
    
    return results[:top_k], query_embedding


def create_side_by_side_visualization(
    query_frame: Dict,
    query_embedding: np.ndarray,
    matches: List[Dict],
    output_dir: str = "visualizations"
) -> None:
    """
    Create individual side-by-side visualizations for each match pair.
    Shows bounding boxes and connecting lines between matched objects.
    Similar to cross_temporal_pipeline.py visualization style.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    query_path = query_frame["image_path"]
    query_id = query_frame["frame_id"]
    
    # Check if query image exists
    if not os.path.exists(query_path):
        print(f"Warning: Query image not found at {query_path}")
        return
    
    query_img = Image.open(query_path).convert("RGB")
    query_dets = query_frame.get("detections", [])
    
    # Create one visualization per match
    for i, match in enumerate(matches[:5]):  # Top 5 matches
        match_path = match["image_path"]
        
        # Check if match image exists
        if not os.path.exists(match_path):
            print(f"Warning: Match image not found at {match_path}")
            continue
        
        match_img = Image.open(match_path).convert("RGB")
        match_dets = match["frame_data"].get("detections", [])
        
        # Create side-by-side canvas
        q_width, q_height = query_img.size
        m_width, m_height = match_img.size
        
        canvas_width = q_width + m_width + 40  # 40px gap
        canvas_height = max(q_height, m_height) + 120  # Space for title and legend
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        canvas.paste(query_img, (0, 60))
        canvas.paste(match_img, (q_width + 40, 60))
        
        draw = ImageDraw.Draw(canvas)
        
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
            title_font = ImageFont.truetype("Arial.ttf", 18)
            small_font = ImageFont.truetype("Arial.ttf", 11)
        except:
            font = ImageFont.load_default()
            title_font = font
            small_font = font
        
        # Draw titles
        draw.text((q_width // 2 - 60, 10), f"Query: {query_id}", fill='black', font=title_font)
        draw.text((q_width + 40 + m_width // 2 - 60, 10), 
                 f"Match {i+1}: {match['frame_id']}", fill='black', font=title_font)
        
        # Draw scores below titles
        score_text = f"Similarity: {match['final_score']:.3f} (V:{match['visual_score']:.3f}, S:{match['semantic_score']:.3f})"
        if "quality_metrics" in match:
            qm = match["quality_metrics"]
            score_text += f" | Quality: {qm['overall_quality_score']:.3f} (IoU:{qm['avg_iou']:.3f})"
        draw.text((canvas_width // 2 - 200, 35), score_text, fill='black', font=small_font)
        
        # Match bounding boxes by label and IoU
        # For each query detection, find best match in winter detections
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
                
                m_label = m_det["label"]
                m_bbox = m_det["bbox"]
                
                # Prioritize same-label matches
                iou = compute_iou(q_bbox, m_bbox)
                
                # Boost IoU if labels match
                effective_iou = iou * 1.5 if q_label == m_label else iou
                
                if effective_iou > best_iou and effective_iou > 0.1:  # Minimum threshold
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
        
        # Draw matched bounding boxes with connecting lines
        for bbox_match in matches_found:
            q_det = bbox_match["query"]
            m_det = bbox_match["match"]
            iou = bbox_match["iou"]
            same_label = bbox_match["same_label"]
            
            # Color based on match quality
            if same_label and iou > 0.5:
                color = 'green'  # Good match
            elif same_label and iou > 0.3:
                color = 'yellow'  # Decent match
            elif iou > 0.3:
                color = 'orange'  # Different label but spatial overlap
            else:
                color = 'red'  # Weak match
            
            # Draw query bbox
            x1, y1, x2, y2 = q_det["bbox"]
            draw.rectangle([x1, y1 + 60, x2, y2 + 60], outline=color, width=3)
            
            # Draw match bbox
            x1, y1, x2, y2 = m_det["bbox"]
            draw.rectangle([x1 + q_width + 40, y1 + 60, x2 + q_width + 40, y2 + 60], 
                          outline=color, width=3)
            
            # Draw connecting line
            q_center_x = (q_det["bbox"][0] + q_det["bbox"][2]) / 2
            q_center_y = (q_det["bbox"][1] + q_det["bbox"][3]) / 2 + 60
            m_center_x = (m_det["bbox"][0] + m_det["bbox"][2]) / 2 + q_width + 40
            m_center_y = (m_det["bbox"][1] + m_det["bbox"][3]) / 2 + 60
            
            draw.line([q_center_x, q_center_y, m_center_x, m_center_y], 
                     fill=color, width=2)
        
        # Draw unmatched bounding boxes in blue
        matched_q_indices = {id(m["query"]) for m in matches_found}
        for q_det in query_dets:
            if id(q_det) not in matched_q_indices:
                x1, y1, x2, y2 = q_det["bbox"]
                draw.rectangle([x1, y1 + 60, x2, y2 + 60], outline='blue', width=2)
        
        matched_m_indices = {id(m["match"]) for m in matches_found}
        for m_det in match_dets:
            if id(m_det) not in matched_m_indices:
                x1, y1, x2, y2 = m_det["bbox"]
                draw.rectangle([x1 + q_width + 40, y1 + 60, x2 + q_width + 40, y2 + 60], 
                              outline='blue', width=2)
        
        # Draw legend at bottom
        legend_y = canvas_height - 50
        legend_items = [
            ('green', 'Good match (same label, IoU>0.5)'),
            ('yellow', 'Decent match (same label, IoU>0.3)'),
            ('orange', 'Spatial overlap only'),
            ('blue', 'Unmatched')
        ]
        
        x_offset = 20
        for color, label in legend_items:
            draw.rectangle([x_offset, legend_y, x_offset + 15, legend_y + 15], 
                          outline=color, fill=color, width=2)
            draw.text((x_offset + 20, legend_y + 2), label, fill='black', font=small_font)
            x_offset += 200
        
        # Add statistics
        stats_y = legend_y + 25
        stats_text = (
            f"Matched objects: {len(matches_found)} | "
            f"Query detections: {len(query_dets)} | "
            f"Match detections: {len(match_dets)}"
        )
        draw.text((20, stats_y), stats_text, fill='black', font=small_font)
        
        # Save individual visualization
        output_path = os.path.join(output_dir, f"baseline_a_match_pair_{query_id}_rank{i+1}_{match['frame_id']}.png")
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
                "final_score", "visual_score", "semantic_score",
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
                f"{match['final_score']:.4f}",
                f"{match['visual_score']:.4f}",
                f"{match['semantic_score']:.4f}",
                f"{qm.get('label_jaccard', 0.0):.4f}",
                f"{qm.get('avg_iou', 0.0):.4f}",
                f"{qm.get('match_rate', 0.0):.4f}",
                f"{qm.get('overall_quality_score', 0.0):.4f}"
            ]
            writer.writerow(row)
    
    print(f"  ✓ Logged results to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Find matching frames across seasons (bidirectional).")
    parser.add_argument("--detections", type=str, default="detections.json", help="Path to detections.json")
    parser.add_argument("--model-path", type=str, default="../checkpoints/llava-fastvithd_0.5b_stage2", help="Path to FastVLM model")
    
    # Mutually exclusive group for query selection
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--autumn-idx", type=int, help="Index of the autumn frame to use as query")
    query_group.add_argument("--winter-idx", type=int, help="Index of the winter frame to use as query")
    
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to retrieve")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidate frames to search (for testing)")
    parser.add_argument("--device", type=str, default="mps", help="Device to use (mps, cuda, cpu)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed embedding calculations")
    parser.add_argument("--visualize", action="store_true", help="Create side-by-side visualizations")
    parser.add_argument("--log-file", type=str, default="baseline_a_results.csv", help="Path to CSV log file")
    
    args = parser.parse_args()
    
    # Check paths
    if not os.path.exists(args.detections):
        print(f"Error: {args.detections} not found.")
        return
    
    # Ensure at least one index is specified
    if args.autumn_idx is None and args.winter_idx is None:
        print("Error: Must specify either --autumn-idx or --winter-idx")
        return

    # Load Data
    data = load_data(args.detections)
    autumn_frames = data.get("autumn", [])
    winter_frames = data.get("winter", [])
    
    if not autumn_frames or not winter_frames:
        print("Error: Missing autumn or winter data in JSON.")
        return
    
    # Determine query and candidate frames based on which index is specified
    if args.autumn_idx is not None:
        # Query: autumn, Candidates: winter
        if args.autumn_idx >= len(autumn_frames):
            print(f"Error: Autumn index {args.autumn_idx} out of range (0-{len(autumn_frames)-1}).")
            return
        
        query_frame = autumn_frames[args.autumn_idx]
        candidate_frames = winter_frames
        query_season = "autumn"
        candidate_season = "winter"
    else:
        # Query: winter, Candidates: autumn
        if args.winter_idx >= len(winter_frames):
            print(f"Error: Winter index {args.winter_idx} out of range (0-{len(winter_frames)-1}).")
            return
        
        query_frame = winter_frames[args.winter_idx]
        candidate_frames = autumn_frames
        query_season = "winter"
        candidate_season = "autumn"
    
    if args.limit:
        candidate_frames = candidate_frames[:args.limit]
        print(f"Limiting search to first {args.limit} {candidate_season} frames.")

    # Initialize Matcher
    matcher = ImageMatcher(model_path=args.model_path, device=args.device)
    
    # Run Matching
    top_matches, query_embedding = find_matches(
        matcher, query_frame, candidate_frames, 
        top_k=args.top_k, verbose=args.verbose
    )
    
    # Display Results
    print(f"\n{'='*60}")
    print(f"MATCHING RESULTS FOR {query_frame['frame_id']} ({query_season.upper()})")
    print(f"Searching in {candidate_season.upper()} dataset")
    print(f"{'='*60}")
    print(f"Query Keywords: {matcher.get_semantic_keywords(query_frame)}")
    print("-" * 60)
    
    for i, match in enumerate(top_matches):
        print(f"Rank {i+1}: {match['frame_id']}")
        print(f"  Similarity Score: {match['final_score']:.4f} (Visual: {match['visual_score']:.4f}, Semantic: {match['semantic_score']:.4f})")
        
        if "quality_metrics" in match:
            qm = match["quality_metrics"]
            print(f"  Quality Score: {qm['overall_quality_score']:.4f} (Label: {qm['label_jaccard']:.3f}, IoU: {qm['avg_iou']:.3f}, Match Rate: {qm['match_rate']:.1%})")
            print(f"  Detections: Query={qm['query_detection_count']}, Match={qm['match_detection_count']}")
        
        print(f"  Keywords: {match['keywords']}")
        print(f"  Path: {match['image_path']}")
        print("-" * 60)
    
    # Create visualization if requested
    if args.visualize and query_embedding is not None:
        create_side_by_side_visualization(query_frame, query_embedding, top_matches)
    
    # Log results
    if top_matches:
        log_results(top_matches, query_frame['frame_id'], args.log_file)
    
    # Print summary statistics
    if top_matches and "quality_metrics" in top_matches[0]:
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        avg_quality = np.mean([m["quality_metrics"]["overall_quality_score"] for m in top_matches])
        avg_label_jaccard = np.mean([m["quality_metrics"]["label_jaccard"] for m in top_matches])
        avg_iou = np.mean([m["quality_metrics"]["avg_iou"] for m in top_matches])
        avg_match_rate = np.mean([m["quality_metrics"]["match_rate"] for m in top_matches])
        
        print(f"Average Quality Score: {avg_quality:.3f}")
        print(f"Average Label Overlap: {avg_label_jaccard:.3f}")
        print(f"Average IoU: {avg_iou:.3f}")
        print(f"Average Match Rate: {avg_match_rate:.1%}")
        
        # Identify best quality match
        best_quality_idx = np.argmax([m["quality_metrics"]["overall_quality_score"] for m in top_matches])
        best_similarity_idx = np.argmax([m["final_score"] for m in top_matches])
        
        print(f"\nBest Quality Match: Rank {best_quality_idx + 1} - {top_matches[best_quality_idx]['frame_id']}")
        print(f"Best Similarity Match: Rank {best_similarity_idx + 1} - {top_matches[best_similarity_idx]['frame_id']}")
        
        if best_quality_idx != best_similarity_idx:
            print(f"\n⚠️  Note: Best quality match differs from highest similarity match!")
            print(f"   This suggests visual similarity doesn't always correlate with spatial layout similarity.")

if __name__ == "__main__":
    main()
