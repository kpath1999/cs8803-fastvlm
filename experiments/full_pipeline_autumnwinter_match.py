#!/usr/bin/env python3
"""
Full Pipeline Autumn-Winter Image Matching Script

This script implements the complete hierarchical matching pipeline:
1. Visual Similarity: FastVLM visual embeddings (Stage 3)
2. Semantic Similarity: Text-based keyword matching (Stage 3)
3. Depth Consistency: 3D spatial layout validation (Stage 4)
4. Geometric Verification: Keypoint matching with RANSAC (Stage 5)

Usage:
    # Basic usage - match autumn_0000 against first 30 winter frames
    python full_pipeline_autumnwinter_match.py --limit 30
    
    # Full search with specific autumn frame and visualization
    python full_pipeline_autumnwinter_match.py --autumn-idx 50 --verbose --visualize
    
    # Reverse: match winter frame against autumn dataset
    python full_pipeline_autumnwinter_match.py --winter-idx 15 --verbose --visualize
    
    # Full dataset search (slow, ~15-20 minutes)
    python full_pipeline_autumnwinter_match.py --autumn-idx 0 --visualize

Design Decisions:
-----------------
1. Hierarchical Filtering: Progresses from broad (embedding) to precise (geometric)
2. Multi-Modal Fusion: Combines visual, semantic, depth, and geometric cues
3. Weighted Scoring: 0.3*embedding + 0.2*semantic + 0.2*depth + 0.3*geometric
4. Invariant Classification: Uses depth consistency to distinguish permanent vs temporal
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
import cv2
import csv
import time

# Add parent directory to path to import pipeline components
sys.path.append(str(Path(__file__).parent.parent))

try:
    from cross_temporal_pipeline import FastVLMAnalyzer, DepthValidator, KeypointMatcher
except ImportError:
    print("Error: Could not import from cross_temporal_pipeline.py")
    sys.exit(1)


class FullPipelineMatcher:
    """
    Handles matching using all pipeline stages: visual, semantic, depth, and geometric.
    """
    
    def __init__(self, model_path: str, device: str = "mps"):
        """Initialize the full pipeline matcher."""
        print(f"Initializing Full Pipeline Matcher with model: {model_path}")
        try:
            self.analyzer = FastVLMAnalyzer(model_path=model_path, device=device)
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Failed to load model ({e}).")
            self.model_loaded = False
            self.analyzer = None
        
        # Initialize depth validator and keypoint matcher
        self.depth_validator = DepthValidator()
        self.keypoint_matcher = KeypointMatcher(method="orb")
        
        # Cache for embeddings
        self.embedding_cache = {}
    
    def get_visual_embedding(self, image_path: str) -> np.ndarray:
        """Compute visual embedding with caching."""
        if image_path in self.embedding_cache:
            return self.embedding_cache[image_path]
        
        if not self.model_loaded:
            return np.zeros(512)
        
        try:
            if not os.path.exists(image_path):
                return np.zeros(1)
            
            image = Image.open(image_path).convert("RGB")
            embedding = self.analyzer.get_vision_embedding(image)
            self.embedding_cache[image_path] = embedding
            return embedding
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return np.zeros(1)
    
    def get_semantic_keywords(self, frame_data: Dict) -> Set[str]:
        """Extract semantic keywords from detections."""
        detections = frame_data.get("detections", [])
        return {d["label"] for d in detections}
    
    def compute_visual_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        if emb1.shape != emb2.shape or len(emb1.shape) == 0:
            return 0.0
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def compute_semantic_similarity(self, keywords1: Set[str], keywords2: Set[str]) -> float:
        """Compute Jaccard similarity between keyword sets."""
        if not keywords1 and not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return float(intersection / union) if union > 0 else 0.0
    
    def compute_depth_consistency(
        self, 
        query_path: str, 
        candidate_path: str,
        depth_dir_query: Optional[Path] = None,
        depth_dir_candidate: Optional[Path] = None
    ) -> float:
        """Compute depth consistency between two frames."""
        if depth_dir_query is None or depth_dir_candidate is None:
            return 0.5  # Neutral score if depth not available
        
        try:
            # Find corresponding depth files
            query_depth_path = self._find_depth_file(query_path, depth_dir_query)
            candidate_depth_path = self._find_depth_file(candidate_path, depth_dir_candidate)
            
            if query_depth_path is None or candidate_depth_path is None:
                return 0.5
            
            # Load depth images
            query_depth = cv2.imread(str(query_depth_path), cv2.IMREAD_UNCHANGED)
            candidate_depth = cv2.imread(str(candidate_depth_path), cv2.IMREAD_UNCHANGED)
            
            if query_depth is None or candidate_depth is None:
                return 0.5
            
            # Compute statistics
            query_stats = self.depth_validator.compute_depth_statistics(query_depth)
            candidate_stats = self.depth_validator.compute_depth_statistics(candidate_depth)
            
            # Compare median depths (simple consistency check)
            if query_stats["median"] > 0 and candidate_stats["median"] > 0:
                depth_ratio = min(query_stats["median"], candidate_stats["median"]) / \
                             max(query_stats["median"], candidate_stats["median"])
                return float(depth_ratio)
            
            return 0.5
        except Exception as e:
            return 0.5
    
    def compute_geometric_confidence(
        self,
        query_path: str,
        query_bbox: List[float],
        candidate_path: str,
        candidate_bbox: List[float]
    ) -> Tuple[int, float]:
        """Compute geometric confidence using keypoint matching."""
        try:
            if not os.path.exists(query_path) or not os.path.exists(candidate_path):
                return 0, 0.0
            
            query_img = Image.open(query_path).convert("RGB")
            candidate_img = Image.open(candidate_path).convert("RGB")
            
            num_inliers, confidence = self.keypoint_matcher.match_regions(
                query_img, query_bbox,
                candidate_img, candidate_bbox
            )
            
            return num_inliers, confidence
        except Exception as e:
            return 0, 0.0
    
    def _find_depth_file(self, rgb_path: str, depth_dir: Path) -> Optional[Path]:
        """Find corresponding depth file for an RGB image."""
        rgb_filename = os.path.basename(rgb_path)
        depth_path = depth_dir / rgb_filename
        
        if depth_path.exists():
            return depth_path
        
        # Try with different extensions
        stem = rgb_filename.rsplit('.', 1)[0]
        for ext in ['.png', '.tiff', '.tif']:
            depth_path = depth_dir / f"{stem}{ext}"
            if depth_path.exists():
                return depth_path
        
        return None


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
    
    # Aggregate metrics
    avg_iou = np.mean(best_ious) if best_ious else 0.0
    avg_same_label_iou = np.mean(same_label_ious) if same_label_ious else 0.0
    max_iou = max(best_ious) if best_ious else 0.0
    good_matches = sum(1 for iou in best_ious if iou > 0.5)
    match_rate = good_matches / len(query_dets) if query_dets else 0.0
    
    count_diff = abs(len(query_dets) - len(match_dets))
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


def find_matches(
    matcher: FullPipelineMatcher,
    query_frame: Dict,
    candidate_frames: List[Dict],
    depth_dir_query: Optional[Path] = None,
    depth_dir_candidate: Optional[Path] = None,
    top_k: int = 5,
    verbose: bool = False
) -> Tuple[List[Dict], np.ndarray]:
    """
    Find top_k matches using full pipeline (all stages).
    
    Scoring: 0.3*visual + 0.2*semantic + 0.2*depth + 0.3*geometric
    """
    query_path = query_frame["image_path"]
    print(f"\nProcessing Query Frame: {query_frame['frame_id']}")
    print(f"  Path: {query_path}")
    
    # Stage 1: Get query features
    query_embedding = matcher.get_visual_embedding(query_path)
    if len(query_embedding) <= 1:
        print("Error: Could not extract embedding for query image.")
        return [], None
    
    query_keywords = matcher.get_semantic_keywords(query_frame)
    query_dets = query_frame.get("detections", [])
    print(f"  Keywords: {query_keywords}")
    print(f"  Detections: {len(query_dets)}")
    
    if verbose:
        print(f"\n  Query Embedding Statistics:")
        print(f"    Shape: {query_embedding.shape}")
        print(f"    Mean: {np.mean(query_embedding):.4f}")
        print(f"    Std: {np.std(query_embedding):.4f}")
    
    # Stage 2: Precompute candidate embeddings
    print(f"\nPrecomputing embeddings for {len(candidate_frames)} candidate frames...")
    candidate_embeddings = []
    for cand_frame in tqdm(candidate_frames, desc="Embedding", unit="frame"):
        emb = matcher.get_visual_embedding(cand_frame["image_path"])
        candidate_embeddings.append(emb)
    
    print(f"✓ Embeddings cached. Now computing multi-modal similarities...")
    
    results = []
    
    # Stage 3: Compute all similarity scores
    iterator = tqdm(zip(candidate_frames, candidate_embeddings),
                   total=len(candidate_frames),
                   desc="Multi-modal matching",
                   unit="frame")
    
    for idx, (cand_frame, cand_embedding) in enumerate(iterator):
        cand_path = cand_frame["image_path"]
        cand_dets = cand_frame.get("detections", [])
        
        if len(cand_embedding) <= 1:
            continue
        
        # Visual similarity (Stage 3)
        visual_score = matcher.compute_visual_similarity(query_embedding, cand_embedding)
        
        # Semantic similarity (Stage 3)
        cand_keywords = matcher.get_semantic_keywords(cand_frame)
        semantic_score = matcher.compute_semantic_similarity(query_keywords, cand_keywords)
        
        # Depth consistency (Stage 4)
        depth_score = matcher.compute_depth_consistency(
            query_path, cand_path,
            depth_dir_query, depth_dir_candidate
        )
        
        # Geometric verification (Stage 5) - only for top candidates by visual+semantic
        # To save time, we compute geometry only if initial scores are promising
        preliminary_score = 0.5 * visual_score + 0.5 * semantic_score
        
        if preliminary_score > 0.3 and len(query_dets) > 0 and len(cand_dets) > 0:
            # Find best bbox match for geometric verification
            best_geom_confidence = 0.0
            best_num_inliers = 0
            
            for q_det in query_dets[:3]:  # Check top 3 query detections
                q_bbox = q_det["bbox"]
                
                for c_det in cand_dets[:3]:  # Check top 3 candidate detections
                    c_bbox = c_det["bbox"]
                    
                    num_inliers, geom_conf = matcher.compute_geometric_confidence(
                        query_path, q_bbox,
                        cand_path, c_bbox
                    )
                    
                    if geom_conf > best_geom_confidence:
                        best_geom_confidence = geom_conf
                        best_num_inliers = num_inliers
            
            geometric_score = best_geom_confidence
            num_inliers = best_num_inliers
        else:
            geometric_score = 0.0
            num_inliers = 0
        
        # Final weighted score: 30% visual, 20% semantic, 20% depth, 30% geometric
        final_score = (
            0.3 * visual_score +
            0.2 * semantic_score +
            0.2 * depth_score +
            0.3 * geometric_score
        )
        
        # Classify as invariant if depth consistency is high
        is_invariant = depth_score > 0.7
        
        results.append({
            "frame_id": cand_frame["frame_id"],
            "image_path": cand_path,
            "embedding_similarity": visual_score,
            "semantic_similarity": semantic_score,
            "depth_consistency": depth_score,
            "geometric_confidence": geometric_score,
            "num_inliers": num_inliers,
            "final_confidence": final_score,
            "is_invariant": is_invariant,
            "keywords": list(cand_keywords),
            "embedding": cand_embedding,
            "frame_data": cand_frame
        })
    
    # Sort and get top K
    results.sort(key=lambda x: x["final_confidence"], reverse=True)
    
    # Evaluate match quality
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
            print(f"  Final Confidence: {result['final_confidence']:.3f}")
            print(f"  Visual: {result['embedding_similarity']:.3f}, Semantic: {result['semantic_similarity']:.3f}")
            print(f"  Depth: {result['depth_consistency']:.3f}, Geometric: {result['geometric_confidence']:.3f}")
            print(f"  Inliers: {result['num_inliers']}, Invariant: {result['is_invariant']}")
            print(f"  Quality Score: {match_quality['overall_quality_score']:.3f}")
    
    return results[:top_k], query_embedding


def create_visualization(
    query_frame: Dict,
    query_embedding: np.ndarray,
    matches: List[Dict],
    output_dir: str = "visualizations"
) -> None:
    """Create individual visualizations for each match with all stage scores."""
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
        
        # Multi-stage scores
        score_text = f"Final: {match['final_confidence']:.3f} | "
        score_text += f"Visual: {match['embedding_similarity']:.3f} | "
        score_text += f"Semantic: {match['semantic_similarity']:.3f} | "
        score_text += f"Depth: {match['depth_consistency']:.3f} | "
        score_text += f"Geometric: {match['geometric_confidence']:.3f} (Inliers: {match['num_inliers']})"
        draw.text((20, 40), score_text, fill='black', font=small_font)
        
        # Classification
        inv_text = f"Classification: {'INVARIANT' if match['is_invariant'] else 'TEMPORAL'}"
        draw.text((20, 55), inv_text, 
                 fill='darkgreen' if match['is_invariant'] else 'darkorange', 
                 font=small_font)
        
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
                
                m_label = m_det["label"]
                m_bbox = m_det["bbox"]
                
                iou = compute_iou(q_bbox, m_bbox)
                effective_iou = iou * 1.5 if q_label == m_label else iou
                
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
            
            # Color based on invariant classification and match quality
            if match['is_invariant'] and same_label and iou > 0.5:
                color = 'green'  # Invariant, good match
            elif match['is_invariant']:
                color = 'yellow'  # Invariant, lower quality
            elif same_label and iou > 0.5:
                color = 'orange'  # Temporal, good match
            else:
                color = 'red'  # Temporal, lower quality
            
            # Draw boxes and lines
            x1, y1, x2, y2 = q_det["bbox"]
            draw.rectangle([x1, y1 + 70, x2, y2 + 70], outline=color, width=3)
            
            x1, y1, x2, y2 = m_det["bbox"]
            draw.rectangle([x1 + q_width + 40, y1 + 70, x2 + q_width + 40, y2 + 70],
                          outline=color, width=3)
            
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
            ('green', 'Invariant (high quality)'),
            ('yellow', 'Invariant (low quality)'),
            ('orange', 'Temporal (matched)'),
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
                                   f"full_pipeline_match_pair_{query_id}_rank{i+1}_{match['frame_id']}.png")
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
                "final_confidence", "visual_score", "semantic_score", 
                "depth_score", "geometric_score", "num_inliers", "is_invariant",
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
                f"{match['final_confidence']:.4f}",
                f"{match['embedding_similarity']:.4f}",
                f"{match['semantic_similarity']:.4f}",
                f"{match['depth_consistency']:.4f}",
                f"{match['geometric_confidence']:.4f}",
                match['num_inliers'],
                match['is_invariant'],
                f"{qm.get('label_jaccard', 0.0):.4f}",
                f"{qm.get('avg_iou', 0.0):.4f}",
                f"{qm.get('match_rate', 0.0):.4f}",
                f"{qm.get('overall_quality_score', 0.0):.4f}"
            ]
            writer.writerow(row)
    
    print(f"  ✓ Logged results to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Full pipeline matching across seasons (bidirectional).")
    parser.add_argument("--detections", type=str, default="detections.json", 
                       help="Path to detections.json")
    parser.add_argument("--model-path", type=str, default="../checkpoints/llava-fastvithd_0.5b_stage2",
                       help="Path to FastVLM model")
    
    # Query selection
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--autumn-idx", type=int, help="Index of autumn frame to use as query")
    query_group.add_argument("--winter-idx", type=int, help="Index of winter frame to use as query")
    
    # Depth directories
    parser.add_argument("--winter-depth", type=str, 
                       default="/Volumes/KAUSAR/rover_dataset/2024-01-13/realsense_D435i/depth",
                       help="Path to winter depth directory")
    parser.add_argument("--autumn-depth", type=str,
                       default="/Volumes/KAUSAR/rover_dataset/2024-04-11/realsense_D435i/depth",
                       help="Path to autumn depth directory")
    
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to retrieve")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit number of candidate frames (for testing)")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cuda, cpu)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed calculations")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--log-file", type=str, default="full_pipeline_results.csv", help="Path to CSV log file")
    
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
        depth_dir_query = Path(args.autumn_depth) if os.path.exists(args.autumn_depth) else None
        depth_dir_candidate = Path(args.winter_depth) if os.path.exists(args.winter_depth) else None
    else:
        if args.winter_idx >= len(winter_frames):
            print(f"Error: Winter index {args.winter_idx} out of range.")
            return
        
        query_frame = winter_frames[args.winter_idx]
        candidate_frames = autumn_frames
        query_season = "winter"
        candidate_season = "autumn"
        depth_dir_query = Path(args.winter_depth) if os.path.exists(args.winter_depth) else None
        depth_dir_candidate = Path(args.autumn_depth) if os.path.exists(args.autumn_depth) else None
    
    if args.limit:
        candidate_frames = candidate_frames[:args.limit]
        print(f"Limiting search to first {args.limit} {candidate_season} frames.")
    
    # Initialize matcher
    matcher = FullPipelineMatcher(model_path=args.model_path, device=args.device)
    
    # Run matching
    top_matches, query_embedding = find_matches(
        matcher, query_frame, candidate_frames,
        depth_dir_query, depth_dir_candidate,
        top_k=args.top_k, verbose=args.verbose
    )
    
    # Display results
    print(f"\n{'='*60}")
    print(f"FULL PIPELINE RESULTS FOR {query_frame['frame_id']} ({query_season.upper()})")
    print(f"Searching in {candidate_season.upper()} dataset")
    print(f"{'='*60}")
    print(f"Query Keywords: {matcher.get_semantic_keywords(query_frame)}")
    print("-" * 60)
    
    for i, match in enumerate(top_matches):
        print(f"\nRank {i+1}: {match['frame_id']}")
        print(f"  Final Confidence: {match['final_confidence']:.4f}")
        print(f"    └─ Visual: {match['embedding_similarity']:.4f}")
        print(f"    └─ Semantic: {match['semantic_similarity']:.4f}")
        print(f"    └─ Depth: {match['depth_consistency']:.4f}")
        print(f"    └─ Geometric: {match['geometric_confidence']:.4f} ({match['num_inliers']} inliers)")
        print(f"  Classification: {'INVARIANT' if match['is_invariant'] else 'TEMPORAL'}")
        print(f"  Keywords: {match['keywords']}")
        
        if "quality_metrics" in match:
            qm = match["quality_metrics"]
            print(f"  Quality Score: {qm['overall_quality_score']:.4f} "
                  f"(IoU: {qm['avg_iou']:.3f}, Match Rate: {qm['match_rate']:.1%})")
        
        print("-" * 60)
    
    # Create visualizations
    if args.visualize and query_embedding is not None:
        create_visualization(query_frame, query_embedding, top_matches)
    
    # Log results
    if top_matches:
        log_results(top_matches, query_frame['frame_id'], args.log_file)
    
    # Summary statistics
    if top_matches:
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        
        avg_final = np.mean([m["final_confidence"] for m in top_matches])
        avg_visual = np.mean([m["embedding_similarity"] for m in top_matches])
        avg_semantic = np.mean([m["semantic_similarity"] for m in top_matches])
        avg_depth = np.mean([m["depth_consistency"] for m in top_matches])
        avg_geometric = np.mean([m["geometric_confidence"] for m in top_matches])
        avg_inliers = np.mean([m["num_inliers"] for m in top_matches])
        
        num_invariant = sum(1 for m in top_matches if m["is_invariant"])
        
        print(f"Average Scores:")
        print(f"  Final Confidence: {avg_final:.3f}")
        print(f"  Visual Similarity: {avg_visual:.3f}")
        print(f"  Semantic Similarity: {avg_semantic:.3f}")
        print(f"  Depth Consistency: {avg_depth:.3f}")
        print(f"  Geometric Confidence: {avg_geometric:.3f}")
        print(f"  Average Inliers: {avg_inliers:.1f}")
        print(f"\nClassification:")
        print(f"  Invariant: {num_invariant}/{len(top_matches)}")
        print(f"  Temporal: {len(top_matches) - num_invariant}/{len(top_matches)}")


if __name__ == "__main__":
    main()
