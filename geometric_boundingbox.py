#!/usr/bin/env python3
"""
*** GEOMETRIC BOUNDING BOX MATCHING PIPELINE ***

This script implements a detection-based matching pipeline using classical computer
vision techniques. It compares detected objects between seasonal datasets (winter and
autumn) by matching bounding box crops with ORB/SIFT, then performs full-image matching
on top candidates.

PIPELINE STAGES:
================

Stage 1: LOAD FILTERED IMAGES
    - Parse artifacts/detections.json for winter and autumn images
    - Filter to first n autumn images and m winter images
    - Load image paths and detection metadata (labels, bboxes, scores)
    - Output: Filtered dataset with detections

Stage 2: DETECTION-LEVEL MATCHING
    - For each autumn image:
        - For each winter image:
            - Compare detections with matching labels
            - Crop regions using bboxes
            - Use ORB/SIFT to match cropped regions
            - Compute similarity score for each detection pair
            - Aggregate scores to create candidate score
    - Output: Candidate matches with aggregated detection scores

Stage 3: CANDIDATE SELECTION
    - Select top-K winter images with highest aggregated scores
    - Apply minimum threshold to filter weak candidates
    - Output: Top-K candidate matches per autumn image

Stage 4: FULL-IMAGE MATCHING
    - Load full images for candidates
    - Use ORB/SIFT on entire image (HxW)
    - Compute final similarity scores
    - Rank candidates by full-image score
    - Output: Best winter match per autumn image

Stage 5: VISUALIZATION
    - Create side-by-side comparison of best match
    - Draw matched detection bounding boxes
    - Annotate with detection-level scores
    - Display final full-image score prominently
    - Output: Visualizations saved to output directory

Stage 6: EXPORT RESULTS
    - Save ranked matches to JSON
    - Export summary statistics
    - Print results to console

USAGE:
======
# Basic usage with small dataset
python geometric_boundingbox.py \\
    --n-autumn 10 \\
    --m-winter 50 \\
    --method orb \\
    --output-dir geometric_bbox_results

# Using SIFT for more robust matching
python geometric_boundingbox.py \\
    --n-autumn 100 \\
    --m-winter 500 \\
    --method sift \\
    --top-k 10 \\
    --output-dir sift_bbox_results

KEY OPTIONS:
  --n-autumn N          Number of autumn images to process (required)
  --m-winter M          Number of winter images to process (required)
  --method METHOD       Classical CV method: orb, sift, or akaze (default: orb)
  --top-k K             Number of candidate matches to consider (default: 5)
  --detection-threshold Minimum detection match score (default: 0.3)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================


@dataclass
class DetectionMatch:
    """Represents a matched detection pair between autumn and winter."""
    
    autumn_label: str
    winter_label: str
    autumn_bbox: List[float]
    winter_bbox: List[float]
    similarity_score: float
    num_keypoints: int
    num_inliers: int
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class CandidateMatch:
    """Represents a candidate winter image match for an autumn image."""
    
    winter_frame_id: str
    winter_image_path: str
    detection_matches: List[DetectionMatch]
    aggregated_score: float
    num_matching_labels: int
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'winter_frame_id': self.winter_frame_id,
            'winter_image_path': self.winter_image_path,
            'detection_matches': [dm.to_dict() for dm in self.detection_matches],
            'aggregated_score': float(self.aggregated_score),
            'num_matching_labels': self.num_matching_labels,
        }


@dataclass
class FinalMatch:
    """Represents the final matched pair with full-image similarity."""
    
    autumn_frame_id: str
    autumn_image_path: str
    winter_frame_id: str
    winter_image_path: str
    detection_level_score: float
    full_image_score: float
    full_image_keypoints: int
    full_image_inliers: int
    visualization_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


# ===========================================================================
# STAGE 2 & 4: CLASSICAL CV MATCHING
# ===========================================================================


class ClassicalCVMatcher:
    """
    Performs classical CV matching using ORB, SIFT, or AKAZE.
    
    This class handles both detection-level matching (cropped bboxes)
    and full-image matching.
    """
    
    def __init__(self, method: str = "orb", n_features: int = 1000):
        """
        Initialize keypoint detector and matcher.
        
        Args:
            method: Keypoint method ('orb', 'sift', or 'akaze')
            n_features: Maximum number of features to detect
        """
        self.method = method.lower()
        self.n_features = n_features
        
        if self.method == "sift":
            # SIFT: Scale-invariant feature transform
            self.detector = cv2.SIFT_create(nfeatures=n_features)
            # FLANN-based matcher for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.method == "akaze":
            # AKAZE: Accelerated-KAZE
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:  # Default to ORB
            # ORB: Oriented FAST and Rotated BRIEF
            self.detector = cv2.ORB_create(nfeatures=n_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def match_regions(
        self,
        img1: Image.Image,
        bbox1: List[float],
        img2: Image.Image,
        bbox2: List[float],
        ratio_threshold: float = 0.75
    ) -> Tuple[int, int, float]:
        """
        Match keypoints between two bounding box regions.
        
        Args:
            img1, img2: PIL Images
            bbox1, bbox2: Bounding boxes [x1, y1, x2, y2]
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            Tuple of (total_matches, inliers, confidence_score)
        """
        # Convert PIL to OpenCV format
        cv_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        cv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
        
        # Crop regions
        x1, y1, x2, y2 = [int(v) for v in bbox1]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(cv_img1.shape[1], x2)
        y2 = min(cv_img1.shape[0], y2)
        crop1 = cv_img1[y1:y2, x1:x2]
        
        x1, y1, x2, y2 = [int(v) for v in bbox2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(cv_img2.shape[1], x2)
        y2 = min(cv_img2.shape[0], y2)
        crop2 = cv_img2[y1:y2, x1:x2]
        
        if crop1.size == 0 or crop2.size == 0:
            return 0, 0, 0.0
        
        # Detect keypoints and compute descriptors
        kp1, des1 = self.detector.detectAndCompute(crop1, None)
        kp2, des2 = self.detector.detectAndCompute(crop2, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return 0, 0, 0.0
        
        # Match descriptors
        try:
            matches = self.matcher.knnMatch(des1, des2, k=2)
        except cv2.error:
            return 0, 0, 0.0
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return 0, 0, 0.0
        
        # Extract matched keypoint locations
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # RANSAC to find inliers
        try:
            _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            if mask is None:
                return len(good_matches), 0, 0.0
            
            inliers = int(np.sum(mask))
            total = len(good_matches)
            confidence = inliers / total if total > 0 else 0.0
            
            return total, inliers, float(confidence)
        except cv2.error:
            return len(good_matches), 0, 0.0
    
    def match_full_images(
        self,
        img1: Image.Image,
        img2: Image.Image,
        ratio_threshold: float = 0.75
    ) -> Tuple[int, int, float]:
        """
        Match keypoints between two full images.
        
        Args:
            img1, img2: PIL Images
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            Tuple of (total_matches, inliers, confidence_score)
        """
        # Convert PIL to OpenCV format
        cv_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        cv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
        
        # Detect keypoints and compute descriptors
        kp1, des1 = self.detector.detectAndCompute(cv_img1, None)
        kp2, des2 = self.detector.detectAndCompute(cv_img2, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return 0, 0, 0.0
        
        # Match descriptors
        try:
            matches = self.matcher.knnMatch(des1, des2, k=2)
        except cv2.error:
            return 0, 0, 0.0
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return 0, 0, 0.0
        
        # Extract matched keypoint locations
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # RANSAC to find inliers
        try:
            _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            if mask is None:
                return len(good_matches), 0, 0.0
            
            inliers = int(np.sum(mask))
            total = len(good_matches)
            confidence = inliers / total if total > 0 else 0.0
            
            return total, inliers, float(confidence)
        except cv2.error:
            return len(good_matches), 0, 0.0


# ===========================================================================
# PIPELINE ORCHESTRATION
# ===========================================================================


class GeometricBoundingBoxPipeline:
    """
    Main pipeline orchestrator for detection-based frame matching.
    
    Coordinates all stages from loading detections through final visualization.
    """
    
    def __init__(
        self,
        detections_json_path: Path,
        output_dir: Path,
        n_autumn: int,
        m_winter: int,
        method: str = "orb",
        top_k: int = 5,
        detection_threshold: float = 0.3,
    ):
        """
        Initialize pipeline components.
        
        Args:
            detections_json_path: Path to detections.json
            output_dir: Directory for results
            n_autumn: Number of autumn images to process
            m_winter: Number of winter images to process
            method: Classical CV method (orb, sift, akaze)
            top_k: Number of candidate matches to consider
            detection_threshold: Minimum detection match score
        """
        self.detections_json_path = detections_json_path
        self.output_dir = output_dir
        self.n_autumn = n_autumn
        self.m_winter = m_winter
        self.method = method
        self.top_k = top_k
        self.detection_threshold = detection_threshold
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize matcher
        self.matcher = ClassicalCVMatcher(method=method)
        
        print(f"\n{'='*70}")
        print(f"GEOMETRIC BOUNDING BOX MATCHING PIPELINE")
        print(f"{'='*70}")
        print(f"  Method: {method.upper()}")
        print(f"  Autumn images: {n_autumn}")
        print(f"  Winter images: {m_winter}")
        print(f"  Top-K candidates: {top_k}")
        print(f"  Detection threshold: {detection_threshold}")
        print(f"  Output directory: {output_dir}")
        print(f"{'='*70}\n")
    
    def load_detections(self) -> Tuple[List[Dict], List[Dict]]:
        """
        STAGE 1: Load and filter detections from JSON.
        
        Returns:
            Tuple of (autumn_data, winter_data)
        """
        print(f"\n[Stage 1] Loading detections from {self.detections_json_path}...")
        
        with open(self.detections_json_path, 'r') as f:
            data = json.load(f)
        
        # Filter to first n autumn and m winter images
        autumn_data = data['autumn'][:self.n_autumn]
        winter_data = data['winter'][:self.m_winter]
        
        print(f"  ✓ Loaded {len(autumn_data)} autumn images")
        print(f"  ✓ Loaded {len(winter_data)} winter images")
        
        return autumn_data, winter_data
    
    def match_detections(
        self,
        autumn_entry: Dict,
        winter_entry: Dict
    ) -> Tuple[List[DetectionMatch], float]:
        """
        STAGE 2: Compare detections between autumn and winter images.
        
        Args:
            autumn_entry: Autumn image entry with detections
            winter_entry: Winter image entry with detections
            
        Returns:
            Tuple of (detection_matches, aggregated_score)
        """
        # Load images
        autumn_img = Image.open(autumn_entry['image_path']).convert('RGB')
        winter_img = Image.open(winter_entry['image_path']).convert('RGB')
        
        detection_matches = []
        scores = []
        
        # Compare detections with matching labels
        for autumn_det in autumn_entry['detections']:
            for winter_det in winter_entry['detections']:
                # Only compare detections with matching labels
                if autumn_det['label'] == winter_det['label']:
                    # Match cropped regions
                    total_matches, inliers, confidence = self.matcher.match_regions(
                        autumn_img, autumn_det['bbox'],
                        winter_img, winter_det['bbox']
                    )
                    
                    # Record all matches for scoring, even if confidence is low
                    scores.append(confidence)
                    
                    # Only add to detection_matches if above threshold
                    if confidence >= self.detection_threshold:
                        detection_matches.append(DetectionMatch(
                            autumn_label=autumn_det['label'],
                            winter_label=winter_det['label'],
                            autumn_bbox=autumn_det['bbox'],
                            winter_bbox=winter_det['bbox'],
                            similarity_score=confidence,
                            num_keypoints=total_matches,
                            num_inliers=inliers
                        ))
        
        # Aggregate scores: average of all matching detection pairs (even if below threshold)
        aggregated_score = np.mean(scores) if len(scores) > 0 else 0.0
        
        return detection_matches, float(aggregated_score)
    
    def select_candidates(
        self,
        autumn_entry: Dict,
        winter_data: List[Dict]
    ) -> List[CandidateMatch]:
        """
        STAGE 3: Select top-K candidate matches for an autumn image.
        
        Args:
            autumn_entry: Autumn image entry
            winter_data: List of winter image entries
            
        Returns:
            List of top-K candidate matches
        """
        candidates = []
        
        for winter_entry in tqdm(winter_data, desc=f"  Matching {autumn_entry['frame_id']}", leave=False):
            detection_matches, aggregated_score = self.match_detections(
                autumn_entry, winter_entry
            )
            
            if aggregated_score > 0:
                candidates.append(CandidateMatch(
                    winter_frame_id=winter_entry['frame_id'],
                    winter_image_path=winter_entry['image_path'],
                    detection_matches=detection_matches,
                    aggregated_score=aggregated_score,
                    num_matching_labels=len(detection_matches)
                ))
        
        # Sort by aggregated score and return top-K
        candidates.sort(key=lambda x: x.aggregated_score, reverse=True)
        return candidates[:self.top_k]
    
    def match_full_images(
        self,
        autumn_entry: Dict,
        candidates: List[CandidateMatch]
    ) -> Optional[FinalMatch]:
        """
        STAGE 4: Perform full-image matching on top candidates.
        
        Args:
            autumn_entry: Autumn image entry
            candidates: List of candidate matches
            
        Returns:
            Best match based on full-image similarity
        """
        if len(candidates) == 0:
            return None
        
        autumn_img = Image.open(autumn_entry['image_path']).convert('RGB')
        
        best_match = None
        best_score = -1
        
        for candidate in candidates:
            winter_img = Image.open(candidate.winter_image_path).convert('RGB')
            
            # Full-image matching
            total_matches, inliers, confidence = self.matcher.match_full_images(
                autumn_img, winter_img
            )
            
            if confidence > best_score:
                best_score = confidence
                best_match = FinalMatch(
                    autumn_frame_id=autumn_entry['frame_id'],
                    autumn_image_path=autumn_entry['image_path'],
                    winter_frame_id=candidate.winter_frame_id,
                    winter_image_path=candidate.winter_image_path,
                    detection_level_score=candidate.aggregated_score,
                    full_image_score=confidence,
                    full_image_keypoints=total_matches,
                    full_image_inliers=inliers
                )
        
        return best_match
    
    def visualize_match(
        self,
        match: FinalMatch,
        autumn_entry: Dict,
        winter_entry: Dict,
        rank: int
    ):
        """
        STAGE 5: Create visualization for a match.
        
        Args:
            match: FinalMatch to visualize
            autumn_entry: Autumn image entry with detections
            winter_entry: Winter image entry with detections
            rank: Rank of this autumn image
        """
        # Load images
        autumn_img = Image.open(match.autumn_image_path).convert('RGB')
        winter_img = Image.open(match.winter_image_path).convert('RGB')
        
        # Resize to same height for side-by-side comparison
        target_height = 600
        autumn_ratio = target_height / autumn_img.height
        winter_ratio = target_height / winter_img.height
        
        autumn_resized = autumn_img.resize(
            (int(autumn_img.width * autumn_ratio), target_height),
            Image.Resampling.LANCZOS
        )
        winter_resized = winter_img.resize(
            (int(winter_img.width * winter_ratio), target_height),
            Image.Resampling.LANCZOS
        )
        
        # Create side-by-side canvas with space for title
        canvas_width = autumn_resized.width + winter_resized.width + 30  # 30px gap
        canvas_height = target_height + 100  # 100px for title and scores
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
        
        # Paste images
        canvas.paste(autumn_resized, (10, 80))
        canvas.paste(winter_resized, (autumn_resized.width + 20, 80))
        
        # Draw bounding boxes on the images
        draw = ImageDraw.Draw(canvas)
        
        # Try to load a nice font, fall back to default if unavailable
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            score_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            score_font = ImageFont.load_default()
        
        # Draw title
        title = f"Match #{rank}: {match.autumn_frame_id} ↔ {match.winter_frame_id}"
        draw.text((10, 10), title, fill='black', font=title_font)
        
        # Draw scores
        score_text = f"Detection Score: {match.detection_level_score:.3f} | Full-Image Score: {match.full_image_score:.3f} | Inliers: {match.full_image_inliers}"
        draw.text((10, 45), score_text, fill='darkgreen', font=score_font)
        
        # Draw season labels
        draw.text((autumn_resized.width // 2 - 30, 60), "AUTUMN", fill='darkorange', font=label_font)
        draw.text((autumn_resized.width + 20 + winter_resized.width // 2 - 30, 60), "WINTER", fill='steelblue', font=label_font)
        
        # Draw bounding boxes for matched detections
        # Find matching labels between autumn and winter
        for autumn_det in autumn_entry['detections']:
            for winter_det in winter_entry['detections']:
                if autumn_det['label'] == winter_det['label']:
                    # Draw autumn bbox
                    bbox = autumn_det['bbox']
                    x1, y1, x2, y2 = [int(v * autumn_ratio) for v in bbox]
                    draw.rectangle(
                        [(10 + x1, 80 + y1), (10 + x2, 80 + y2)],
                        outline='darkorange',
                        width=2
                    )
                    draw.text((10 + x1 + 2, 80 + y1 + 2), autumn_det['label'], fill='darkorange', font=label_font)
                    
                    # Draw winter bbox
                    bbox = winter_det['bbox']
                    x1, y1, x2, y2 = [int(v * winter_ratio) for v in bbox]
                    draw.rectangle(
                        [(autumn_resized.width + 20 + x1, 80 + y1), (autumn_resized.width + 20 + x2, 80 + y2)],
                        outline='steelblue',
                        width=2
                    )
                    draw.text((autumn_resized.width + 20 + x1 + 2, 80 + y1 + 2), winter_det['label'], fill='steelblue', font=label_font)
        
        # Save visualization
        output_path = self.output_dir / f"match_{rank:03d}_{match.autumn_frame_id}.png"
        canvas.save(output_path)
        match.visualization_path = str(output_path)
        
        print(f"  ✓ Saved visualization: {output_path}")
    
    def run(self) -> List[FinalMatch]:
        """
        Execute the complete pipeline.
        
        Returns:
            List of final matches
        """
        # Stage 1: Load detections
        autumn_data, winter_data = self.load_detections()
        
        # Stages 2-5: Process each autumn image
        print(f"\n[Stages 2-4] Matching detections and finding best matches...")
        
        final_matches = []
        
        for rank, autumn_entry in enumerate(tqdm(autumn_data, desc="Processing autumn images"), start=1):
            # Stage 2 & 3: Match detections and select candidates
            candidates = self.select_candidates(autumn_entry, winter_data)
            
            if len(candidates) == 0:
                print(f"  ⚠ No candidates found for {autumn_entry['frame_id']}")
                continue
            
            # Stage 4: Full-image matching
            best_match = self.match_full_images(autumn_entry, candidates)
            
            if best_match is None:
                print(f"  ⚠ No match found for {autumn_entry['frame_id']}")
                continue
            
            # Stage 5: Visualization
            # Find winter entry for visualization
            winter_entry = next(w for w in winter_data if w['frame_id'] == best_match.winter_frame_id)
            self.visualize_match(best_match, autumn_entry, winter_entry, rank)
            
            final_matches.append(best_match)
        
        # Stage 6: Export results
        print(f"\n[Stage 6] Exporting results...")
        self.export_results(final_matches)
        
        return final_matches
    
    def export_results(self, matches: List[FinalMatch]):
        """
        STAGE 6: Export results to JSON and print summary.
        
        Args:
            matches: List of final matches
        """
        # Save to JSON
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'metadata': {
                    'method': self.method,
                    'n_autumn': self.n_autumn,
                    'm_winter': self.m_winter,
                    'top_k': self.top_k,
                    'detection_threshold': self.detection_threshold,
                    'total_matches': len(matches),
                },
                'matches': [m.to_dict() for m in matches]
            }, f, indent=2)
        
        print(f"  ✓ Saved results to {results_path}")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"  Total autumn images processed: {self.n_autumn}")
        print(f"  Total matches found: {len(matches)}")
        
        if len(matches) > 0:
            avg_detection_score = np.mean([m.detection_level_score for m in matches])
            avg_full_image_score = np.mean([m.full_image_score for m in matches])
            avg_inliers = np.mean([m.full_image_inliers for m in matches])
            
            print(f"  Average detection-level score: {avg_detection_score:.3f}")
            print(f"  Average full-image score: {avg_full_image_score:.3f}")
            print(f"  Average inliers: {avg_inliers:.1f}")
            
            print(f"\n  Top 5 matches by full-image score:")
            sorted_matches = sorted(matches, key=lambda x: x.full_image_score, reverse=True)[:5]
            for i, match in enumerate(sorted_matches, 1):
                print(f"    {i}. {match.autumn_frame_id} ↔ {match.winter_frame_id}")
                print(f"       Detection: {match.detection_level_score:.3f}, Full-image: {match.full_image_score:.3f}, Inliers: {match.full_image_inliers}")
        
        print(f"{'='*70}\n")


# ===========================================================================
# MAIN
# ===========================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Geometric Bounding Box Matching Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--n-autumn',
        type=int,
        required=True,
        help='Number of autumn images to process'
    )
    parser.add_argument(
        '--m-winter',
        type=int,
        required=True,
        help='Number of winter images to process'
    )
    parser.add_argument(
        '--detections-json',
        type=Path,
        default=Path('artifacts/detections.json'),
        help='Path to detections.json (default: artifacts/detections.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('geometric_bbox_results'),
        help='Output directory (default: geometric_bbox_results)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='orb',
        choices=['orb', 'sift', 'akaze'],
        help='Classical CV method (default: orb)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of candidate matches to consider (default: 5)'
    )
    parser.add_argument(
        '--detection-threshold',
        type=float,
        default=0.3,
        help='Minimum detection match score (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = GeometricBoundingBoxPipeline(
        detections_json_path=args.detections_json,
        output_dir=args.output_dir,
        n_autumn=args.n_autumn,
        m_winter=args.m_winter,
        method=args.method,
        top_k=args.top_k,
        detection_threshold=args.detection_threshold
    )
    
    pipeline.run()


if __name__ == '__main__':
    main()
