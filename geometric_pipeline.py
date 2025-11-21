#!/usr/bin/env python3
"""
*** GEOMETRIC PIPELINE: CLASSICAL COMPUTER VISION FRAME MATCHING ***

This script uses classical computer vision techniques (no deep learning) to match
frames between autumn and winter datasets. It identifies the most similar frame
pairs based on color histograms and ORB keypoint matching.

PIPELINE STAGES:
================

Stage 1: IMAGE LOADING
    - Load RGB frames from winter and autumn datasets
    - Filter out system files and establish frame inventory
    - Output: Lists of frame paths for each dataset

Stage 2: FRAME PAIR GENERATION
    - Create all possible pairs between autumn and winter frames
    - Support limiting via max_pairs argument
    - Output: List of (winter_path, autumn_path) tuples

Stage 3: CLASSICAL CV MATCHING
    - Compute HSV color histograms for global scene similarity
    - Extract ORB keypoints for local geometric matching
    - Apply ratio test and RANSAC for robust matching
    - Output: Similarity scores for each pair

Stage 4: SCORING AND RANKING
    - Combine histogram similarity and keypoint confidence
    - Rank all pairs by combined score
    - Identify best match for each autumn frame
    - Output: Ranked list of matches

Stage 5: VISUALIZATION
    - Create side-by-side comparisons for top matches
    - Draw ORB keypoint correspondences
    - Annotate with similarity scores
    - Output: Visual results in output directory

Stage 6: EXPORT RESULTS
    - Save ranked matches to JSON
    - Export summary statistics
    - Print top matches to console

USAGE:
======
# Recommended: Fast keyframe selection for large datasets
python geometric_pipeline.py \
    --winter-rgb data/winter/realsense_D435i/rgb \
    --autumn-rgb data/autumn/realsense_D435i/rgb \
    --output-dir geometric_results \
    --use-keyframing \
    --keyframe-threshold 0.95 \
    --top-k 10

# Basic usage with limited pairs for testing
python geometric_pipeline.py \
    --winter-rgb data/winter/realsense_D435i/rgb \
    --autumn-rgb data/autumn/realsense_D435i/rgb \
    --output-dir geometric_results \
    --max-pairs 20 \
    --top-k 5

# Full dataset processing without keyframe selection
python geometric_pipeline.py \
    --winter-rgb data/winter/realsense_D435i/rgb \
    --autumn-rgb data/autumn/realsense_D435i/rgb \
    --output-dir geometric_results \
    --top-k 10

KEY OPTIONS:
  --use-keyframing      Enable fast histogram-based keyframe selection (highly recommended for large datasets)
  --keyframe-threshold  Histogram correlation threshold for keyframe selection (default: 0.95)
  --max-pairs N         Process at most N frame pairs (useful for testing)
  --top-k K             Visualize top K matches (default: 10)
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
class GeometricMatch:
    """Represents a geometric match between two frames."""
    
    winter_frame_id: str
    autumn_frame_id: str
    winter_image_path: str
    autumn_image_path: str
    
    # Color histogram similarity (0-1, higher is better)
    histogram_similarity: float
    
    # ORB keypoint matching results
    orb_matches_count: int          # Number of good matches after ratio test
    orb_inliers_count: int          # Number of inliers after RANSAC
    orb_confidence: float           # Inliers / total matches
    
    # Combined score (weighted average)
    final_score: float
    
    # Visualization path
    visualization_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


# ===========================================================================
# STAGE 3: CLASSICAL CV MATCHING
# ===========================================================================

class ColorHistogramMatcher:
    """
    Computes and compares color histograms for frame similarity.
    
    Uses HSV color space with 8 bins per channel for a compact
    512-dimensional feature vector (8x8x8).
    """
    
    @staticmethod
    def compute_histogram(image: Image.Image) -> np.ndarray:
        """
        Compute HSV color histogram for an image.
        
        Args:
            image: PIL Image in RGB format
            
        Returns:
            Normalized histogram as flattened numpy array
        """
        # Convert PIL to OpenCV format (RGB -> BGR)
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to HSV color space (more perceptually uniform than RGB)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        # Compute 3D histogram: 8 bins per channel
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                           [0, 180, 0, 256, 0, 256])
        
        # Normalize to make it scale-invariant
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    @staticmethod
    def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compare two histograms using correlation.
        
        Args:
            hist1, hist2: Normalized histograms
            
        Returns:
            Correlation score [0, 1], where 1 = identical
        """
        correlation = cv2.compareHist(
            hist1.astype(np.float32), 
            hist2.astype(np.float32), 
            cv2.HISTCMP_CORREL
        )
        # Clamp to [0, 1] range (correlation can be slightly negative)
        return max(0.0, min(1.0, float(correlation)))


class ORBKeypointMatcher:
    """
    Performs local geometric matching using ORB keypoints.
    
    ORB (Oriented FAST and Rotated BRIEF) is a fast, patent-free
    alternative to SIFT for keypoint detection and matching.
    """
    
    def __init__(self, n_features: int = 1000):
        """
        Initialize ORB detector.
        
        Args:
            n_features: Maximum number of features to detect
        """
        self.detector = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def match_frames(
        self,
        img1: Image.Image,
        img2: Image.Image,
        ratio_threshold: float = 0.75
    ) -> Tuple[int, int, float, List, List, List]:
        """
        Match ORB keypoints between two frames.
        
        Uses Lowe's ratio test to filter good matches and RANSAC
        for geometric verification.
        
        Args:
            img1, img2: PIL Images to match
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            Tuple of (total_matches, inliers, confidence, good_matches, kp1, kp2)
        """
        # Convert PIL to OpenCV grayscale
        cv_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        cv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
        
        # Detect keypoints and compute descriptors
        kp1, des1 = self.detector.detectAndCompute(cv_img1, None)
        kp2, des2 = self.detector.detectAndCompute(cv_img2, None)
        
        # Check if we have enough keypoints
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return 0, 0, 0.0, [], [], []
        
        # Match descriptors using KNN (k=2 for ratio test)
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return 0, 0, 0.0, [], kp1, kp2
        
        # Extract matched keypoint locations
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # RANSAC to find inliers
        try:
            _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            if mask is None:
                return len(good_matches), 0, 0.0, good_matches, kp1, kp2
            
            inliers = int(np.sum(mask))
            total = len(good_matches)
            confidence = inliers / total if total > 0 else 0.0
            
            return total, inliers, confidence, good_matches, kp1, kp2
        except:
            return len(good_matches), 0, 0.0, good_matches, kp1, kp2


# ===========================================================================
# PIPELINE ORCHESTRATION
# ===========================================================================

class GeometricPipeline:
    """
    Main pipeline orchestrator for classical CV frame matching.
    
    Coordinates all stages from image loading through final visualization.
    """
    
    def __init__(
        self,
        winter_rgb_dir: Path,
        autumn_rgb_dir: Path,
        output_dir: Path,
        histogram_weight: float = 0.4,
        keypoint_weight: float = 0.6,
        use_keyframing: bool = False,
        keyframe_similarity_threshold: float = 0.95,
    ):
        """
        Initialize pipeline components.
        
        Args:
            winter_rgb_dir: Path to winter RGB images
            autumn_rgb_dir: Path to autumn RGB images
            output_dir: Directory for results
            histogram_weight: Weight for histogram similarity in final score
            keypoint_weight: Weight for keypoint confidence in final score
            use_keyframing: Enable keyframe selection to reduce the number of images
            keyframe_similarity_threshold: Histogram correlation threshold for keyframe selection
        """
        self.winter_rgb_dir = winter_rgb_dir
        self.autumn_rgb_dir = autumn_rgb_dir
        self.output_dir = output_dir
        self.histogram_weight = histogram_weight
        self.keypoint_weight = keypoint_weight
        self.use_keyframing = use_keyframing
        self.keyframe_similarity_threshold = keyframe_similarity_threshold
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        # Initialize matchers
        self.histogram_matcher = ColorHistogramMatcher()
        self.orb_matcher = ORBKeypointMatcher(n_features=1000)
        
        print("\n" + "="*70)
        print("GEOMETRIC PIPELINE: CLASSICAL CV FRAME MATCHING")
        print("="*70)
        print(f"Histogram weight: {histogram_weight}")
        print(f"Keypoint weight: {keypoint_weight}")
        if use_keyframing:
            print(f"Using keyframe selection with similarity threshold: {keyframe_similarity_threshold}")
        print("="*70)
    
    def load_frames(self) -> Tuple[List[Path], List[Path]]:
        """
        STAGE 1: Load image files from directories.
        
        Returns:
            Tuple of (winter_images, autumn_images)
        """
        print("\n" + "="*70)
        print("STAGE 1: Loading image frames")
        print("="*70)
        
        # Get all image files, filtering out macOS resource fork files (._*)
        winter_images = sorted([
            p for p in self.winter_rgb_dir.glob("*.png") 
            if not p.name.startswith("._")
        ]) + sorted([
            p for p in self.winter_rgb_dir.glob("*.jpg") 
            if not p.name.startswith("._")
        ])
        
        autumn_images = sorted([
            p for p in self.autumn_rgb_dir.glob("*.png") 
            if not p.name.startswith("._")
        ]) + sorted([
            p for p in self.autumn_rgb_dir.glob("*.jpg") 
            if not p.name.startswith("._")
        ])
        
        print(f"Found {len(winter_images)} winter images")
        print(f"Found {len(autumn_images)} autumn images")
        
        if len(winter_images) > 0:
            print(f"First winter image: {winter_images[0].name}")
        if len(autumn_images) > 0:
            print(f"First autumn image: {autumn_images[0].name}")
        
        return winter_images, autumn_images
    
    def _compute_color_histogram(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Compute a color histogram for fast frame comparison.
        
        Uses HSV color space with 8 bins per channel for a compact 
        512-dimensional feature vector (8x8x8). This is much faster
        than deep learning embeddings.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized histogram or None if loading fails
        """
        try:
            # Load image efficiently
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert to HSV color space (more perceptually uniform than RGB)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Compute 3D histogram: 8 bins per channel
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                               [0, 180, 0, 256, 0, 256])
            
            # Normalize to make it scale-invariant
            hist = cv2.normalize(hist, hist).flatten()
            
            return hist
        except Exception as e:
            print(f"Warning: Could not process {image_path}, skipping. Error: {e}")
            return None
    
    def _select_keyframes(
        self,
        image_paths: List[Path],
        similarity_threshold: float,
        dataset_name: str
    ) -> List[Path]:
        """
        STAGE 0: Select keyframes using fast histogram-based comparison.
        
        Uses color histograms instead of deep learning embeddings for
        significantly faster processing (~100x speedup). A new keyframe 
        is selected when the histogram difference exceeds the threshold.
        
        Args:
            image_paths: Chronologically sorted list of image paths.
            similarity_threshold: Histogram correlation threshold [0, 1]. 
                                  A new keyframe is chosen if correlation is *below* this value.
                                  Typical values: 0.90-0.98 (higher = more keyframes)
            dataset_name: Name of the dataset for logging (e.g., "Winter").
            
        Returns:
            A reduced list of image paths representing keyframes.
        """
        if not image_paths:
            return []

        print(f"\n" + "-"*30)
        print(f"STAGE 0: Selecting keyframes for {dataset_name} dataset")
        print(f"  Total images to process: {len(image_paths)}")
        print(f"  Similarity threshold: < {similarity_threshold} (histogram correlation)")
        print(f"  Using fast color histogram approach (no neural network)")
        print("-" * 30)

        keyframes = []
        last_keyframe_hist = None

        for image_path in tqdm(image_paths, desc=f"Selecting {dataset_name} keyframes"):
            current_hist = self._compute_color_histogram(image_path)
            if current_hist is None:
                continue

            # The first image is always a keyframe
            if last_keyframe_hist is None:
                keyframes.append(image_path)
                last_keyframe_hist = current_hist
                continue

            # Compare histograms using correlation
            # cv2.HISTCMP_CORREL returns values in [0, 1] where 1 = identical
            correlation = cv2.compareHist(
                last_keyframe_hist.astype(np.float32), 
                current_hist.astype(np.float32), 
                cv2.HISTCMP_CORREL
            )

            # Select as keyframe if correlation is below threshold
            # (i.e., images are sufficiently different)
            if correlation < similarity_threshold:
                keyframes.append(image_path)
                last_keyframe_hist = current_hist
        
        print(f"  → Selected {len(keyframes)} keyframes from {len(image_paths)} total images")
        print(f"  → Reduction: {len(image_paths)} → {len(keyframes)} ({100 * len(keyframes) / len(image_paths):.1f}%)")
        return keyframes
    
    def generate_pairs(
        self,
        winter_images: List[Path],
        autumn_images: List[Path],
        max_pairs: Optional[int] = None
    ) -> List[Tuple[Path, Path]]:
        """
        STAGE 2: Generate frame pairs for matching.
        
        Creates all possible pairs (cross-product) between autumn and winter.
        
        Args:
            winter_images: List of winter frame paths
            autumn_images: List of autumn frame paths
            max_pairs: Maximum number of pairs to generate
            
        Returns:
            List of (winter_path, autumn_path) tuples
        """
        print("\n" + "="*70)
        print("STAGE 2: Generating frame pairs")
        print("="*70)
        
        pairs = []
        total_possible = len(winter_images) * len(autumn_images)
        print(f"Total possible pairs: {total_possible}")
        
        for a_img in autumn_images:
            for w_img in winter_images:
                pairs.append((w_img, a_img))
                if max_pairs and len(pairs) >= max_pairs:
                    break
            if max_pairs and len(pairs) >= max_pairs:
                break
        
        print(f"Generated {len(pairs)} pairs to analyze")
        if max_pairs and len(pairs) < total_possible:
            print(f"  (limited to max_pairs={max_pairs})")
        
        return pairs
    
    def match_pair(
        self,
        winter_path: Path,
        autumn_path: Path
    ) -> GeometricMatch:
        """
        STAGE 3: Match a single frame pair using classical CV techniques.
        
        Args:
            winter_path: Path to winter image
            autumn_path: Path to autumn image
            
        Returns:
            GeometricMatch object with all metrics
        """
        # Load images
        winter_img = Image.open(winter_path).convert('RGB')
        autumn_img = Image.open(autumn_path).convert('RGB')
        
        # Compute color histogram similarity
        winter_hist = self.histogram_matcher.compute_histogram(winter_img)
        autumn_hist = self.histogram_matcher.compute_histogram(autumn_img)
        hist_similarity = self.histogram_matcher.compare_histograms(winter_hist, autumn_hist)
        
        # Match ORB keypoints
        total_matches, inliers, orb_confidence, _, _, _ = self.orb_matcher.match_frames(
            winter_img, autumn_img
        )
        
        # Compute final score (weighted combination)
        final_score = (
            self.histogram_weight * hist_similarity +
            self.keypoint_weight * orb_confidence
        )
        
        return GeometricMatch(
            winter_frame_id=winter_path.stem,
            autumn_frame_id=autumn_path.stem,
            winter_image_path=str(winter_path),
            autumn_image_path=str(autumn_path),
            histogram_similarity=float(hist_similarity),
            orb_matches_count=total_matches,
            orb_inliers_count=inliers,
            orb_confidence=float(orb_confidence),
            final_score=float(final_score)
        )
    
    def visualize_match(
        self,
        match: GeometricMatch,
        output_path: Path,
        rank: int
    ) -> None:
        """
        STAGE 5: Create visualization for a matched pair.
        
        Args:
            match: GeometricMatch to visualize
            output_path: Where to save visualization
            rank: Rank of this match (for title)
        """
        # Load images
        winter_img = Image.open(match.winter_image_path).convert('RGB')
        autumn_img = Image.open(match.autumn_image_path).convert('RGB')
        
        # Get ORB keypoints and matches for visualization
        _, _, _, good_matches, kp1, kp2 = self.orb_matcher.match_frames(
            winter_img, autumn_img
        )
        
        # Convert PIL to OpenCV for drawing
        cv_winter = cv2.cvtColor(np.array(winter_img), cv2.COLOR_RGB2BGR)
        cv_autumn = cv2.cvtColor(np.array(autumn_img), cv2.COLOR_RGB2BGR)
        
        # Draw matches if we have any
        if len(good_matches) > 0:
            # Draw only top 50 matches for clarity
            display_matches = good_matches[:50]
            matched_img = cv2.drawMatches(
                cv_winter, kp1,
                cv_autumn, kp2,
                display_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        else:
            # Just concatenate images if no matches
            matched_img = np.hstack([cv_winter, cv_autumn])
        
        # Convert back to PIL for text annotation
        result_img = Image.fromarray(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
        
        # Add text annotations
        draw = ImageDraw.Draw(result_img)
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
            title_font = ImageFont.truetype("Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            title_font = font
        
        # Add title and scores
        title = f"Rank #{rank} | Final Score: {match.final_score:.3f}"
        draw.text((10, 10), title, fill='yellow', font=title_font)
        
        info = (
            f"Histogram Similarity: {match.histogram_similarity:.3f} | "
            f"ORB Matches: {match.orb_matches_count} | "
            f"Inliers: {match.orb_inliers_count} | "
            f"ORB Confidence: {match.orb_confidence:.3f}"
        )
        draw.text((10, 40), info, fill='yellow', font=font)
        
        # Add frame IDs
        draw.text((10, 70), f"Winter: {match.winter_frame_id}", fill='cyan', font=font)
        draw.text((10, 95), f"Autumn: {match.autumn_frame_id}", fill='cyan', font=font)
        
        # Save
        result_img.save(output_path, quality=90)
        print(f"  Saved visualization: {output_path.name}")
    
    def run(
        self,
        max_pairs: Optional[int] = None,
        top_k: int = 10
    ) -> List[GeometricMatch]:
        """
        Run the complete pipeline.
        
        Args:
            max_pairs: Maximum number of pairs to evaluate
            top_k: Number of top matches to visualize
            
        Returns:
            List of GeometricMatch results, sorted by score (descending)
        """
        # Stage 1: Load frames
        winter_images, autumn_images = self.load_frames()
        
        if len(winter_images) == 0 or len(autumn_images) == 0:
            print("\nERROR: No images found in one or both directories!")
            return []
        
        # STAGE 0: Optional Keyframe Selection
        if self.use_keyframing:
            winter_images = self._select_keyframes(winter_images, self.keyframe_similarity_threshold, "Winter")
            autumn_images = self._select_keyframes(autumn_images, self.keyframe_similarity_threshold, "Autumn")
        
        # Stage 2: Generate pairs
        pairs = self.generate_pairs(winter_images, autumn_images, max_pairs)
        
        if len(pairs) == 0:
            print("\nERROR: No frame pairs to process!")
            return []
        
        # Stage 3: Match all pairs
        print("\n" + "="*70)
        print("STAGE 3: Matching frame pairs with classical CV")
        print("="*70)
        
        matches = []
        for winter_path, autumn_path in tqdm(pairs, desc="Matching pairs"):
            match = self.match_pair(winter_path, autumn_path)
            matches.append(match)
        
        # Stage 4: Sort by final score (descending)
        print("\n" + "="*70)
        print("STAGE 4: Ranking matches by score")
        print("="*70)
        
        matches.sort(key=lambda m: m.final_score, reverse=True)
        
        print(f"Top 10 matches:")
        for i, match in enumerate(matches[:10], 1):
            print(f"  {i}. Score: {match.final_score:.3f} | "
                  f"Winter: {match.winter_frame_id} <-> Autumn: {match.autumn_frame_id}")
        
        # Stage 5: Visualize top-k matches
        print("\n" + "="*70)
        print(f"STAGE 5: Visualizing top {top_k} matches")
        print("="*70)
        
        for i, match in enumerate(matches[:top_k], 1):
            vis_path = self.output_dir / "visualizations" / f"rank_{i:02d}_score_{match.final_score:.3f}.jpg"
            self.visualize_match(match, vis_path, i)
            match.visualization_path = str(vis_path)
        
        # Stage 6: Export results
        print("\n" + "="*70)
        print("STAGE 6: Exporting results")
        print("="*70)
        
        output_file = self.output_dir / "data" / "matches.json"
        with open(output_file, 'w') as f:
            json.dump([m.to_dict() for m in matches], f, indent=2)
        
        print(f"Saved match data to: {output_file}")
        
        # Print summary
        self._print_summary(matches)
        
        return matches
    
    def _print_summary(self, matches: List[GeometricMatch]) -> None:
        """Print summary statistics."""
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        
        if len(matches) == 0:
            print("No matches found!")
            return
        
        scores = [m.final_score for m in matches]
        hist_sims = [m.histogram_similarity for m in matches]
        orb_confs = [m.orb_confidence for m in matches]
        
        print(f"Total pairs evaluated: {len(matches)}")
        print(f"\nFinal Score Statistics:")
        print(f"  Mean: {np.mean(scores):.3f}")
        print(f"  Median: {np.median(scores):.3f}")
        print(f"  Max: {np.max(scores):.3f}")
        print(f"  Min: {np.min(scores):.3f}")
        
        print(f"\nHistogram Similarity:")
        print(f"  Mean: {np.mean(hist_sims):.3f}")
        print(f"  Median: {np.median(hist_sims):.3f}")
        
        print(f"\nORB Keypoint Confidence:")
        print(f"  Mean: {np.mean(orb_confs):.3f}")
        print(f"  Median: {np.median(orb_confs):.3f}")
        
        print(f"\nResults saved to: {self.output_dir}")


# ===========================================================================
# COMMAND LINE INTERFACE
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Geometric pipeline for classical CV frame matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--winter-rgb",
        type=str,
        required=True,
        help="Directory containing winter RGB images"
    )
    parser.add_argument(
        "--autumn-rgb",
        type=str,
        required=True,
        help="Directory containing autumn RGB images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="geometric_results",
        help="Output directory for results (default: geometric_results)"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Maximum number of frame pairs to evaluate"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top matches to visualize (default: 10)"
    )
    parser.add_argument(
        "--histogram-weight",
        type=float,
        default=0.4,
        help="Weight for histogram similarity in final score (default: 0.4)"
    )
    parser.add_argument(
        "--keypoint-weight",
        type=float,
        default=0.6,
        help="Weight for keypoint confidence in final score (default: 0.6)"
    )
    parser.add_argument(
        "--use-keyframing",
        action="store_true",
        help="Enable fast histogram-based keyframe selection (recommended for large datasets)"
    )
    parser.add_argument(
        "--keyframe-threshold",
        type=float,
        default=0.95,
        help="Histogram correlation threshold for keyframe selection (default: 0.95). "
             "Lower values = fewer keyframes. Typical range: 0.90-0.98"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the pipeline."""
    args = parse_args()
    
    # Validate weights sum to 1.0
    total_weight = args.histogram_weight + args.keypoint_weight
    if abs(total_weight - 1.0) > 0.01:
        print(f"WARNING: Weights sum to {total_weight}, normalizing to 1.0")
        args.histogram_weight /= total_weight
        args.keypoint_weight /= total_weight
    
    pipeline = GeometricPipeline(
        winter_rgb_dir=Path(args.winter_rgb),
        autumn_rgb_dir=Path(args.autumn_rgb),
        output_dir=Path(args.output_dir),
        histogram_weight=args.histogram_weight,
        keypoint_weight=args.keypoint_weight,
        use_keyframing=args.use_keyframing,
        keyframe_similarity_threshold=args.keyframe_threshold,
    )    
    # Run pipeline
    results = pipeline.run(max_pairs=args.max_pairs, top_k=args.top_k)
    
    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
