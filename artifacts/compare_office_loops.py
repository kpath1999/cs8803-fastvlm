#!/usr/bin/env python3
"""
Office Loop Route Comparison
=============================

Compares multiple videos of the same route taken under different conditions
(weather, lighting, seasons) to identify:
1. Route similarity (are these the same path?)
2. Condition-invariant landmarks
3. Condition-specific differences
4. Temporal progression along the route

This is a direct application of semantic SLAM concepts to your office loop dataset.

Usage:
    python compare_office_loops.py \
        --video-dir data/fourseasons/officeloop \
        --output-dir ./office_loop_analysis
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict

import torch
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# Import from video_analysis_experiments
import sys
sys.path.insert(0, os.path.dirname(__file__))
from video_analysis_experiments import FastVLMVideoAnalyzer, VideoProcessor, FrameAnalysis


@dataclass
class RouteSegment:
    """Represents a segment of the route with semantic description"""
    video_name: str
    frame_number: int
    timestamp_ms: float
    route_position: float  # 0.0 to 1.0 representing progress along route
    description: str
    landmarks: List[str]
    embedding: np.ndarray
    condition: str  # e.g., "sunny", "rainy", "snowy"
    
    def to_dict(self):
        d = asdict(self)
        d['embedding'] = self.embedding.tolist() if self.embedding is not None else None
        return d


@dataclass
class RouteMatch:
    """Represents a matched segment across two videos"""
    video1: str
    video2: str
    segment1: RouteSegment
    segment2: RouteSegment
    embedding_similarity: float
    semantic_similarity_score: float  # 0-1
    condition_differences: List[str]  # What changed between videos
    invariant_features: List[str]     # What stayed the same


class OfficeLoopAnalyzer:
    """
    Specialized analyzer for comparing multiple traversals of the same route.
    """
    
    def __init__(self, analyzer: FastVLMVideoAnalyzer):
        self.analyzer = analyzer
        self.videos_data = {}  # Store analysis for each video
    
    def extract_route_segments(self, video_path: str, 
                               num_segments: int = 10) -> List[RouteSegment]:
        """
        Extract evenly-spaced segments along the route.
        
        Args:
            video_path: Path to video file
            num_segments: Number of segments to extract along the route
        
        Returns:
            List of RouteSegment objects
        """
        video = VideoProcessor(video_path)
        video_name = Path(video_path).stem
        
        # Extract condition from filename (date gives us seasonal info)
        condition = self._infer_condition(video_name)
        
        print(f"\n📹 Processing {video_name}")
        print(f"   Condition: {condition}")
        print(f"   Extracting {num_segments} route segments...")
        
        segments = []
        
        # Extract frames evenly distributed across the video
        for i in range(num_segments):
            route_position = i / (num_segments - 1) if num_segments > 1 else 0.0
            frame_num = int(route_position * (video.total_frames - 1))
            
            img = video.extract_frame(frame_num)
            if img is None:
                continue
            
            timestamp_ms = (frame_num / video.fps) * 1000
            
            # Get rich semantic description
            desc = self._describe_route_segment(img)
            
            # Extract landmarks
            landmarks = self._extract_landmarks_list(img)
            
            # Get embedding
            embedding = self.analyzer.get_vision_embedding(img)
            
            segment = RouteSegment(
                video_name=video_name,
                frame_number=frame_num,
                timestamp_ms=timestamp_ms,
                route_position=route_position,
                description=desc,
                landmarks=landmarks,
                embedding=embedding,
                condition=condition
            )
            
            segments.append(segment)
        
        print(f"   ✓ Extracted {len(segments)} segments")
        return segments
    
    def _infer_condition(self, video_name: str) -> str:
        """Infer weather/season condition from filename date"""
        # Extract date from filename: recording_2020-03-24_...
        try:
            date_part = video_name.split('_')[1]
            year, month, day = date_part.split('-')
            month = int(month)
            
            # Simple heuristic based on month
            if month in [12, 1, 2]:
                return "winter"
            elif month in [3, 4, 5]:
                return "spring"
            elif month in [6, 7, 8]:
                return "summer"
            else:
                return "fall"
        except:
            return "unknown"
    
    def _describe_route_segment(self, image: Image.Image) -> str:
        """Get detailed description of a route segment"""
        prompt = (
            "Describe this scene as if giving directions. "
            "Focus on: 1) Permanent landmarks (buildings, signs, distinctive features), "
            "2) The type of area (parking lot, street, courtyard, etc.), "
            "3) Approximate direction or orientation. "
            "Be concise but distinctive."
        )
        return self.analyzer.analyze_frame(image, prompt, temperature=0.2)
    
    def _extract_landmarks_list(self, image: Image.Image) -> List[str]:
        """Extract a list of distinct landmarks"""
        prompt = (
            "List 3-5 prominent, permanent landmarks visible in this scene "
            "(e.g., 'red brick building', 'parking sign', 'large tree'). "
            "Separate with commas."
        )
        response = self.analyzer.analyze_frame(image, prompt, temperature=0.2)
        return [lm.strip() for lm in response.split(',') if lm.strip()]
    
    def match_route_segments(self, segments1: List[RouteSegment], 
                            segments2: List[RouteSegment]) -> List[RouteMatch]:
        """
        Match segments from two different traversals of the same route.
        
        This uses both embedding similarity and route position to find correspondences.
        """
        print(f"\n🔗 Matching {segments1[0].video_name} ↔ {segments2[0].video_name}")
        
        matches = []
        
        for seg1 in segments1:
            best_match = None
            best_score = -1
            
            for seg2 in segments2:
                # Combined score: embedding similarity + route position similarity
                emb_sim = self._cosine_similarity(seg1.embedding, seg2.embedding)
                pos_sim = 1.0 - abs(seg1.route_position - seg2.route_position)
                
                # Weighted combination (favor position for route matching)
                combined_score = 0.7 * pos_sim + 0.3 * emb_sim
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = seg2
                    best_emb_sim = emb_sim
            
            if best_match and best_score > 0.6:  # Threshold for valid match
                # Analyze differences and similarities
                diffs, invariants = self._analyze_segment_differences(seg1, best_match)
                
                match = RouteMatch(
                    video1=seg1.video_name,
                    video2=best_match.video_name,
                    segment1=seg1,
                    segment2=best_match,
                    embedding_similarity=float(best_emb_sim),
                    semantic_similarity_score=float(best_score),
                    condition_differences=diffs,
                    invariant_features=invariants
                )
                matches.append(match)
        
        print(f"   ✓ Found {len(matches)} segment matches")
        return matches
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def _analyze_segment_differences(self, seg1: RouteSegment, 
                                     seg2: RouteSegment) -> Tuple[List[str], List[str]]:
        """
        Analyze what's different and what's the same between two segments.
        
        Returns:
            (differences, invariant_features)
        """
        # Find common landmarks (invariants)
        landmarks1_lower = [lm.lower() for lm in seg1.landmarks]
        landmarks2_lower = [lm.lower() for lm in seg2.landmarks]
        
        # Simple overlap detection
        common = []
        for lm1 in seg1.landmarks:
            for lm2 in seg2.landmarks:
                if self._landmarks_similar(lm1.lower(), lm2.lower()):
                    common.append(lm1)
                    break
        
        # Condition differences
        differences = []
        if seg1.condition != seg2.condition:
            differences.append(f"Condition: {seg1.condition} vs {seg2.condition}")
        
        # Simple heuristic for appearance differences
        if self._cosine_similarity(seg1.embedding, seg2.embedding) < 0.85:
            differences.append("Significant appearance change")
        
        return differences, common
    
    def _landmarks_similar(self, lm1: str, lm2: str) -> bool:
        """Check if two landmark descriptions refer to the same thing"""
        # Simple word overlap heuristic
        words1 = set(lm1.split())
        words2 = set(lm2.split())
        overlap = len(words1 & words2)
        return overlap >= 2 or (overlap >= 1 and (len(words1) <= 2 or len(words2) <= 2))
    
    def compute_route_similarity_matrix(self, all_segments: Dict[str, List[RouteSegment]]) -> np.ndarray:
        """
        Compute pairwise similarity between all video pairs.
        
        Returns:
            Similarity matrix (n_videos x n_videos)
        """
        video_names = list(all_segments.keys())
        n_videos = len(video_names)
        
        similarity_matrix = np.zeros((n_videos, n_videos))
        
        print(f"\n📊 Computing route similarity matrix for {n_videos} videos...")
        
        for i, vid1 in enumerate(video_names):
            for j, vid2 in enumerate(video_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:  # Compute only upper triangle
                    # Average embedding similarity across matched segments
                    sims = []
                    for seg1 in all_segments[vid1]:
                        for seg2 in all_segments[vid2]:
                            if abs(seg1.route_position - seg2.route_position) < 0.15:
                                sim = self._cosine_similarity(seg1.embedding, seg2.embedding)
                                sims.append(sim)
                    
                    avg_sim = np.mean(sims) if sims else 0.0
                    similarity_matrix[i, j] = avg_sim
                    similarity_matrix[j, i] = avg_sim
        
        return similarity_matrix, video_names
    
    def identify_invariant_landmarks(self, all_segments: Dict[str, List[RouteSegment]]) -> List[Dict]:
        """
        Find landmarks that appear consistently across all videos.
        These are the best candidates for season/condition-invariant features.
        """
        print(f"\n🏛️ Identifying invariant landmarks across all videos...")
        
        # Group segments by route position
        position_bins = defaultdict(list)
        
        for video_name, segments in all_segments.items():
            for seg in segments:
                # Bin by route position (0.1 bins)
                bin_idx = int(seg.route_position * 10)
                position_bins[bin_idx].append(seg)
        
        invariant_landmarks = []
        
        for bin_idx, segments in position_bins.items():
            if len(segments) < len(all_segments) * 0.5:  # Need at least half the videos
                continue
            
            # Find common landmarks across all segments in this bin
            all_landmarks = []
            for seg in segments:
                all_landmarks.extend([lm.lower() for lm in seg.landmarks])
            
            # Count occurrences
            from collections import Counter
            landmark_counts = Counter(all_landmarks)
            
            # Landmarks appearing in multiple videos at this position
            for landmark, count in landmark_counts.items():
                if count >= len(segments) * 0.5:  # In at least half of the segments
                    invariant_landmarks.append({
                        'route_position': bin_idx / 10.0,
                        'landmark': landmark,
                        'frequency': count / len(segments),
                        'videos_seen': count
                    })
        
        print(f"   ✓ Found {len(invariant_landmarks)} invariant landmarks")
        return invariant_landmarks


def visualize_similarity_matrix(similarity_matrix: np.ndarray, 
                                video_names: List[str], 
                                output_path: Path):
    """Create a heatmap of route similarity"""
    plt.figure(figsize=(12, 10))
    
    # Shorten video names for display
    short_names = [name.replace('recording_', '').replace('_5x', '') for name in video_names]
    
    sns.heatmap(similarity_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='YlOrRd',
                xticklabels=short_names,
                yticklabels=short_names,
                cbar_kws={'label': 'Route Similarity'})
    
    plt.title('Office Loop Route Similarity Matrix\n(Higher = More Similar Routes)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Video Recording', fontsize=12)
    plt.ylabel('Video Recording', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved similarity matrix to {output_path}")


def visualize_route_progression(all_segments: Dict[str, List[RouteSegment]], 
                                output_path: Path):
    """Visualize how embeddings change along the route for each video"""
    plt.figure(figsize=(14, 8))
    
    for video_name, segments in all_segments.items():
        positions = [seg.route_position for seg in segments]
        # Use first PCA component of embeddings as y-axis
        embeddings = np.array([seg.embedding for seg in segments])
        
        # Handle both 1D and 2D embedding arrays
        if embeddings.ndim == 1:
            # Already 1D, use directly
            embedding_norms = embeddings
        else:
            # 2D array, compute norm along correct axis
            embedding_norms = np.linalg.norm(embeddings, axis=1)
        
        short_name = video_name.replace('recording_', '').replace('_5x', '')
        plt.plot(positions, embedding_norms, 'o-', label=short_name, alpha=0.7)
    
    plt.xlabel('Route Position (0 = Start, 1 = End)', fontsize=12)
    plt.ylabel('Embedding Magnitude', fontsize=12)
    plt.title('Route Progression: Embedding Evolution', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved route progression to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple traversals of office loop route")
    parser.add_argument("--video-dir", type=str, 
                       default="data/fourseasons/officeloop",
                       help="Directory containing office loop videos")
    parser.add_argument("--model-path", type=str,
                       default="/Users/kausar/Documents/ml-fastvlm/checkpoints/llava-fastvithd_1.5b_stage2",
                       help="Path to FastVLM model checkpoint")
    parser.add_argument("--output-dir", type=str, 
                       default="./office_loop_analysis",
                       help="Directory to save results")
    parser.add_argument("--num-segments", type=int, default=10,
                       help="Number of segments to extract per video")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device to run on (mps, cuda, cpu)")
    parser.add_argument("--max-videos", type=int, default=None,
                       help="Maximum number of videos to process (for testing)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("Office Loop Route Comparison Analysis")
    print("="*70)
    
    # Find all videos
    video_dir = Path(args.video_dir)
    video_files = sorted(video_dir.glob("*.mp4"))
    
    if args.max_videos:
        video_files = video_files[:args.max_videos]
    
    print(f"\nFound {len(video_files)} videos to analyze:")
    for vf in video_files:
        print(f"  • {vf.name}")
    
    # Initialize analyzer
    print("\nInitializing FastVLM...")
    vlm_analyzer = FastVLMVideoAnalyzer(args.model_path, device=args.device)
    loop_analyzer = OfficeLoopAnalyzer(vlm_analyzer)
    
    # Extract route segments from all videos
    all_segments = {}
    
    for video_file in video_files:
        segments = loop_analyzer.extract_route_segments(
            str(video_file), 
            num_segments=args.num_segments
        )
        all_segments[video_file.stem] = segments
    
    # Compute similarity matrix
    similarity_matrix, video_names = loop_analyzer.compute_route_similarity_matrix(all_segments)
    
    # Find invariant landmarks
    invariant_landmarks = loop_analyzer.identify_invariant_landmarks(all_segments)
    
    # Pairwise matching (for first few videos)
    print(f"\n🔄 Computing pairwise matches...")
    all_matches = []
    video_pairs_processed = 0
    
    for i, vid1 in enumerate(video_names[:3]):  # Limit to avoid too much processing
        for vid2 in video_names[i+1:i+2]:  # Match each with next one
            matches = loop_analyzer.match_route_segments(
                all_segments[vid1],
                all_segments[vid2]
            )
            all_matches.extend(matches)
            video_pairs_processed += 1
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # 1. Similarity matrix
    np.save(output_dir / "similarity_matrix.npy", similarity_matrix)
    with open(output_dir / "video_names.json", 'w') as f:
        json.dump(video_names, f, indent=2)
    
    # 2. All segments data
    segments_export = {}
    for video_name, segments in all_segments.items():
        segments_export[video_name] = [seg.to_dict() for seg in segments]
    
    with open(output_dir / "route_segments.json", 'w') as f:
        json.dump(segments_export, f, indent=2)
    
    # 3. Invariant landmarks
    with open(output_dir / "invariant_landmarks.json", 'w') as f:
        json.dump(invariant_landmarks, f, indent=2)
    
    # 4. Sample matches
    matches_export = []
    for match in all_matches[:20]:  # Save first 20
        matches_export.append({
            'video1': match.video1,
            'video2': match.video2,
            'route_position_1': match.segment1.route_position,
            'route_position_2': match.segment2.route_position,
            'embedding_similarity': float(match.embedding_similarity),
            'semantic_similarity': float(match.semantic_similarity_score),
            'description_1': match.segment1.description,
            'description_2': match.segment2.description,
            'condition_differences': match.condition_differences,
            'invariant_features': match.invariant_features
        })
    
    with open(output_dir / "sample_matches.json", 'w') as f:
        json.dump(matches_export, f, indent=2)
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    visualize_similarity_matrix(similarity_matrix, video_names, 
                                output_dir / "similarity_matrix.png")
    visualize_route_progression(all_segments, 
                                output_dir / "route_progression.png")
    
    # Print summary report
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\n📹 Videos Analyzed: {len(video_files)}")
    print(f"   Segments per video: {args.num_segments}")
    print(f"   Total segments: {sum(len(segs) for segs in all_segments.values())}")
    
    print(f"\n🔗 Route Similarity:")
    avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    print(f"   Average similarity: {avg_similarity:.3f}")
    
    # Get min/max excluding diagonal (which is always 1.0)
    off_diagonal = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    if len(off_diagonal) > 0:
        print(f"   Min similarity: {np.min(off_diagonal):.3f}")
        print(f"   Max similarity: {np.max(off_diagonal):.3f}")
    else:
        print(f"   (Only one video - no comparisons to make)")
    
    if avg_similarity > 0.7:
        print(f"   ✓ HIGH similarity - These appear to be the SAME ROUTE")
    elif avg_similarity > 0.5:
        print(f"   ⚠️  MODERATE similarity - Likely same route with variations")
    else:
        print(f"   ❌ LOW similarity - Routes may be different")
    
    print(f"\n🏛️ Invariant Landmarks:")
    print(f"   Found {len(invariant_landmarks)} condition-invariant features")
    
    # Show top invariant landmarks
    sorted_landmarks = sorted(invariant_landmarks, 
                             key=lambda x: x['frequency'], 
                             reverse=True)[:5]
    
    for lm in sorted_landmarks:
        print(f"   • '{lm['landmark']}' at position {lm['route_position']:.1f} "
              f"(seen in {lm['videos_seen']}/{len(video_files)} videos)")
    
    print(f"\n📊 Condition Breakdown:")
    conditions = {}
    for video_name, segments in all_segments.items():
        cond = segments[0].condition if segments else "unknown"
        conditions[cond] = conditions.get(cond, 0) + 1
    
    for cond, count in sorted(conditions.items()):
        print(f"   {cond.capitalize()}: {count} video(s)")
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   • similarity_matrix.png - Heatmap of route similarities")
    print(f"   • route_progression.png - How scenes change along route")
    print(f"   • route_segments.json - All extracted segments with descriptions")
    print(f"   • invariant_landmarks.json - Condition-invariant features")
    print(f"   • sample_matches.json - Example segment matches")
    
    print("\n" + "="*70)
    print("✓ Analysis Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
