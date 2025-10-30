#!/usr/bin/env python3
"""
Cross-Temporal Landmark Matching with LLM Verification

This script demonstrates Phase 2 of your semantic SLAM pipeline:
Context-aware landmark association using FastVLM embeddings and LLM reasoning.

This is a direct precursor to matching landmarks across seasonal changes.

Usage:
    python cross_temporal_matching.py \
        --video1 summer_video.mp4 \
        --video2 winter_video.mp4 \
        --model-path ./checkpoints/llava-fastvithd_1.5b_stage2
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass

import torch
from PIL import Image
from tqdm import tqdm

# Import from your FastVLM installation
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# Import video processing from experiments script
import sys
sys.path.insert(0, os.path.dirname(__file__))
from video_analysis_experiments import FastVLMVideoAnalyzer, VideoProcessor, FrameAnalysis


@dataclass
class LandmarkMatch:
    """Represents a potential landmark match between two observations"""
    landmark_a: FrameAnalysis
    landmark_b: FrameAnalysis
    embedding_similarity: float  # Cosine similarity of embeddings
    llm_verification_score: float  # Probability from LLM (0.0 to 1.0)
    llm_reasoning: str  # LLM's explanation
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        return self.llm_verification_score >= threshold
    
    def to_dict(self):
        return {
            'frame_a': self.landmark_a.frame_number,
            'timestamp_a_ms': self.landmark_a.timestamp_ms,
            'description_a': self.landmark_a.description,
            'frame_b': self.landmark_b.frame_number,
            'timestamp_b_ms': self.landmark_b.timestamp_ms,
            'description_b': self.landmark_b.description,
            'embedding_similarity': float(self.embedding_similarity),
            'llm_verification_score': float(self.llm_verification_score),
            'llm_reasoning': self.llm_reasoning,
            'high_confidence': self.is_high_confidence()
        }


class CrossTemporalMatcher:
    """
    Matches landmarks across different temporal observations using:
    1. Fast embedding-based filtering
    2. Slow LLM-based semantic verification
    """
    
    def __init__(self, analyzer: FastVLMVideoAnalyzer, 
                 embedding_threshold: float = 0.7,
                 use_external_llm: bool = False):
        """
        Args:
            analyzer: FastVLM analyzer for generating embeddings and descriptions
            embedding_threshold: Minimum cosine similarity to consider as candidate match
            use_external_llm: If True, use external API (GPT-4, etc). If False, use FastVLM itself
        """
        self.analyzer = analyzer
        self.embedding_threshold = embedding_threshold
        self.use_external_llm = use_external_llm
    
    def find_candidate_matches(self, 
                               landmarks_a: List[FrameAnalysis], 
                               landmarks_b: List[FrameAnalysis]) -> List[Tuple[int, int, float]]:
        """
        Phase 1: Fast filtering using embedding similarity.
        
        Returns:
            List of (index_a, index_b, similarity) tuples
        """
        print("\n🔍 Finding candidate matches using embeddings...")
        
        candidates = []
        
        for i, lm_a in enumerate(landmarks_a):
            for j, lm_b in enumerate(landmarks_b):
                # Compute cosine similarity
                emb_a = lm_a.embedding
                emb_b = lm_b.embedding
                
                similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
                
                if similarity >= self.embedding_threshold:
                    candidates.append((i, j, similarity))
        
        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        print(f"  Found {len(candidates)} candidate matches (similarity >= {self.embedding_threshold})")
        return candidates
    
    def verify_match_with_llm(self, landmark_a: FrameAnalysis, landmark_b: FrameAnalysis,
                             context_a: str = "", context_b: str = "") -> Tuple[float, str]:
        """
        Phase 2: Slow, deep reasoning using LLM.
        
        This is where you'd integrate with GPT-4 or another powerful LLM.
        For now, we use FastVLM itself to demonstrate the concept.
        
        Args:
            landmark_a: First landmark observation
            landmark_b: Second landmark observation
            context_a: Additional context (e.g., "observed in Summer")
            context_b: Additional context (e.g., "observed in Winter")
        
        Returns:
            (probability, reasoning) where probability is 0.0 to 1.0
        """
        if self.use_external_llm:
            # Placeholder for external LLM API call
            return self._verify_with_external_llm(landmark_a, landmark_b, context_a, context_b)
        else:
            # Use FastVLM itself (less ideal but demonstrates the concept)
            return self._verify_with_fastvlm(landmark_a, landmark_b, context_a, context_b)
    
    def _verify_with_fastvlm(self, landmark_a: FrameAnalysis, landmark_b: FrameAnalysis,
                            context_a: str, context_b: str) -> Tuple[float, str]:
        """
        Use FastVLM to verify if two landmarks are the same.
        
        Note: This is a workaround. In production, you'd use a more powerful LLM
        like GPT-4 which is better at this kind of reasoning.
        """
        # Create a text-only reasoning prompt
        # Since FastVLM needs an image, we'll create a dummy prompt and parse the response
        
        prompt = f"""Two landmarks were observed at different times. Analyze if they are the same physical object:

Landmark A {context_a}:
"{landmark_a.description}"

Landmark B {context_b}:
"{landmark_b.description}"

Question: What is the probability (from 0.0 to 1.0) that these descriptions refer to the same physical landmark observed under different conditions? Consider that appearance may change due to lighting, season, or viewing angle, but the core identity should remain.

Respond with ONLY a number between 0.0 and 1.0, followed by a brief explanation."""
        
        # We need an image for FastVLM, so we'll use a simple text-based heuristic instead
        # In a real implementation, you'd call an external LLM API here
        
        # Simple heuristic: look for common object types
        common_objects = set(landmark_a.objects) & set(landmark_b.objects)
        
        # If they share objects and descriptions have some overlap, higher probability
        desc_a_words = set(landmark_a.description.lower().split())
        desc_b_words = set(landmark_b.description.lower().split())
        word_overlap = len(desc_a_words & desc_b_words) / max(len(desc_a_words), len(desc_b_words))
        
        # Combine signals
        if len(common_objects) > 0:
            probability = min(0.5 + word_overlap * 0.5, 1.0)
            reasoning = f"Shared objects: {common_objects}. Word overlap: {word_overlap:.2f}"
        else:
            probability = word_overlap * 0.5
            reasoning = f"No shared objects. Word overlap: {word_overlap:.2f}"
        
        return probability, reasoning
    
    def _verify_with_external_llm(self, landmark_a: FrameAnalysis, landmark_b: FrameAnalysis,
                                  context_a: str, context_b: str) -> Tuple[float, str]:
        """
        Placeholder for external LLM API integration.
        
        In production, you would:
        1. Format the prompt for GPT-4/Claude/Gemini
        2. Call the API
        3. Parse the response to extract probability and reasoning
        """
        # Example with OpenAI API (requires openai package and API key)
        # import openai
        # 
        # prompt = f"""..."""  # Your carefully engineered prompt
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # 
        # # Parse response to extract probability and reasoning
        # ...
        
        raise NotImplementedError("External LLM integration not yet implemented. Set use_external_llm=False.")
    
    def match_landmarks(self, 
                       landmarks_a: List[FrameAnalysis], 
                       landmarks_b: List[FrameAnalysis],
                       context_a: str = "Video A",
                       context_b: str = "Video B") -> List[LandmarkMatch]:
        """
        Full two-phase matching pipeline:
        1. Fast embedding-based filtering
        2. LLM-based semantic verification
        
        Args:
            landmarks_a: Landmarks from first video/session
            landmarks_b: Landmarks from second video/session
            context_a: Description of first context (e.g., "Summer 2023")
            context_b: Description of second context (e.g., "Winter 2024")
        
        Returns:
            List of verified landmark matches
        """
        # Phase 1: Fast filtering
        candidates = self.find_candidate_matches(landmarks_a, landmarks_b)
        
        if not candidates:
            print("  No candidates found!")
            return []
        
        # Phase 2: LLM verification
        print(f"\n🤔 Verifying {min(len(candidates), 10)} top candidates with LLM reasoning...")
        print("  (This would use GPT-4 in production; using heuristic for demo)")
        
        matches = []
        
        # Verify top candidates (limit to avoid slow processing)
        for i, j, emb_sim in tqdm(candidates[:10], desc="Verifying"):
            lm_a = landmarks_a[i]
            lm_b = landmarks_b[j]
            
            prob, reasoning = self.verify_match_with_llm(
                lm_a, lm_b, 
                context_a=f"from {context_a}",
                context_b=f"from {context_b}"
            )
            
            match = LandmarkMatch(
                landmark_a=lm_a,
                landmark_b=lm_b,
                embedding_similarity=emb_sim,
                llm_verification_score=prob,
                llm_reasoning=reasoning
            )
            
            matches.append(match)
        
        # Sort by LLM verification score
        matches.sort(key=lambda m: m.llm_verification_score, reverse=True)
        
        return matches


def extract_landmarks_from_video(analyzer: FastVLMVideoAnalyzer, 
                                 video_path: str,
                                 sample_interval: float = 5.0,
                                 max_frames: int = 10) -> List[FrameAnalysis]:
    """Extract landmarks from a video at regular intervals"""
    print(f"\n📹 Processing video: {video_path}")
    
    video = VideoProcessor(video_path)
    frames = video.extract_frames_at_interval(sample_interval)
    
    # Limit frames for faster prototyping
    frames = frames[:max_frames]
    
    print(f"  Extracting landmarks from {len(frames)} frames...")
    
    landmarks = []
    for frame_num, timestamp_ms, img in tqdm(frames, desc="Analyzing frames"):
        analysis = analyzer.extract_landmarks(img)
        analysis.frame_number = frame_num
        analysis.timestamp_ms = timestamp_ms
        landmarks.append(analysis)
    
    return landmarks


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Temporal Landmark Matching (Phase 2 of Semantic SLAM)"
    )
    parser.add_argument("--video1", type=str, required=True,
                       help="Path to first video (e.g., summer)")
    parser.add_argument("--video2", type=str, required=True,
                       help="Path to second video (e.g., winter)")
    parser.add_argument("--model-path", type=str,
                       default="/Users/kausar/Documents/ml-fastvlm/checkpoints/llava-fastvithd_1.5b_stage2",
                       help="Path to FastVLM model")
    parser.add_argument("--context1", type=str, default="Video 1",
                       help="Description of first video context (e.g., 'Summer 2023')")
    parser.add_argument("--context2", type=str, default="Video 2",
                       help="Description of second video context (e.g., 'Winter 2024')")
    parser.add_argument("--output-dir", type=str, default="./matching_results",
                       help="Directory to save results")
    parser.add_argument("--sample-interval", type=float, default=5.0,
                       help="Sample frames every N seconds")
    parser.add_argument("--max-frames", type=int, default=10,
                       help="Maximum frames to process per video (for faster prototyping)")
    parser.add_argument("--embedding-threshold", type=float, default=0.7,
                       help="Minimum embedding similarity for candidate matches")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device: mps, cuda, or cpu")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*60)
    print("Cross-Temporal Landmark Matching")
    print("="*60)
    print(f"\nVideo 1 ({args.context1}): {args.video1}")
    print(f"Video 2 ({args.context2}): {args.video2}")
    
    # Initialize FastVLM
    analyzer = FastVLMVideoAnalyzer(args.model_path, device=args.device)
    
    # Extract landmarks from both videos
    landmarks_1 = extract_landmarks_from_video(
        analyzer, args.video1, args.sample_interval, args.max_frames
    )
    landmarks_2 = extract_landmarks_from_video(
        analyzer, args.video2, args.sample_interval, args.max_frames
    )
    
    print(f"\n✓ Extracted {len(landmarks_1)} landmarks from {args.context1}")
    print(f"✓ Extracted {len(landmarks_2)} landmarks from {args.context2}")
    
    # Match landmarks
    matcher = CrossTemporalMatcher(
        analyzer, 
        embedding_threshold=args.embedding_threshold,
        use_external_llm=False  # Set to True if you have LLM API access
    )
    
    matches = matcher.match_landmarks(
        landmarks_1, landmarks_2,
        context_a=args.context1,
        context_b=args.context2
    )
    
    # Display results
    print("\n" + "="*60)
    print("MATCHING RESULTS")
    print("="*60)
    
    high_conf_matches = [m for m in matches if m.is_high_confidence()]
    
    print(f"\nTotal candidate matches: {len(matches)}")
    print(f"High-confidence matches (≥0.8): {len(high_conf_matches)}")
    
    print("\n--- Top 5 Matches ---")
    for i, match in enumerate(matches[:5], 1):
        print(f"\n#{i} - Confidence: {match.llm_verification_score:.2f}, "
              f"Embedding Sim: {match.embedding_similarity:.2f}")
        print(f"  {args.context1} (frame {match.landmark_a.frame_number}):")
        print(f"    '{match.landmark_a.description[:80]}...'")
        print(f"  {args.context2} (frame {match.landmark_b.frame_number}):")
        print(f"    '{match.landmark_b.description[:80]}...'")
        print(f"  Reasoning: {match.llm_reasoning}")
    
    # Save results
    results_file = output_dir / "landmark_matches.json"
    with open(results_file, 'w') as f:
        json.dump({
            'context_1': args.context1,
            'context_2': args.context2,
            'video_1': args.video1,
            'video_2': args.video2,
            'num_landmarks_1': len(landmarks_1),
            'num_landmarks_2': len(landmarks_2),
            'total_matches': len(matches),
            'high_confidence_matches': len(high_conf_matches),
            'matches': [m.to_dict() for m in matches]
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Integrate with a powerful LLM (GPT-4, Claude, Gemini) for better verification
2. Add geometric consistency checks using SLAM pose estimates
3. Build a unified landmark database with multiple appearance descriptors
4. Test on ROVER dataset's seasonal sequences
5. Implement pose graph optimization with semantic constraints
    """)


if __name__ == "__main__":
    main()
