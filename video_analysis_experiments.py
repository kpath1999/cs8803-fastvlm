#!/usr/bin/env python3
"""
FastVLM Video Analysis Experiments
==================================

This script demonstrates FastVLM's capabilities for video analysis, with experiments
designed as stepping stones toward semantic SLAM and multi-agent mapping.

Experiments:
1. Frame-by-Frame Semantic Extraction
2. Temporal Change Detection
3. Semantic Keyframe Selection
4. Object Tracking via Description Matching
5. Scene Graph Construction

Usage:
    python video_analysis_experiments.py --video-path <path_to_video> --experiment <experiment_name>
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

import torch
import cv2
from PIL import Image
from tqdm import tqdm

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


@dataclass
class FrameAnalysis:
    """Stores the analysis results for a single video frame"""
    frame_number: int
    timestamp_ms: float
    description: str
    landmarks: List[str]  # Extracted landmark descriptions
    objects: List[str]    # Detected objects
    spatial_layout: str   # Overall scene layout description
    embedding: Optional[np.ndarray] = None  # Vision encoder embedding
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        if self.embedding is not None:
            d['embedding'] = self.embedding.tolist()
        return d


@dataclass
class TemporalChange:
    """Represents a detected change between frames"""
    frame_from: int
    frame_to: int
    change_description: str
    confidence: float


class FastVLMVideoAnalyzer:
    """Wrapper for FastVLM model tailored for video analysis"""
    
    def __init__(self, model_path: str, device: str = "mps"):
        print("Initializing FastVLM Video Analyzer...")
        disable_torch_init()
        
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(model_path, None, model_name, device=device)
        
        self.device = device
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        print(f"Model loaded on device: {device}")
    
    def get_vision_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extract the vision encoder embedding for an image.
        This is useful for comparing visual similarity between frames.
        """
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        
        with torch.no_grad():
            # Get vision tower features
            vision_tower = self.model.get_vision_tower()
            image_features = vision_tower(image_tensor.unsqueeze(0).half().to(self.device))
            
            # Average pool to get a global descriptor
            embedding = image_features.mean(dim=(1, 2)).squeeze().cpu().numpy()
        
        return embedding
    
    def analyze_frame(self, image: Image.Image, prompt: str, 
                     conv_mode: str = "qwen_2", temperature: float = 0.2) -> str:
        """
        Run FastVLM inference on a single frame with a custom prompt.
        """
        # Construct prompt
        qs = prompt
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_formatted, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        
        # Process image
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(self.device),
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=256,
                use_cache=True
            )
            
            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        return output
    
    def extract_landmarks(self, image: Image.Image) -> FrameAnalysis:
        """
        Extract semantic landmarks from a frame.
        This is the foundation for Phase 1 of your SLAM system.
        """
        # Get detailed landmark description
        landmark_prompt = (
            "Describe up to five distinct, stationary landmarks visible in this image. "
            "For each landmark, provide: 1) what it is, 2) its distinctive features, "
            "3) its approximate location in the scene (left/right/center, near/far). "
            "Focus on permanent features like trees, rocks, structures, or distinctive terrain."
        )
        landmark_desc = self.analyze_frame(image, landmark_prompt)
        
        # Get object list
        object_prompt = "List all visible objects in this image, separated by commas."
        objects_str = self.analyze_frame(image, object_prompt, temperature=0.0)
        
        # Get spatial layout
        layout_prompt = "Describe the overall spatial layout of this scene in one sentence."
        layout = self.analyze_frame(image, layout_prompt, temperature=0.0)
        
        # Get embedding
        embedding = self.get_vision_embedding(image)
        
        # Parse outputs (simple parsing - could be enhanced with LLM)
        objects = [obj.strip() for obj in objects_str.split(',') if obj.strip()]
        
        return FrameAnalysis(
            frame_number=-1,  # Set by caller
            timestamp_ms=-1,  # Set by caller
            description=landmark_desc,
            landmarks=[landmark_desc],  # For now, treat whole description as one landmark
            objects=objects,
            spatial_layout=layout,
            embedding=embedding
        )
    
    def detect_change(self, frame1: Image.Image, frame2: Image.Image) -> str:
        """
        Detect what changed between two frames using VLM reasoning.
        """
        # Create a composite prompt with both frame descriptions
        desc1 = self.analyze_frame(frame1, "Describe this image in detail.", temperature=0.0)
        desc2 = self.analyze_frame(frame2, "Describe this image in detail.", temperature=0.0)
        
        # In a real implementation, you'd feed both descriptions to an LLM
        # For now, return both descriptions for manual comparison
        change_summary = f"Frame 1: {desc1}\nFrame 2: {desc2}"
        return change_summary


class VideoProcessor:
    """Handles video file I/O and frame extraction"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video Info: {self.total_frames} frames @ {self.fps} fps, {self.width}x{self.height}")
    
    def extract_frame(self, frame_number: int) -> Optional[Image.Image]:
        """Extract a specific frame from the video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        # Convert BGR to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    
    def extract_frames_at_interval(self, interval_seconds: float) -> List[Tuple[int, float, Image.Image]]:
        """Extract frames at regular time intervals"""
        interval_frames = int(interval_seconds * self.fps)
        frames = []
        
        for frame_num in range(0, self.total_frames, interval_frames):
            img = self.extract_frame(frame_num)
            if img is not None:
                timestamp_ms = (frame_num / self.fps) * 1000
                frames.append((frame_num, timestamp_ms, img))
        
        return frames
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_1_frame_extraction(analyzer: FastVLMVideoAnalyzer, video: VideoProcessor, 
                                  output_dir: Path, interval: float = 2.0):
    """
    Experiment 1: Frame-by-Frame Semantic Extraction
    
    Extract semantic descriptions from video frames at regular intervals.
    This demonstrates FastVLM's ability to understand individual frames.
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Frame-by-Frame Semantic Extraction")
    print(f"{'='*60}\n")
    
    frames = video.extract_frames_at_interval(interval)
    print(f"Analyzing {len(frames)} frames (one every {interval}s)...")
    
    results = []
    for frame_num, timestamp_ms, img in tqdm(frames):
        analysis = analyzer.extract_landmarks(img)
        analysis.frame_number = frame_num
        analysis.timestamp_ms = timestamp_ms
        results.append(analysis)
        
        # Print sample
        if len(results) <= 3:
            print(f"\nFrame {frame_num} ({timestamp_ms/1000:.1f}s):")
            print(f"  Landmarks: {analysis.description[:100]}...")
            print(f"  Objects: {', '.join(analysis.objects[:5])}")
    
    # Save results
    output_file = output_dir / "experiment_1_frame_extraction.json"
    with open(output_file, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    return results


def experiment_2_temporal_changes(analyzer: FastVLMVideoAnalyzer, video: VideoProcessor,
                                  output_dir: Path, interval: float = 1.0):
    """
    Experiment 2: Temporal Change Detection
    
    Detect what changes between consecutive frames. This is crucial for understanding
    dynamic scenes and identifying when new landmarks appear.
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Temporal Change Detection")
    print(f"{'='*60}\n")
    
    frames = video.extract_frames_at_interval(interval)
    print(f"Analyzing changes across {len(frames)} frames...")
    
    changes = []
    for i in tqdm(range(len(frames) - 1)):
        frame_num1, ts1, img1 = frames[i]
        frame_num2, ts2, img2 = frames[i + 1]
        
        # Simple approach: describe each frame and compare
        desc1 = analyzer.analyze_frame(img1, "Describe what you see in one sentence.", temperature=0.0)
        desc2 = analyzer.analyze_frame(img2, "Describe what you see in one sentence.", temperature=0.0)
        
        change = {
            'frame_from': frame_num1,
            'frame_to': frame_num2,
            'time_from_s': ts1 / 1000,
            'time_to_s': ts2 / 1000,
            'description_1': desc1,
            'description_2': desc2,
        }
        changes.append(change)
        
        if i < 2:
            print(f"\nFrame {frame_num1} -> {frame_num2}:")
            print(f"  Before: {desc1}")
            print(f"  After:  {desc2}")
    
    output_file = output_dir / "experiment_2_temporal_changes.json"
    with open(output_file, 'w') as f:
        json.dump(changes, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    return changes


def experiment_3_keyframe_selection(analyzer: FastVLMVideoAnalyzer, video: VideoProcessor,
                                    output_dir: Path, sample_interval: float = 0.5):
    """
    Experiment 3: Semantic Keyframe Selection
    
    Select keyframes based on visual similarity (using embeddings). This is important
    for efficient SLAM - you don't want to process every single frame.
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: Semantic Keyframe Selection")
    print(f"{'='*60}\n")
    
    frames = video.extract_frames_at_interval(sample_interval)
    print(f"Sampling {len(frames)} frames to find keyframes...")
    
    # Extract embeddings for all frames
    embeddings = []
    for frame_num, timestamp_ms, img in tqdm(frames, desc="Extracting embeddings"):
        emb = analyzer.get_vision_embedding(img)
        embeddings.append((frame_num, timestamp_ms, emb))
    
    # Select keyframes based on embedding distance
    keyframes = [embeddings[0]]  # Always include first frame
    similarity_threshold = 0.95  # Cosine similarity threshold
    
    for i in range(1, len(embeddings)):
        frame_num, ts, emb = embeddings[i]
        
        # Compare with last keyframe
        last_kf_emb = keyframes[-1][2]
        similarity = np.dot(emb, last_kf_emb) / (np.linalg.norm(emb) * np.linalg.norm(last_kf_emb))
        
        # If sufficiently different, it's a new keyframe
        if similarity < similarity_threshold:
            keyframes.append((frame_num, ts, emb))
            print(f"  Keyframe selected: frame {frame_num} (similarity: {similarity:.3f})")
    
    print(f"\n✓ Selected {len(keyframes)} keyframes from {len(frames)} frames")
    
    # Save keyframe info (without embeddings to keep file small)
    keyframe_info = [
        {'frame_number': kf[0], 'timestamp_ms': kf[1]}
        for kf in keyframes
    ]
    
    output_file = output_dir / "experiment_3_keyframes.json"
    with open(output_file, 'w') as f:
        json.dump(keyframe_info, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")
    return keyframes


def experiment_4_object_tracking(analyzer: FastVLMVideoAnalyzer, video: VideoProcessor,
                                output_dir: Path, target_object: str = "animal"):
    """
    Experiment 4: Object Tracking via Description Matching
    
    Track a specific object (e.g., the animal) across frames by matching descriptions.
    This is a precursor to landmark matching across different viewing conditions.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 4: Object Tracking - '{target_object}'")
    print(f"{'='*60}\n")
    
    frames = video.extract_frames_at_interval(1.0)
    print(f"Tracking '{target_object}' across {len(frames)} frames...")
    
    tracking_results = []
    for frame_num, timestamp_ms, img in tqdm(frames):
        # Ask specifically about the target object
        prompt = f"Is there an {target_object} visible in this image? If yes, describe its location and what it's doing. If no, say 'No {target_object} visible.'"
        response = analyzer.analyze_frame(img, prompt, temperature=0.0)
        
        tracking_results.append({
            'frame_number': frame_num,
            'timestamp_ms': timestamp_ms,
            'tracking_response': response,
            'object_detected': target_object.lower() in response.lower() and 'no' not in response.lower()[:10]
        })
    
    # Summary
    detected_frames = [r for r in tracking_results if r['object_detected']]
    print(f"\n✓ '{target_object}' detected in {len(detected_frames)}/{len(frames)} frames")
    
    output_file = output_dir / f"experiment_4_tracking_{target_object}.json"
    with open(output_file, 'w') as f:
        json.dump(tracking_results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")
    return tracking_results


def experiment_5_scene_graph(analyzer: FastVLMVideoAnalyzer, video: VideoProcessor,
                             output_dir: Path):
    """
    Experiment 5: Scene Graph Construction
    
    Build a structured representation of the scene by extracting objects and
    their spatial relationships. This is the "unified scene graph" you mentioned.
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT 5: Scene Graph Construction")
    print(f"{'='*60}\n")
    
    # Analyze first frame to build initial scene graph
    first_frame = video.extract_frame(0)
    
    # Extract objects
    objects_prompt = "List all objects, living things, and landmarks visible in this scene. Format as a bullet list."
    objects_response = analyzer.analyze_frame(first_frame, objects_prompt)
    
    # Extract spatial relationships
    spatial_prompt = "Describe the spatial relationships between objects in this scene. Use terms like 'in front of', 'behind', 'to the left of', etc."
    spatial_response = analyzer.analyze_frame(first_frame, spatial_prompt)
    
    # Build scene graph structure
    scene_graph = {
        'frame_number': 0,
        'timestamp_ms': 0.0,
        'objects': objects_response,
        'spatial_relationships': spatial_response,
        'metadata': {
            'video_path': video.video_path,
            'analysis_date': datetime.now().isoformat()
        }
    }
    
    print("\nScene Graph:")
    print(f"\nObjects:\n{objects_response}")
    print(f"\nSpatial Relationships:\n{spatial_response}")
    
    output_file = output_dir / "experiment_5_scene_graph.json"
    with open(output_file, 'w') as f:
        json.dump(scene_graph, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    return scene_graph


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FastVLM Video Analysis Experiments")
    parser.add_argument("--video-path", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--model-path", type=str,
                       default="/Users/kausar/Documents/ml-fastvlm/checkpoints/llava-fastvithd_1.5b_stage2",
                       help="Path to FastVLM model checkpoint")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=['all', '1', '2', '3', '4', '5'],
                       help="Which experiment to run (default: all)")
    parser.add_argument("--output-dir", type=str, default="./video_analysis_output",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device to run on (mps, cuda, cpu)")
    parser.add_argument("--interval", type=float, default=2.0,
                       help="Frame sampling interval in seconds (for applicable experiments)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize
    print("\n" + "="*60)
    print("FastVLM Video Analysis Experiments")
    print("="*60)
    
    analyzer = FastVLMVideoAnalyzer(args.model_path, device=args.device)
    video = VideoProcessor(args.video_path)
    
    # Run experiments
    experiments = {
        '1': lambda: experiment_1_frame_extraction(analyzer, video, output_dir, args.interval),
        '2': lambda: experiment_2_temporal_changes(analyzer, video, output_dir, args.interval),
        '3': lambda: experiment_3_keyframe_selection(analyzer, video, output_dir),
        '4': lambda: experiment_4_object_tracking(analyzer, video, output_dir),
        '5': lambda: experiment_5_scene_graph(analyzer, video, output_dir),
    }
    
    if args.experiment == 'all':
        for exp_num in sorted(experiments.keys()):
            experiments[exp_num]()
    else:
        experiments[args.experiment]()
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
