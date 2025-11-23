#!/usr/bin/env python3
"""
*** HIERARCHICAL CROSS-TEMPORAL LANDMARK MATCHING PIPELINE ***

This script implements a rigorous 7-stage pipeline for matching landmarks across
seasonal datasets (winter 2024-01-13 and autumn 2024-04-11). The pipeline progresses
from broad semantic matching to precise geometric verification.

PIPELINE STAGES:
================

Stage 1: IMAGE STREAM INGESTION
    - Load corresponding RGB frames from winter and autumn datasets
    - Establish temporal correspondence between datasets
    - Output: Paired frame candidates for matching

Stage 2: OPEN-VOCABULARY OBJECT DETECTION
    - Apply OWL-ViT to detect objects of interest in both seasons
    - Generate bounding boxes for candidate landmarks
    - Output: Detected objects with confidence scores and spatial locations

Stage 3: SEMANTIC ENRICHMENT WITH FASTVLM
    - Extract high-level semantic descriptions for whole scenes
    - Generate detailed descriptions for each detected object
    - Compute visual embeddings for similarity comparison
    - Output: Rich semantic metadata for each detection

Stage 4: DEPTH-BASED SPATIAL VALIDATION
    - Load depth frames corresponding to RGB images
    - Overlay depth information on detected landmarks
    - Compute 3D spatial layout consistency across seasons
    - Output: Depth-augmented confidence scores for matches

Stage 5: GEOMETRIC KEYPOINT MATCHING
    - Extract keypoints within matched bounding boxes (SIFT/ORB/SuperPoint)
    - Match keypoints across seasons using robust matchers
    - Apply RANSAC to filter outlier correspondences
    - Output: Geometrically verified landmark matches

Stage 6: VISUALIZATION AND RESULT GENERATION
    - Create side-by-side comparisons of matched frames
    - Apply color-coded bounding boxes (green=invariant, red=temporal)
    - Annotate matches with confidence scores and reasoning
    - Output: Visual results in pipeline_results/ directory

Stage 7: SEMANTIC SEGMENTATION (STRETCH GOAL)
    - Apply semantic segmentation to both frames
    - Identify invariant vs. temporal scene elements
    - Visualize changes at pixel-level granularity
    - Output: Segmentation masks highlighting seasonal changes

USAGE:
======
# Recommended: Fast keyframe selection with focused object detection
python cross_temporal_pipeline.py \
    --winter-rgb /Volumes/KAUSAR/rover_dataset/2024-01-13/realsense_D435i/rgb \
    --winter-depth /Volumes/KAUSAR/rover_dataset/2024-01-13/realsense_D435i/depth \
    --autumn-rgb /Volumes/KAUSAR/rover_dataset/2024-04-11/realsense_D435i/rgb \
    --autumn-depth /Volumes/KAUSAR/rover_dataset/2024-04-11/realsense_D435i/depth \
    --model-path checkpoints/llava-fastvithd_0.5b_stage2 \
    --output-dir pipeline_results \
    --use-keyframing \
    --adaptive-thresholds \
    --focused-queries \
    --max-pairs 100

# Basic: Process all frames without keyframe selection
python cross_temporal_pipeline.py \
    --winter-rgb /path/to/winter/rgb \
    --winter-depth /path/to/winter/depth \
    --autumn-rgb /path/to/autumn/rgb \
    --autumn-depth /path/to/autumn/depth \
    --model-path checkpoints/llava-fastvithd_0.5b_stage2 \
    --output-dir pipeline_results \
    --max-pairs 10

KEY OPTIONS:
  --use-keyframing         Enable fast histogram-based keyframe selection (highly recommended for large datasets)
  --adaptive-thresholds    Use per-category detection thresholds for better accuracy
  --focused-queries        Use only permanent landmark queries (house, tree, etc.)
  --max-pairs N            Process at most N frame pairs (useful for testing)
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


# ---------------------------------------------------------------------------
# Warning management
# ---------------------------------------------------------------------------

try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning
except ImportError:  # pragma: no cover - environments without pydantic
    UnsupportedFieldAttributeWarning = None

if UnsupportedFieldAttributeWarning is not None:
    warnings.filterwarnings(
        "ignore",
        message=r"The 'repr' attribute.*Field\(\) function",
        category=UnsupportedFieldAttributeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The 'frozen' attribute.*Field\(\) function",
        category=UnsupportedFieldAttributeWarning,
    )


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================


@dataclass
class Detection:
    """Represents a single object detection with semantic metadata."""
    
    label: str              # Object category (e.g., "tree", "building")
    bbox: List[float]       # Bounding box [x1, y1, x2, y2]
    score: float            # Detection confidence [0, 1]
    description: str        # FastVLM semantic description
    embedding: np.ndarray   # Visual feature embedding
    depth_stats: Optional[Dict[str, float]] = None  # Depth statistics within bbox
    keypoints: Optional[List] = None  # Matched keypoints (if any)
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        data = asdict(self)
        data['embedding'] = self.embedding.tolist() if self.embedding is not None else None
        return data


@dataclass
class LandmarkMatch:
    """Represents a matched landmark across two temporal frames."""
    
    winter_detection: Detection
    autumn_detection: Detection
    
    # Similarity scores from different matching stages
    embedding_similarity: float     # Cosine similarity of visual embeddings
    semantic_similarity: float      # Text-based semantic similarity
    depth_consistency: float        # Depth layout consistency score
    geometric_confidence: float     # Keypoint matching confidence
    
    # Overall match confidence (weighted combination)
    final_confidence: float
    
    # Classification of this landmark
    is_invariant: bool  # True if landmark is permanent, False if temporal
    
    # Explanation for human interpretation
    reasoning: str
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'winter': self.winter_detection.to_dict(),
            'autumn': self.autumn_detection.to_dict(),
            'scores': {
                'embedding_similarity': float(self.embedding_similarity),
                'semantic_similarity': float(self.semantic_similarity),
                'depth_consistency': float(self.depth_consistency),
                'geometric_confidence': float(self.geometric_confidence),
                'final_confidence': float(self.final_confidence),
            },
            'is_invariant': self.is_invariant,
            'reasoning': self.reasoning,
        }


@dataclass
class FramePairAnalysis:
    """Results from analyzing a pair of winter/autumn frames."""
    
    winter_frame_id: str
    autumn_frame_id: str
    winter_image_path: str
    autumn_image_path: str
    
    # All detections in each frame
    winter_detections: List[Detection]
    autumn_detections: List[Detection]
    
    # Matched landmarks between frames
    matches: List[LandmarkMatch]
    
    # Unmatched detections (only in one season)
    winter_only: List[Detection]
    autumn_only: List[Detection]
    
    # Visualization paths
    visualization_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'winter_frame_id': self.winter_frame_id,
            'autumn_frame_id': self.autumn_frame_id,
            'winter_image_path': self.winter_image_path,
            'autumn_image_path': self.autumn_image_path,
            'matches': [m.to_dict() for m in self.matches],
            'winter_only': [d.to_dict() for d in self.winter_only],
            'autumn_only': [d.to_dict() for d in self.autumn_only],
            'visualization_path': self.visualization_path,
            'summary': {
                'total_matches': len(self.matches),
                'invariant_landmarks': sum(1 for m in self.matches if m.is_invariant),
                'temporal_objects': sum(1 for m in self.matches if not m.is_invariant),
                'winter_unique': len(self.winter_only),
                'autumn_unique': len(self.autumn_only),
            }
        }


# ===========================================================================
# STAGE 2: OPEN-VOCABULARY OBJECT DETECTION
# ===========================================================================


class OpenVocabularyDetector:
    """
    OWL-ViT based detector for open-vocabulary object detection.
    
    This allows us to detect a wide range of objects using natural language
    queries rather than being limited to a fixed set of categories.
    """

    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        device: str = "cpu",
        score_threshold: float = 0.2,
        use_adaptive_thresholds: bool = False,
    ):
        """
        Initialize the detector.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run inference on (cpu/cuda/mps)
            score_threshold: Minimum confidence for detections (0.2 is good for garden scenes)
            use_adaptive_thresholds: If True, use per-category thresholds instead of global
        """
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.score_threshold = score_threshold
        self.use_adaptive_thresholds = use_adaptive_thresholds
        
        print(f"      OWL-ViT loaded on device: {self.device}")
        
        if use_adaptive_thresholds:
            try:
                from adaptive_threshold_config import get_threshold
                self.get_threshold = get_threshold
                print("      Using adaptive per-category thresholds")
            except ImportError:
                print("      WARNING: adaptive_threshold_config not found, using global threshold")
                self.use_adaptive_thresholds = False

    def detect(
        self, 
        image: Image.Image, 
        queries: Sequence[str]
    ) -> List[Dict]:
        """
        Detect objects in an image based on text queries.
        
        Args:
            image: Input PIL Image
            queries: List of object categories to detect
            
        Returns:
            List of detections with label, score, and bbox
        """
        # DEBUG
        print(f"      DEBUG [OWL-ViT]: Processing image {image.size} with {len(queries)} queries")
        
        inputs = self.processor(text=list(queries), images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert to image coordinates
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=0.0
        )[0]

        detections = []
        all_scores = []
        all_detections_with_labels = []  # Store ALL detections with labels for analysis
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score_value = score.item()
            label_text = queries[label.item()]
            bbox = box.tolist()
            
            all_scores.append(score_value)
            all_detections_with_labels.append({
                "label": label_text,
                "score": score_value,
                "bbox": bbox,
            })
            
            # Determine threshold for this category
            if self.use_adaptive_thresholds:
                category_threshold = self.get_threshold(label_text)
            else:
                category_threshold = self.score_threshold
            
            if score_value < category_threshold:
                continue
                
            detections.append({
                "label": label_text,
                "score": score_value,
                "bbox": bbox,
            })
        
        # DEBUG - Show top 10 detections with labels to see what's being detected
        print(f"      DEBUG [OWL-ViT]: Found {len(detections)} detections above threshold {self.score_threshold}")
        if len(all_scores) > 0:
            print(f"      DEBUG [OWL-ViT]: Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
            
            # Show top 10 detections WITH LABELS to see what's being detected
            top_10 = sorted(all_detections_with_labels, key=lambda x: x["score"], reverse=True)[:10]
            print(f"      DEBUG [OWL-ViT]: Top 10 detections (label: score):")
            for i, det in enumerate(top_10, 1):
                status = "✓" if det["score"] >= self.score_threshold else "✗"
                print(f"        {i}. {status} {det['label']}: {det['score']:.3f}")
        
        # Apply per-category NMS: keep only highest-scoring detection per category
        detections = self._apply_per_category_nms(detections)
        print(f"      DEBUG [OWL-ViT]: After per-category NMS: {len(detections)} unique detections")
        
        if len(detections) > 0:
            print(f"      DEBUG [OWL-ViT]: ✓ {len(detections)} unique detections")
            # Show which categories were kept
            kept_categories = [f"{d['label']}({d['score']:.3f})" for d in sorted(detections, key=lambda x: x['score'], reverse=True)]
            print(f"      DEBUG [OWL-ViT]: Kept: {', '.join(kept_categories)}")
        else:
            print(f"      DEBUG [OWL-ViT]: ⚠️  No detections above threshold {self.score_threshold}!")
            if len(all_detections_with_labels) > 0:
                best = all_detections_with_labels[0]
                print(f"      DEBUG [OWL-ViT]: Best detection was '{best['label']}' at {best['score']:.3f}")
                print(f"      DEBUG [OWL-ViT]: Consider lowering threshold to {best['score'] - 0.01:.2f}")
        
        return detections
    
    def _apply_per_category_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Keep only the highest-scoring detection per category.
        
        This eliminates duplicate detections of the same object type,
        significantly reducing the number of detections to process.
        
        Args:
            detections: List of detections with label, score, bbox
            
        Returns:
            Filtered list with one detection per category
        """
        if len(detections) == 0:
            return []
        
        # Group by category
        category_detections = {}
        for det in detections:
            label = det["label"]
            if label not in category_detections or det["score"] > category_detections[label]["score"]:
                category_detections[label] = det
        
        # Return only the best detection per category
        return list(category_detections.values())


# ===========================================================================
# STAGE 3: SEMANTIC ENRICHMENT WITH FASTVLM
# ===========================================================================


class FastVLMAnalyzer:
    """
    FastVLM wrapper for semantic scene understanding.
    
    Provides high-level semantic descriptions that complement the geometric
    matching with human-interpretable reasoning.
    """

    def __init__(self, model_path: str, device: str = "mps"):
        """
        Initialize FastVLM model.
        
        Args:
            model_path: Path to FastVLM checkpoint
            device: Device for inference
        """
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, device=device
        )
        self.device = device
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _format_prompt(self, prompt: str) -> str:
        """Add image tokens to prompt based on model configuration."""
        if self.model.config.mm_use_im_start_end:
            return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        return DEFAULT_IMAGE_TOKEN + "\n" + prompt

    def _generate(self, image: Image.Image, prompt: str, temperature: float = 0.2) -> str:
        """
        Generate text response from image and prompt.
        
        Args:
            image: Input PIL Image
            prompt: Text prompt
            temperature: Sampling temperature (0 = deterministic)
            
        Returns:
            Generated text response
        """
        formatted_prompt = self._format_prompt(prompt)
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], formatted_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        with torch.inference_mode():
            gen_kwargs = {
                "images": image_tensor.unsqueeze(0).to(self.device, dtype=torch.float16),
                "image_sizes": [image.size],
                "do_sample": temperature > 0,
                "max_new_tokens": 256,
                "use_cache": True,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
            
            output_ids = self.model.generate(input_ids, **gen_kwargs)

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def describe_object(self, image: Image.Image) -> str:
        """
        Generate detailed description of an object.
        
        Focuses on invariant properties: material, color, shape, texture.
        These features help match landmarks across seasonal changes.
        """
        prompt = (
            "Describe this object in detail, including its material, color, shape, texture, "
            "and any identifying features that would help recognize it across different seasons."
        )
        return self._generate(image, prompt, temperature=0.2)

    def get_vision_embedding(self, image: Image.Image) -> np.ndarray:
        
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        with torch.no_grad():
            vision_tower = self.model.get_vision_tower()
            vision_feats = vision_tower(
                image_tensor.unsqueeze(0).to(self.device, dtype=torch.float16)
            )

            # Some FastVLM implementations return dicts
            if isinstance(vision_feats, dict):
                vision_feats = vision_feats.get("image_features", vision_feats["feats"])

            # Case 1: (B, C, H, W)
            if vision_feats.ndim == 4:
                B, C, H, W = vision_feats.shape
                vision_feats = vision_feats.permute(0, 2, 3, 1).reshape(B, H * W, C)

            # Case 2: (B, S, D)
            if vision_feats.ndim == 3:
                embedding = vision_feats.mean(dim=1)[0].cpu().numpy().astype(np.float32)
                return embedding

            # Case 3: (B, D)
            if vision_feats.ndim == 2:
                return vision_feats[0].cpu().numpy().astype(np.float32)

            # Anything else is unexpected
            raise ValueError(f"Unexpected vision_feats shape: {vision_feats.shape}")

    
    def compare_objects(self, desc1: str, desc2: str) -> Tuple[float, str]:
        """
        Compare semantic similarity of two object descriptions.
        
        Uses a simple heuristic: count shared keywords weighted by importance.
        More sophisticated approaches could use sentence embeddings.
        
        Returns:
            Similarity score [0, 1] and explanation
        """
        # Keywords that indicate permanent vs temporal objects
        permanent_keywords = ['building', 'structure', 'brick', 'concrete', 'metal', 
                            'stone', 'pole', 'sign', 'pavement', 'sidewalk']
        temporal_keywords = ['tree', 'leaves', 'grass', 'flowers', 'car', 'vehicle',
                           'person', 'snow', 'green', 'brown']
        
        desc1_lower = desc1.lower()
        desc2_lower = desc2.lower()
        
        # Extract common words
        words1 = set(desc1_lower.split())
        words2 = set(desc2_lower.split())
        common_words = words1 & words2
        
        # Simple Jaccard similarity
        if len(words1) == 0 or len(words2) == 0:
            return 0.0, "Empty descriptions"
        
        similarity = len(common_words) / len(words1 | words2)
        
        # Check for permanent vs temporal indicators
        perm_count = sum(1 for kw in permanent_keywords if kw in desc1_lower and kw in desc2_lower)
        temp_count = sum(1 for kw in temporal_keywords if kw in desc1_lower or kw in desc2_lower)
        
        explanation = f"Shared words: {len(common_words)}, Permanent indicators: {perm_count}"
        
        return similarity, explanation


# ===========================================================================
# STAGE 4: DEPTH-BASED SPATIAL VALIDATION
# ===========================================================================


class DepthValidator:
    """
    Validates landmark matches using depth information.
    
    Depth data provides geometric constraints: objects at the same location
    should have similar depth profiles across seasons.
    """
    
    @staticmethod
    def load_depth_image(depth_path: str) -> Optional[np.ndarray]:
        """
        Load depth image from file.
        
        Assumes depth is stored as 16-bit PNG or similar format.
        
        Returns:
            Depth array or None if loading fails
        """
        if not os.path.exists(depth_path):
            return None
        
        try:
            # Load as grayscale
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth is None:
                # Try loading as regular image if depth format fails
                depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            return depth
        except Exception as e:
            print(f"Warning: Could not load depth image {depth_path}: {e}")
            return None
    
    @staticmethod
    def extract_depth_stats(
        depth: np.ndarray, 
        bbox: Sequence[float]
    ) -> Dict[str, float]:
        """
        Compute depth statistics within a bounding box.
        
        Args:
            depth: Depth image array
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with mean, median, std of depth values
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth.shape[1] - 1, x2)
        y2 = min(depth.shape[0] - 1, y2)
        
        depth_crop = depth[y1:y2, x1:x2]
        
        # Filter out invalid depth values (often 0 or very large)
        valid_depths = depth_crop[depth_crop > 0]
        
        if len(valid_depths) == 0:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'valid_ratio': 0.0
            }
        
        return {
            'mean': float(np.mean(valid_depths)),
            'median': float(np.median(valid_depths)),
            'std': float(np.std(valid_depths)),
            'valid_ratio': float(len(valid_depths) / depth_crop.size)
        }
    
    @staticmethod
    def compute_depth_consistency(
        stats1: Dict[str, float], 
        stats2: Dict[str, float]
    ) -> float:
        """
        Compute depth consistency score between two bounding boxes.
        
        Permanent landmarks should have similar depth distributions.
        Large differences suggest different objects or significant structural changes.
        
        Args:
            stats1: Depth statistics from first frame
            stats2: Depth statistics from second frame
            
        Returns:
            Consistency score [0, 1], higher is more consistent
        """
        if stats1['valid_ratio'] < 0.3 or stats2['valid_ratio'] < 0.3:
            # Insufficient depth data
            return 0.5  # Neutral score
        
        # Compare mean depths (normalized by average)
        mean1 = stats1['mean']
        mean2 = stats2['mean']
        avg_mean = (mean1 + mean2) / 2
        
        if avg_mean < 1e-6:
            return 0.5
        
        mean_diff = abs(mean1 - mean2) / avg_mean
        
        # Consistency decreases with difference
        # Use exponential decay: exp(-k * diff)
        consistency = np.exp(-2.0 * mean_diff)
        
        return float(consistency)


# ===========================================================================
# STAGE 5: GEOMETRIC KEYPOINT MATCHING
# ===========================================================================


class KeypointMatcher:
    """
    Performs geometric verification using keypoint matching.
    
    This provides the strongest evidence for landmark correspondence by
    matching local geometric features between images.
    """
    
    def __init__(self, method: str = "orb"):
        """
        Initialize keypoint detector and matcher.
        
        Args:
            method: Keypoint method ('orb', 'sift', or 'akaze')
        """
        self.method = method.lower()
        
        if self.method == "sift":
            # SIFT: Scale-invariant feature transform (patented but available)
            self.detector = cv2.SIFT_create()
            # FLANN-based matcher for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.method == "akaze":
            # AKAZE: Accelerated-KAZE, good for local features
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:  # Default to ORB
            # ORB: Oriented FAST and Rotated BRIEF, fast and free
            self.detector = cv2.ORB_create(nfeatures=500)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def match_regions(
        self, 
        img1: Image.Image, 
        bbox1: Sequence[float],
        img2: Image.Image,
        bbox2: Sequence[float],
        ratio_threshold: float = 0.75
    ) -> Tuple[int, float]:
        """
        Match keypoints between two bounding box regions.
        
        Uses Lowe's ratio test to filter good matches and RANSAC for
        geometric verification.
        
        Args:
            img1, img2: PIL Images
            bbox1, bbox2: Bounding boxes to compare
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            (num_inliers, confidence_score)
        """
        # Convert PIL to OpenCV format
        cv_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        cv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
        
        # Crop regions
        x1, y1, x2, y2 = [int(v) for v in bbox1]
        crop1 = cv_img1[y1:y2, x1:x2]
        
        x1, y1, x2, y2 = [int(v) for v in bbox2]
        crop2 = cv_img2[y1:y2, x1:x2]
        
        if crop1.size == 0 or crop2.size == 0:
            return 0, 0.0
        
        # Detect keypoints and compute descriptors
        kp1, des1 = self.detector.detectAndCompute(crop1, None)
        kp2, des2 = self.detector.detectAndCompute(crop2, None)
        
        # DEBUG
        print(f"      DEBUG [Keypoints]: Crop1 size: {crop1.shape}, Crop2 size: {crop2.shape}")
        print(f"      DEBUG [Keypoints]: KP1: {len(kp1) if kp1 else 0}, KP2: {len(kp2) if kp2 else 0}")
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            print(f"      DEBUG [Keypoints]: Insufficient keypoints or descriptors")
            return 0, 0.0
        
        # Match descriptors
        if self.method == "sift":
            # KNN matching for SIFT
            matches = self.matcher.knnMatch(des1, des2, k=2)
        else:
            # KNN matching for binary descriptors
            matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        print(f"      DEBUG [Keypoints]: Total matches: {len(matches)}, Good matches: {len(good_matches)}")
        
        if len(good_matches) < 4:
            print(f"      DEBUG [Keypoints]: Insufficient good matches (< 4)")
            return 0, 0.0
        
        # Extract matched keypoint locations
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # RANSAC to find inliers
        try:
            _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            if mask is None:
                return 0, 0.0
            
            inliers = np.sum(mask)
            total = len(good_matches)
            confidence = inliers / total if total > 0 else 0.0
            
            return int(inliers), float(confidence)
        except:
            return 0, 0.0


# ===========================================================================
# STAGE 1 & 6: PIPELINE ORCHESTRATION AND VISUALIZATION
# ===========================================================================


class CrossTemporalPipeline:
    """
    Main pipeline orchestrator.
    
    Coordinates all stages from image loading through final visualization.
    """
    
    def __init__(
        self,
        winter_rgb_dir: Path,
        winter_depth_dir: Path,
        autumn_rgb_dir: Path,
        autumn_depth_dir: Path,
        model_path: str,
        output_dir: Path,
        device: str = "mps",
        detection_threshold: float = 0.2,
        queries: Optional[List[str]] = None,
        use_adaptive_thresholds: bool = False,
        use_focused_queries: bool = False,
        use_keyframing: bool = False,
        keyframe_similarity_threshold: float = 0.95,
    ):
        """
        Initialize pipeline components.
        
        Args:
            winter_rgb_dir: Path to winter RGB images
            winter_depth_dir: Path to winter depth images
            autumn_rgb_dir: Path to autumn RGB images
            autumn_depth_dir: Path to autumn depth images
            model_path: Path to FastVLM checkpoint
            output_dir: Directory for results
            device: Inference device
            detection_threshold: Default score threshold for OWL-ViT
            queries: Object categories to detect
            use_adaptive_thresholds: Use per-category thresholds based on empirical analysis
            use_focused_queries: Use only high-confidence queries (ignores queries argument)
            use_keyframing: Enable keyframe selection to reduce the number of images
            keyframe_similarity_threshold: Histogram correlation threshold for keyframe selection
        """
        self.winter_rgb_dir = winter_rgb_dir
        self.winter_depth_dir = winter_depth_dir
        self.autumn_rgb_dir = autumn_rgb_dir
        self.autumn_depth_dir = autumn_depth_dir
        self.output_dir = output_dir
        self.use_keyframing = use_keyframing
        self.keyframe_similarity_threshold = keyframe_similarity_threshold
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

        # Initialize models
        print("\n" + "="*70)
        print("INITIALIZING MODELS")
        print("="*70)
        
        print("Loading FastVLM for semantic analysis...")
        self.vlm = FastVLMAnalyzer(model_path, device=device)
        
        print("Loading OWL-ViT for object detection...")
        self.detector = OpenVocabularyDetector(
            device=device, 
            score_threshold=detection_threshold,
            use_adaptive_thresholds=use_adaptive_thresholds
        )
        
        print("Loading Keypoint Matcher...")
        self.keypoint_matcher = KeypointMatcher()
        
        print("Initializing Depth Validator...")
        self.depth_validator = DepthValidator()
        
        self.queries = queries or DEFAULT_QUERIES
        if use_focused_queries:
            self.queries = list(PERMANENT_CATEGORIES)
            print(f"Using focused queries: {self.queries}")
        
        print("\n" + "="*70)
        print("PIPELINE READY")
        print("="*70)
        if use_adaptive_thresholds:
            print("Using adaptive per-category thresholds")
        if self.use_keyframing:
            print(f"Using keyframe selection with similarity threshold: {self.keyframe_similarity_threshold}")

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
    
    def find_frame_pairs(
        self, 
        max_pairs: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """
        STAGE 1: Establish correspondence between winter and autumn frames.
        
        Strategy: Cross-product matching - compare EVERY autumn frame against 
        EVERY winter frame to find the best matches regardless of index.
        This allows finding matches like autumn #1 with winter #41.
        
        Args:
            max_pairs: Maximum number of pairs to process
            
        Returns:
            List of (winter_path, autumn_path) tuples
        """
        print("\n" + "="*70)
        print("STAGE 1: Finding corresponding frame pairs (cross-product)")
        print("="*70)
        
        # DEBUG: Check if directories exist
        print(f"DEBUG: Checking winter RGB directory: {self.winter_rgb_dir}")
        print(f"DEBUG: Directory exists: {self.winter_rgb_dir.exists()}")
        print(f"DEBUG: Is directory: {self.winter_rgb_dir.is_dir()}")
        
        print(f"DEBUG: Checking autumn RGB directory: {self.autumn_rgb_dir}")
        print(f"DEBUG: Directory exists: {self.autumn_rgb_dir.exists()}")
        print(f"DEBUG: Is directory: {self.autumn_rgb_dir.is_dir()}")
        
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
        
        # STAGE 0: Optional Keyframe Selection
        if self.use_keyframing:
            winter_images = self._select_keyframes(winter_images, self.keyframe_similarity_threshold, "Winter")
            autumn_images = self._select_keyframes(autumn_images, self.keyframe_similarity_threshold, "Autumn")

        # DEBUG: Show what files were found
        print(f"DEBUG: Found {len(winter_images)} winter images to be used for matching.")
        if len(winter_images) > 0:
            print(f"DEBUG: First 3 winter images: {[str(p.name) for p in winter_images[:3]]}")
        else:
            # Try listing directory contents
            try:
                all_winter_files = list(self.winter_rgb_dir.iterdir())
                print(f"DEBUG: Total files in winter directory: {len(all_winter_files)}")
                print(f"DEBUG: First 10 files: {[str(p.name) for p in all_winter_files[:10]]}")
            except Exception as e:
                print(f"DEBUG: Error listing winter directory: {e}")
        
        print(f"DEBUG: Found {len(autumn_images)} autumn images")
        if len(autumn_images) > 0:
            print(f"DEBUG: First 3 autumn images: {[str(p.name) for p in autumn_images[:3]]}")
        else:
            # Try listing directory contents
            try:
                all_autumn_files = list(self.autumn_rgb_dir.iterdir())
                print(f"DEBUG: Total files in autumn directory: {len(all_autumn_files)}")
                print(f"DEBUG: First 10 files: {[str(p.name) for p in all_autumn_files[:10]]}")
            except Exception as e:
                print(f"DEBUG: Error listing autumn directory: {e}")
        
        # Cross-product: create all possible pairs (every autumn with every winter)
        pairs = []
        total_possible = len(winter_images) * len(autumn_images)
        print(f"DEBUG: Total possible frame pairs: {total_possible}")
        
        for a_img in autumn_images:
            for w_img in winter_images:
                pairs.append((str(w_img), str(a_img)))
                if max_pairs and len(pairs) >= max_pairs:
                    break
            if max_pairs and len(pairs) >= max_pairs:
                break
        
        print(f"Found {len(pairs)} frame pairs to analyze")
        if max_pairs:
            print(f"  (limited to max_pairs={max_pairs} from {total_possible} total possible)")
        return pairs
    
    def detect_landmarks(
        self, 
        image: Image.Image,
        depth_path: Optional[str] = None
    ) -> List[Detection]:
        """
        STAGE 2-3: Detect objects and enrich with semantics.
        
        Combines open-vocabulary detection with FastVLM descriptions
        and visual embeddings.
        
        Args:
            image: Input PIL Image
            depth_path: Optional path to corresponding depth image
            
        Returns:
            List of Detection objects with full metadata
        """
        # DEBUG: Image info
        print(f"    DEBUG: Processing image size: {image.size}, mode: {image.mode}")
        
        # Stage 2: Object detection
        raw_detections = self.detector.detect(image, self.queries)
        
        # DEBUG: Detection results
        print(f"    DEBUG: OWL-ViT found {len(raw_detections)} raw detections")
        if len(raw_detections) > 0:
            print(f"    DEBUG: Sample detection: {raw_detections[0]}")
        
        # Load depth if available
        depth = None
        if depth_path:
            print(f"    DEBUG: Loading depth from: {depth_path}")
            depth = self.depth_validator.load_depth_image(depth_path)
            if depth is not None:
                print(f"    DEBUG: Depth loaded successfully, shape: {depth.shape}")
            else:
                print(f"    DEBUG: Failed to load depth image")
        
        detections = []
        width, height = image.size
        
        for det in raw_detections:
            # Clamp and pad bounding box
            bbox = self._clamp_bbox(det['bbox'], width, height)
            
            # Crop object region
            crop = image.crop(bbox)
            
            # Stage 3: Semantic enrichment with FastVLM
            description = self.vlm.describe_object(crop)
            embedding = self.vlm.get_vision_embedding(crop)
            
            # DEBUG: Check embedding shape
            print(f"    DEBUG: Embedding shape: {embedding.shape}, dtype: {embedding.dtype}, ndim: {embedding.ndim}")
            if embedding.ndim > 0:
                print(f"    DEBUG: Embedding first 5 values: {embedding[:5]}")
            else:
                print(f"    DEBUG: Embedding is scalar: {embedding}")
            
            # Stage 4: Extract depth statistics if available
            depth_stats = None
            if depth is not None:
                depth_stats = self.depth_validator.extract_depth_stats(depth, bbox)
            
            detections.append(Detection(
                label=det['label'],
                bbox=bbox,
                score=det['score'],
                description=description,
                embedding=embedding,
                depth_stats=depth_stats
            ))
        
        return detections
    
    def match_landmarks(
        self,
        winter_img: Image.Image,
        winter_dets: List[Detection],
        autumn_img: Image.Image,
        autumn_dets: List[Detection]
    ) -> Tuple[List[LandmarkMatch], List[Detection], List[Detection]]:
        """
        STAGE 4-5: Match detections across seasons using multiple cues.
        
        Implements hierarchical matching:
        1. Embedding similarity (coarse matching)
        2. Semantic similarity (verification)
        3. Depth consistency (spatial validation)
        4. Keypoint matching (geometric verification)
        
        Args:
            winter_img, autumn_img: Full images
            winter_dets, autumn_dets: Detections in each image
            
        Returns:
            (matches, winter_only, autumn_only)
        """
        print(f"  Matching {len(winter_dets)} winter detections with {len(autumn_dets)} autumn detections")
        
        if len(winter_dets) == 0 or len(autumn_dets) == 0:
            return [], winter_dets, autumn_dets
        
        # Compute pairwise embedding similarities
        # Stack embeddings properly: each embedding should be a row
        winter_embeddings = np.vstack([d.embedding.reshape(1, -1) if d.embedding.ndim == 1 
                                       else d.embedding for d in winter_dets])
        autumn_embeddings = np.vstack([d.embedding.reshape(1, -1) if d.embedding.ndim == 1 
                                       else d.embedding for d in autumn_dets])
        
        # DEBUG: Check embedding shapes
        print(f"  DEBUG: Winter embeddings shape: {winter_embeddings.shape}")
        print(f"  DEBUG: Autumn embeddings shape: {autumn_embeddings.shape}")
        
        # Cosine similarity matrix (n_winter x n_autumn)
        sim_matrix = cosine_similarity(winter_embeddings, autumn_embeddings)
        
        # Greedy matching: match each winter detection to best autumn detection
        matches = []
        matched_autumn = set()
        matched_winter = set()
        
        # Sort by similarity score
        candidates = []
        for i in range(len(winter_dets)):
            for j in range(len(autumn_dets)):
                candidates.append((i, j, sim_matrix[i, j]))
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        for w_idx, a_idx, emb_sim in candidates:
            if w_idx in matched_winter or a_idx in matched_autumn:
                continue
            
            # Only consider if embedding similarity is reasonable
            if emb_sim < 0.5:
                continue
            
            w_det = winter_dets[w_idx]
            a_det = autumn_dets[a_idx]
            
            # Must be same category
            if w_det.label != a_det.label:
                continue
            
            # Compute semantic similarity
            sem_sim, sem_explanation = self.vlm.compare_objects(
                w_det.description, 
                a_det.description
            )
            
            # Compute depth consistency if available
            depth_consistency = 0.5  # Neutral default
            if w_det.depth_stats and a_det.depth_stats:
                depth_consistency = self.depth_validator.compute_depth_consistency(
                    w_det.depth_stats,
                    a_det.depth_stats
                )
            
            # Geometric verification with keypoint matching
            num_inliers, geo_confidence = self.keypoint_matcher.match_regions(
                winter_img, w_det.bbox,
                autumn_img, a_det.bbox
            )
            
            # Combine scores with weights
            # Embedding: 30%, Semantic: 20%, Depth: 20%, Geometric: 30%
            final_confidence = (
                0.30 * emb_sim +
                0.20 * sem_sim +
                0.20 * depth_consistency +
                0.30 * geo_confidence
            )
            
            # Decide if this landmark is invariant or temporal
            # Permanent structures should have high geometric and depth consistency
            is_invariant = (geo_confidence > 0.3 and depth_consistency > 0.6) or \
                          (w_det.label in PERMANENT_CATEGORIES)
            
            reasoning = (
                f"Embedding: {emb_sim:.2f}, Semantic: {sem_sim:.2f}, "
                f"Depth: {depth_consistency:.2f}, Keypoints: {num_inliers} inliers "
                f"({geo_confidence:.2f} confidence)"
            )
            
            match = LandmarkMatch(
                winter_detection=w_det,
                autumn_detection=a_det,
                embedding_similarity=emb_sim,
                semantic_similarity=sem_sim,
                depth_consistency=depth_consistency,
                geometric_confidence=geo_confidence,
                final_confidence=final_confidence,
                is_invariant=is_invariant,
                reasoning=reasoning
            )
            
            matches.append(match)
            matched_winter.add(w_idx)
            matched_autumn.add(a_idx)
        
        # Identify unmatched detections
        winter_only = [d for i, d in enumerate(winter_dets) if i not in matched_winter]
        autumn_only = [d for i, d in enumerate(autumn_dets) if i not in matched_autumn]
        
        print(f"    → {len(matches)} matches, {len(winter_only)} winter-only, {len(autumn_only)} autumn-only")
        
        return matches, winter_only, autumn_only
    
    def visualize_comparison(
        self,
        winter_img: Image.Image,
        autumn_img: Image.Image,
        matches: List[LandmarkMatch],
        winter_only: List[Detection],
        autumn_only: List[Detection],
        output_path: Path
    ) -> None:
        """
        STAGE 6: Create side-by-side visualization with color-coded boxes.
        
        Color coding:
        - Green: Invariant landmarks (matched with high confidence)
        - Yellow: Uncertain matches (low confidence)
        - Red: Temporal objects (seasonal changes)
        - Blue: Unmatched (only in one season)
        
        Args:
            winter_img, autumn_img: Input images
            matches: Matched landmarks
            winter_only, autumn_only: Unmatched detections
            output_path: Where to save visualization
        """
        # Create side-by-side canvas
        w_width, w_height = winter_img.size
        a_width, a_height = autumn_img.size
        
        canvas_width = w_width + a_width + 40  # 40px gap
        canvas_height = max(w_height, a_height) + 100  # Space for legend
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        canvas.paste(winter_img, (0, 50))
        canvas.paste(autumn_img, (w_width + 40, 50))
        
        draw = ImageDraw.Draw(canvas)
        
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
            title_font = ImageFont.truetype("Arial.ttf", 18)
        except:
            font = ImageFont.load_default()
            title_font = font
        
        # Draw titles
        draw.text((w_width // 2 - 50, 10), "Winter (2024-01-13)", fill='black', font=title_font)
        draw.text((w_width + 40 + a_width // 2 - 50, 10), "Autumn (2024-04-11)", fill='black', font=title_font)
        
        # Draw matched landmarks with color coding
        for match in matches:
            # Determine color based on confidence and invariance
            if match.is_invariant and match.final_confidence > 0.7:
                color = 'green'  # High-confidence invariant
            elif match.is_invariant:
                color = 'yellow'  # Low-confidence invariant
            elif match.final_confidence > 0.6:
                color = 'orange'  # Matched but temporal
            else:
                color = 'red'  # Low-confidence temporal
            
            # Draw winter bbox
            x1, y1, x2, y2 = match.winter_detection.bbox
            draw.rectangle([x1, y1 + 50, x2, y2 + 50], outline=color, width=3)
            
            # Draw autumn bbox
            x1, y1, x2, y2 = match.autumn_detection.bbox
            draw.rectangle([x1 + w_width + 40, y1 + 50, x2 + w_width + 40, y2 + 50], 
                          outline=color, width=3)
            
            # Draw connecting line
            w_center_x = (match.winter_detection.bbox[0] + match.winter_detection.bbox[2]) / 2
            w_center_y = (match.winter_detection.bbox[1] + match.winter_detection.bbox[3]) / 2 + 50
            a_center_x = (match.autumn_detection.bbox[0] + match.autumn_detection.bbox[2]) / 2 + w_width + 40
            a_center_y = (match.autumn_detection.bbox[1] + match.autumn_detection.bbox[3]) / 2 + 50
            
            draw.line([w_center_x, w_center_y, a_center_x, a_center_y], 
                     fill=color, width=2)
        
        # Draw unmatched detections in blue
        for det in winter_only:
            x1, y1, x2, y2 = det.bbox
            draw.rectangle([x1, y1 + 50, x2, y2 + 50], outline='blue', width=2)
        
        for det in autumn_only:
            x1, y1, x2, y2 = det.bbox
            draw.rectangle([x1 + w_width + 40, y1 + 50, x2 + w_width + 40, y2 + 50], 
                          outline='blue', width=2)
        
        # Draw legend at bottom
        legend_y = canvas_height - 80
        legend_items = [
            ('green', 'Invariant (high conf)'),
            ('yellow', 'Invariant (low conf)'),
            ('orange', 'Temporal (matched)'),
            ('red', 'Temporal (low conf)'),
            ('blue', 'Unmatched')
        ]
        
        x_offset = 20
        for color, label in legend_items:
            draw.rectangle([x_offset, legend_y, x_offset + 20, legend_y + 20], 
                          outline=color, fill=color, width=2)
            draw.text((x_offset + 30, legend_y + 5), label, fill='black', font=font)
            x_offset += 200
        
        # Add statistics
        stats_y = legend_y + 30
        stats_text = (
            f"Matches: {len(matches)} | "
            f"Invariant: {sum(1 for m in matches if m.is_invariant)} | "
            f"Temporal: {sum(1 for m in matches if not m.is_invariant)} | "
            f"Winter-only: {len(winter_only)} | "
            f"Autumn-only: {len(autumn_only)}"
        )
        draw.text((20, stats_y), stats_text, fill='black', font=font)
        
        # Save
        canvas.save(output_path)
        print(f"  Saved visualization to {output_path}")
    
    def process_frame_pair(
        self,
        winter_path: str,
        autumn_path: str,
        pair_id: str
    ) -> FramePairAnalysis:
        """
        Process a single pair of winter/autumn frames.
        
        Args:
            winter_path: Path to winter RGB image
            autumn_path: Path to autumn RGB image
            pair_id: Identifier for this pair
            
        Returns:
            FramePairAnalysis with all results
        """
        print(f"\nProcessing pair {pair_id}:")
        print(f"  Winter: {Path(winter_path).name}")
        print(f"  Autumn: {Path(autumn_path).name}")
        
        # DEBUG: Check if files exist
        print(f"  DEBUG: Winter file exists: {Path(winter_path).exists()}")
        print(f"  DEBUG: Autumn file exists: {Path(autumn_path).exists()}")
        
        # Load images
        try:
            winter_img = Image.open(winter_path).convert('RGB')
            print(f"  DEBUG: Winter image loaded successfully: {winter_img.size}")
        except Exception as e:
            print(f"  ERROR: Failed to load winter image: {e}")
            raise
        
        try:
            autumn_img = Image.open(autumn_path).convert('RGB')
            print(f"  DEBUG: Autumn image loaded successfully: {autumn_img.size}")
        except Exception as e:
            print(f"  ERROR: Failed to load autumn image: {e}")
            raise
        
        # Find corresponding depth images
        # Note: RGB and depth may have slightly different timestamps
        # Find the closest matching depth file by timestamp
        winter_stem = Path(winter_path).stem
        autumn_stem = Path(autumn_path).stem
        
        winter_depth_path = self._find_closest_depth_file(self.winter_depth_dir, winter_stem)
        autumn_depth_path = self._find_closest_depth_file(self.autumn_depth_dir, autumn_stem)
        
        # DEBUG: Depth paths
        print(f"  DEBUG: Winter RGB stem: {winter_stem}")
        print(f"  DEBUG: Winter depth path: {winter_depth_path}")
        print(f"  DEBUG: Autumn RGB stem: {autumn_stem}")
        print(f"  DEBUG: Autumn depth path: {autumn_depth_path}")
        
        # Detect landmarks in both frames
        print("  Detecting landmarks in winter frame...")
        winter_dets = self.detect_landmarks(winter_img, winter_depth_path)
        
        print("  Detecting landmarks in autumn frame...")
        autumn_dets = self.detect_landmarks(autumn_img, autumn_depth_path)
        
        # DEBUG: Detection counts
        print(f"  DEBUG: Total winter detections: {len(winter_dets)}")
        print(f"  DEBUG: Total autumn detections: {len(autumn_dets)}")
        
        # Match landmarks
        print("  Matching landmarks across seasons...")
        matches, winter_only, autumn_only = self.match_landmarks(
            winter_img, winter_dets,
            autumn_img, autumn_dets
        )
        
        # Create visualization
        vis_path = self.output_dir / "visualizations" / f"{pair_id}_comparison.jpg"
        self.visualize_comparison(
            winter_img, autumn_img,
            matches, winter_only, autumn_only,
            vis_path
        )
        
        return FramePairAnalysis(
            winter_frame_id=winter_stem,
            autumn_frame_id=autumn_stem,
            winter_image_path=winter_path,
            autumn_image_path=autumn_path,
            winter_detections=winter_dets,
            autumn_detections=autumn_dets,
            matches=matches,
            winter_only=winter_only,
            autumn_only=autumn_only,
            visualization_path=str(vis_path)
        )
    
    def run(self, max_pairs: Optional[int] = None) -> List[FramePairAnalysis]:
        """
        Run the complete pipeline.
        
        Args:
            max_pairs: Maximum number of frame pairs to process
            
        Returns:
            List of FramePairAnalysis results
        """
        print("\n" + "="*70)
        print("CROSS-TEMPORAL LANDMARK MATCHING PIPELINE")
        print("="*70)
        
        # Stage 1: Find frame pairs
        pairs = self.find_frame_pairs(max_pairs)
        
        # DEBUG: Exit early if no pairs found
        if len(pairs) == 0:
            print("\nWARNING: No frame pairs found! Pipeline cannot proceed.")
            print("Please check:")
            print("  1. Directory paths are correct")
            print("  2. Directories contain .png or .jpg files")
            print("  3. Both winter and autumn directories have matching files")
            return []
        
        # Process each pair
        results = []
        for idx, (winter_path, autumn_path) in enumerate(tqdm(pairs, desc="Processing pairs")):
            pair_id = f"pair_{idx:03d}"
            result = self.process_frame_pair(winter_path, autumn_path, pair_id)
            results.append(result)
        
        # Export results
        self._export_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _export_results(self, results: List[FramePairAnalysis]) -> None:
        """Export results to JSON."""
        output_file = self.output_dir / "data" / "landmark_matches.json"
        
        with open(output_file, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        print(f"\nResults exported to {output_file}")
    
    def _print_summary(self, results: List[FramePairAnalysis]) -> None:
        """Print summary statistics."""
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        
        total_matches = sum(len(r.matches) for r in results)
        total_invariant = sum(sum(1 for m in r.matches if m.is_invariant) for r in results)
        total_temporal = total_matches - total_invariant
        
        avg_confidence = np.mean([
            m.final_confidence 
            for r in results 
            for m in r.matches
        ]) if total_matches > 0 else 0
        
        print(f"Processed {len(results)} frame pairs")
        print(f"Total matched landmarks: {total_matches}")
        print(f"  - Invariant (permanent): {total_invariant}")
        print(f"  - Temporal (seasonal): {total_temporal}")
        print(f"Average match confidence: {avg_confidence:.3f}")
        print(f"\nResults saved to: {self.output_dir}")
    
    def _clamp_bbox(
        self, 
        bbox: Sequence[float], 
        width: int, 
        height: int, 
        padding: float = 0.02
    ) -> List[float]:
        """Clamp and pad bounding box to image boundaries.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            width: Image width
            height: Image height
            padding: Padding ratio (0.02 = 2% padding on each side)
        
        Returns:
            Clamped bounding box
        """
        x1, y1, x2, y2 = bbox
        pad_w = (x2 - x1) * padding
        pad_h = (y2 - y1) * padding
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width - 1, x2 + pad_w)
        y2 = min(height - 1, y2 + pad_h)
        return [x1, y1, x2, y2]
    
    def _find_closest_depth_file(self, depth_dir: Path, rgb_stem: str) -> Optional[str]:
        """
        Find the depth file with the closest matching timestamp to an RGB file.
        
        RGB and depth files may have slightly different timestamps due to
        sensor synchronization. This finds the depth file with the closest
        timestamp within a reasonable tolerance (100ms).
        
        Args:
            depth_dir: Directory containing depth images
            rgb_stem: Stem of RGB filename (e.g., "1705136563.3253479")
            
        Returns:
            Path to closest depth file, or None if not found
        """
        try:
            # Parse the RGB timestamp
            rgb_timestamp = float(rgb_stem)
        except ValueError:
            return None
        
        # Find all depth files
        depth_files = []
        for ext in ['.png', '.jpg', '.tiff']:
            depth_files.extend(depth_dir.glob(f"*{ext}"))
        
        # Filter out macOS resource fork files
        depth_files = [f for f in depth_files if not f.name.startswith("._")]
        
        if not depth_files:
            return None
        
        # Find closest timestamp
        best_match = None
        best_diff = float('inf')
        
        for depth_file in depth_files:
            try:
                depth_timestamp = float(depth_file.stem)
                diff = abs(depth_timestamp - rgb_timestamp)
                
                # Only consider files within 100ms (0.1 seconds)
                if diff < 0.1 and diff < best_diff:
                    best_diff = diff
                    best_match = depth_file
            except ValueError:
                continue
        
        if best_match:
            return str(best_match)
        return None


# ===========================================================================
# CONFIGURATION AND CONSTANTS
# ===========================================================================


DEFAULT_QUERIES = [
    # Natural landmarks (invariant across seasons) - simplified for better detection
    "tree",
    "rock",
    "stone",
    "statue",
    "fountain",
    "bench",
    "fence",
    "gate",
    "path",
    "walkway",
    # Structural elements
    "lamp",
    "post",
    "wall",
    "planter",
    "pot",
    # Vegetation (temporal - changes with seasons) - MORE SPECIFIC
    "bush",
    "shrub",
    "flowers",
    "grass",
    "hedge",
    "plant",
    "leaves",
    "branch",
    "foliage",
    # Small seasonal elements
    "snow",
    "ice",
    "puddle",
    "mud",
    # Buildings/structures
    "house",
    "building",
    "shed",
    "window",
    "door",
    "roof",
]

# Categories that are typically permanent/invariant
PERMANENT_CATEGORIES = {
    "tree",
    "rock",
    "stone",
    "statue",
    "fountain",
    "bench",
    "fence",
    "gate",
    "path",
    "walkway",
    "lamp",
    "post",
    "wall",
    "planter",
    "pot",
    "house",
    "building",
    "shed",
}


# ===========================================================================
# COMMAND LINE INTERFACE
# ===========================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-temporal landmark matching pipeline",
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
        "--winter-depth",
        type=str,
        required=True,
        help="Directory containing winter depth images"
    )
    parser.add_argument(
        "--autumn-rgb",
        type=str,
        required=True,
        help="Directory containing autumn RGB images"
    )
    parser.add_argument(
        "--autumn-depth",
        type=str,
        required=True,
        help="Directory containing autumn depth images"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to FastVLM checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pipeline_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device for inference (cpu/cuda/mps)"
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.2,
        help="Minimum confidence for object detections (ignored if --adaptive-thresholds is used)"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Maximum number of frame pairs to process"
    )
    parser.add_argument(
        "--adaptive-thresholds",
        action="store_true",
        help="Use per-category thresholds based on empirical analysis (recommended for garden scenes)"
    )
    parser.add_argument(
        "--focused-queries",
        action="store_true",
        help="Use only high-confidence queries (house, tree, window, etc.) instead of full query set"
    )
    parser.add_argument(
        "--use-keyframing",
        action="store_true",
        help="Enable fast histogram-based keyframe selection (recommended for large datasets)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the pipeline."""
    args = parse_args()
    
    pipeline = CrossTemporalPipeline(
        winter_rgb_dir=Path(args.winter_rgb),
        winter_depth_dir=Path(args.winter_depth),
        autumn_rgb_dir=Path(args.autumn_rgb),
        autumn_depth_dir=Path(args.autumn_depth),
        model_path=args.model_path,
        output_dir=Path(args.output_dir),
        device=args.device,
        detection_threshold=args.detection_threshold,
        use_adaptive_thresholds=args.adaptive_thresholds,
        use_focused_queries=args.focused_queries,
        use_keyframing=args.use_keyframing,
    )
    
    # Run pipeline
    results = pipeline.run(max_pairs=args.max_pairs)
    
    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
