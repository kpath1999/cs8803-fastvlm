#!/usr/bin/env python3
"""Unified Landmark Analysis Pipeline
=======================================

This script combines keyframe extraction, open-vocabulary object detection,
and FastVLM-based semantic reasoning to produce both visual and textual
summaries of persistent landmarks across multiple video streams. It merges
and extends the functionality from the former ``compare_office_loops`` and
``video_analysis_experiments`` scripts so that a single command performs the
end-to-end workflow requested for the office loop dataset.

Main capabilities
-----------------
1. **Keyframe sampling** at configurable time intervals from each video.
2. **Open-vocabulary object detection** on every keyframe using OWL-ViT.
3. **Semantic enrichment with FastVLM** for whole scenes and per-object crops.
4. **Visual outputs** with annotated frames and cropped landmarks.
5. **Structured exports** (JSON + CSV) for downstream analysis of landmarks
   across conditions and seasons.

Usage example::

python landmark_analysis.py \
  --video-dir data/fourseasons/officeloop \
  --model-path checkpoints/llava-fastvithd_0.5b_stage2 \
  --keyframe-interval 1.5

The script assumes that FastVLM checkpoints are already available locally and
that the ``transformers`` library is installed for OWL-ViT.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class LandmarkDetection:
	"""Represents a single detected landmark with semantic metadata."""

	label: str
	bbox: List[float]
	score: float
	description: str
	visual_embedding: List[float]
	position_2d: List[float]

	def to_dict(self) -> Dict:
		return asdict(self)


@dataclass
class KeyframeAnalysis:
	"""Stores the analysis for one keyframe."""

	video_id: str
	frame_number: int
	timestamp_s: float
	keyframe_path: str
	annotated_path: Optional[str]
	scene_description: str
	scene_objects: List[str]
	detections: List[LandmarkDetection] = field(default_factory=list)

	def to_dict(self) -> Dict:
		data = asdict(self)
		data["detections"] = [det.to_dict() for det in self.detections]
		return data


# ---------------------------------------------------------------------------
# Video ingestion utilities
# ---------------------------------------------------------------------------


class VideoProcessor:
	"""Handles video frame extraction."""

	def __init__(self, video_path: str):
		self.video_path = video_path
		self.cap = cv2.VideoCapture(video_path)

		if not self.cap.isOpened():
			raise ValueError(f"Cannot open video file: {video_path}")

		self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
		self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

	def extract_frame(self, frame_idx: int) -> Optional[Image.Image]:
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
		success, frame = self.cap.read()
		if not success:
			return None
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		return Image.fromarray(frame_rgb)

	def iter_keyframes(self, interval_seconds: float) -> Iterable[Tuple[int, float, Image.Image]]:
		frame_step = max(int(interval_seconds * self.fps), 1)
		for frame_idx in range(0, self.total_frames, frame_step):
			frame = self.extract_frame(frame_idx)
			if frame is None:
				continue
			timestamp = frame_idx / self.fps
			yield frame_idx, timestamp, frame

	def __del__(self) -> None:
		if hasattr(self, "cap"):
			self.cap.release()


# ---------------------------------------------------------------------------
# FastVLM wrapper
# ---------------------------------------------------------------------------


class FastVLMVideoAnalyzer:
	"""Wrapper around FastVLM for semantic reasoning tasks."""

	def __init__(self, model_path: str, device: str = "mps"):
		disable_torch_init()
		model_name = get_model_name_from_path(model_path)
		self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
			model_path, None, model_name, device=device
		)
		self.device = device
		self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

	def _format_prompt(self, prompt: str) -> str:
		if self.model.config.mm_use_im_start_end:
			return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
		return DEFAULT_IMAGE_TOKEN + "\n" + prompt

	def _generate(self, image: Image.Image, prompt: str, temperature: float = 0.2) -> str:
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
			# Fix warning: only pass temperature when do_sample=True
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

	def describe_scene(self, image: Image.Image) -> Tuple[str, List[str]]:
		scene_prompt = (
			"Provide a concise yet detailed description of this scene focusing on permanent "
			"landmarks, spatial layout, and orientation cues."
		)
		objects_prompt = "List all prominent objects or structures visible in the scene, comma separated."
		description = self._generate(image, scene_prompt, temperature=0.2)
		objects_raw = self._generate(image, objects_prompt, temperature=0.0)
		objects = [obj.strip() for obj in objects_raw.split(",") if obj.strip()]
		return description, objects

	def describe_object(self, image: Image.Image) -> str:
		prompt = (
			"Describe this object in detail, including its material, color, shape, texture, "
			"and any identifying features that would help recognize it again."
		)
		return self._generate(image, prompt, temperature=0.2)

	def get_vision_embedding(self, image: Image.Image) -> np.ndarray:
		image_tensor = process_images([image], self.image_processor, self.model.config)[0]
		with torch.no_grad():
			vision_tower = self.model.get_vision_tower()
			vision_feats = vision_tower(image_tensor.unsqueeze(0).to(self.device, dtype=torch.float16))
			embedding = vision_feats.mean(dim=(1, 2)).squeeze().cpu().numpy()
		return embedding


# ---------------------------------------------------------------------------
# Open-vocabulary detector
# ---------------------------------------------------------------------------


class OpenVocabularyDetector:
	"""OWL-ViT detector with configurable text queries."""

	def __init__(
		self,
		model_name: str = "google/owlvit-base-patch32",
		device: str = "cpu",
		score_threshold: float = 0.35,
	):
		self.processor = OwlViTProcessor.from_pretrained(model_name)
		self.model = OwlViTForObjectDetection.from_pretrained(model_name)
		self.device = torch.device(device)
		self.model.to(self.device)
		self.score_threshold = score_threshold

	def detect(self, image: Image.Image, queries: Sequence[str]) -> List[Dict]:
		inputs = self.processor(text=list(queries), images=image, return_tensors="pt")
		inputs = {k: v.to(self.device) for k, v in inputs.items()}

		with torch.no_grad():
			outputs = self.model(**inputs)

		target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
		# Use post_process_object_detection instead of deprecated post_process
		results = self.processor.post_process_object_detection(
			outputs=outputs, 
			target_sizes=target_sizes, 
			threshold=0.0
		)[0]

		detections = []
		for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
			score_value = score.item()
			if score_value < self.score_threshold:
				continue
			bbox = box.tolist()  # [x1, y1, x2, y2]
			detections.append(
				{
					"label": queries[label.item()],
					"score": score_value,
					"bbox": bbox,
				}
			)
		return detections


# ---------------------------------------------------------------------------
# Landmark analysis orchestrator
# ---------------------------------------------------------------------------


DEFAULT_QUERIES = [
	"office building",
	"brick building",
	"glass building",
	"parking lot",
	"parking structure",
	"tree",
	"large tree",
	"light pole",
	"stop sign",
	"street sign",
	"crosswalk",
	"sidewalk",
	"road",
	"car",
	"truck",
	"bike rack",
	"bench",
	"statue",
	"fountain",
	"flower bed",
	"awning",
	"flag",
	"entrance",
	"zebra crossing"
]


def ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def clamp_bbox(bbox: Sequence[float], width: int, height: int, padding: float = 0.02) -> Tuple[int, int, int, int]:
	x1, y1, x2, y2 = bbox
	pad_w = (x2 - x1) * padding
	pad_h = (y2 - y1) * padding
	x1 = max(0, int(x1 - pad_w))
	y1 = max(0, int(y1 - pad_h))
	x2 = min(width - 1, int(x2 + pad_w))
	y2 = min(height - 1, int(y2 + pad_h))
	return x1, y1, x2, y2


def draw_annotations(image: Image.Image, detections: Sequence[LandmarkDetection]) -> Image.Image:
	annotated = image.copy()
	draw = ImageDraw.Draw(annotated)
	try:
		font = ImageFont.truetype("Arial.ttf", 16)
	except OSError:
		font = ImageFont.load_default()

	for det in detections:
		x1, y1, x2, y2 = det.bbox
		draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
		label_text = f"{det.label} ({det.score:.2f})"
		
		# Handle both old and new Pillow versions
		try:
			# Pillow >= 10.0.0
			text_bbox = draw.textbbox((0, 0), label_text, font=font)
			text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
		except AttributeError:
			# Pillow < 10.0.0
			text_w, text_h = draw.textsize(label_text, font=font)
		
		draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill="yellow")
		draw.text((x1 + 2, y1 - text_h - 2), label_text, fill="black", font=font)
	return annotated


def crop_image(image: Image.Image, bbox: Sequence[int]) -> Image.Image:
	x1, y1, x2, y2 = bbox
	return image.crop((x1, y1, x2, y2))


class LandmarkAnalysisPipeline:
	"""Coordinates keyframe sampling, detection, and semantic reasoning."""

	def __init__(
		self,
		model_path: str,
		detector_model: str,
		device: str,
		query_terms: Sequence[str],
		detection_threshold: float,
		output_dir: Path,
		keyframe_interval: float,
		max_videos: Optional[int] = None,
	):
		self.output_dir = ensure_dir(output_dir)
		self.keyframe_dir = ensure_dir(self.output_dir / "keyframes")
		self.annotated_dir = ensure_dir(self.output_dir / "annotated")
		self.crops_dir = ensure_dir(self.output_dir / "crops")
		self.data_dir = ensure_dir(self.output_dir / "data")

		self.vlm = FastVLMVideoAnalyzer(model_path, device=device)
		self.detector = OpenVocabularyDetector(
			model_name=detector_model,
			device=device,
			score_threshold=detection_threshold,
		)
		self.query_terms = list(query_terms)
		self.keyframe_interval = keyframe_interval
		self.max_videos = max_videos

	def process_directory(self, video_dir: Path) -> List[KeyframeAnalysis]:
		video_files = sorted(video_dir.glob("*.mp4"))
		if self.max_videos:
			video_files = video_files[: self.max_videos]

		analyses: List[KeyframeAnalysis] = []

		for video_path in tqdm(video_files, desc="Videos"):
			analyses.extend(self.process_video(video_path))

		return analyses

	def process_video(self, video_path: Path) -> List[KeyframeAnalysis]:
		processor = VideoProcessor(str(video_path))
		video_id = video_path.stem
		video_keyframe_dir = ensure_dir(self.keyframe_dir / video_id)
		video_annotated_dir = ensure_dir(self.annotated_dir / video_id)
		video_crop_dir = ensure_dir(self.crops_dir / video_id)

		results: List[KeyframeAnalysis] = []
		
		print(f"\nProcessing video: {video_id}")

		for frame_idx, timestamp, frame in processor.iter_keyframes(self.keyframe_interval):
			scene_desc, scene_objects = self.vlm.describe_scene(frame)
			detections_raw = self.detector.detect(frame, self.query_terms)
			
			print(f"  Frame {frame_idx} ({timestamp:.1f}s): {len(detections_raw)} detections found")

			width, height = frame.size
			landmark_detections: List[LandmarkDetection] = []

			for det_idx, det in enumerate(detections_raw):
				print(f"    - {det['label']}: score={det['score']:.3f}, bbox={det['bbox']}")
				bbox = clamp_bbox(det["bbox"], width, height)
				crop = crop_image(frame, bbox)
				crop_path = video_crop_dir / f"{frame_idx:06d}_{det_idx:02d}.jpg"
				crop.save(crop_path)

				description = self.vlm.describe_object(crop)
				embedding = self.vlm.get_vision_embedding(crop).tolist()
				cx = (bbox[0] + bbox[2]) / 2
				cy = (bbox[1] + bbox[3]) / 2

				landmark_detections.append(
					LandmarkDetection(
						label=det["label"],
						bbox=[float(v) for v in bbox],
						score=float(det["score"]),
						description=description,
						visual_embedding=embedding,
						position_2d=[float(cx), float(cy)],
					)
				)

			# Save keyframe image
			keyframe_path = video_keyframe_dir / f"{frame_idx:06d}.jpg"
			frame.save(keyframe_path)

			# Save annotated image only if there are detections
			if len(landmark_detections) > 0:
				annotated_image = draw_annotations(frame, landmark_detections)
				annotated_path = video_annotated_dir / f"{frame_idx:06d}.jpg"
				annotated_image.save(annotated_path)
				annotated_path_str = str(annotated_path.relative_to(self.output_dir))
			else:
				annotated_path_str = None

			keyframe_analysis = KeyframeAnalysis(
				video_id=video_id,
				frame_number=frame_idx,
				timestamp_s=timestamp,
				keyframe_path=str(keyframe_path.relative_to(self.output_dir)),
				annotated_path=annotated_path_str,
				scene_description=scene_desc,
				scene_objects=scene_objects,
				detections=landmark_detections,
			)
			results.append(keyframe_analysis)

		return results


def load_queries(labels_file: Optional[Path]) -> List[str]:
	if labels_file is None:
		return DEFAULT_QUERIES
	with open(labels_file) as f:
		labels = [line.strip() for line in f if line.strip()]
	return labels or DEFAULT_QUERIES


def export_results(results: Sequence[KeyframeAnalysis], output_dir: Path) -> None:
	data_dir = ensure_dir(output_dir / "data")
	json_path = data_dir / "landmarks.json"
	with open(json_path, "w") as f:
		json.dump([res.to_dict() for res in results], f, indent=2)

	# CSV summary for quick scanning
	try:
		import csv

		csv_path = data_dir / "landmarks.csv"
		with open(csv_path, "w", newline="") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(
				[
					"video_id",
					"frame_number",
					"timestamp_s",
					"label",
					"score",
					"bbox_x1",
					"bbox_y1",
					"bbox_x2",
					"bbox_y2",
					"center_x",
					"center_y",
					"description",
				]
			)
			for res in results:
				for det in res.detections:
					writer.writerow(
						[
							res.video_id,
							res.frame_number,
							f"{res.timestamp_s:.2f}",
							det.label,
							f"{det.score:.3f}",
							*[f"{coord:.1f}" for coord in det.bbox],
							f"{det.position_2d[0]:.1f}",
							f"{det.position_2d[1]:.1f}",
							det.description,
						]
					)
	except Exception as exc:  # pragma: no cover - optional convenience export
		print(f"Warning: could not export CSV summary ({exc})")


def summarize_landmarks(results: Sequence[KeyframeAnalysis], output_dir: Path) -> None:
	summary: Dict[str, Dict[str, float]] = {}
	for res in results:
		for det in res.detections:
			summary.setdefault(det.label, {"count": 0, "avg_score": 0.0})
			entry = summary[det.label]
			entry["count"] += 1
			entry["avg_score"] += det.score

	for label, data in summary.items():
		count = data["count"]
		data["avg_score"] = data["avg_score"] / count if count else 0.0

	summary_path = ensure_dir(output_dir / "data") / "summary.json"
	with open(summary_path, "w") as f:
		json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Landmark extraction with FastVLM + OWL-ViT")
	parser.add_argument("--video-dir", type=str, required=True, help="Directory containing input videos")
	parser.add_argument(
		"--model-path",
		type=str,
		required=True,
		help="Path to FastVLM checkpoint (e.g., ./checkpoints/llava-fastvithd_1.5b_stage2)",
	)
	parser.add_argument(
		"--detector-model",
		type=str,
		default="google/owlvit-base-patch32",
		help="Hugging Face model name for OWL-ViT",
	)
	parser.add_argument(
		"--labels-file",
		type=str,
		help="Optional text file with detection queries (one per line)",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="./landmark_analysis_results",
		help="Directory to store results",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="mps" if torch.backends.mps.is_available() else "cpu",
		help="Device for both FastVLM and detector (mps, cuda, or cpu)",
	)
	parser.add_argument(
		"--keyframe-interval",
		type=float,
		default=1.5,
		help="Keyframe sampling interval in seconds",
	)
	parser.add_argument(
		"--detection-threshold",
		type=float,
		default=0.35,
		help="Confidence threshold for detector outputs",
	)
	parser.add_argument("--max-videos", type=int, help="Optional limit on number of videos to process")
	parser.add_argument("--test-mode", action="store_true", help="Test mode: process only first 2 videos")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	video_dir = Path(args.video_dir).expanduser()
	if not video_dir.exists():
		raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

	output_root = Path(args.output_dir).expanduser()

	# Derive output subdirectory mirroring the input video path (minus the top-level folder)
	try:
		relative_input = video_dir.relative_to(Path.cwd())
	except ValueError:
		relative_input = video_dir.resolve()

	relative_parts = [part for part in relative_input.parts if part not in (relative_input.anchor, "")]
	if len(relative_parts) > 1:
		subpath = Path(*relative_parts[1:])
	elif relative_parts:
		subpath = Path(relative_parts[0])
	else:
		subpath = Path(video_dir.name)

	output_dir = output_root / subpath
	queries = load_queries(Path(args.labels_file) if args.labels_file else None)

	# Handle test mode
	max_videos = args.max_videos
	if args.test_mode:
		max_videos = 2
		print("\n" + "="*60)
		print("TEST MODE: Processing only first 2 videos")
		print("="*60 + "\n")

	pipeline = LandmarkAnalysisPipeline(
		model_path=args.model_path,
		detector_model=args.detector_model,
		device=args.device,
		query_terms=queries,
		detection_threshold=args.detection_threshold,
		output_dir=output_dir,
		keyframe_interval=args.keyframe_interval,
		max_videos=max_videos,
	)

	results = pipeline.process_directory(video_dir)
	export_results(results, output_dir)
	summarize_landmarks(results, output_dir)

	print("\n" + "="*60)
	print("Analysis complete!")
	print("="*60)
	print(f"\nTotal keyframes processed: {len(results)}")
	total_detections = sum(len(r.detections) for r in results)
	print(f"Total detections: {total_detections}")
	
	if total_detections == 0:
		print("\n⚠️  WARNING: No detections found!")
		print("Possible reasons:")
		print("  1. Detection threshold too high (current: {:.2f})".format(args.detection_threshold))
		print("     → Try lowering with --detection-threshold 0.1")
		print("  2. Query terms don't match scene content")
		print("     → Check DEFAULT_QUERIES or provide custom --labels-file")
		print("  3. OWL-ViT model needs better queries for your scene")
		print("     → Try more specific terms like 'car', 'person', 'door', 'window'")
	else:
		print(f"\nDetections per keyframe (avg): {total_detections/len(results):.1f}")
	
	print(f"\nOutput directories:")
	print(f"  Keyframes: {output_dir / 'keyframes'}")
	print(f"  Annotated frames: {output_dir / 'annotated'}")
	print(f"  Crops: {output_dir / 'crops'}")
	print(f"  Data exports: {output_dir / 'data'}")
	print()


if __name__ == "__main__":
	main()
