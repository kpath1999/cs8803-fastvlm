# Cross-Temporal Landmark Matching Pipeline

Vision-language pipeline for matching landmarks across seasonal datasets using FastVLM and OWL-ViT. Identifies invariant landmarks and temporal changes in outdoor environments captured under different conditions (winter vs autumn).

**Project Documentation:** [Google Docs](https://docs.google.com/document/d/1B7olGH20mLg_vbxM9kXPgtVxORdQn5aJza6HhdTuOmo/edit?usp=sharing)

## Overview

The pipeline implements a 7-stage hierarchical matching system:

1. **Image Stream Ingestion** - Pairs corresponding frames from winter/autumn datasets
2. **Open-Vocabulary Detection** - OWL-ViT detects objects using natural language queries
3. **Semantic Enrichment** - FastVLM generates descriptions and visual embeddings
4. **Depth Validation** - Validates matches using depth consistency
5. **Geometric Verification** - Keypoint matching (ORB/SIFT) with RANSAC
6. **Visualization** - Color-coded comparisons (green=invariant, red=temporal, blue=unmatched)
7. **Semantic Segmentation** - (Stretch goal) Pixel-level change detection

## Quick Start

```bash
# Activate environment
conda activate fastvlm

# Run pipeline
python cross_temporal_pipeline.py \
    --winter-rgb /Volumes/KAUSAR/rover_dataset/2024-01-13/realsense_D435i/rgb \
    --winter-depth /Volumes/KAUSAR/rover_dataset/2024-01-13/realsense_D435i/depth \
    --autumn-rgb /Volumes/KAUSAR/rover_dataset/2024-04-11/realsense_D435i/rgb \
    --autumn-depth /Volumes/KAUSAR/rover_dataset/2024-04-11/realsense_D435i/depth \
    --model-path checkpoints/llava-fastvithd_0.5b_stage2 \
    --output-dir pipeline_results \
    --max-pairs 10
```

## Output

Results are saved to `pipeline_results/`:
- `visualizations/` - Side-by-side frame comparisons with annotated matches
- `data/landmark_matches.json` - Structured match data with confidence scores
- `crops/` - Cropped detection regions for inspection

## Architecture

**Detection Models:**
- OWL-ViT (`google/owlvit-base-patch32`) for open-vocabulary object detection
- FastVLM (`llava-fastvithd_0.5b_stage2`) for semantic understanding

**Matching Strategy:**
- Embedding similarity (cosine distance on visual features)
- Semantic similarity (text-based keyword matching)
- Depth consistency (3D spatial validation)
- Geometric confidence (keypoint matching with RANSAC)

**Classification:**
- **Invariant landmarks** - Permanent structures (buildings, signs, poles)
- **Temporal objects** - Seasonal changes (trees, vegetation, vehicles)

## Datasets

- [ROVER](https://iis-esslingen.github.io/rover/pages/dataset_overview/) - Multi-seasonal outdoor navigation
- [FourSeasons](https://cvg.cit.tum.de/data/datasets/4seasons-dataset/download) - Long-term visual localization

## Dependencies

Core requirements:
- PyTorch 2.0+
- Transformers (Hugging Face)
- FastVLM (llava)
- OpenCV
- Pillow, NumPy, scikit-learn

See `pyproject.toml` for full dependencies.

## Notes

- The pipeline filters out macOS resource fork files (`._*`) automatically
- Supports MPS (Apple Silicon), CUDA, and CPU backends
- Detection threshold (default 0.3) can be adjusted via `--detection-threshold`
- Depth images are optional but improve match quality
