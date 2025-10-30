# FastVLM Video Analysis Experiments

This guide explains how to use `video_analysis_experiments.py` to explore FastVLM's capabilities on video data, with experiments designed as stepping stones toward your semantic SLAM goal.

## Overview

The experimental script demonstrates five key capabilities:

### 1. **Frame-by-Frame Semantic Extraction**
Extracts rich semantic descriptions from video frames at regular intervals. This is the foundation for Phase 1 of your SLAM pipeline.

**What it does:**
- Processes frames at configurable intervals (default: every 2 seconds)
- Extracts landmark descriptions using targeted prompts
- Lists detected objects
- Describes spatial layout
- Generates FastViTHD embeddings for each frame

**Why it matters for SLAM:**
This demonstrates the keyframe processing you'll need in your system. Instead of storing raw pixel data, you're creating semantic representations that are robust to appearance changes.

### 2. **Temporal Change Detection**
Detects what changes between consecutive frames using VLM reasoning.

**What it does:**
- Compares descriptions of sequential frames
- Identifies when new objects appear or disappear
- Tracks scene dynamics

**Why it matters for SLAM:**
Understanding change is crucial for:
- Identifying when to create new keyframes
- Detecting dynamic vs. static elements
- Building temporal consistency in your map

### 3. **Semantic Keyframe Selection**
Intelligently selects keyframes based on visual similarity using embeddings.

**What it does:**
- Extracts FastViTHD embeddings from all frames
- Uses cosine similarity to identify significant visual changes
- Selects minimal set of keyframes that capture scene diversity

**Why it matters for SLAM:**
This is exactly what you'll need in Phase 1. Instead of processing every frame, you:
- Reduce computational load by 10-100x
- Focus on frames with new visual information
- Create a sparse but comprehensive map

### 4. **Object Tracking via Description Matching**
Tracks specific objects (like your animal) across frames using semantic matching.

**What it does:**
- Queries each frame for a specific object
- Tracks object presence/absence over time
- Describes object state and actions

**Why it matters for SLAM:**
This is a simplified version of landmark association (Phase 2). You're learning to:
- Track the same entity across time
- Handle varying descriptions of the same object
- Build temporal object persistence

### 5. **Scene Graph Construction**
Builds a structured, semantic representation of the scene.

**What it does:**
- Extracts all objects and landmarks
- Identifies spatial relationships ("tree is behind the rock")
- Creates a unified semantic description

**Why it matters for SLAM:**
This is your "invariant picture into the surroundings." A scene graph:
- Abstracts away pixel-level details
- Captures semantic relationships
- Enables reasoning about scene structure

## Installation & Setup

### Prerequisites

```bash
# Install OpenCV for video processing
pip install opencv-python tqdm

# You should already have the FastVLM dependencies installed
```

### Quick Start

```bash
# Run all experiments on your animal video
python video_analysis_experiments.py \
    --video-path /path/to/your/animal_video.mp4 \
    --experiment all \
    --output-dir ./animal_analysis_results
```

### Run Individual Experiments

```bash
# Experiment 1: Frame extraction (good first test)
python video_analysis_experiments.py \
    --video-path /path/to/your/video.mp4 \
    --experiment 1 \
    --interval 3.0  # Sample every 3 seconds

# Experiment 3: Keyframe selection (most relevant for SLAM)
python video_analysis_experiments.py \
    --video-path /path/to/your/video.mp4 \
    --experiment 3

# Experiment 4: Track the animal
python video_analysis_experiments.py \
    --video-path /path/to/your/video.mp4 \
    --experiment 4
```

### Command-Line Options

```
--video-path        Path to your input video file (required)
--model-path        Path to FastVLM checkpoint (default: 1.5b stage2)
--experiment        Which experiment to run: 1, 2, 3, 4, 5, or 'all'
--output-dir        Where to save JSON results (default: ./video_analysis_output)
--device            Device to use: mps (Mac), cuda, or cpu
--interval          Frame sampling interval in seconds (default: 2.0)
```

## Understanding the Output

Each experiment saves JSON files with detailed results:

### `experiment_1_frame_extraction.json`
```json
[
  {
    "frame_number": 0,
    "timestamp_ms": 0.0,
    "description": "The image shows a deer eating leaves from a tree...",
    "landmarks": ["..."],
    "objects": ["deer", "tree", "leaves", "grass"],
    "spatial_layout": "Outdoor forest scene with a deer in the center",
    "embedding": [0.234, -0.156, ...]  // 768-dim vector
  },
  ...
]
```

**Key insight:** The `embedding` field is what you'll use for landmark matching in Phase 2. When two frames have similar embeddings (cosine similarity > 0.95), they're likely showing the same scene.

### `experiment_3_keyframes.json`
```json
[
  {"frame_number": 0, "timestamp_ms": 0.0},
  {"frame_number": 45, "timestamp_ms": 1500.0},
  {"frame_number": 120, "timestamp_ms": 4000.0}
]
```

**Key insight:** These are the frames you'd process in a real SLAM system. Note how much data reduction this achieves!

## Connection to Your SLAM Goals

### How This Maps to Your 3-Phase System

**Phase 1: Frame-Level Semantic Landmark Extraction**
- ✅ **Experiment 1** shows you how to extract semantic landmarks
- ✅ **Experiment 3** demonstrates keyframe selection
- ✅ Both generate the embeddings you need for matching

**Phase 2: Context-Aware Landmark Association**
- ✅ **Experiment 4** is a simplified version of semantic matching
- ⏭️ Next step: Use an LLM to match landmarks between videos
- ⏭️ Next step: Implement the probabilistic matching prompt you outlined

**Phase 3: Map Merging and Pose Graph Optimization**
- ⏭️ This requires integration with ORB-SLAM3 or similar
- ⏭️ The semantic matches from Phase 2 become constraints
- ⏭️ Will likely need to move to a more powerful machine

### Bridging to ROVER Dataset

Once you're comfortable with these experiments, you can:

1. **Apply to ROVER sequences:**
   ```bash
   # Process a summer sequence
   python video_analysis_experiments.py \
       --video-path /path/to/ROVER/campus_large_summer/.../video.mp4 \
       --experiment 3 \
       --output-dir ./rover_summer_keyframes
   
   # Process a winter sequence
   python video_analysis_experiments.py \
       --video-path /path/to/ROVER/campus_large_winter/.../video.mp4 \
       --experiment 3 \
       --output-dir ./rover_winter_keyframes
   ```

2. **Match landmarks across seasons:**
   - Load keyframe results from summer and winter
   - Extract embeddings for candidate pairs
   - Use LLM to verify: "Is this green tree the same as this snowy tree?"

3. **Build season-invariant maps:**
   - Create a unified landmark database
   - Each landmark has multiple appearance descriptors
   - Use semantic similarity instead of visual similarity

## Example: Processing Your Animal Video

Let's say you have `deer_eating_leaves.mp4`:

```bash
# Step 1: Extract keyframes and landmarks
python video_analysis_experiments.py \
    --video-path deer_eating_leaves.mp4 \
    --experiment 1 \
    --interval 2.0

# Step 2: Find interesting moments (keyframes)
python video_analysis_experiments.py \
    --video-path deer_eating_leaves.mp4 \
    --experiment 3

# Step 3: Track the deer
python video_analysis_experiments.py \
    --video-path deer_eating_leaves.mp4 \
    --experiment 4

# Step 4: Build scene graph
python video_analysis_experiments.py \
    --video-path deer_eating_leaves.mp4 \
    --experiment 5
```

**Expected insights:**
- **Experiment 1:** See how FastVLM describes the deer's actions frame-by-frame
- **Experiment 3:** Identify when the deer moves to new locations
- **Experiment 4:** Track the deer even as it moves and changes pose
- **Experiment 5:** Understand the overall scene structure

## Performance Expectations on Mac

On a MacBook Air with M-series chip:

- **Model loading:** ~30-60 seconds
- **Per-frame inference:** ~2-5 seconds (1.5B model)
- **Embedding extraction:** ~0.5-1 second
- **Video processing:** For a 1-minute video at 2-second intervals:
  - ~30 frames to process
  - ~90-150 seconds total (2-3 minutes)

**Tips for speed:**
- Use the 0.5B model for faster iteration (lower quality)
- Increase `--interval` to process fewer frames
- Run specific experiments instead of `--experiment all`

## Next Steps: Advanced Experiments

Once you've mastered these basics, try:

### A. Cross-Video Landmark Matching
1. Process two different videos of the same scene
2. Extract keyframes and embeddings from both
3. Find potential landmark matches using cosine similarity
4. Verify matches using LLM reasoning

### B. Temporal Scene Graph Evolution
1. Build scene graphs at multiple timestamps
2. Track how relationships change over time
3. Identify persistent vs. transient elements

### C. Integration with SLAM
1. Set up ORB-SLAM3 on your Mac
2. Run SLAM on your video to get camera poses
3. Associate FastVLM landmarks with SLAM map points
4. Create a semantic-geometric hybrid map

## Troubleshooting

**"CUDA out of memory" / "MPS out of memory":**
- Use the 0.5B model instead of 1.5B
- Increase `--interval` to process fewer frames
- Process shorter video clips

**Video won't open:**
- Make sure you have `opencv-python` installed
- Check video codec (try converting to H.264/MP4)
- Verify file path is correct

**Slow inference:**
- Expected on MacBook Air - be patient!
- Consider using GPU workstation or cloud for large-scale processing
- For prototyping, use short videos (10-30 seconds)

## Questions to Explore

As you run these experiments, consider:

1. **How stable are the embeddings?** Do similar scenes produce similar embeddings?
2. **What makes a good keyframe?** How does the similarity threshold affect keyframe density?
3. **How well does FastVLM track objects?** Can it maintain object identity across poses?
4. **How structured are the scene graphs?** What relationships does the VLM capture?
5. **What's missing for SLAM?** What additional information do you need for pose estimation?

## Resources

- **FastVLM Paper:** Understanding the FastViTHD architecture
- **LLaVA Documentation:** Prompt engineering for VLMs
- **ORB-SLAM3:** For geometric SLAM integration
- **ROVER Dataset:** Multi-season benchmark data

---

**Ready to start?** Try running Experiment 1 on your animal video and see what FastVLM can tell you!
