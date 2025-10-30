# Quick Start Guide: FastVLM for Semantic SLAM

This guide will help you get started with using FastVLM for your semantic SLAM project using your animal video.

## 📁 What You Have

Three new experimental scripts:

1. **`video_analysis_experiments.py`** - 5 foundational experiments
2. **`cross_temporal_matching.py`** - Landmark matching across videos (Phase 2 demo)
3. **`convert_rover_to_video.py`** - Helper to convert ROVER image sequences to video

## 🚀 Quick Start (5 minutes)

### Step 1: Test FastVLM on a Single Frame

First, verify FastVLM works on a frame from your video:

```bash
# Extract a frame from your video using ffmpeg (if you have it)
ffmpeg -i your_animal_video.mp4 -vframes 1 -ss 00:00:05 test_frame.jpg

# Or use Python to extract a frame:
python -c "
import cv2
cap = cv2.VideoCapture('your_animal_video.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)  # 5 seconds in
ret, frame = cap.read()
cv2.imwrite('test_frame.jpg', frame)
"

# Test FastVLM on this frame
python predict.py \
  --model-path /Users/kausar/Documents/ml-fastvlm/checkpoints/llava-fastvithd_1.5b_stage2 \
  --image-file test_frame.jpg \
  --prompt "Describe this scene in detail, focusing on permanent landmarks like trees, rocks, and terrain features."
```

### Step 2: Run Your First Video Experiment

Start with Experiment 5 (Scene Graph) - it's fast and gives interesting results:

```bash
python video_analysis_experiments.py \
  --video-path your_animal_video.mp4 \
  --experiment 5 \
  --output-dir ./my_first_experiment
```

**Expected output:**
- A scene graph describing the overall scene structure
- JSON file with objects and spatial relationships

### Step 3: Extract Keyframes

Now run Experiment 3 to see how FastVLM intelligently selects keyframes:

```bash
python video_analysis_experiments.py \
  --video-path your_animal_video.mp4 \
  --experiment 3 \
  --output-dir ./my_first_experiment
```

**This will:**
- Sample your video densely (every 0.5s by default)
- Extract FastViTHD embeddings
- Select keyframes based on visual novelty
- Show you massive data reduction (90%+ fewer frames to process)

### Step 4: Full Analysis (if you have time)

Run all experiments on a short clip:

```bash
# First, create a short test clip (first 30 seconds)
ffmpeg -i your_animal_video.mp4 -t 30 test_clip_30s.mp4

# Run all experiments
python video_analysis_experiments.py \
  --video-path test_clip_30s.mp4 \
  --experiment all \
  --interval 3.0 \
  --output-dir ./full_analysis
```

**Expected time:** 5-10 minutes on MacBook Air for 30-second clip

## 🎯 Project Ideas Ranked by Difficulty

### ⭐ Level 1: Understanding FastVLM (Today)

**Goal:** Learn what FastVLM can "see" in your video

**Steps:**
1. Run Experiment 5 (Scene Graph) on your animal video
2. Run Experiment 4 (Object Tracking) to track the animal
3. Compare FastVLM's descriptions at different timestamps

**What you'll learn:**
- How stable are FastVLM's descriptions?
- Can it track the animal reliably?
- What landmarks does it identify?

### ⭐⭐ Level 2: Keyframe-Based Analysis (This Week)

**Goal:** Build a sparse semantic representation of your video

**Steps:**
1. Run Experiment 3 to extract keyframes
2. Run Experiment 1 on those keyframes only
3. Analyze the embeddings - which frames are similar?

**What you'll learn:**
- How to select informative frames
- Embedding-based similarity for scene comparison
- Foundation for Phase 1 of your SLAM system

**Experiment:**
```bash
# Step 1: Get keyframes
python video_analysis_experiments.py \
  --video-path your_animal_video.mp4 \
  --experiment 3 \
  --output-dir ./keyframe_analysis

# Step 2: Analyze the keyframe selection
python -c "
import json
with open('./keyframe_analysis/experiment_3_keyframes.json') as f:
    kf = json.load(f)
print(f'Selected {len(kf)} keyframes')
print('Timestamps:', [k['timestamp_ms']/1000 for k in kf])
"
```

### ⭐⭐⭐ Level 3: Cross-Temporal Matching (Next Week)

**Goal:** Match the same landmarks across different parts of your video

**Steps:**
1. Split your video into two segments (early vs late)
2. Run `cross_temporal_matching.py` to find common landmarks
3. Visualize the matches

**Experiment:**
```bash
# Split your video
ffmpeg -i your_animal_video.mp4 -t 30 segment1.mp4
ffmpeg -i your_animal_video.mp4 -ss 30 segment2.mp4

# Match landmarks across segments
python cross_temporal_matching.py \
  --video1 segment1.mp4 \
  --video2 segment2.mp4 \
  --context1 "First 30 seconds" \
  --context2 "After 30 seconds" \
  --max-frames 5 \
  --output-dir ./matching_test
```

**What you'll learn:**
- Embedding-based candidate pairing
- Semantic verification (Phase 2 foundation)
- How well landmarks persist over time

### ⭐⭐⭐⭐ Level 4: ROVER Multi-Season Matching (Next Month)

**Goal:** Match landmarks between summer and winter ROVER sequences

**Prerequisites:**
- Download one ROVER summer sequence (~40GB)
- Download matching winter sequence (~30GB)

**Steps:**
1. Convert ROVER image sequences to video:
```bash
python convert_rover_to_video.py \
  --input-dir /path/to/ROVER/campus_large_summer/realsense_D435i/rgb \
  --output-path rover_summer_d435i.mp4 \
  --fps 30
```

2. Extract keyframes from both:
```bash
python video_analysis_experiments.py \
  --video-path rover_summer_d435i.mp4 \
  --experiment 3 \
  --output-dir ./rover_summer_keyframes

python video_analysis_experiments.py \
  --video-path rover_winter_d435i.mp4 \
  --experiment 3 \
  --output-dir ./rover_winter_keyframes
```

3. Match landmarks across seasons:
```bash
python cross_temporal_matching.py \
  --video1 rover_summer_d435i.mp4 \
  --video2 rover_winter_d435i.mp4 \
  --context1 "Summer 2023" \
  --context2 "Winter 2024" \
  --max-frames 20 \
  --output-dir ./rover_seasonal_matching
```

**What you'll learn:**
- Real-world seasonal invariance challenges
- How well semantic matching handles drastic appearance changes
- Foundation for multi-agent SLAM

### ⭐⭐⭐⭐⭐ Level 5: Full Semantic SLAM Pipeline (Long-term Goal)

**Goal:** Implement all 3 phases with geometric SLAM integration

**Requirements:**
- ORB-SLAM3 or similar visual SLAM system
- Access to more powerful hardware (Linux workstation or cloud)
- Integration with pose graph optimization (g2o or Ceres)

**Components:**
1. Phase 1: Keyframe semantic extraction (you're learning this now!)
2. Phase 2: LLM-based landmark matching (cross_temporal_matching.py is a start)
3. Phase 3: Pose graph optimization with semantic constraints

**This is research-level work** and will take several months to develop properly.

## 📊 Expected Performance

On **MacBook Air M2/M3** with the **1.5B model**:

| Operation | Time per Frame | Notes |
|-----------|---------------|-------|
| Single description | 2-5 seconds | With prompt |
| Embedding extraction | 0.5-1 second | FastViTHD features |
| Keyframe selection | 1 sec/frame | Embedding-only |
| LLM verification | 3-5 seconds | Per pair (demo version) |

**For a 1-minute video:**
- Experiment 3 (keyframes): ~2-3 minutes
- Experiment 1 (full analysis at 2s intervals): ~3-5 minutes
- Cross-temporal matching (10 frames each): ~10-15 minutes

## 🎓 Learning Path

### Week 1: FastVLM Basics
- ✅ Run `predict.py` on individual frames
- ✅ Run Experiment 5 (Scene Graph)
- ✅ Run Experiment 4 (Object Tracking)
- **Deliverable:** Understand what FastVLM "sees"

### Week 2: Keyframe Analysis
- ✅ Run Experiment 3 (Keyframe Selection)
- ✅ Analyze embedding similarity patterns
- ✅ Manually inspect selected keyframes
- **Deliverable:** Sparse semantic map of your video

### Week 3: Temporal Matching
- ✅ Run `cross_temporal_matching.py` on video segments
- ✅ Analyze matching accuracy
- ✅ Experiment with different prompts
- **Deliverable:** Understanding of semantic matching

### Week 4: ROVER Dataset
- ✅ Download small ROVER sequence
- ✅ Convert to video
- ✅ Run experiments on real-world data
- **Deliverable:** Baseline results on seasonal data

### Month 2-3: SLAM Integration
- Set up ORB-SLAM3
- Integrate semantic landmarks with geometric map
- Implement pose graph optimization
- **Deliverable:** Prototype semantic SLAM system

## 🐛 Troubleshooting

**"ModuleNotFoundError: No module named 'tqdm'"**
```bash
pip install tqdm opencv-python
```

**"Video won't open"**
```bash
# Check video codec
ffmpeg -i your_video.mp4

# Convert to standard H.264 if needed
ffmpeg -i your_video.mp4 -c:v libx264 -c:a aac output.mp4
```

**"Too slow / Out of memory"**
- Use the 0.5B model instead of 1.5B
- Increase `--interval` to process fewer frames
- Use shorter video clips for testing
- Reduce `--max-frames` in cross_temporal_matching.py

**"Embeddings don't seem to match similar scenes"**
- This is expected! Vision embeddings are appearance-based
- That's why we need LLM reasoning for semantic matching
- Try adjusting `--embedding-threshold` (lower = more candidates)

## 💡 Tips for Success

1. **Start small:** Use 10-30 second clips for initial experiments
2. **Iterate on prompts:** The quality of FastVLM's output depends heavily on your prompts
3. **Visualize results:** Look at the actual frames that get selected/matched
4. **Keep notes:** Document which experiments give useful results
5. **Be patient:** Inference on Mac is slow but sufficient for prototyping

## 📚 Next Steps

After running these experiments, you'll want to:

1. **Integrate external LLM:** Replace heuristic matching with GPT-4 API
2. **Add geometric constraints:** Incorporate depth data and camera poses
3. **Scale up:** Test on full ROVER sequences
4. **Optimize:** Move to GPU hardware for real-time processing
5. **Publish:** This could be a strong research contribution!

## 🔗 Resources

- **FastVLM Paper:** [Link to understand the architecture]
- **ROVER Dataset:** https://sites.google.com/view/rover-dataset
- **ORB-SLAM3:** https://github.com/UZ-SLAMLab/ORB_SLAM3
- **LLaVA:** https://github.com/haotian-liu/LLaVA

---

**Ready to start?** Run this command:

```bash
python video_analysis_experiments.py \
  --video-path your_animal_video.mp4 \
  --experiment 5 \
  --output-dir ./getting_started
```

Then open `./getting_started/experiment_5_scene_graph.json` to see what FastVLM discovered! 🚀
