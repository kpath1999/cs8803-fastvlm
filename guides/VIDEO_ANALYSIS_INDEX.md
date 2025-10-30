# FastVLM Video Analysis Experiments - Index

## 📚 Documentation

Start here to understand the project:

1. **[QUICK_START.md](QUICK_START.md)** ⭐ **START HERE**
   - 5-minute quick start guide
   - Step-by-step learning path
   - Troubleshooting and tips

2. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**
   - Complete overview of what was built
   - Feasibility assessment for your SLAM goals
   - Success metrics and milestones

3. **[VIDEO_EXPERIMENTS_README.md](VIDEO_EXPERIMENTS_README.md)**
   - Detailed explanation of each experiment
   - Connection to semantic SLAM phases
   - Performance expectations and examples

## 🛠️ Scripts

### Core Analysis Scripts

1. **[video_analysis_experiments.py](video_analysis_experiments.py)**
   - Main experimental framework
   - 5 different experiments
   - Command: `python video_analysis_experiments.py --help`

2. **[cross_temporal_matching.py](cross_temporal_matching.py)**
   - Landmark matching across videos/time
   - Phase 2 demonstration
   - Command: `python cross_temporal_matching.py --help`

3. **[visualize_results.py](visualize_results.py)**
   - Create visual outputs from results
   - Keyframe grids, timelines, similarity plots
   - Command: `python visualize_results.py --help`

### Helper Scripts

4. **[convert_rover_to_video.py](convert_rover_to_video.py)**
   - Convert ROVER image sequences to video
   - Command: `python convert_rover_to_video.py --help`

5. **[install_video_deps.sh](install_video_deps.sh)**
   - Install required dependencies
   - Command: `bash install_video_deps.sh`

## 🚀 Quick Start Commands

### First Time Setup

```bash
# Install dependencies
bash install_video_deps.sh

# Or manually:
pip install opencv-python tqdm matplotlib
```

### Run Your First Experiment

```bash
# Build a scene graph of your video (fastest experiment)
python video_analysis_experiments.py \
  --video-path your_animal_video.mp4 \
  --experiment 5 \
  --output-dir ./first_experiment

# Check the results
cat ./first_experiment/experiment_5_scene_graph.json
```

### Run All Experiments

```bash
# Full analysis pipeline
python video_analysis_experiments.py \
  --video-path your_animal_video.mp4 \
  --experiment all \
  --interval 2.0 \
  --output-dir ./full_analysis

# Create visualizations
python visualize_results.py \
  --video-path your_animal_video.mp4 \
  --results-dir ./full_analysis \
  --output-dir ./visualizations
```

### Cross-Temporal Matching

```bash
# Match landmarks between two videos
python cross_temporal_matching.py \
  --video1 summer_video.mp4 \
  --video2 winter_video.mp4 \
  --context1 "Summer" \
  --context2 "Winter" \
  --output-dir ./matching_results
```

## 📋 Experiment Guide

### Experiment 1: Frame-by-Frame Semantic Extraction
- **Time:** ~3-5 min for 1-min video
- **Best for:** Understanding what FastVLM "sees"
- **Outputs:** Frame descriptions, landmarks, objects, embeddings

### Experiment 2: Temporal Change Detection
- **Time:** ~5-8 min for 1-min video
- **Best for:** Understanding scene dynamics
- **Outputs:** Frame-to-frame change descriptions

### Experiment 3: Semantic Keyframe Selection ⭐
- **Time:** ~2-3 min for 1-min video
- **Best for:** Efficient video processing
- **Outputs:** Selected keyframes based on visual novelty

### Experiment 4: Object Tracking
- **Time:** ~3-5 min for 1-min video
- **Best for:** Following specific objects/animals
- **Outputs:** Tracking timeline, detection status

### Experiment 5: Scene Graph Construction ⚡
- **Time:** ~10-20 sec
- **Best for:** Quick scene understanding
- **Outputs:** Structured scene representation

## 🎯 Project Roadmap

### ✅ Phase 1: Foundation (This Week)
- Run experiments on animal video
- Understand FastVLM capabilities
- Explore keyframe selection

### ⬜ Phase 2: Temporal Analysis (Next Week)
- Test cross-temporal matching
- Experiment with different prompts
- Analyze matching accuracy

### ⬜ Phase 3: ROVER Integration (This Month)
- Download ROVER dataset
- Process seasonal sequences
- Match landmarks across seasons

### ⬜ Phase 4: SLAM Integration (Long Term)
- Set up ORB-SLAM3
- Integrate geometric and semantic
- Implement pose graph optimization

## 📊 Expected Results

### What FastVLM Excels At
✅ Rich semantic descriptions  
✅ Object identification  
✅ Spatial relationship understanding  
✅ Robust to viewing angle changes  
✅ Natural language reasoning  

### Current Limitations
⚠️ Not real-time on Mac  
⚠️ Embeddings change with appearance  
⚠️ No geometric understanding  
⚠️ Requires external LLM for best matching  
⚠️ Limited by prompt engineering  

## 🔧 Troubleshooting

**Import errors:**
```bash
bash install_video_deps.sh
```

**Video won't open:**
```bash
# Convert to H.264
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
```

**Too slow:**
- Use shorter video clips (10-30 seconds)
- Increase `--interval` parameter
- Use 0.5B model instead of 1.5B
- Reduce `--max-frames` in matching script

**Out of memory:**
- Close other applications
- Process fewer frames at once
- Use smaller model (0.5B)

## 📖 Learning Resources

### Understanding the Code
- Read docstrings in each script
- Start with `video_analysis_experiments.py`
- Check `FrameAnalysis` and `LandmarkMatch` dataclasses

### SLAM Background
- ORB-SLAM3 paper: Visual SLAM with relocalization
- ROVER dataset paper: Multi-season robotics dataset
- Your own notes on semantic SLAM phases

### FastVLM Architecture
- FastViTHD: Vision encoder
- LLaVA: Vision-language integration
- Qwen-2: Language model backbone

## 🤝 Contributing

Feel free to extend these experiments:

1. Add new prompts for different landmark types
2. Implement better LLM integration (GPT-4, Claude)
3. Add geometric consistency checks
4. Create better visualizations
5. Optimize for real-time performance

## 📝 Citation

If you use this in research:

```bibtex
@software{fastvlm_video_analysis,
  title={FastVLM Video Analysis for Semantic SLAM},
  author={Your Name},
  year={2025},
  note={Experimental framework for season-invariant landmark detection}
}
```

## 📞 Support

If you get stuck:
1. Check [QUICK_START.md](QUICK_START.md) troubleshooting section
2. Review command-line help: `python script.py --help`
3. Look at example outputs in documentation
4. Start with simpler experiments first

---

**Ready to start?** Open [QUICK_START.md](QUICK_START.md) and follow the 5-minute guide!
