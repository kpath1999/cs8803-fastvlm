# Office Loop Route Comparison Guide

## What This Does

The `compare_office_loops.py` script is specifically designed to analyze your office loop dataset with 8 videos recorded on different dates/conditions. It answers three key questions:

1. **Are these the same route?** - Computes similarity scores across all video pairs
2. **What landmarks are invariant?** - Identifies features visible in all conditions
3. **What changes between conditions?** - Highlights seasonal/weather differences

## Quick Start

### Basic Usage

```bash
# Install matplotlib and seaborn for visualizations
pip install matplotlib seaborn

# Run on all videos in the office loop directory
python compare_office_loops.py \
  --video-dir data/fourseasons/officeloop \
  --output-dir ./office_loop_results
```

### Testing on Subset

```bash
# Process only first 3 videos (faster for testing)
python compare_office_loops.py \
  --video-dir data/fourseasons/officeloop \
  --max-videos 3 \
  --num-segments 5 \
  --output-dir ./office_loop_test
```

## What You'll Get

### 1. Similarity Matrix (`similarity_matrix.png`)

A heatmap showing how similar each video pair is:
- **High scores (>0.7)**: Same route, different conditions
- **Medium scores (0.5-0.7)**: Similar but with variations
- **Low scores (<0.5)**: Different routes or major changes

### 2. Route Progression (`route_progression.png`)

Shows how the visual scene changes as you progress along the route:
- Each line represents one video
- Similar patterns = same route
- Diverging patterns = route variations

### 3. Route Segments (`route_segments.json`)

Detailed data for each video segment:
```json
{
  "recording_2020-03-24_17-36-22_5x": [
    {
      "frame_number": 0,
      "route_position": 0.0,
      "description": "Driving through parking lot with brick building on left...",
      "landmarks": ["red brick building", "parking sign", "tree line"],
      "condition": "spring"
    },
    ...
  ]
}
```

### 4. Invariant Landmarks (`invariant_landmarks.json`)

Features that appear across all videos:
```json
[
  {
    "route_position": 0.2,
    "landmark": "brick building",
    "frequency": 0.875,
    "videos_seen": 7
  }
]
```

These are your **season-invariant features** - perfect for SLAM!

### 5. Sample Matches (`sample_matches.json`)

Examples of matched segments across videos:
```json
{
  "video1": "recording_2020-03-24_...",
  "video2": "recording_2021-01-07_...",
  "route_position_1": 0.3,
  "route_position_2": 0.33,
  "embedding_similarity": 0.82,
  "description_1": "Parking lot with spring foliage...",
  "description_2": "Same parking lot in winter, snow visible...",
  "condition_differences": ["Condition: spring vs winter"],
  "invariant_features": ["brick building", "parking structure"]
}
```

## Understanding the Results

### Route Similarity Interpretation

**Average Similarity > 0.7:**
```
✓ These are definitely the SAME ROUTE
→ Perfect for multi-condition SLAM
→ Invariant landmarks are reliable
```

**Average Similarity 0.5-0.7:**
```
⚠️  Likely the same route with some variations
→ May have taken slightly different paths
→ Some landmarks still usable
```

**Average Similarity < 0.5:**
```
❌ Different routes or major reconstruction
→ Not ideal for direct matching
→ May need different approach
```

### Using Invariant Landmarks

The landmarks with high `frequency` scores are your best candidates for:
- Season-invariant localization
- Loop closure detection
- Cross-condition place recognition

Example:
```json
{
  "route_position": 0.4,
  "landmark": "corner brick building",
  "frequency": 0.875,  // Seen in 87.5% of videos
  "videos_seen": 7     // Appeared in 7 out of 8 videos
}
```

This landmark at 40% along the route is reliable across seasons!

## Advanced Usage

### Adjust Number of Segments

```bash
# More segments = finer granularity (but slower)
python compare_office_loops.py \
  --num-segments 20 \
  --output-dir ./detailed_analysis
```

### Use Different Model

```bash
# Try the 0.5B model for faster processing
python compare_office_loops.py \
  --model-path ./checkpoints/llava-fastvithd_0.5b_stage2 \
  --output-dir ./fast_analysis
```

## Expected Performance

On MacBook Air with 8 videos, 10 segments each:

| Operation | Time Estimate |
|-----------|--------------|
| Model loading | 30-60 sec |
| Per-segment analysis | 3-5 sec |
| Total for 80 segments | 4-7 minutes |
| Matching & visualization | 30 sec |
| **Total runtime** | **5-8 minutes** |

## Interpreting Your Office Loop Data

Based on your 8 videos spanning from March 2020 to May 2021:

**Seasonal Coverage:**
- **Spring**: March-May videos (4 videos)
- **Summer**: June video (1 video)
- **Winter**: January-February videos (2 videos)
- **Fall**: None in your dataset

**What to Look For:**

1. **Do all videos show the same starting point?**
   - Check `route_position: 0.0` descriptions

2. **How do landmarks change across seasons?**
   - Compare descriptions at same `route_position` values
   - Look for invariant vs. condition-specific features

3. **Which segments are most reliable?**
   - High embedding similarity across all videos
   - Consistent landmark descriptions

## Next Steps After Analysis

### 1. Identify Best Keyframes for SLAM

```python
import json

# Load results
with open('./office_loop_results/invariant_landmarks.json') as f:
    landmarks = json.load(f)

# Find most reliable landmarks
reliable = [lm for lm in landmarks if lm['frequency'] > 0.75]
print(f"Found {len(reliable)} highly reliable landmarks")

# Use these positions for keyframe extraction
keyframe_positions = [lm['route_position'] for lm in reliable]
```

### 2. Build Condition-Invariant Map

The invariant landmarks form your semantic map:
- Each landmark is a "node" in your map
- Position along route provides ordering
- Descriptions enable re-identification

### 3. Test Loop Closure Detection

Pick two videos with different conditions:
```bash
# First, run the comparison
python compare_office_loops.py --max-videos 2

# Check if the end matches the beginning
# (This tests if loop closure is detectable)
```

### 4. Extend to Full ROVER Dataset

Once you understand the office loop results, apply the same approach to ROVER's campus sequences!

## Troubleshooting

**"Similarity is very low (<0.3)":**
- Videos might not be the same route
- Try increasing `--num-segments` for better coverage
- Check if videos are aligned (same starting point)

**"All similarities are >0.95":**
- Videos might be nearly identical (same day, similar conditions)
- This is actually good for validation!

**"Not enough invariant landmarks":**
- Try lowering the frequency threshold in the code
- Increase `--num-segments` to sample more of the route
- Some routes naturally have fewer distinctive features

## Example Analysis Workflow

```bash
# 1. Quick test with 2 videos
python compare_office_loops.py \
  --max-videos 2 \
  --num-segments 5 \
  --output-dir ./test

# 2. Check the similarity
cat ./test/similarity_matrix.png  # Should be high if same route

# 3. Full analysis
python compare_office_loops.py \
  --output-dir ./full_results

# 4. Examine invariant landmarks
cat ./full_results/invariant_landmarks.json | head -20

# 5. Look at a specific match
cat ./full_results/sample_matches.json | head -30
```

## Connection to Your SLAM Goals

This script implements:

✅ **Phase 1**: Semantic landmark extraction at route segments  
✅ **Phase 2 (partial)**: Condition-aware matching using embeddings  
⬜ **Phase 3**: Would require geometric SLAM integration  

The invariant landmarks you discover here are exactly what you need for:
- Multi-season localization
- Loop closure in changing conditions
- Semantic map building

---

**Ready to analyze?** Run:

```bash
python compare_office_loops.py
```

Then check `./office_loop_analysis/similarity_matrix.png` to see if these are the same route!
