# Frame Comparisons Summary

## Overview

Successfully generated **40 comparison figures** organized in `experiments/results_presentation/frame_comparisons/`

## Structure

### 1. Single-Method Comparisons (30 files)
Each query-match pair has 3 individual comparison figures:

**Format:** `{query_id}_{match_id}_{method}.png`

**Example for autumn_1726:**
- `autumn_1726_winter_0839_baseline_a.png` - Baseline A (Visual + Semantic)
- `autumn_1726_winter_0000_baseline_b.png` - Baseline B (Geometric Keypoints)  
- `autumn_1726_winter_0839_full_pipeline.png` - Full Pipeline (Hierarchical)

**Content:**
- Left panel: Query image with cyan bounding boxes
- Right panel: Top-1 match with lime bounding boxes
- Title includes: Method name, IoU score, additional metrics (Jaccard, inliers)

### 2. Cross-Method Comparisons (10 files)
One comprehensive comparison per query showing all methods:

**Format:** `{query_id}_{match_id}_all_methods.png`

**Examples:**
- `autumn_0000_winter_0456_all_methods.png`
- `autumn_0305_winter_0345_all_methods.png`
- `autumn_0864_winter_0743_all_methods.png`
- `autumn_1136_winter_0983_all_methods.png`
- `autumn_1726_winter_0839_all_methods.png`
- `winter_0000_autumn_0894_all_methods.png`
- `winter_0309_autumn_0894_all_methods.png`
- `winter_0491_autumn_1147_all_methods.png`
- `winter_0797_autumn_0736_all_methods.png`
- `winter_1109_autumn_0736_all_methods.png`

**Content:**
- Column 1: Query image (cyan boxes)
- Column 2: Baseline A top-1 match (lime boxes)
- Column 3: Baseline B top-1 match (lime boxes)
- Column 4: Full Pipeline top-1 match (lime boxes)
- Each match annotated with IoU score and color-coded:
  - 🟢 Green: IoU > 0.30 (good match)
  - 🟡 Orange: IoU 0.15-0.30 (moderate)
  - 🔴 Red: IoU < 0.15 (poor)

## Query Coverage

**Total Queries:** 10 (5 autumn, 5 winter)

### Autumn → Winter Queries
1. `autumn_0000` → `winter_0456` (BA), `winter_0002` (BB), `winter_0413` (FP)
2. `autumn_0305` → `winter_0345` (BA, FP), `winter_0006` (BB)
3. `autumn_0864` → `winter_0743` (BA), `winter_0000` (BB), `winter_0686` (FP)
4. `autumn_1136` → `winter_0983` (BA), `winter_0001` (BB), `winter_0259` (FP)
5. `autumn_1726` → `winter_0839` (BA, FP), `winter_0000` (BB)

### Winter → Autumn Queries
1. `winter_0000` → `autumn_0894` (BA, BB, FP)
2. `winter_0309` → `autumn_0894` (BA, BB, FP)
3. `winter_0491` → `autumn_1147` (BA, FP), `autumn_0000` (BB)
4. `winter_0797` → `autumn_0736` (BA, BB, FP)
5. `winter_1109` → `autumn_0736` (BA, FP), `autumn_0000` (BB)

## File Statistics

```
Total files: 40
├── baseline_a:        10 files
├── baseline_b:        10 files
├── full_pipeline:     10 files
└── all_methods:       10 files
```

**Average file size:** ~1.2 MB per image (150 DPI, high quality)

## Usage for Presentation

### For Individual Method Slides
Use single-method comparison files to show:
- Baseline A performance examples
- Baseline B performance examples
- Full Pipeline performance examples

### For Comparative Analysis
Use cross-method comparison files to show:
- How different methods perform on the same query
- Highlight where methods agree/disagree
- Demonstrate strengths/weaknesses of each approach

### Recommended Figures for Slides

**Best Matches (IoU > 0.40):**
- `autumn_0864_winter_0743_all_methods.png` - Strong performance across methods
- `autumn_1726_winter_0839_all_methods.png` - Good visual and semantic alignment

**Method Disagreement Cases:**
- `winter_0491_autumn_1147_all_methods.png` - BA fails (0.010), BB succeeds (0.325)
- `winter_0000_autumn_0894_all_methods.png` - BB wins (0.325), BA/FP weaker

**Moderate Performance:**
- `autumn_0000_winter_0456_all_methods.png` - Consistent ~0.25-0.30 across all

## Color Coding Legend

**Bounding Boxes:**
- **Cyan:** Query frame detections
- **Lime:** Match frame detections

**Title Colors (IoU quality):**
- **Green:** Good match (IoU > 0.30)
- **Orange:** Moderate match (IoU 0.15-0.30)
- **Red:** Poor match (IoU < 0.15)

## Regeneration

To regenerate all comparisons:
```bash
python experiments/create_slide_figure.py
```

This will:
1. Load frame mappings from `detections.json`
2. Load top-1 results from all CSV files
3. Generate 30 single-method + 10 cross-method comparisons
4. Save to `experiments/results_presentation/frame_comparisons/`

---

**Generated:** November 24, 2025  
**Script:** `experiments/create_slide_figure.py`  
**Output Directory:** `experiments/results_presentation/frame_comparisons/`
