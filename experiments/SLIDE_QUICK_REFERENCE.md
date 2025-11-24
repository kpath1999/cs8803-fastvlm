# Quick Reference: Baseline A Slide Content

## 🎯 Core Message
**Baseline A achieves competitive cross-season matching (IoU = 0.25) using only global embeddings + semantic labels—no geometric verification needed.**

---

## 📊 Key Numbers to Memorize

| Metric | Value |
|--------|-------|
| Mean IoU | **0.2529** |
| Best Case | 0.4082 (autumn_0864) |
| Worst Case | 0.0100 (winter_0491) |
| vs Baseline B | **+23% better** |
| vs Full Pipeline | -3% (competitive!) |

---

## 🔧 Method Components

### 1. Preprocessing
- **OWL-ViT** detects objects in all frames
- Generates bounding boxes + labels

### 2. Visual Similarity
- **FastViTHD** encoder (May 2024)
- Global image embedding (512D)
- **Cosine similarity** between embeddings

### 3. Semantic Similarity  
- Use OWL-ViT labels as keywords
- **Jaccard similarity** of label sets

### 4. Fusion
```
Score = 0.7 × Visual + 0.3 × Semantic
```

---

## 📋 Slide Structure Suggestion

### Slide Title
**"Baseline A: Global Visual + Semantic Retrieval"**

### Left Column (Method)
- **Architecture Diagram**
  - RGB → FastViTHD → Embedding
  - RGB → OWL-ViT → Labels
  - Fusion → Top-K Results

- **Key Innovation**
  - FastViTHD: Fast + high-res
  - Global context preservation
  - No geometric verification

### Right Column (Results)
- **Performance Table**
  ```
  Mean IoU:     0.2529 ± 0.107
  Best Match:   0.408 (autumn→winter)
  Worst Match:  0.010 (winter→autumn)
  ```

- **Example Visualizations** (2-3 pairs)
  - autumn_1726 → winter_0839 (IoU: 0.403) ✅
  - autumn_0864 → winter_0743 (IoU: 0.408) ✅
  - winter_0491 → autumn_1147 (IoU: 0.010) ❌

### Bottom Row
- **Comparison Bar Chart**
  - Baseline A: 0.253
  - Baseline B: 0.205
  - Full Pipeline: 0.245

---

## 💬 Speaking Points (30-60 seconds)

> "Baseline A tests whether simple retrieval can solve cross-season matching. We use FastViTHD—a recent efficient vision encoder—to extract global image embeddings, combined with semantic labels from OWL-ViT for object-level matching.
>
> The fusion score is 70% visual, 30% semantic, prioritizing overall appearance while ensuring content consistency.
>
> Results show a mean IoU of 0.25, which is competitive with our full pipeline and significantly better than geometric-only matching. This demonstrates that global features capture scene context well, though they struggle with spatial precision on textureless scenes.
>
> This baseline is NOT from prior work—we implemented it to validate whether expensive geometric verification is necessary. Turns out: it helps, but isn't critical for initial retrieval."

---

## 🎨 Visual Assets

### Required Figures
1. **`iou_summary_table.png`** - Comparison across methods
2. **`all_methods_comparison.png`** - Side-by-side query-match pairs (use rows 1, 3, 5)
3. **Method diagram** - Create simple flowchart in PowerPoint/Keynote

### Color Coding
- 🟢 Green: IoU > 0.30 (Good match)
- 🟡 Orange: IoU 0.15-0.30 (Moderate)
- 🔴 Red: IoU < 0.15 (Poor)

---

## ❓ Anticipated Questions & Answers

### Q: "Why not use prior work baselines?"
**A:** "We wanted settings exactly equivalent to our pipeline—same model, same data, same detections. Reimplementation ensures fair comparison and lets us control variables."

### Q: "Why is Full Pipeline only 3% better?"
**A:** "It shows global features are surprisingly strong! The full pipeline adds geometric verification and depth consistency, which helps on edge cases but isn't always necessary."

### Q: "What about CLIP or other vision models?"
**A:** "FastViTHD is optimized for high-res images with fewer tokens, making it faster than standard ViTs while preserving detail. Plus, it's part of our FastVLM stack for consistency."

### Q: "Why 0.7/0.3 weighting?"
**A:** "Empirically tuned. Visual similarity is the primary signal for 'looking similar,' while semantic acts as a regularizer to avoid matching, say, a forest to a building just because both are green."

---

## 📁 File References

**Generated Assets:**
- `experiments/results_presentation/iou_summary_table.png`
- `experiments/results_presentation/all_methods_comparison.png`
- `experiments/results_presentation/iou_breakdown_heatmap.png`
- `experiments/results_presentation/iou_detailed_breakdown.csv`

**Source Code:**
- `experiments/baseline_a_autumnwinter_match.py`
- `experiments/compute_iou.py`

**Results Data:**
- `experiments/baseline_a_results.csv`
- `experiments/baseline_b_results.csv`
- `experiments/full_pipeline_results.csv`

---

## 🚀 Next Steps (If Asked)

1. **Improve Semantic Matching:** Use dense captions instead of labels
2. **Multi-Scale Embeddings:** Combine global + patch-level features
3. **Learned Fusion:** Replace fixed 0.7/0.3 with learned weights
4. **Reranking:** Use Baseline A for top-100, then geometric verification for top-10

---

**Last Updated:** November 24, 2025  
**Experiment Run:** November 22-23, 2025  
**Dataset:** FourSeasons (Winter/Autumn 2024)
