# Experimental Evaluation Framework

This directory contains scripts for controlled experimentation comparing three matching approaches:
- **Baseline A**: Embedding-only matching (FastVLM + cosine similarity)
- **Baseline B**: Geometric-only matching (ORB/SIFT keypoints)
- **Full Pipeline**: Multi-modal fusion (embedding + semantic + depth + geometric)

## Experimental Workflow

### Stage 1: Generate Controlled Detections

Run OWL-ViT **once** on all frame pairs and save detections to ensure all methods use identical bounding boxes:

```bash
python experiments/generate_detections.py \
    --winter-rgb /Volumes/KAUSAR/rover_dataset/2024-01-13/realsense_D435i/rgb \
    --autumn-rgb /Volumes/KAUSAR/rover_dataset/2024-04-11/realsense_D435i/rgb \
    --output experiments/detections.json \
    --adaptive-thresholds \
    --use-keyframing \
    --max-pairs 100
```

**Important**: Use `--adaptive-thresholds` (without `--focused-queries`) to ensure **broad category coverage**:
- **37 categories** covering permanent structures (house, tree, fence) + temporal objects (bush, grass, flowers)
- **Per-category optimized thresholds** based on empirical analysis (e.g., house: 0.089, grass: 0.012)
- See **[DETECTION_STRATEGY.md](/archive/DETECTION_STRATEGY.md)** for details on query selection and thresholds

**Output**: `experiments/detections.json` containing all detected bounding boxes, labels, and scores for each frame pair.

---

### Stage 2: Run Baseline A (Embedding-Only)

Match detections using **only** FastVLM visual embeddings and cosine similarity:

```bash
python experiments/baseline_a_embedding.py \
    --detections experiments/detections.json \
    --winter-rgb /Volumes/KAUSAR/rover_dataset/2024-01-13/realsense_D435i/rgb \
    --autumn-rgb /Volumes/KAUSAR/rover_dataset/2024-04-11/realsense_D435i/rgb \
    --model-path checkpoints/llava-fastvithd_0.5b_stage2 \
    --output experiments/results_baseline_a.json \
    --similarity-threshold 0.7
```

**Output**: `experiments/results_baseline_a.json` containing matches based purely on embedding similarity.

---

### Stage 3: Run Baseline B (Geometric-Only)

Match detections using **only** ORB/SIFT keypoint matching within bounding boxes:

```bash
python experiments/baseline_b_geometric.py \
    --detections experiments/detections.json \
    --output experiments/results_baseline_b.json \
    --method orb \
    --min-inliers 4 \
    --min-confidence 0.3
```

**Output**: `experiments/results_baseline_b.json` containing matches based purely on geometric keypoint correspondences.

---

### Stage 4: Run Full Pipeline (Multi-Modal Fusion)

Match detections using **all** cues (embedding + semantic + depth + geometric):

```bash
python experiments/full_pipeline_comparison.py \
    --detections experiments/detections.json \
    --winter-rgb /Volumes/KAUSAR/rover_dataset/2024-01-13/realsense_D435i/rgb \
    --winter-depth /Volumes/KAUSAR/rover_dataset/2024-01-13/realsense_D435i/depth \
    --autumn-rgb /Volumes/KAUSAR/rover_dataset/2024-04-11/realsense_D435i/rgb \
    --autumn-depth /Volumes/KAUSAR/rover_dataset/2024-04-11/realsense_D435i/depth \
    --model-path checkpoints/llava-fastvithd_0.5b_stage2 \
    --output experiments/results_full_pipeline.json \
    --embedding-threshold 0.7 \
    --min-inliers 4 \
    --min-geometric-conf 0.3
```

**Output**: `experiments/results_full_pipeline.json` containing matches using hierarchical multi-modal fusion.

---

### Stage 5: Evaluate and Compare Methods

Generate comprehensive evaluation report answering research questions RQ1-RQ5:

```bash
python experiments/evaluate_methods.py \
    --baseline-a experiments/results_baseline_a.json \
    --baseline-b experiments/results_baseline_b.json \
    --full-pipeline experiments/results_full_pipeline.json \
    --output experiments/evaluation_report.json
```

**Output**: `experiments/evaluation_report.json` containing metrics and findings for all research questions.

---

### Optional: Generate Visualizations

Create publication-quality plots for all research questions:

```bash
python experiments/visualize_results.py \
    --evaluation experiments/evaluation_report.json \
    --output experiments/figures/
```

**Output**: PNG figures in `experiments/figures/` for each research question.

---

### Optional: Generate Results Table

Create a markdown summary table for quick reference:

```bash
python experiments/generate_results_table.py \
    --evaluation experiments/evaluation_report.json \
    --output experiments/results_table.md
```

**Output**: `experiments/results_table.md` with formatted comparison tables.

---

## Research Questions

### RQ1: Multi-Modal Fusion Value
**Question**: Does combining multiple cues produce higher-quality matches than single-cue approaches?

**Metrics**:
- Average final confidence score
- Percentage of high-confidence matches (>0.7)
- Geometric inlier counts across methods

**Expected Finding**: Full pipeline should have higher average confidence and more geometric inliers.

---

### RQ2: Geometric vs. Semantic Reliability
**Question**: Which is more reliable—geometric features or semantic features?

**Metrics**:
- Total number of matches found per method
- Distribution of confidence scores
- Qualitative analysis of failure cases

**Expected Finding**: Geometric matching produces fewer but higher-precision matches; embedding matching finds more candidates but with lower precision.

---

### RQ3: Hierarchical Filtering Effectiveness
**Question**: Does hierarchical progression from embedding → semantic → depth → geometric successfully filter false positives?

**Metrics**:
- Track candidate pairs passing each stage
- Analyze the "funnel" effect from initial similarity to final verification

**Expected Finding**: Many pairs have high embedding similarity, but only a subset also have high geometric consistency.

---

### RQ4: Invariant vs. Temporal Classification
**Question**: Can depth consistency distinguish permanent structures from seasonal objects?

**Metrics**:
- Average depth consistency for "invariant" vs. "temporal" matches
- Statistical separation (t-test or KS-test)

**Expected Finding**: Invariant landmarks cluster at high depth consistency (>0.8); temporal objects show lower consistency.

---

### RQ5: Robustness to Appearance Change
**Question**: How well does each method handle challenging seasonal appearance changes?

**Metrics**:
- Manually identify 10-15 "challenging" frame pairs
- Compare success rates (% correct matches) across methods

**Expected Finding**:
- Embedding-only fails on visually similar but distinct objects
- Geometric-only fails when texture changes drastically (e.g., snow)
- Full pipeline succeeds by requiring agreement across modalities

---

## File Structure

```
experiments/
├── README.md                          # This file
├── SUMMARY.md                         # Quick reference guide
├── run_all_experiments.sh             # One-command runner for all stages
├── generate_detections.py             # Stage 1: OWL-ViT detection
├── baseline_a_embedding.py            # Stage 2: Embedding-only matching
├── baseline_b_geometric.py            # Stage 3: Geometric-only matching
├── full_pipeline_comparison.py        # Stage 4: Full multi-modal pipeline
├── evaluate_methods.py                # Stage 5: Evaluation and comparison
├── visualize_results.py               # Optional: Generate plots
├── generate_results_table.py          # Optional: Generate markdown table
├── detections.json                    # Controlled detections (from Stage 1)
├── results_baseline_a.json            # Baseline A results (from Stage 2)
├── results_baseline_b.json            # Baseline B results (from Stage 3)
├── results_full_pipeline.json         # Full pipeline results (from Stage 4)
├── evaluation_report.json             # Final evaluation report (from Stage 5)
├── results_table.md                   # Summary table (optional)
└── figures/                           # Visualization plots (optional)
    ├── rq1_multimodal_fusion.png
    ├── rq2_geometric_vs_semantic.png
    ├── rq3_hierarchical_filtering.png
    ├── rq4_invariant_vs_temporal.png
    └── rq5_robustness.png
```
├── full_pipeline_comparison.py        # Stage 4: Full multi-modal pipeline
├── evaluate_methods.py                # Stage 5: Evaluation and comparison
├── detections.json                    # Controlled detections (from Stage 1)
├── results_baseline_a.json            # Baseline A results (from Stage 2)
├── results_baseline_b.json            # Baseline B results (from Stage 3)
├── results_full_pipeline.json         # Full pipeline results (from Stage 4)
└── evaluation_report.json             # Final evaluation report (from Stage 5)
```

---

## Key Design Principles

1. **Controlled Detection**: All methods use identical OWL-ViT detections (same bounding boxes, labels, scores)
2. **Isolated Comparison**: Each method uses different matching strategies on the same input
3. **Reproducibility**: All scripts are deterministic and results are saved to JSON
4. **Comprehensive Metrics**: Evaluation covers all research questions with statistical rigor

---

## Notes

- All scripts import functions from `cross_temporal_pipeline.py` to ensure consistency
- Results are saved as JSON for easy analysis and visualization
- The evaluation framework provides both quantitative metrics and qualitative insights
- For RQ5 (robustness), manual annotation of "challenging" pairs is recommended for ground truth
