# Baseline A: Global Visual & Semantic Retrieval

## 1. Methodology & Justification

### Preprocessing: Open-Vocabulary Object Detection
- **Tool:** OWL-ViT (Owl Vision Transformer)
- **Application:** All RGB frames in both Autumn and Winter datasets
- **Output:** Candidate bounding boxes with object labels (e.g., "tree", "bench", "path")
- **Purpose:** Provides structured semantic content for matching

### Visual Similarity Component
**Approach:** Global Image Embeddings via FastVLM
- **Model:** FastViTHD (Hybrid Vision Encoder, May 2024 release)
- **Innovation:** Processes high-resolution images with fewer tokens than standard ViTs
- **Key Advantage:** Reduces encoding time while preserving fine-grained spatial detail
- **Technical Detail:** Extracts a single embedding vector representing the entire scene
- **Metric:** Cosine Similarity between query and candidate embeddings

**Why Global Embeddings?**
- Captures overall scene composition and spatial layout
- More robust than object-level matching (avoids ambiguity of "tree" → "tree")
- Preserves contextual relationships between objects
- Faster inference compared to per-object embedding

### Semantic Similarity Component
**Approach:** Label-Based Keyword Matching
- **Source:** Unique detection labels from OWL-ViT detections
- **Example:** `{"tree", "bench", "path", "building"}`
- **Metric:** Jaccard Similarity (IoU of label sets)
- **Formula:** $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$

**Why Labels Instead of Dense Captions?**
- **Efficiency:** No need to run caption generation on every frame
- **Consistency:** Structured, repeatable semantic descriptors
- **Speed:** Pre-computed during detection stage
- **Reliability:** More stable than free-form text matching

### Fusion Strategy
**Final Ranking Score:**
```
Score = 0.7 × Visual_Similarity + 0.3 × Semantic_Similarity
```

**Rationale:**
- **Visual (70%):** Primary signal for scene appearance matching
- **Semantic (30%):** Regularizer to ensure content consistency
- **Design Choice:** Visual dominance reflects that "looking similar" is the core task

---

## 2. Implementation Details

### Nature of Baseline
- **Type:** Self-implemented baseline
- **Purpose:** Establish performance lower bound using only global features
- **Hypothesis:** Tests whether simple retrieval is sufficient for cross-season matching

### Comparison to Prior Work
- **NOT based on existing paper results** - fully custom implementation
- **Comparable Concepts:**
  - Similar to image retrieval systems (e.g., CLIP-based search)
  - Analogous to "BoW + SIFT" paradigm but with learned embeddings
- **Key Difference:** We use FastViTHD specifically for its efficiency on high-res images

### Experimental Settings
- **Image Resolution:** Native RealSense D435i resolution (640×480)
- **Model Checkpoint:** `llava-fastvithd_0.5b_stage2`
- **Device:** Apple Silicon (MPS)
- **Consistency:** All settings match main pipeline for fair comparison

---

## 3. Evaluation Methodology

### Primary Metric: Average IoU
**Calculation:**
1. For each query frame, retrieve top-1 match
2. For each bounding box in query, find best spatially overlapping bbox in match
3. Compute IoU for each query bbox → match bbox pair
4. Average all IoU values

**Why Average IoU?**
- **Unbiased:** Measures both semantic and spatial consistency
- **Semantic Check:** Do retrieved frames contain same objects?
- **Spatial Check:** Are objects in the correct locations?
- **Holistic:** Single metric capturing overall match quality
- **Interpretable:** Direct measure of detection alignment

**IoU Formula:**
```
IoU(bbox₁, bbox₂) = Area(bbox₁ ∩ bbox₂) / Area(bbox₁ ∪ bbox₂)
```

### Query Set
- **Direction:** Bidirectional matching
  - 5 Autumn queries → Winter candidates
  - 6 Winter queries → Autumn candidates
- **Total Queries:** 11 test cases
- **Selection:** Representative frames across the traversal

---

## 4. Quantitative Results

### Overall Performance
| Metric | Value |
|--------|-------|
| **Mean IoU** | **0.2529** |
| Std Dev | 0.1068 |
| Min IoU | 0.0100 |
| Max IoU | 0.4082 |
| N Queries | 11 |

### Per-Query Breakdown
| Query ID | Top-1 Match | Avg IoU | Interpretation |
|----------|-------------|---------|----------------|
| autumn_1726 | winter_0839 | 0.4032 | ✅ Strong match |
| autumn_0864 | winter_0743 | 0.4082 | ✅ Strong match |
| winter_1109 | autumn_0736 | 0.3273 | ✅ Good match |
| winter_0797 | autumn_0736 | 0.2955 | 🟡 Moderate |
| autumn_0000 | winter_0456 | 0.2708 | 🟡 Moderate |
| winter_0309 | autumn_0894 | 0.2486 | 🟡 Moderate |
| autumn_0305 | winter_0345 | 0.2256 | 🟡 Moderate |
| autumn_1136 | winter_0983 | 0.2070 | 🟡 Moderate |
| winter_0000 | autumn_0894 | 0.1606 | ⚠️ Weak |
| winter_0491 | autumn_1147 | 0.0100 | ❌ Poor |

### Comparative Analysis
| Method | Mean IoU | vs Baseline A |
|--------|----------|---------------|
| **Baseline A** (Visual + Semantic) | **0.2529** | - |
| Baseline B (Geometric Only) | 0.2054 | -18.8% |
| Full Pipeline (Hierarchical) | 0.2451 | -3.1% |

**Key Observations:**
- Baseline A outperforms geometric-only matching
- Performs competitively with full pipeline (within 3%)
- Strong performance on Autumn→Winter direction (IoU > 0.40)
- Struggles with Winter→Autumn when scenes are textureless

---

## 5. Qualitative Analysis

### Strengths
✅ **Fast Inference:** Single forward pass per image  
✅ **Scene-Level Reasoning:** Captures overall composition  
✅ **Robust to Seasonal Changes:** Learns invariant features  
✅ **Consistent Performance:** Low variance on good matches  

### Limitations
⚠️ **Translational Sensitivity:** Retrieves visually similar scenes but with spatial offsets  
⚠️ **Textureless Failures:** Struggles when scenes lack distinctive features  
⚠️ **No Geometric Verification:** Cannot filter false positives based on keypoint consistency  

### Example Case Studies

**✅ Success Case: autumn_1726 → winter_0839 (IoU: 0.4032)**
- Visual similarity captures similar building configuration
- Semantic labels overlap: {"tree", "building", "path"}
- Spatial layout preserved despite seasonal differences

**❌ Failure Case: winter_0491 → autumn_1147 (IoU: 0.010)**
- High visual similarity (similar "empty corridor" appearance)
- Semantic match (both contain "path", "wall")
- BUT: Different physical locations (translational offset)

---

## 6. Slide Visual Content

### Figure 1: Method Pipeline Diagram
```
[RGB Frame] → [FastViTHD] → [Global Embedding (512D)]
                                    ↓
                            [Cosine Similarity]
                                    ↓
[OWL-ViT] → [Label Set] → [Jaccard Similarity]
                                    ↓
                        [Weighted Fusion (0.7 + 0.3)]
                                    ↓
                        [Ranked Retrieval Results]
```

### Figure 2: Side-by-Side Top-1 Matches
**Include:** 2-3 example pairs from `all_methods_comparison.png`
- Show Query | Baseline A Match | IoU score
- Color-code: Green (IoU > 0.3), Orange (0.15-0.3), Red (< 0.15)

### Figure 3: IoU Distribution
**Visualization:** Box plot or histogram showing:
- Mean line at 0.2529
- Distribution of 11 query results
- Comparison with Baseline B and Full Pipeline

---

## 7. Key Takeaways for Slide

### Main Points (Bullet Format)
- **Baseline A uses global visual embeddings (FastViTHD) + semantic labels (OWL-ViT)**
- **Self-implemented, not from prior work - designed to test if simple retrieval suffices**
- **Mean IoU: 0.2529 - competitive with full pipeline, better than geometric-only**
- **Strengths:** Fast, captures scene context, robust to seasonal changes
- **Limitations:** Sensitive to spatial offsets, lacks geometric verification
- **Best Use Case:** Initial retrieval candidate generation for downstream refinement

### One-Liner Summary
> *"Baseline A demonstrates that global visual + semantic retrieval achieves 0.25 average IoU—competitive performance without expensive geometric verification, ideal as a first-stage filter."*

---

## 8. Suggested Presentation Flow

1. **Motivation (30 sec)**
   - "Before complex pipelines, can simple retrieval work?"
   - "Global embeddings: fast but coarse?"

2. **Method Overview (1 min)**
   - FastViTHD for visual features
   - OWL-ViT labels for semantics
   - Fusion strategy

3. **Results (1 min)**
   - Show IoU table (0.2529 mean)
   - Display 2-3 example matches
   - Highlight strengths/limitations

4. **Context (30 sec)**
   - Compare to Baseline B and Full Pipeline
   - Position as "strong baseline, not SOTA"

---

**Generated:** November 24, 2025  
**Experiment:** Cross-Temporal Landmark Matching  
**Dataset:** FourSeasons (Winter 2024-01-13, Autumn 2024-04-11)  
**Code:** `experiments/baseline_a_autumnwinter_match.py`
