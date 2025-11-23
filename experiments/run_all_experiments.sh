#!/bin/bash
#
# RUN ALL EXPERIMENTS
#
# This script runs all three matching methods (Baseline A, Baseline B, Full Pipeline)
# on a curated set of test frames to enable systematic comparison.
#
# Test Frames:
#   Autumn: 0, 305, 864, 1136, 1726
#   Winter: 0, 309, 491, 797, 1109
#
# Each method finds top-5 matches in the opposite season dataset.
# Results are logged to CSV files for quantitative analysis.
#
# Usage:
#   bash experiments/run_all_experiments.sh

set -e  # Exit on error

echo "========================================================================"
echo "EXPERIMENTAL EVALUATION PIPELINE"
echo "========================================================================"
echo ""

# Configuration
MODEL_PATH="checkpoints/llava-fastvithd_0.5b_stage2"
DETECTIONS="experiments/detections.json"
WINTER_DEPTH="/Volumes/KAUSAR/rover_dataset/2024-01-13/realsense_D435i/depth"
AUTUMN_DEPTH="/Volumes/KAUSAR/rover_dataset/2024-04-11/realsense_D435i/depth"

# Test frame indices
AUTUMN_FRAMES=(0 305 864 1136 1726)
WINTER_FRAMES=(0 309 491 797 1109)

# Check if detections.json exists
if [ ! -f "$DETECTIONS" ]; then
    echo "Error: $DETECTIONS not found!"
    echo "Please generate detections first using generate_detections.py"
    exit 1
fi

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Detections: $DETECTIONS"
echo "  Autumn test frames: ${AUTUMN_FRAMES[@]}"
echo "  Winter test frames: ${WINTER_FRAMES[@]}"
echo ""

# Stage 1: Baseline A (Embedding + Semantic)
echo "========================================================================"
echo "STAGE 1: BASELINE A (EMBEDDING + SEMANTIC MATCHING)"
echo "========================================================================"
echo ""

echo "Processing Autumn → Winter matches..."
for idx in "${AUTUMN_FRAMES[@]}"; do
    echo "  Running autumn_$(printf '%04d' $idx)..."
    python experiments/baseline_a_autumnwinter_match.py \
        --detections "$DETECTIONS" \
        --model-path "$MODEL_PATH" \
        --autumn-idx $idx \
        --top-k 5 \
        --visualize \
        --log-file experiments/baseline_a_results.csv
done

echo ""
echo "Processing Winter → Autumn matches..."
for idx in "${WINTER_FRAMES[@]}"; do
    echo "  Running winter_$(printf '%04d' $idx)..."
    python experiments/baseline_a_autumnwinter_match.py \
        --detections "$DETECTIONS" \
        --model-path "$MODEL_PATH" \
        --winter-idx $idx \
        --top-k 5 \
        --visualize \
        --log-file experiments/baseline_a_results.csv
done

echo ""
echo "✓ Stage 1 complete: Results logged to experiments/baseline_a_results.csv"
echo ""

# Stage 2: Baseline B (Geometric-Only)
echo "========================================================================"
echo "STAGE 2: BASELINE B (GEOMETRIC-ONLY MATCHING)"
echo "========================================================================"
echo ""

echo "Processing Autumn → Winter matches..."
for idx in "${AUTUMN_FRAMES[@]}"; do
    echo "  Running autumn_$(printf '%04d' $idx)..."
    python experiments/baseline_b_autumnwinter_match.py \
        --detections "$DETECTIONS" \
        --autumn-idx $idx \
        --top-k 5 \
        --method orb \
        --min-inliers 4 \
        --visualize \
        --log-file experiments/baseline_b_results.csv
done

echo ""
echo "Processing Winter → Autumn matches..."
for idx in "${WINTER_FRAMES[@]}"; do
    echo "  Running winter_$(printf '%04d' $idx)..."
    python experiments/baseline_b_autumnwinter_match.py \
        --detections "$DETECTIONS" \
        --winter-idx $idx \
        --top-k 5 \
        --method orb \
        --min-inliers 4 \
        --visualize \
        --log-file experiments/baseline_b_results.csv
done

echo ""
echo "✓ Stage 2 complete: Results logged to experiments/baseline_b_results.csv"
echo ""

# Stage 3: Full Pipeline (Multi-Modal Fusion)
echo "========================================================================"
echo "STAGE 3: FULL PIPELINE (MULTI-MODAL FUSION)"
echo "========================================================================"
echo ""

echo "Processing Autumn → Winter matches..."
for idx in "${AUTUMN_FRAMES[@]}"; do
    echo "  Running autumn_$(printf '%04d' $idx)..."
    python experiments/full_pipeline_autumnwinter_match.py \
        --detections "$DETECTIONS" \
        --model-path "$MODEL_PATH" \
        --winter-depth "$WINTER_DEPTH" \
        --autumn-depth "$AUTUMN_DEPTH" \
        --autumn-idx $idx \
        --top-k 5 \
        --visualize \
        --log-file experiments/full_pipeline_results.csv
done

echo ""
echo "Processing Winter → Autumn matches..."
for idx in "${WINTER_FRAMES[@]}"; do
    echo "  Running winter_$(printf '%04d' $idx)..."
    python experiments/full_pipeline_autumnwinter_match.py \
        --detections "$DETECTIONS" \
        --model-path "$MODEL_PATH" \
        --winter-depth "$WINTER_DEPTH" \
        --autumn-depth "$AUTUMN_DEPTH" \
        --winter-idx $idx \
        --top-k 5 \
        --visualize \
        --log-file experiments/full_pipeline_results.csv
done

echo ""
echo "✓ Stage 3 complete: Results logged to experiments/full_pipeline_results.csv"
echo ""

# Summary
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  Baseline A CSV:     experiments/baseline_a_results.csv"
echo "  Baseline B CSV:     experiments/baseline_b_results.csv"
echo "  Full Pipeline CSV:  experiments/full_pipeline_results.csv"
echo "  Visualizations:     experiments/visualizations/"
echo ""
echo "Statistics:"
echo "  Test frames:        ${#AUTUMN_FRAMES[@]} autumn + ${#WINTER_FRAMES[@]} winter = $((${#AUTUMN_FRAMES[@]} + ${#WINTER_FRAMES[@]})) total"
echo "  Matches per frame:  5"
echo "  Total comparisons:  $(((${{#AUTUMN_FRAMES[@]}} + ${{#WINTER_FRAMES[@]}}) * 5)) matches logged per method"
echo ""
echo "Next steps:"
echo "  1. Analyze CSV files to compare methods quantitatively"
echo "  2. Review visualizations in experiments/visualizations/"
echo "  3. Generate comparison plots (precision, recall, confidence distributions)"
echo "  4. Identify failure cases for qualitative analysis"
echo ""
