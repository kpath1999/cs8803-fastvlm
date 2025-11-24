#!/usr/bin/env python3
"""
Compute Average IoU Across All Methods and Create Compact Visualizations

This script:
1. Computes average IoU for top-1 matches across all methods (Baseline A, B, Full Pipeline)
2. Creates compact 2x3 grid visualizations showing query-match pairs
3. Uses frame mappings from detections.json to ensure correct images

Methods compared:
- Baseline A: Global visual embeddings + semantic similarity
- Baseline B: Geometric keypoint matching only
- Full Pipeline: Multi-stage hierarchical matching

Queries tested (bidirectional):
- Autumn queries: autumn_0000, autumn_0305, autumn_0864, autumn_1136, autumn_1726
- Winter queries: winter_0000, winter_0309, winter_0491, winter_0797, winter_1109

Total comparisons: 3 methods × 10 queries = 30 data points (showing top 6 representative)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple

# File paths
BASELINE_A_CSV = Path('experiments/baseline_a_results.csv')
BASELINE_B_CSV = Path('experiments/baseline_b_results.csv')
FULL_PIPELINE_CSV = Path('experiments/full_pipeline_results.csv')
DETECTIONS_JSON = Path('experiments/detections.json')
OUTPUT_DIR = Path('experiments/results_presentation')

# Frame mappings (loaded from detections.json)
FRAME_TO_PATH = {}


def load_frame_mappings():
    """Load frame_id to image_path mapping from detections.json."""
    global FRAME_TO_PATH
    
    if not DETECTIONS_JSON.exists():
        print(f"Error: {DETECTIONS_JSON} not found")
        return False
    
    with open(DETECTIONS_JSON, 'r') as f:
        data = json.load(f)
    
    # Map winter frames
    for frame in data.get('winter', []):
        FRAME_TO_PATH[frame['frame_id']] = frame['image_path']
    
    # Map autumn frames
    for frame in data.get('autumn', []):
        FRAME_TO_PATH[frame['frame_id']] = frame['image_path']
    
    print(f"✓ Loaded {len(FRAME_TO_PATH)} frame mappings")
    return True


def compute_average_iou_per_method(csv_path: Path, method_name: str) -> Dict:
    """
    Compute average IoU for top-1 matches from a results CSV.
    
    Returns:
        Dictionary with statistics: mean_iou, std_iou, query_results
    """
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Filter for top-1 matches only
    top1 = df[df['rank'] == 1].copy()
    
    # Extract IoU values
    iou_values = top1['avg_iou'].values
    
    results = {
        'method': method_name,
        'mean_iou': float(np.mean(iou_values)),
        'std_iou': float(np.std(iou_values)),
        'min_iou': float(np.min(iou_values)),
        'max_iou': float(np.max(iou_values)),
        'n_queries': len(iou_values),
        'query_results': top1[['query_id', 'match_id', 'avg_iou']].to_dict('records')
    }
    
    return results


def load_image_safe(frame_id: str) -> Image.Image:
    """Load image from frame ID with error handling."""
    if frame_id not in FRAME_TO_PATH:
        print(f"Warning: {frame_id} not in mappings")
        # Create placeholder
        return Image.new('RGB', (640, 480), color='gray')
    
    img_path = Path(FRAME_TO_PATH[frame_id])
    
    if not img_path.exists():
        print(f"Warning: {img_path} not found")
        return Image.new('RGB', (640, 480), color='gray')
    
    return Image.open(img_path).convert('RGB')


def create_compact_visualization(
    query_ids: List[str],
    baseline_a_df: pd.DataFrame,
    baseline_b_df: pd.DataFrame,
    full_pipeline_df: pd.DataFrame,
    output_path: Path
):
    """
    Create ultra-compact 2x3 grid showing query-match pairs for all methods.
    
    Layout:
    [Query 1 Autumn] [B.A Match] [B.B Match] [Full Match]
    [Query 2 Winter] [B.A Match] [B.B Match] [Full Match]
    [Query 3 Autumn] [B.A Match] [B.B Match] [Full Match]
    [Query 4 Winter] [B.A Match] [B.B Match] [Full Match]
    [Query 5 Autumn] [B.A Match] [B.B Match] [Full Match]
    [Query 6 Winter] [B.A Match] [B.B Match] [Full Match]
    """
    n_queries = len(query_ids)
    
    # Create figure with tight spacing
    fig, axes = plt.subplots(n_queries, 4, figsize=(16, n_queries * 3.5))
    fig.subplots_adjust(wspace=0.02, hspace=0.15, left=0.02, right=0.98, top=0.96, bottom=0.02)
    
    for row_idx, query_id in enumerate(query_ids):
        # Get top-1 matches from each method
        ba_match = baseline_a_df[(baseline_a_df['query_id'] == query_id) & (baseline_a_df['rank'] == 1)]
        bb_match = baseline_b_df[(baseline_b_df['query_id'] == query_id) & (baseline_b_df['rank'] == 1)]
        fp_match = full_pipeline_df[(full_pipeline_df['query_id'] == query_id) & (full_pipeline_df['rank'] == 1)]
        
        if ba_match.empty or bb_match.empty or fp_match.empty:
            print(f"Warning: Missing data for {query_id}")
            continue
        
        ba_row = ba_match.iloc[0]
        bb_row = bb_match.iloc[0]
        fp_row = fp_match.iloc[0]
        
        # Load images
        query_img = load_image_safe(query_id)
        ba_img = load_image_safe(ba_row['match_id'])
        bb_img = load_image_safe(bb_row['match_id'])
        fp_img = load_image_safe(fp_row['match_id'])
        
        # Resize to uniform size
        target_size = (640, 480)
        query_img = query_img.resize(target_size, Image.LANCZOS)
        ba_img = ba_img.resize(target_size, Image.LANCZOS)
        bb_img = bb_img.resize(target_size, Image.LANCZOS)
        fp_img = fp_img.resize(target_size, Image.LANCZOS)
        
        # Column 0: Query
        ax = axes[row_idx, 0]
        ax.imshow(query_img)
        ax.axis('off')
        season = query_id.split('_')[0].upper()
        ax.set_title(f'{season}\n{query_id}', fontsize=10, fontweight='bold', pad=5)
        
        # Column 1: Baseline A
        ax = axes[row_idx, 1]
        ax.imshow(ba_img)
        ax.axis('off')
        ba_iou = ba_row['avg_iou']
        color = 'green' if ba_iou > 0.3 else 'orange' if ba_iou > 0.15 else 'red'
        ax.set_title(f'Baseline A\nIoU: {ba_iou:.3f}', fontsize=9, color=color, fontweight='bold', pad=5)
        
        # Column 2: Baseline B
        ax = axes[row_idx, 2]
        ax.imshow(bb_img)
        ax.axis('off')
        bb_iou = bb_row['avg_iou']
        color = 'green' if bb_iou > 0.3 else 'orange' if bb_iou > 0.15 else 'red'
        ax.set_title(f'Baseline B\nIoU: {bb_iou:.3f}', fontsize=9, color=color, fontweight='bold', pad=5)
        
        # Column 3: Full Pipeline
        ax = axes[row_idx, 3]
        ax.imshow(fp_img)
        ax.axis('off')
        fp_iou = fp_row['avg_iou']
        color = 'green' if fp_iou > 0.3 else 'orange' if fp_iou > 0.15 else 'red'
        ax.set_title(f'Full Pipeline\nIoU: {fp_iou:.3f}', fontsize=9, color=color, fontweight='bold', pad=5)
    
    plt.suptitle('Cross-Temporal Matching: Query vs Top-1 Matches (All Methods)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"\n✓ Saved visualization to {output_path}")
    plt.close()


def create_summary_table(
    baseline_a_stats: Dict,
    baseline_b_stats: Dict,
    full_pipeline_stats: Dict,
    output_path: Path
):
    """Create a summary table comparing all methods."""
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Method', 'Mean IoU', 'Std IoU', 'Min IoU', 'Max IoU', 'N Queries'],
        [
            'Baseline A\n(Visual + Semantic)',
            f"{baseline_a_stats['mean_iou']:.4f}",
            f"{baseline_a_stats['std_iou']:.4f}",
            f"{baseline_a_stats['min_iou']:.4f}",
            f"{baseline_a_stats['max_iou']:.4f}",
            f"{baseline_a_stats['n_queries']}"
        ],
        [
            'Baseline B\n(Geometric Only)',
            f"{baseline_b_stats['mean_iou']:.4f}",
            f"{baseline_b_stats['std_iou']:.4f}",
            f"{baseline_b_stats['min_iou']:.4f}",
            f"{baseline_b_stats['max_iou']:.4f}",
            f"{baseline_b_stats['n_queries']}"
        ],
        [
            'Full Pipeline\n(Hierarchical)',
            f"{full_pipeline_stats['mean_iou']:.4f}",
            f"{full_pipeline_stats['std_iou']:.4f}",
            f"{full_pipeline_stats['min_iou']:.4f}",
            f"{full_pipeline_stats['max_iou']:.4f}",
            f"{full_pipeline_stats['n_queries']}"
        ]
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code by performance
    methods = [baseline_a_stats['mean_iou'], baseline_b_stats['mean_iou'], full_pipeline_stats['mean_iou']]
    colors = ['#FFE699', '#C6E0B4', '#B4C7E7']  # Yellow, Green, Blue
    
    for i in range(1, 4):
        table[(i, 0)].set_facecolor(colors[i-1])
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Average IoU Comparison: Top-1 Matches Across Methods', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved summary table to {output_path}")
    plt.close()


def create_detailed_breakdown(
    baseline_a_stats: Dict,
    baseline_b_stats: Dict,
    full_pipeline_stats: Dict,
    output_path: Path
):
    """Create a detailed per-query breakdown table."""
    
    # Merge all query results
    all_queries = set()
    for stats in [baseline_a_stats, baseline_b_stats, full_pipeline_stats]:
        all_queries.update([r['query_id'] for r in stats['query_results']])
    
    all_queries = sorted(list(all_queries))
    
    # Build comparison table
    rows = []
    for query_id in all_queries:
        row = {'Query': query_id}
        
        # Find IoU for each method
        for stats, label in [(baseline_a_stats, 'Baseline A'), 
                             (baseline_b_stats, 'Baseline B'), 
                             (full_pipeline_stats, 'Full Pipeline')]:
            matches = [r for r in stats['query_results'] if r['query_id'] == query_id]
            if matches:
                row[label] = matches[0]['avg_iou']
            else:
                row[label] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_path = output_path.parent / 'iou_detailed_breakdown.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Saved detailed breakdown to {csv_path}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap
    data_matrix = df[['Baseline A', 'Baseline B', 'Full Pipeline']].values
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.5)
    
    # Set ticks
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(len(all_queries)))
    ax.set_xticklabels(['Baseline A', 'Baseline B', 'Full Pipeline'])
    ax.set_yticklabels(all_queries)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(all_queries)):
        for j in range(3):
            val = data_matrix[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.3f}', ha="center", va="center", 
                             color="black" if val > 0.25 else "white", fontsize=9)
    
    ax.set_title("Per-Query IoU Breakdown Across Methods", fontsize=14, fontweight='bold', pad=15)
    fig.colorbar(im, ax=ax, label='Average IoU')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved breakdown heatmap to {output_path}")
    plt.close()


def main():
    """Main execution."""
    print("="*70)
    print("COMPUTING AVERAGE IOU FOR ALL METHODS")
    print("="*70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load frame mappings
    if not load_frame_mappings():
        print("Error: Could not load frame mappings. Exiting.")
        return
    
    # Compute statistics for each method
    print("\n" + "-"*70)
    print("1. BASELINE A: Visual + Semantic Similarity")
    print("-"*70)
    baseline_a_stats = compute_average_iou_per_method(BASELINE_A_CSV, 'Baseline A')
    if baseline_a_stats:
        print(f"   Mean IoU: {baseline_a_stats['mean_iou']:.4f} ± {baseline_a_stats['std_iou']:.4f}")
        print(f"   Range: [{baseline_a_stats['min_iou']:.4f}, {baseline_a_stats['max_iou']:.4f}]")
        print(f"   N = {baseline_a_stats['n_queries']} queries")
    
    print("\n" + "-"*70)
    print("2. BASELINE B: Geometric Keypoint Matching")
    print("-"*70)
    baseline_b_stats = compute_average_iou_per_method(BASELINE_B_CSV, 'Baseline B')
    if baseline_b_stats:
        print(f"   Mean IoU: {baseline_b_stats['mean_iou']:.4f} ± {baseline_b_stats['std_iou']:.4f}")
        print(f"   Range: [{baseline_b_stats['min_iou']:.4f}, {baseline_b_stats['max_iou']:.4f}]")
        print(f"   N = {baseline_b_stats['n_queries']} queries")
    
    print("\n" + "-"*70)
    print("3. FULL PIPELINE: Hierarchical Multi-Stage")
    print("-"*70)
    full_pipeline_stats = compute_average_iou_per_method(FULL_PIPELINE_CSV, 'Full Pipeline')
    if full_pipeline_stats:
        print(f"   Mean IoU: {full_pipeline_stats['mean_iou']:.4f} ± {full_pipeline_stats['std_iou']:.4f}")
        print(f"   Range: [{full_pipeline_stats['min_iou']:.4f}, {full_pipeline_stats['max_iou']:.4f}]")
        print(f"   N = {full_pipeline_stats['n_queries']} queries")
    
    # Validate all methods have data
    if not all([baseline_a_stats, baseline_b_stats, full_pipeline_stats]):
        print("\nError: Missing data from one or more methods. Cannot proceed.")
        return
    
    # Create summary table
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    create_summary_table(
        baseline_a_stats,
        baseline_b_stats,
        full_pipeline_stats,
        OUTPUT_DIR / 'iou_summary_table.png'
    )
    
    # Create detailed breakdown
    create_detailed_breakdown(
        baseline_a_stats,
        baseline_b_stats,
        full_pipeline_stats,
        OUTPUT_DIR / 'iou_breakdown_heatmap.png'
    )
    
    # Select representative queries (3 autumn, 3 winter)
    # Mix of high and low performers for each direction
    selected_queries = [
        'autumn_0000',   # Row 1
        'winter_0000',   # Row 2
        'autumn_0864',   # Row 3
        'winter_0309',   # Row 4
        'autumn_1726',   # Row 5
        'winter_1109',   # Row 6
    ]
    
    # Load CSVs for visualization
    ba_df = pd.read_csv(BASELINE_A_CSV)
    bb_df = pd.read_csv(BASELINE_B_CSV)
    fp_df = pd.read_csv(FULL_PIPELINE_CSV)
    
    # Create compact visualization
    create_compact_visualization(
        selected_queries,
        ba_df,
        bb_df,
        fp_df,
        OUTPUT_DIR / 'all_methods_comparison.png'
    )
    
    # Print final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Baseline A Avg IoU: {baseline_a_stats['mean_iou']:.4f}")
    print(f"Baseline B Avg IoU: {baseline_b_stats['mean_iou']:.4f}")
    print(f"Full Pipeline Avg IoU: {full_pipeline_stats['mean_iou']:.4f}")
    
    improvement_over_a = ((full_pipeline_stats['mean_iou'] - baseline_a_stats['mean_iou']) / baseline_a_stats['mean_iou']) * 100
    improvement_over_b = ((full_pipeline_stats['mean_iou'] - baseline_b_stats['mean_iou']) / baseline_b_stats['mean_iou']) * 100
    
    print(f"\nFull Pipeline Improvement:")
    print(f"  vs Baseline A: {improvement_over_a:+.1f}%")
    print(f"  vs Baseline B: {improvement_over_b:+.1f}%")
    
    # Print per-query details
    print("\n" + "-"*70)
    print("PER-QUERY RESULTS (IoU)")
    print("-"*70)
    
    # Get common queries
    common_queries = set([r['query_id'] for r in baseline_a_stats['query_results']])
    common_queries &= set([r['query_id'] for r in baseline_b_stats['query_results']])
    common_queries &= set([r['query_id'] for r in full_pipeline_stats['query_results']])
    
    for query_id in sorted(common_queries):
        ba_iou = next(r['avg_iou'] for r in baseline_a_stats['query_results'] if r['query_id'] == query_id)
        bb_iou = next(r['avg_iou'] for r in baseline_b_stats['query_results'] if r['query_id'] == query_id)
        fp_iou = next(r['avg_iou'] for r in full_pipeline_stats['query_results'] if r['query_id'] == query_id)
        
        print(f"{query_id:15s} | A:{ba_iou:.3f} | B:{bb_iou:.3f} | Full:{fp_iou:.3f}")
    
    print("\n" + "="*70)
    print("✓ All results saved to experiments/results_presentation/")
    print("="*70)


if __name__ == '__main__':
    main()