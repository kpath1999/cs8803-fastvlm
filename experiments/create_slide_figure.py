#!/usr/bin/env python3
"""
Create individual frame comparison figures for presentation.

This script generates:
1. Individual comparisons for each query-match pair per method
   - autumn_1726_winter_0839_baseline_a.png
   - autumn_1726_winter_0839_baseline_b.png
   - autumn_1726_winter_0839_full_pipeline.png

2. Cross-method comparisons showing all methods for a single query-match pair
   - autumn_1726_winter_0839_all_methods.png

All outputs are saved to experiments/results_presentation/frame_comparisons/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import pandas as pd
from typing import Dict, List, Tuple

# Paths
BASELINE_A_CSV = Path('experiments/baseline_a_results.csv')
BASELINE_B_CSV = Path('experiments/baseline_b_results.csv')
FULL_PIPELINE_CSV = Path('experiments/full_pipeline_results.csv')
DETECTIONS_JSON = Path('experiments/detections.json')
OUTPUT_DIR = Path('experiments/results_presentation')
FRAME_COMP_DIR = OUTPUT_DIR / 'frame_comparisons'

# Load frame mappings
FRAME_TO_PATH = {}

def load_frame_mappings():
    """Load frame_id to image_path mapping."""
    global FRAME_TO_PATH
    with open(DETECTIONS_JSON, 'r') as f:
        data = json.load(f)
    for frame in data.get('winter', []) + data.get('autumn', []):
        FRAME_TO_PATH[frame['frame_id']] = frame['image_path']
    print(f"✓ Loaded {len(FRAME_TO_PATH)} frame mappings")

def load_image_safe(frame_id: str) -> Image.Image:
    """Load image with fallback."""
    img_path = Path(FRAME_TO_PATH.get(frame_id, ''))
    if img_path.exists():
        return Image.open(img_path).convert('RGB')
    return Image.new('RGB', (640, 480), color='gray')


def get_bboxes_for_frame(frame_id: str) -> List[Dict]:
    """Get bounding boxes for a frame from detections.json."""
    try:
        with open(DETECTIONS_JSON, 'r') as f:
            data = json.load(f)
        
        season = frame_id.split('_')[0]
        for frame in data.get(season, []):
            if frame['frame_id'] == frame_id:
                return frame.get('detections', [])
    except Exception as e:
        print(f"Warning: Could not load bboxes for {frame_id}: {e}")
    return []


def draw_bboxes_on_image(img: Image.Image, bboxes: List[Dict], color='green', thickness=2) -> Image.Image:
    """Draw bounding boxes on image."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for bbox_data in bboxes:
        bbox = bbox_data.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
            
            # Add label
            label = bbox_data.get('label', '')
            if label:
                # Try to use a font, fallback to default
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
                except:
                    font = ImageFont.load_default()
                
                draw.text((x1, y1 - 15), label, fill=color, font=font)
    
    return img_copy


def create_single_method_comparison(
    query_id: str,
    match_id: str,
    method_name: str,
    iou: float,
    additional_metrics: Dict = None,
    output_path: Path = None
):
    """
    Create a compact side-by-side comparison for a single method.
    
    Layout: [Query Image] [Match Image]
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.1, left=0.05, right=0.95, top=0.85, bottom=0.05)
    
    # Load images
    query_img = load_image_safe(query_id)
    match_img = load_image_safe(match_id)
    
    # Get bounding boxes
    query_bboxes = get_bboxes_for_frame(query_id)
    match_bboxes = get_bboxes_for_frame(match_id)
    
    # Draw bboxes
    query_img_annotated = draw_bboxes_on_image(query_img, query_bboxes, color='cyan', thickness=2)
    match_img_annotated = draw_bboxes_on_image(match_img, match_bboxes, color='lime', thickness=2)
    
    # Query
    axes[0].imshow(query_img_annotated)
    axes[0].axis('off')
    season = query_id.split('_')[0].upper()
    axes[0].set_title(f'QUERY: {query_id}\n({season})', fontsize=11, fontweight='bold', pad=15)
    
    # Match
    axes[1].imshow(match_img_annotated)
    axes[1].axis('off')
    match_season = match_id.split('_')[0].upper()
    color = 'green' if iou > 0.3 else 'orange' if iou > 0.15 else 'red'
    axes[1].set_title(f'TOP-1 MATCH: {match_id}\n({match_season})', 
                     fontsize=11, fontweight='bold', color=color, pad=15)
    
    # Main title with metrics
    # title = f'{method_name}\n'
    # title += f'Average IoU: {iou:.4f}'
    
    # if additional_metrics:
    #     if 'label_jaccard' in additional_metrics:
    #         title += f' | Label Jaccard: {additional_metrics["label_jaccard"]:.3f}'
    #     if 'num_inliers' in additional_metrics:
    #         title += f' | Inliers: {int(additional_metrics["num_inliers"])}'
    
    # fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    # Save
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_cross_method_comparison(
    query_id: str,
    match_ids: Dict[str, str],  # {method_name: match_id}
    ious: Dict[str, float],
    additional_metrics: Dict[str, Dict] = None,
    output_path: Path = None
):
    """
    Create a compact comparison showing all methods for a single query.
    
    Layout:
    Row 1: [Query] [Baseline A Match] [Baseline B Match] [Full Pipeline Match]
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.80, bottom=0.02)
    
    # Query image
    query_img = load_image_safe(query_id)
    query_bboxes = get_bboxes_for_frame(query_id)
    query_img_annotated = draw_bboxes_on_image(query_img, query_bboxes, color='cyan', thickness=2)
    
    axes[0].imshow(query_img_annotated)
    axes[0].axis('off')
    season = query_id.split('_')[0].upper()
    axes[0].set_title(f'QUERY\n{query_id}\n({season})', fontsize=10, fontweight='bold', pad=5)
    
    # Method matches
    methods = ['Baseline A', 'Baseline B', 'Full Pipeline']
    for idx, method in enumerate(methods, start=1):
        match_id = match_ids.get(method, '')
        iou = ious.get(method, 0.0)
        
        if match_id:
            match_img = load_image_safe(match_id)
            match_bboxes = get_bboxes_for_frame(match_id)
            match_img_annotated = draw_bboxes_on_image(match_img, match_bboxes, color='lime', thickness=2)
            
            axes[idx].imshow(match_img_annotated)
            axes[idx].axis('off')
            
            color = 'green' if iou > 0.3 else 'orange' if iou > 0.15 else 'red'
            title = f'{method}\n{match_id}\nIoU: {iou:.4f}'
            
            axes[idx].set_title(title, fontsize=9, fontweight='bold', color=color, pad=5)
        else:
            axes[idx].axis('off')
            axes[idx].text(0.5, 0.5, 'No Match', ha='center', va='center', fontsize=12)
    
    # Main title
    fig.suptitle(f'Cross-Method Comparison: {query_id}', 
                fontsize=14, fontweight='bold', y=0.95)
    
    # Save
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_all_comparisons():
    """Generate all individual and cross-method comparisons."""
    
    print("="*70)
    print("GENERATING FRAME COMPARISONS")
    print("="*70)
    
    # Create output directory
    FRAME_COMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all CSVs
    df_ba = pd.read_csv(BASELINE_A_CSV)
    df_bb = pd.read_csv(BASELINE_B_CSV)
    df_fp = pd.read_csv(FULL_PIPELINE_CSV)
    
    # Filter for top-1 matches
    top1_ba = df_ba[df_ba['rank'] == 1]
    top1_bb = df_bb[df_bb['rank'] == 1]
    top1_fp = df_fp[df_fp['rank'] == 1]
    
    # Get all unique queries
    all_queries = set(top1_ba['query_id'].unique()) | \
                  set(top1_bb['query_id'].unique()) | \
                  set(top1_fp['query_id'].unique())
    
    all_queries = sorted(list(all_queries))
    
    print(f"\nProcessing {len(all_queries)} query frames...\n")
    
    # Track statistics
    single_method_count = 0
    cross_method_count = 0
    
    for query_id in all_queries:
        print(f"Processing {query_id}...")
        
        # Get matches for each method
        ba_match = top1_ba[top1_ba['query_id'] == query_id]
        bb_match = top1_bb[top1_bb['query_id'] == query_id]
        fp_match = top1_fp[top1_fp['query_id'] == query_id]
        
        match_ids = {}
        ious = {}
        metrics = {}
        
        # Baseline A
        if not ba_match.empty:
            row = ba_match.iloc[0]
            match_ids['Baseline A'] = row['match_id']
            ious['Baseline A'] = row['avg_iou']
            metrics['Baseline A'] = {
                'label_jaccard': row.get('label_jaccard', 0),
                'visual_score': row.get('visual_score', 0),
                'semantic_score': row.get('semantic_score', 0)
            }
            
            # Create single-method figure for Baseline A
            output_file = FRAME_COMP_DIR / f"{query_id}_{row['match_id']}_baseline_a.png"
            create_single_method_comparison(
                query_id, row['match_id'], 'Baseline A (Visual + Semantic)',
                row['avg_iou'], metrics['Baseline A'], output_file
            )
            single_method_count += 1
        
        # Baseline B
        if not bb_match.empty:
            row = bb_match.iloc[0]
            match_ids['Baseline B'] = row['match_id']
            ious['Baseline B'] = row['avg_iou']
            metrics['Baseline B'] = {
                'label_jaccard': row.get('label_jaccard', 0),
                'num_inliers': row.get('num_inliers', 0),
                'geometric_confidence': row.get('geometric_confidence', 0)
            }
            
            # Create single-method figure for Baseline B
            output_file = FRAME_COMP_DIR / f"{query_id}_{row['match_id']}_baseline_b.png"
            create_single_method_comparison(
                query_id, row['match_id'], 'Baseline B (Geometric Keypoints)',
                row['avg_iou'], metrics['Baseline B'], output_file
            )
            single_method_count += 1
        
        # Full Pipeline
        if not fp_match.empty:
            row = fp_match.iloc[0]
            match_ids['Full Pipeline'] = row['match_id']
            ious['Full Pipeline'] = row['avg_iou']
            metrics['Full Pipeline'] = {
                'label_jaccard': row.get('label_jaccard', 0),
                'num_inliers': row.get('num_inliers', 0),
                'visual_score': row.get('visual_score', 0),
                'depth_score': row.get('depth_score', 0)
            }
            
            # Create single-method figure for Full Pipeline
            output_file = FRAME_COMP_DIR / f"{query_id}_{row['match_id']}_full_pipeline.png"
            create_single_method_comparison(
                query_id, row['match_id'], 'Full Pipeline (Hierarchical)',
                row['avg_iou'], metrics['Full Pipeline'], output_file
            )
            single_method_count += 1
        
        # Create cross-method comparison if we have at least 2 methods
        if len(match_ids) >= 2:
            # Use the most common match_id for filename
            match_id_for_filename = match_ids.get('Baseline A', 
                                   match_ids.get('Full Pipeline',
                                   match_ids.get('Baseline B', 'unknown')))
            
            output_file = FRAME_COMP_DIR / f"{query_id}_{match_id_for_filename}_all_methods.png"
            create_cross_method_comparison(
                query_id, match_ids, ious, metrics, output_file
            )
            cross_method_count += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Single-method comparisons: {single_method_count}")
    print(f"Cross-method comparisons: {cross_method_count}")
    print(f"Total figures generated: {single_method_count + cross_method_count}")
    print(f"\n✓ All comparisons saved to {FRAME_COMP_DIR}")
    print("="*70)


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_frame_mappings()
    
    # Generate all frame comparisons
    generate_all_comparisons()
    
    print("\n" + "="*70)
    print("Done! Check the following directory:")
    print(f"  - Frame comparisons: {FRAME_COMP_DIR}")
    print("="*70)
