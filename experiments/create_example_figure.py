#!/usr/bin/env python3
"""
Create a visual summary figure showing example query-match pairs for the presentation.

This script generates a publication-quality figure with:
- Top 3 successful matches (high quality)
- Top 2 challenging cases (method disagreement)
- Annotated with quality scores and component breakdowns
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np
import json

# Paths
BASELINE_A = Path('experiments/baseline_a_results.csv')
BASELINE_B = Path('experiments/baseline_b_results.csv')
FULL_PIPELINE = Path('experiments/full_pipeline_results.csv')
OUTPUT_DIR = Path('experiments/results_presentation')
DETECTIONS_JSON = Path('experiments/detections.json')

# Load frame ID to image path mapping
FRAME_TO_PATH = {}

def load_frame_mappings():
    """Load frame_id to image_path mapping from detections.json."""
    global FRAME_TO_PATH
    
    if not DETECTIONS_JSON.exists():
        print(f"Warning: {DETECTIONS_JSON} not found")
        return
    
    with open(DETECTIONS_JSON, 'r') as f:
        data = json.load(f)
    
    # Map winter frames
    for frame in data.get('winter', []):
        FRAME_TO_PATH[frame['frame_id']] = frame['image_path']
    
    # Map autumn frames
    for frame in data.get('autumn', []):
        FRAME_TO_PATH[frame['frame_id']] = frame['image_path']
    
    print(f"Loaded {len(FRAME_TO_PATH)} frame mappings")

def load_image(frame_id: str):
    """Load image from frame ID (e.g., 'autumn_0305' or 'winter_0840')."""
    if frame_id not in FRAME_TO_PATH:
        print(f"Warning: {frame_id} not found in mappings")
        return None
    
    img_path = Path(FRAME_TO_PATH[frame_id])
    
    if img_path.exists():
        return Image.open(img_path).convert('RGB')
    else:
        print(f"Warning: {img_path} not found")
        return None

def create_example_figure():
    """Create figure showing top examples and challenging cases."""
    
    # Load data
    df_full = pd.read_csv(FULL_PIPELINE)
    
    # Get top-1 matches sorted by quality
    top1 = df_full[df_full['rank'] == 1].sort_values('overall_quality_score', ascending=False)
    
    # Select examples
    # Top 3 successes
    successes = top1.head(3)
    
    # Bottom 2 challenges
    challenges = top1.tail(2)
    
    examples = pd.concat([successes, challenges])
    
    print("Selected Examples:")
    print(examples[['query_id', 'match_id', 'final_confidence', 'overall_quality_score']])
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    for idx, (_, row) in enumerate(examples.iterrows()):
        query_id = row['query_id']
        match_id = row['match_id']
        quality = row['overall_quality_score']
        
        # Load images
        query_img = load_image(query_id)
        match_img = load_image(match_id)
        
        if query_img is None or match_img is None:
            continue
        
        # Resize for display
        query_img.thumbnail((400, 300))
        match_img.thumbnail((400, 300))
        
        # Query image
        ax_query = plt.subplot(5, 4, idx*4 + 1)
        ax_query.imshow(query_img)
        ax_query.axis('off')
        season = query_id.split('_')[0].capitalize()
        ax_query.set_title(f'Query: {query_id}', fontsize=10, fontweight='bold')
        
        # Arrow
        ax_arrow = plt.subplot(5, 4, idx*4 + 2)
        ax_arrow.text(0.5, 0.5, '→', fontsize=40, ha='center', va='center',
                     color='green' if quality > 0.5 else 'orange')
        ax_arrow.axis('off')
        
        # Match image
        ax_match = plt.subplot(5, 4, idx*4 + 3)
        ax_match.imshow(match_img)
        ax_match.axis('off')
        match_season = match_id.split('_')[0].capitalize()
        ax_match.set_title(f'Match: {match_id}', fontsize=10, fontweight='bold')
        
        # Metrics panel
        ax_metrics = plt.subplot(5, 4, idx*4 + 4)
        ax_metrics.axis('off')
        
        metrics_text = f"""Quality: {quality:.3f}
        
Visual: {row.get('visual_score', 0):.2f}
Semantic: {row.get('semantic_score', 0):.2f}
Depth: {row.get('depth_score', 0.5):.2f}
Geometric: {row.get('geometric_score', 0):.2f}

Label J: {row.get('label_jaccard', 0):.2f}
IoU: {row.get('avg_iou', 0):.2f}
Inliers: {int(row.get('num_inliers', 0))}"""
        
        color = 'green' if quality > 0.5 else 'orange' if quality > 0.4 else 'red'
        ax_metrics.text(0.1, 0.5, metrics_text, fontsize=8, va='center',
                       family='monospace', color=color)
    
    plt.suptitle('Cross-Season Landmark Matching: Example Results', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Add legend
    fig.text(0.5, 0.01, 
            'Top 3 rows: High-quality matches | Bottom 2 rows: Challenging cases',
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'example_matches.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved example matches figure to {OUTPUT_DIR / 'example_matches.png'}")

if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load frame mappings from detections.json
    load_frame_mappings()
    
    if len(FRAME_TO_PATH) == 0:
        print("Error: No frame mappings loaded. Cannot create figure.")
    else:
        create_example_figure()
