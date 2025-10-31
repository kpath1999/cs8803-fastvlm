#!/usr/bin/env python3
"""
Visualize Video Analysis Results

Creates visual outputs from the experiment results to help understand
what FastVLM is detecting and how keyframes are selected.

Usage:
    python visualize_results.py \
        --video-path your_video.mp4 \
        --results-dir ./video_analysis_output \
        --output-dir ./visualizations
"""

import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import List, Dict


def create_keyframe_grid(video_path: str, keyframe_file: str, output_path: str):
    """
    Create a grid visualization of selected keyframes.
    """
    # Load keyframe data
    with open(keyframe_file) as f:
        keyframes = json.load(f)
    
    print(f"Creating keyframe grid with {len(keyframes)} frames...")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Extract keyframe images
    images = []
    for kf in keyframes[:16]:  # Limit to 16 for 4x4 grid
        frame_num = kf['frame_number']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append((frame_num, kf['timestamp_ms'], frame_rgb))
    
    cap.release()
    
    # Create grid
    n_images = len(images)
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (frame_num, timestamp_ms, img) in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        ax.imshow(img)
        ax.set_title(f"Frame {frame_num}\n{timestamp_ms/1000:.1f}s", fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved keyframe grid to {output_path}")
    plt.close()


def visualize_tracking_timeline(tracking_file: str, output_path: str):
    """
    Create a timeline showing when the tracked object was detected.
    """
    with open(tracking_file) as f:
        tracking_data = json.load(f)
    
    print(f"Creating tracking timeline with {len(tracking_data)} data points...")
    
    timestamps = [d['timestamp_ms'] / 1000 for d in tracking_data]
    detected = [1 if d['object_detected'] else 0 for d in tracking_data]
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Plot detection status
    colors = ['red' if d == 0 else 'green' for d in detected]
    ax.scatter(timestamps, detected, c=colors, s=100, alpha=0.6)
    ax.plot(timestamps, detected, 'k--', alpha=0.3)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Object Detected', fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Not Detected', 'Detected'])
    ax.grid(True, alpha=0.3)
    
    # Add summary
    total = len(detected)
    detected_count = sum(detected)
    ax.set_title(f'Object Tracking Timeline\n'
                f'Detected in {detected_count}/{total} frames '
                f'({100*detected_count/total:.1f}%)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved tracking timeline to {output_path}")
    plt.close()


def create_annotated_frame(video_path: str, frame_analysis_file: str, 
                          frame_index: int, output_path: str):
    """
    Create an annotated frame showing what FastVLM detected.
    """
    with open(frame_analysis_file) as f:
        analyses = json.load(f)
    
    if frame_index >= len(analyses):
        print(f"Frame index {frame_index} out of range (max: {len(analyses)-1})")
        return
    
    analysis = analyses[frame_index]
    frame_num = analysis['frame_number']
    
    # Extract frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Could not read frame {frame_num}")
        return
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create figure with frame and annotations
    fig = plt.figure(figsize=(14, 10))
    
    # Main image
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(frame_rgb)
    ax1.set_title(f"Frame {frame_num} ({analysis['timestamp_ms']/1000:.1f}s)", 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Annotations
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('off')
    
    text = f"FASTIVLM ANALYSIS\n\n"
    text += f"Objects Detected:\n"
    objects = analysis.get('objects', [])
    if objects:
        text += "  • " + "\n  • ".join(objects[:10])
    else:
        text += "  (none)"
    
    text += f"\n\nSpatial Layout:\n"
    text += f"  {analysis.get('spatial_layout', 'N/A')}\n"
    
    text += f"\n\nLandmark Description:\n"
    desc = analysis.get('description', 'N/A')
    # Wrap text
    import textwrap
    wrapped = textwrap.fill(desc, width=80)
    text += f"  {wrapped}"
    
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved annotated frame to {output_path}")
    plt.close()


def plot_embedding_similarity(frame_analysis_file: str, output_path: str):
    """
    Plot embedding similarity between consecutive frames.
    """
    with open(frame_analysis_file) as f:
        analyses = json.load(f)
    
    print(f"Computing embedding similarities for {len(analyses)} frames...")
    
    similarities = []
    timestamps = []
    
    for i in range(len(analyses) - 1):
        emb1 = np.array(analyses[i]['embedding'])
        emb2 = np.array(analyses[i + 1]['embedding'])
        
        # Cosine similarity
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarities.append(sim)
        timestamps.append(analyses[i + 1]['timestamp_ms'] / 1000)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(timestamps, similarities, 'b-', linewidth=2, alpha=0.7)
    ax.axhline(y=0.95, color='r', linestyle='--', label='Keyframe threshold (0.95)')
    ax.fill_between(timestamps, similarities, 0.95, 
                     where=[s < 0.95 for s in similarities],
                     alpha=0.3, color='green', label='Significant change')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Cosine Similarity to Previous Frame', fontsize=12)
    ax.set_title('Temporal Embedding Similarity\n'
                '(Drops indicate scene changes suitable for keyframe selection)', 
                fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.8, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved similarity plot to {output_path}")
    plt.close()


def visualize_landmark_matches(matches_file: str, video1_path: str, 
                               video2_path: str, output_path: str):
    """
    Visualize top landmark matches between two videos.
    """
    with open(matches_file) as f:
        data = json.load(f)
    
    matches = data['matches'][:3]  # Top 3 matches
    
    if not matches:
        print("No matches to visualize")
        return
    
    print(f"Creating visualization for {len(matches)} matches...")
    
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    fig, axes = plt.subplots(len(matches), 2, figsize=(12, 4 * len(matches)))
    if len(matches) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, match in enumerate(matches):
        # Extract frames
        cap1.set(cv2.CAP_PROP_POS_FRAMES, match['frame_a'])
        ret1, frame1 = cap1.read()
        
        cap2.set(cv2.CAP_PROP_POS_FRAMES, match['frame_b'])
        ret2, frame2 = cap2.read()
        
        if ret1 and ret2:
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            
            # Plot
            axes[idx, 0].imshow(frame1_rgb)
            axes[idx, 0].set_title(f"{data['context_1']}\nFrame {match['frame_a']}", 
                                   fontsize=10)
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(frame2_rgb)
            axes[idx, 1].set_title(f"{data['context_2']}\nFrame {match['frame_b']}", 
                                   fontsize=10)
            axes[idx, 1].axis('off')
            
            # Add match quality text
            fig.text(0.5, 1 - (idx + 0.5) / len(matches), 
                    f"Match #{idx+1}: Confidence={match['llm_verification_score']:.2f}, "
                    f"Similarity={match['embedding_similarity']:.2f}",
                    ha='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    cap1.release()
    cap2.release()
    
    plt.suptitle('Cross-Temporal Landmark Matches', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved match visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize video analysis results")
    parser.add_argument("--video-path", type=str, required=True,
                       help="Path to original video file")
    parser.add_argument("--results-dir", type=str, required=True,
                       help="Directory containing experiment results (JSON files)")
    parser.add_argument("--output-dir", type=str, default="./visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--video2-path", type=str,
                       help="Path to second video (for matching visualization)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60 + "\n")
    
    # 1. Keyframe grid
    keyframe_file = results_dir / "experiment_3_keyframes.json"
    if keyframe_file.exists():
        create_keyframe_grid(
            args.video_path, 
            str(keyframe_file),
            str(output_dir / "keyframe_grid.png")
        )
    
    # 2. Tracking timeline
    tracking_files = list(results_dir.glob("experiment_4_tracking_*.json"))
    if tracking_files:
        visualize_tracking_timeline(
            str(tracking_files[0]),
            str(output_dir / "tracking_timeline.png")
        )
    
    # 3. Frame analysis
    frame_file = results_dir / "experiment_1_frame_extraction.json"
    if frame_file.exists():
        # Create annotated frame for first analysis
        create_annotated_frame(
            args.video_path,
            str(frame_file),
            0,
            str(output_dir / "annotated_frame_0.png")
        )
        
        # Plot embedding similarity
        plot_embedding_similarity(
            str(frame_file),
            str(output_dir / "embedding_similarity.png")
        )
    
    # 4. Landmark matches
    matches_file = results_dir / "landmark_matches.json"
    if matches_file.exists() and args.video2_path:
        with open(matches_file) as f:
            data = json.load(f)
        visualize_landmark_matches(
            str(matches_file),
            data.get('video_1', args.video_path),
            data.get('video_2', args.video2_path),
            str(output_dir / "landmark_matches.png")
        )
    
    print("\n" + "="*60)
    print(f"✓ All visualizations saved to {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
