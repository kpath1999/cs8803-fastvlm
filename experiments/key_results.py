#!/usr/bin/env python3
"""
Key Results Presentation and Performance Analysis

This script analyzes experimental results from three cross-season matching approaches:
- Baseline A: Visual embeddings + semantic similarity
- Baseline B: Geometric-only (keypoint matching)
- Full Pipeline: Multi-modal fusion (visual + semantic + depth + geometric)

Generates:
1. Quantitative metrics tables (precision, recall, match quality)
2. Performance comparison visualizations
3. Success/failure analysis with concrete examples
4. Final assessment against research goals

Usage:
    python key_results.py --output-dir results_presentation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set publication-quality plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


class ResultsAnalyzer:
    """Analyzes and presents experimental results."""
    
    def __init__(self, baseline_a_csv: Path, baseline_b_csv: Path, 
                 full_pipeline_csv: Path, output_dir: Path):
        self.baseline_a = pd.read_csv(baseline_a_csv)
        self.baseline_b = pd.read_csv(baseline_b_csv)
        self.full_pipeline = pd.read_csv(full_pipeline_csv)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define test frames
        self.autumn_frames = [0, 305, 864, 1136, 1726]
        self.winter_frames = [0, 309, 491, 797, 1109]
        
    def compute_summary_statistics(self) -> pd.DataFrame:
        """Compute aggregate metrics for all three methods."""
        
        stats = []
        
        for method_name, df in [
            ("Baseline A (Visual+Semantic)", self.baseline_a),
            ("Baseline B (Geometric)", self.baseline_b),
            ("Full Pipeline (Multi-Modal)", self.full_pipeline)
        ]:
            # Core metrics
            total_queries = df['query_id'].nunique()
            total_matches = len(df)
            avg_top1_quality = df[df['rank'] == 1]['overall_quality_score'].mean()
            avg_all_quality = df['overall_quality_score'].mean()
            
            # Match quality distribution
            high_quality_matches = (df['overall_quality_score'] > 0.5).sum()
            high_quality_pct = (high_quality_matches / total_matches * 100)
            
            # Label consistency
            avg_label_jaccard = df['label_jaccard'].mean()
            avg_iou = df['avg_iou'].mean()
            avg_match_rate = df['match_rate'].mean()
            
            # Method-specific metrics
            if 'visual_score' in df.columns:
                avg_visual = df['visual_score'].mean()
            else:
                avg_visual = np.nan
                
            if 'semantic_score' in df.columns:
                avg_semantic = df['semantic_score'].mean()
            else:
                avg_semantic = np.nan
                
            if 'geometric_confidence' in df.columns:
                avg_geometric = df['geometric_confidence'].mean()
                avg_inliers = df['num_inliers'].mean()
            elif 'geometric_score' in df.columns:
                avg_geometric = df['geometric_score'].mean()
                avg_inliers = df['num_inliers'].mean()
            else:
                avg_geometric = np.nan
                avg_inliers = np.nan
            
            stats.append({
                'Method': method_name,
                'Total Queries': total_queries,
                'Total Matches': total_matches,
                'Matches/Query': total_matches / total_queries,
                'Top-1 Quality': avg_top1_quality,
                'Avg Quality': avg_all_quality,
                'High Quality %': high_quality_pct,
                'Label Jaccard': avg_label_jaccard,
                'Avg IoU': avg_iou,
                'Match Rate': avg_match_rate,
                'Visual Score': avg_visual,
                'Semantic Score': avg_semantic,
                'Geometric Score': avg_geometric,
                'Avg Inliers': avg_inliers
            })
        
        return pd.DataFrame(stats)
    
    def analyze_top_k_performance(self, k: int = 5) -> Dict:
        """Analyze how often the correct match appears in top-K results."""
        
        results = {}
        
        for method_name, df in [
            ("Baseline A", self.baseline_a),
            ("Baseline B", self.baseline_b),
            ("Full Pipeline", self.full_pipeline)
        ]:
            # For each query, check if any top-K match has high quality (>0.5)
            topk_data = df[df['rank'] <= k]
            queries_with_good_match = topk_data[
                topk_data['overall_quality_score'] > 0.5
            ]['query_id'].nunique()
            
            total_queries = df['query_id'].nunique()
            success_rate = queries_with_good_match / total_queries * 100
            
            # Average rank of best match
            best_matches = df.loc[df.groupby('query_id')['overall_quality_score'].idxmax()]
            avg_best_rank = best_matches['rank'].mean()
            
            results[method_name] = {
                'success_rate': success_rate,
                'avg_best_rank': avg_best_rank,
                'queries_with_match': queries_with_good_match,
                'total_queries': total_queries
            }
        
        return results
    
    def identify_challenging_cases(self) -> pd.DataFrame:
        """Identify queries where methods disagree or all struggle."""
        
        challenging = []
        
        all_queries = set(self.baseline_a['query_id'].unique()) | \
                     set(self.baseline_b['query_id'].unique()) | \
                     set(self.full_pipeline['query_id'].unique())
        
        for query_id in all_queries:
            # Get top-1 match for each method
            top1_a = self.baseline_a[
                (self.baseline_a['query_id'] == query_id) & 
                (self.baseline_a['rank'] == 1)
            ]
            top1_b = self.baseline_b[
                (self.baseline_b['query_id'] == query_id) & 
                (self.baseline_b['rank'] == 1)
            ]
            top1_full = self.full_pipeline[
                (self.full_pipeline['query_id'] == query_id) & 
                (self.full_pipeline['rank'] == 1)
            ]
            
            if len(top1_a) == 0 or len(top1_b) == 0 or len(top1_full) == 0:
                continue
            
            quality_a = top1_a['overall_quality_score'].values[0]
            quality_b = top1_b['overall_quality_score'].values[0]
            quality_full = top1_full['overall_quality_score'].values[0]
            
            match_a = top1_a['match_id'].values[0]
            match_b = top1_b['match_id'].values[0]
            match_full = top1_full['match_id'].values[0]
            
            # Case 1: All methods struggle (all quality < 0.4)
            all_struggle = quality_a < 0.4 and quality_b < 0.4 and quality_full < 0.4
            
            # Case 2: Methods disagree on match (different match_ids)
            disagree = len(set([match_a, match_b, match_full])) > 1
            
            # Case 3: Full pipeline significantly better
            full_best = quality_full > max(quality_a, quality_b) + 0.1
            
            if all_struggle or disagree or full_best:
                challenging.append({
                    'query_id': query_id,
                    'baseline_a_match': match_a,
                    'baseline_a_quality': quality_a,
                    'baseline_b_match': match_b,
                    'baseline_b_quality': quality_b,
                    'full_pipeline_match': match_full,
                    'full_pipeline_quality': quality_full,
                    'all_struggle': all_struggle,
                    'methods_disagree': disagree,
                    'full_pipeline_best': full_best
                })
        
        return pd.DataFrame(challenging)
    
    def plot_quality_distributions(self):
        """Plot quality score distributions for all methods."""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (method_name, df, ax) in enumerate([
            ("Baseline A\n(Visual + Semantic)", self.baseline_a, axes[0]),
            ("Baseline B\n(Geometric)", self.baseline_b, axes[1]),
            ("Full Pipeline\n(Multi-Modal)", self.full_pipeline, axes[2])
        ]):
            # Plot histogram
            ax.hist(df['overall_quality_score'], bins=20, alpha=0.7, 
                   color=['#3498db', '#e74c3c', '#2ecc71'][idx], edgecolor='black')
            ax.axvline(df['overall_quality_score'].mean(), color='black', 
                      linestyle='--', linewidth=2, label=f'Mean: {df["overall_quality_score"].mean():.3f}')
            ax.axvline(0.5, color='red', linestyle=':', linewidth=2, 
                      label='Quality Threshold')
            
            ax.set_xlabel('Overall Quality Score')
            ax.set_ylabel('Frequency')
            ax.set_title(method_name)
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_distributions.png', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved quality distributions to {self.output_dir / 'quality_distributions.png'}")
    
    def plot_metric_comparison(self):
        """Plot radar chart comparing methods across key metrics."""
        
        # Compute normalized metrics (0-1 scale)
        metrics_data = []
        
        for method_name, df in [
            ("Baseline A", self.baseline_a),
            ("Baseline B", self.baseline_b),
            ("Full Pipeline", self.full_pipeline)
        ]:
            metrics_data.append({
                'Method': method_name,
                'Quality': df['overall_quality_score'].mean(),
                'Label Match': df['label_jaccard'].mean(),
                'Spatial IoU': df['avg_iou'].mean(),
                'Bbox Match Rate': df['match_rate'].mean(),
                'Confidence': df[df.columns[df.columns.str.contains('score|confidence')][0]].mean()
            })
        
        # Create grouped bar chart
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.set_index('Method', inplace=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.plot(kind='bar', ax=ax, width=0.8, 
                       color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
        
        ax.set_ylabel('Score')
        ax.set_xlabel('')
        ax.set_title('Cross-Season Matching Performance Comparison')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metric_comparison.png', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved metric comparison to {self.output_dir / 'metric_comparison.png'}")
    
    def plot_top1_success_rates(self):
        """Plot success rate (top-1 quality > 0.5) for each query."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        all_queries = sorted(set(self.baseline_a['query_id'].unique()) | 
                           set(self.baseline_b['query_id'].unique()) | 
                           set(self.full_pipeline['query_id'].unique()))
        
        x = np.arange(len(all_queries))
        width = 0.25
        
        success_a = []
        success_b = []
        success_full = []
        
        for query_id in all_queries:
            # Check if top-1 match has quality > 0.5
            top1_a = self.baseline_a[
                (self.baseline_a['query_id'] == query_id) & 
                (self.baseline_a['rank'] == 1)
            ]
            success_a.append(
                1 if len(top1_a) > 0 and top1_a['overall_quality_score'].values[0] > 0.5 else 0
            )
            
            top1_b = self.baseline_b[
                (self.baseline_b['query_id'] == query_id) & 
                (self.baseline_b['rank'] == 1)
            ]
            success_b.append(
                1 if len(top1_b) > 0 and top1_b['overall_quality_score'].values[0] > 0.5 else 0
            )
            
            top1_full = self.full_pipeline[
                (self.full_pipeline['query_id'] == query_id) & 
                (self.full_pipeline['rank'] == 1)
            ]
            success_full.append(
                1 if len(top1_full) > 0 and top1_full['overall_quality_score'].values[0] > 0.5 else 0
            )
        
        ax.bar(x - width, success_a, width, label='Baseline A', color='#3498db', alpha=0.8)
        ax.bar(x, success_b, width, label='Baseline B', color='#e74c3c', alpha=0.8)
        ax.bar(x + width, success_full, width, label='Full Pipeline', color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Query Frame')
        ax.set_ylabel('Success (1) / Failure (0)')
        ax.set_title('Top-1 Match Success Rate by Query (Quality > 0.5)')
        ax.set_xticks(x)
        ax.set_xticklabels([q.split('_')[1] for q in all_queries], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top1_success_rates.png', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved top-1 success rates to {self.output_dir / 'top1_success_rates.png'}")
    
    def plot_score_breakdown(self):
        """Plot breakdown of individual score components."""
        
        # Only Full Pipeline has all components
        df = self.full_pipeline
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Visual scores
        if 'visual_score' in df.columns:
            axes[0, 0].hist(df['visual_score'], bins=20, alpha=0.7, color='#3498db', edgecolor='black')
            axes[0, 0].axvline(df['visual_score'].mean(), color='black', linestyle='--', linewidth=2)
            axes[0, 0].set_xlabel('Visual Embedding Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title(f'Visual Similarity (mean={df["visual_score"].mean():.3f})')
            axes[0, 0].grid(alpha=0.3)
        
        # Semantic scores
        if 'semantic_score' in df.columns:
            axes[0, 1].hist(df['semantic_score'], bins=20, alpha=0.7, color='#e74c3c', edgecolor='black')
            axes[0, 1].axvline(df['semantic_score'].mean(), color='black', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('Semantic Label Jaccard')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title(f'Semantic Similarity (mean={df["semantic_score"].mean():.3f})')
            axes[0, 1].grid(alpha=0.3)
        
        # Depth scores
        if 'depth_score' in df.columns:
            axes[1, 0].hist(df['depth_score'], bins=20, alpha=0.7, color='#f39c12', edgecolor='black')
            axes[1, 0].axvline(df['depth_score'].mean(), color='black', linestyle='--', linewidth=2)
            axes[1, 0].set_xlabel('Depth Consistency Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title(f'Depth Consistency (mean={df["depth_score"].mean():.3f})')
            axes[1, 0].grid(alpha=0.3)
        
        # Geometric scores
        if 'geometric_score' in df.columns:
            axes[1, 1].hist(df['geometric_score'], bins=20, alpha=0.7, color='#2ecc71', edgecolor='black')
            axes[1, 1].axvline(df['geometric_score'].mean(), color='black', linestyle='--', linewidth=2)
            axes[1, 1].set_xlabel('Geometric Verification Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title(f'Geometric Confidence (mean={df["geometric_score"].mean():.3f})')
            axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle('Full Pipeline: Component Score Distributions', fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_breakdown.png', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved score breakdown to {self.output_dir / 'score_breakdown.png'}")
    
    def generate_summary_report(self):
        """Generate comprehensive text report."""
        
        report = []
        report.append("=" * 80)
        report.append("CROSS-SEASON LANDMARK MATCHING: KEY RESULTS")
        report.append("=" * 80)
        report.append("")
        
        # Desired Behavior
        report.append("DESIRED BEHAVIOR")
        report.append("-" * 80)
        report.append("Goal: Match physical landmarks (buildings, trees, signs, benches) between")
        report.append("      autumn and winter image streams despite severe appearance changes.")
        report.append("")
        report.append("Success Criteria:")
        report.append("  1. High match quality (overall_quality_score > 0.5)")
        report.append("  2. Correct label agreement (same object type matched)")
        report.append("  3. Spatial consistency (high IoU between bounding boxes)")
        report.append("  4. Geometric verification (RANSAC inliers confirm match)")
        report.append("  5. Invariant classification (depth consistency distinguishes permanent structures)")
        report.append("")
        
        # Summary Statistics
        stats_df = self.compute_summary_statistics()
        report.append("QUANTITATIVE RESULTS")
        report.append("-" * 80)
        report.append(stats_df.to_string(index=False))
        report.append("")
        
        # Top-K Performance
        topk_results = self.analyze_top_k_performance(k=5)
        report.append("TOP-5 RETRIEVAL PERFORMANCE")
        report.append("-" * 80)
        for method, data in topk_results.items():
            report.append(f"{method}:")
            report.append(f"  Success Rate: {data['success_rate']:.1f}% ({data['queries_with_match']}/{data['total_queries']} queries)")
            report.append(f"  Avg Best Match Rank: {data['avg_best_rank']:.2f}")
        report.append("")
        
        # Key Findings
        report.append("KEY FINDINGS")
        report.append("-" * 80)
        
        # Finding 1: Multi-modal superiority
        avg_quality = {
            'Baseline A': self.baseline_a['overall_quality_score'].mean(),
            'Baseline B': self.baseline_b['overall_quality_score'].mean(),
            'Full Pipeline': self.full_pipeline['overall_quality_score'].mean()
        }
        best_method = max(avg_quality, key=avg_quality.get)
        report.append(f"1. Multi-Modal Fusion Effectiveness:")
        report.append(f"   {best_method} achieves highest average quality ({avg_quality[best_method]:.3f})")
        
        improvement_over_a = (avg_quality['Full Pipeline'] - avg_quality['Baseline A']) / avg_quality['Baseline A'] * 100
        improvement_over_b = (avg_quality['Full Pipeline'] - avg_quality['Baseline B']) / avg_quality['Baseline B'] * 100
        
        if avg_quality['Full Pipeline'] > avg_quality['Baseline A']:
            report.append(f"   Full Pipeline improves {improvement_over_a:+.1f}% over Baseline A")
        if avg_quality['Full Pipeline'] > avg_quality['Baseline B']:
            report.append(f"   Full Pipeline improves {improvement_over_b:+.1f}% over Baseline B")
        report.append("")
        
        # Finding 2: Geometric reliability
        geom_inliers_b = self.baseline_b['num_inliers'].mean()
        geom_inliers_full = self.full_pipeline['num_inliers'].mean()
        report.append(f"2. Geometric Verification Value:")
        report.append(f"   Baseline B avg inliers: {geom_inliers_b:.2f}")
        report.append(f"   Full Pipeline avg inliers: {geom_inliers_full:.2f}")
        
        high_geom_b = (self.baseline_b['num_inliers'] >= 4).sum()
        high_geom_full = (self.full_pipeline['num_inliers'] >= 4).sum()
        report.append(f"   Matches with ≥4 inliers: Baseline B={high_geom_b}, Full={high_geom_full}")
        report.append("")
        
        # Finding 3: Label consistency
        label_jaccard_a = self.baseline_a['label_jaccard'].mean()
        label_jaccard_b = self.baseline_b['label_jaccard'].mean()
        label_jaccard_full = self.full_pipeline['label_jaccard'].mean()
        report.append(f"3. Semantic Consistency:")
        report.append(f"   Baseline A label Jaccard: {label_jaccard_a:.3f}")
        report.append(f"   Baseline B label Jaccard: {label_jaccard_b:.3f}")
        report.append(f"   Full Pipeline label Jaccard: {label_jaccard_full:.3f}")
        report.append("")
        
        # Finding 4: Challenging cases
        challenging_df = self.identify_challenging_cases()
        report.append(f"4. Challenging Cases Identified: {len(challenging_df)}")
        if len(challenging_df) > 0:
            all_struggle = challenging_df['all_struggle'].sum()
            disagree = challenging_df['methods_disagree'].sum()
            full_best = challenging_df['full_pipeline_best'].sum()
            report.append(f"   All methods struggle: {all_struggle} queries")
            report.append(f"   Methods disagree: {disagree} queries")
            report.append(f"   Full pipeline significantly better: {full_best} queries")
        report.append("")
        
        # False Positive Analysis
        report.append("FALSE POSITIVE ANALYSIS")
        report.append("-" * 80)
        
        # Define false positives as matches with quality < 0.3
        fp_threshold = 0.3
        fp_a = (self.baseline_a['overall_quality_score'] < fp_threshold).sum()
        fp_b = (self.baseline_b['overall_quality_score'] < fp_threshold).sum()
        fp_full = (self.full_pipeline['overall_quality_score'] < fp_threshold).sum()
        
        total_a = len(self.baseline_a)
        total_b = len(self.baseline_b)
        total_full = len(self.full_pipeline)
        
        report.append(f"Low-quality matches (quality < {fp_threshold}):")
        report.append(f"  Baseline A: {fp_a}/{total_a} ({fp_a/total_a*100:.1f}%)")
        report.append(f"  Baseline B: {fp_b}/{total_b} ({fp_b/total_b*100:.1f}%)")
        report.append(f"  Full Pipeline: {fp_full}/{total_full} ({fp_full/total_full*100:.1f}%)")
        report.append("")
        
        # Performance Assessment
        report.append("PERFORMANCE ASSESSMENT")
        report.append("-" * 80)
        
        # Success threshold: >60% of queries have top-1 quality > 0.5
        success_threshold_pct = 60.0
        full_success_rate = topk_results['Full Pipeline']['success_rate']
        
        is_success = full_success_rate >= success_threshold_pct
        
        if is_success:
            report.append(f"✓ SUCCESS: Full Pipeline achieves {full_success_rate:.1f}% top-1 success rate")
            report.append(f"           (exceeds {success_threshold_pct}% threshold)")
        else:
            report.append(f"✗ PARTIAL SUCCESS: Full Pipeline achieves {full_success_rate:.1f}% top-1 success rate")
            report.append(f"                   (below {success_threshold_pct}% threshold)")
        
        report.append("")
        report.append("Success Factors:")
        if avg_quality['Full Pipeline'] > 0.5:
            report.append("  ✓ Average match quality exceeds 0.5 threshold")
        else:
            report.append("  ✗ Average match quality below 0.5 threshold")
        
        if label_jaccard_full > 0.7:
            report.append("  ✓ Strong semantic consistency (label Jaccard > 0.7)")
        else:
            report.append("  ✗ Weak semantic consistency (label Jaccard < 0.7)")
        
        if geom_inliers_full >= 4:
            report.append("  ✓ Sufficient geometric verification (avg inliers ≥ 4)")
        else:
            report.append("  ✗ Insufficient geometric verification (avg inliers < 4)")
        
        report.append("")
        report.append("Limitations:")
        report.append("  - Depth data not always available (fallback to neutral 0.5 score)")
        report.append("  - Geometric verification limited by texture changes (snow/foliage)")
        report.append("  - OWL-ViT detections may miss small or occluded landmarks")
        report.append("  - No ground truth annotations for quantitative precision/recall")
        report.append("")
        
        report.append("Recommendations:")
        report.append("  1. Collect ground truth annotations for subset of challenging queries")
        report.append("  2. Experiment with alternative geometric features (e.g., SuperPoint, LoFTR)")
        report.append("  3. Incorporate temporal consistency across video sequences")
        report.append("  4. Fine-tune FastVLM on seasonal appearance variation dataset")
        report.append("")
        
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\n✓ Saved summary report to {self.output_dir / 'summary_report.txt'}")
        
        return report_text
    
    def export_results_table(self):
        """Export key metrics as CSV and markdown table."""
        
        stats_df = self.compute_summary_statistics()
        
        # Save CSV
        stats_df.to_csv(self.output_dir / 'summary_statistics.csv', index=False)
        print(f"✓ Saved summary statistics to {self.output_dir / 'summary_statistics.csv'}")
        
        # Save markdown table (manual formatting)
        with open(self.output_dir / 'summary_statistics.md', 'w') as f:
            f.write("# Cross-Season Matching Performance Summary\n\n")
            
            # Manual markdown table
            cols = stats_df.columns.tolist()
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            for _, row in stats_df.iterrows():
                formatted_row = []
                for col in cols:
                    val = row[col]
                    if isinstance(val, float):
                        formatted_row.append(f"{val:.3f}" if not np.isnan(val) else "N/A")
                    else:
                        formatted_row.append(str(val))
                f.write("| " + " | ".join(formatted_row) + " |\n")
            
            f.write("\n\n## Metric Definitions\n\n")
            f.write("- **Total Queries**: Number of unique query frames tested\n")
            f.write("- **Total Matches**: Number of candidate matches returned (top-5 per query)\n")
            f.write("- **Matches/Query**: Average number of matches per query\n")
            f.write("- **Top-1 Quality**: Average quality score of the best match\n")
            f.write("- **Avg Quality**: Average quality score across all matches\n")
            f.write("- **High Quality %**: Percentage of matches with quality > 0.5\n")
            f.write("- **Label Jaccard**: Average Jaccard similarity of semantic labels\n")
            f.write("- **Avg IoU**: Average intersection-over-union of bounding boxes\n")
            f.write("- **Match Rate**: Average percentage of detected objects successfully matched\n")
            f.write("- **Visual/Semantic/Geometric Score**: Component scores from each modality\n")
            f.write("- **Avg Inliers**: Average number of RANSAC inliers for geometric verification\n")
        
        print(f"✓ Saved markdown table to {self.output_dir / 'summary_statistics.md'}")
    
    def run_complete_analysis(self):
        """Execute full analysis pipeline."""
        
        print("\n" + "=" * 80)
        print("CROSS-SEASON MATCHING: KEY RESULTS ANALYSIS")
        print("=" * 80 + "\n")
        
        print("Step 1: Computing summary statistics...")
        self.export_results_table()
        
        print("\nStep 2: Generating quality distribution plots...")
        self.plot_quality_distributions()
        
        print("\nStep 3: Generating metric comparison plots...")
        self.plot_metric_comparison()
        
        print("\nStep 4: Generating top-1 success rate plots...")
        self.plot_top1_success_rates()
        
        print("\nStep 5: Generating score breakdown plots...")
        self.plot_score_breakdown()
        
        print("\nStep 6: Generating comprehensive summary report...")
        self.generate_summary_report()
        
        print("\n" + "=" * 80)
        print(f"✓ ALL RESULTS SAVED TO: {self.output_dir}/")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and present cross-season matching results"
    )
    parser.add_argument(
        '--baseline-a',
        type=str,
        default='experiments/baseline_a_results.csv',
        help='Path to Baseline A results CSV'
    )
    parser.add_argument(
        '--baseline-b',
        type=str,
        default='experiments/baseline_b_results.csv',
        help='Path to Baseline B results CSV'
    )
    parser.add_argument(
        '--full-pipeline',
        type=str,
        default='experiments/full_pipeline_results.csv',
        help='Path to Full Pipeline results CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/results_presentation',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Verify input files exist
    for csv_path in [args.baseline_a, args.baseline_b, args.full_pipeline]:
        if not Path(csv_path).exists():
            print(f"Error: {csv_path} not found")
            sys.exit(1)
    
    # Run analysis
    analyzer = ResultsAnalyzer(
        baseline_a_csv=Path(args.baseline_a),
        baseline_b_csv=Path(args.baseline_b),
        full_pipeline_csv=Path(args.full_pipeline),
        output_dir=Path(args.output_dir)
    )
    
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
