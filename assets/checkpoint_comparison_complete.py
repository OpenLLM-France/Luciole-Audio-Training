#!/usr/bin/env python3
"""
Complete checkpoint comparison script with all functions.
Combines simple bar charts, line plots, and detailed analysis.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_results():
    """Load results from all checkpoint directories."""
    checkpoints = {
        '5K': 'model2/evaluations_5000/test_speechlm2/results.json',
        '10K': 'model2/evaluations_10000/test_speechlm2/results.json',
        '15K': 'model2/evaluations_15000/test_speechlm2/results.json', 
        '20K': 'model2/evaluations_20000/test_speechlm2/results.json'
    }
    
    results = {}
    for checkpoint, path in checkpoints.items():
        with open(path, 'r') as f:
            results[checkpoint] = json.load(f)
    
    return results

def plot_bleu_comparison(results, checkpoints, colors, ax):
    """Plot BLEU scores comparison."""
    
    # Get all datasets with BLEU scores
    bleu_data = {}
    datasets = []
    
    for dataset in results['5K'].keys():
        if 'bleu' in results['5K'][dataset]:
            datasets.append(dataset)
            bleu_data[dataset] = []
            for ckpt in checkpoints:
                bleu_data[dataset].append(results[ckpt][dataset]['bleu'])
    
    # Create grouped bar chart
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, ckpt in enumerate(checkpoints):
        values = [bleu_data[dataset][i] for dataset in datasets]
        bars = ax.bar(x + i * width, values, width, label=f'{ckpt} steps', 
                     color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    ax.set_title('BLEU Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

def plot_bert_comparison(results, checkpoints, colors, ax):
    """Plot BERT F1 scores comparison."""
    
    # Get all datasets with BERT F1 scores
    bert_data = {}
    datasets = []
    
    for dataset in results['5K'].keys():
        if 'bert_f1' in results['5K'][dataset]:
            datasets.append(dataset)
            bert_data[dataset] = []
            for ckpt in checkpoints:
                # Multiply by 100 to convert to percentage (0-100 range)
                bert_data[dataset].append(results[ckpt][dataset]['bert_f1'] * 100)
    
    # Create grouped bar chart
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, ckpt in enumerate(checkpoints):
        values = [bert_data[dataset][i] for dataset in datasets]
        bars = ax.bar(x + i * width, values, width, label=f'{ckpt} steps', 
                     color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('BERT F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('BERT F1 Score Comparison (0-100%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

def plot_wer_comparison(results, checkpoints, colors, ax):
    """Plot WER comparison."""
    
    # Get all datasets with WER scores
    wer_data = {}
    datasets = []
    
    for dataset in results['5K'].keys():
        if 'wer' in results['5K'][dataset]:
            datasets.append(dataset)
            wer_data[dataset] = []
            for ckpt in checkpoints:
                wer_data[dataset].append(results[ckpt][dataset]['wer'])
    
    # Create grouped bar chart
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, ckpt in enumerate(checkpoints):
        values = [wer_data[dataset][i] for dataset in datasets]
        bars = ax.bar(x + i * width, values, width, label=f'{ckpt} steps', 
                     color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Word Error Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Word Error Rate Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

def plot_summary_comparison(results, checkpoints, colors, ax):
    """Plot overall performance summary."""
    
    # Calculate average performance for each checkpoint
    metrics_summary = {
        'Avg BLEU': [],
        'Avg BERT F1': [],
        'Avg WER': []
    }
    
    for ckpt in checkpoints:
        # BLEU average
        bleu_scores = []
        bert_scores = []
        wer_scores = []
        
        for dataset in results[ckpt].keys():
            if 'bleu' in results[ckpt][dataset]:
                bleu_scores.append(results[ckpt][dataset]['bleu'])
            if 'bert_f1' in results[ckpt][dataset]:
                # Scale BERT F1 to percentage for consistency
                bert_scores.append(results[ckpt][dataset]['bert_f1'] * 100)
            if 'wer' in results[ckpt][dataset]:
                wer_scores.append(results[ckpt][dataset]['wer'])
        
        metrics_summary['Avg BLEU'].append(np.mean(bleu_scores) if bleu_scores else 0)
        metrics_summary['Avg BERT F1'].append(np.mean(bert_scores) if bert_scores else 0)
        metrics_summary['Avg WER'].append(np.mean(wer_scores) if wer_scores else 0)
    
    # Create grouped bar chart
    x = np.arange(len(list(metrics_summary.keys())))
    width = 0.25
    
    for i, ckpt in enumerate(checkpoints):
        values = [metrics_summary[metric][i] for metric in metrics_summary.keys()]
        bars = ax.bar(x + i * width, values, width, label=f'{ckpt} steps', 
                     color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Metric Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(list(metrics_summary.keys()))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

def create_simple_comparison(results):
    """Create simple, clear comparison plots."""
    
    # Create a large figure with clear subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison: 5K vs 10K vs 20K Steps', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    checkpoints = ['5K', '10K', '15K', '20K']
    colors = ['#3498db', '#e74c3c', '#f1c40f', '#2ecc71']  # Blue, Red, Green
    
    # 1. BLEU Scores Comparison (Top Left)
    ax1 = axes[0, 0]
    plot_bleu_comparison(results, checkpoints, colors, ax1)
    
    # 2. BERT F1 Scores Comparison (Top Right)
    ax2 = axes[0, 1]
    plot_bert_comparison(results, checkpoints, colors, ax2)
    
    # 3. WER Comparison (Bottom Left)
    ax3 = axes[1, 0]
    plot_wer_comparison(results, checkpoints, colors, ax3)
    
    # 4. Overall Performance Summary (Bottom Right)
    ax4 = axes[1, 1]
    plot_summary_comparison(results, checkpoints, colors, ax4)
    
    plt.tight_layout()
    plt.savefig('simple_checkpoint_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_simple_summary(results):
    """Print a simple comparison summary."""
    print("\n" + "="*80)
    print("📊 SIMPLE CHECKPOINT COMPARISON SUMMARY")
    print("="*80)
    
    checkpoints = ['5K', '10K', '15K', '20K']
    
    print("\n🏆 BEST PERFORMING CHECKPOINT BY DATASET:")
    print("-" * 50)
    
    for dataset in results['5K'].keys():
        print(f"\n{dataset}:")
        
        for metric in results['5K'][dataset].keys():
            values = {}
            for ckpt in checkpoints:
                values[ckpt] = results[ckpt][dataset][metric]
            
            # Determine best (lower for WER/error metrics, higher for others)
            if metric in ['wer', 'nins', 'ndel', 'nsub']:
                best_ckpt = min(values, key=values.get)
                symbol = "📉"
            else:
                best_ckpt = max(values, key=values.get)
                symbol = "📈"
            
            print(f"  {symbol} {metric}: {best_ckpt} steps ({values[best_ckpt]:.3f})")

def main():
    """Main function."""
    print("Creating simple and clear checkpoint comparisons...")
    results = load_results()
    
    print("Generating comparison plots...")
    create_simple_comparison(results)
    
    print("Creating summary...")
    print_simple_summary(results)
    
    print(f"\n✅ Simple comparison complete!")
    print(f"📊 Main comparison: 'simple_checkpoint_comparison.png'")
    print(f"💡 These plots clearly show which checkpoint performs best for each task!")
    print(f"💡 BERT scores are now scaled to 0-100% for better visibility!")

if __name__ == "__main__":
    main()