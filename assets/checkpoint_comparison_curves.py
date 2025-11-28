#!/usr/bin/env python3
"""
Script to create curve plots showing model improvements across checkpoints.
Shows training progression with trend lines for each metric.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results():
    """Load results from all checkpoint directories."""
    checkpoints = {
        5000: 'model2/evaluations_5000/test_speechlm2/results.json',
        10000: 'model2/evaluations_10000/test_speechlm2/results.json', 
        15000: 'model2/evaluations_15000/test_speechlm2/results.json', 
        20000: 'model2/evaluations_20000/test_speechlm2/results.json'
    }
    
    results = {}
    for checkpoint, path in checkpoints.items():
        with open(path, 'r') as f:
            results[checkpoint] = json.load(f)
    
    return results

def plot_metric_curves(results, datasets, metric, title, steps, ax, lower_is_better=False):
    """Plot metric curves for specified datasets."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    for i, dataset in enumerate(datasets):
        values = []
        for step in steps:
            if step in results and dataset in results[step]:
                if metric in results[step][dataset]:
                    values.append(results[step][dataset][metric])
                else:
                    values.append(None)
            else:
                values.append(None)
        
        # Filter out None values
        valid_steps = [s for s, v in zip(steps, values) if v is not None]
        valid_values = [v for v in values if v is not None]
        
        if valid_values:
            ax.plot(valid_steps, valid_values, marker='o', label=dataset, 
                   color=colors[i], linewidth=2, markersize=8)
    
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    if lower_is_better:
        ax.invert_yaxis()

def plot_average_performance(results, steps, ax):
    """Plot average performance across all metrics."""
    # Calculate average BLEU and WER for each checkpoint
    avg_bleu = []
    avg_wer = []
    
    for step in steps:
        bleu_scores = []
        wer_scores = []
        
        for dataset, metrics in results[step].items():
            if 'bleu' in metrics:
                bleu_scores.append(metrics['bleu'])
            if 'wer' in metrics:
                wer_scores.append(metrics['wer'])
        
        avg_bleu.append(np.mean(bleu_scores) if bleu_scores else 0)
        avg_wer.append(np.mean(wer_scores) if wer_scores else 0)
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(steps, avg_bleu, marker='o', color='#2ecc71', 
                    linewidth=3, markersize=10, label='Avg BLEU')
    line2 = ax2.plot(steps, avg_wer, marker='s', color='#e74c3c', 
                     linewidth=3, markersize=10, label='Avg WER')
    
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average BLEU', fontsize=12, fontweight='bold', color='#2ecc71')
    ax2.set_ylabel('Average WER', fontsize=12, fontweight='bold', color='#e74c3c')
    ax.set_title('Average Performance Evolution', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

def analyze_trends(results):
    """Analyze and print trends across checkpoints."""
    steps = sorted(results.keys())
    
    print("\n" + "="*80)
    print("📈 TRAINING TRENDS ANALYSIS")
    print("="*80)
    
    # Get all datasets
    datasets = list(results[steps[0]].keys())
    
    for dataset in datasets:
        print(f"\n{dataset}:")
        metrics = list(results[steps[0]][dataset].keys())
        
        for metric in metrics:
            values = []
            for step in steps:
                if step in results and dataset in results[step]:
                    if metric in results[step][dataset]:
                        values.append(results[step][dataset][metric])
            
            if len(values) >= 2:
                trend = "📈 Improving" if values[-1] > values[0] else "📉 Declining"
                if metric in ['wer', 'nins', 'ndel', 'nsub']:
                    trend = "📈 Improving" if values[-1] < values[0] else "📉 Declining"
                
                change = abs(values[-1] - values[0])
                print(f"  {metric}: {trend} ({values[0]:.3f} → {values[-1]:.3f}, Δ={change:.3f})")

def create_curve_plots(results):
    """Create comprehensive curve plots for all metrics."""
    
    steps = [5000, 10000, 15000, 20000]
    datasets = list(results[5000].keys())
    
    # Group datasets by task type
    qa_datasets = []
    asr_datasets = []
    translation_datasets = []
    
    for dataset in datasets:
        metrics = list(results[5000][dataset].keys())
        if 'bleu' in metrics and 'bert_p' in metrics:
            qa_datasets.append(dataset)
        elif 'wer' in metrics and 'bleu' not in metrics:
            asr_datasets.append(dataset)
        elif 'wer' in metrics and 'bleu' in metrics:
            translation_datasets.append(dataset)
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(24, 16))
    
    # 1. QA Tasks - BLEU scores
    if qa_datasets:
        ax1 = plt.subplot(3, 3, 1)
        plot_metric_curves(results, qa_datasets, 'bleu', 'BLEU Score Evolution - QA Tasks', steps, ax1)
    
    # 2. QA Tasks - BERT F1 scores  
    if qa_datasets:
        ax2 = plt.subplot(3, 3, 2)
        plot_metric_curves(results, qa_datasets, 'bert_f1', 'BERT F1 Evolution - QA Tasks', steps, ax2)
    
    # 3. QA Tasks - BERT Precision
    if qa_datasets:
        ax3 = plt.subplot(3, 3, 3)
        plot_metric_curves(results, qa_datasets, 'bert_p', 'BERT Precision Evolution - QA Tasks', steps, ax3)
    
    # 4. ASR Tasks - WER
    if asr_datasets:
        ax4 = plt.subplot(3, 3, 4)
        plot_metric_curves(results, asr_datasets, 'wer', 'WER Evolution - ASR Tasks', steps, ax4, lower_is_better=True)
    
    # 5. Translation Tasks - BLEU
    if translation_datasets:
        ax5 = plt.subplot(3, 3, 5)
        plot_metric_curves(results, translation_datasets, 'bleu', 'BLEU Evolution - Translation Tasks', steps, ax5)
    
    # 6. Translation Tasks - WER
    if translation_datasets:
        ax6 = plt.subplot(3, 3, 6)
        plot_metric_curves(results, translation_datasets, 'wer', 'WER Evolution - Translation Tasks', steps, ax6, lower_is_better=True)
    
    # 7. All BLEU scores combined
    ax7 = plt.subplot(3, 3, 7)
    all_bleu_datasets = [d for d in datasets if 'bleu' in results[5000][d]]
    plot_metric_curves(results, all_bleu_datasets, 'bleu', 'All BLEU Scores Evolution', steps, ax7)
    
    # 8. All WER scores combined
    ax8 = plt.subplot(3, 3, 8)
    all_wer_datasets = [d for d in datasets if 'wer' in results[5000][d]]
    plot_metric_curves(results, all_wer_datasets, 'wer', 'All WER Scores Evolution', steps, ax8, lower_is_better=True)
    
    # 9. Performance summary - Average improvement
    ax9 = plt.subplot(3, 3, 9)
    plot_average_performance(results, steps, ax9)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('checkpoint_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function."""
    print("Loading evaluation results for curve analysis...")
    results = load_results()
    
    print("Creating curve plots...")
    create_curve_plots(results)
    
    print("Analyzing trends...")
    analyze_trends(results)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Curve plots saved as 'checkpoint_curves.png'")
    print(f"📈 Check the trends to see which metrics improve over training")

if __name__ == "__main__":
    main()