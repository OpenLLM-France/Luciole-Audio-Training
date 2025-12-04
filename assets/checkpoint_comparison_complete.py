#!/usr/bin/env python3
"""
Complete checkpoint comparison script with all functions.
Combines simple bar charts, line plots, and detailed analysis.
"""

import json
import yaml
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_results(yaml_path):
    """Load results from all checkpoint directories."""
    
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    checkpoints = yaml_data['models']
    
    results = {}
    for checkpoint, path in checkpoints.items():
        with open(path, 'r') as f:
            results[checkpoint] = json.load(f)
    
    return results

def add_bars(plot_data, checkpoints, datasets, ax, colors, higher_better=True):
    x = np.arange(len(datasets))
    width = 1 / (len(checkpoints) + 1)

    if "Avg WER" in datasets:
        best_idx = {}
        for d in datasets:
            vals = plot_data[d]
            if d == datasets[-1]:
                best_idx[d] = int(np.argmin(vals))
            else:
                best_idx[d] = int(np.argmax(vals))
    else:
        if higher_better:
            func = np.argmax
        else:
            func = np.argmin
        best_idx = {d: int(func(plot_data[d])) for d in datasets}

    for i, ckpt in enumerate(checkpoints):
        values = [plot_data[d][i] for d in datasets]
        bars = ax.bar(x + i * width, values, width, label=f'{ckpt}',
                      color=colors[i], alpha=0.8)

        # IMPORTANT: capture dataset index with enumerate
        for d_i, (bar, val) in enumerate(zip(bars, values)):
            dataset_name = datasets[d_i]

            # highlight the best checkpoint for this dataset
            if i == best_idx[dataset_name]:
                bar.set_edgecolor('black')
                bar.set_linewidth(2.5)
                bar.set_alpha(1.0)

            ax.annotate(f'{val:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

    ax.set_xticks(x + width)
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    proxy_handles = []
    for h in handles:
        color = h.patches[0].get_facecolor()
        proxy_handles.append(Patch(facecolor=color, edgecolor='none'))
    ax.legend(proxy_handles, labels)
    ax.grid(True, alpha=0.3, axis='y')


def plot_comparison(results, colors, ax, metric='bleu', metric_name=None, ylim=(0, 100)):
    
    checkpoints = list(results.keys())
    first_row = results[checkpoints[0]]
    
    plot_data = dict()
    datasets = list()
    
    for dataset in results[checkpoints[0]].keys():
        if metric in first_row[dataset]:
            datasets.append(dataset)
            plot_data[dataset] = []
            for ckpt in checkpoints:
                plot_data[dataset].append(results[ckpt][dataset][metric])
    max_value = max(max(values) for values in plot_data.values())
    if max_value<1 and ylim[1] == 100:
        ylim = (0, 1)

    add_bars(plot_data, checkpoints, datasets, ax, colors, higher_better=False if metric == 'wer' else True)
    
    if metric_name is None:
        metric_name = metric.upper()
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric_name} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=45, ha='right')
    ax.set_ylim(ylim)

def plot_summary_comparison(results, checkpoints, colors, ax, ylim=(0, 100)):
    """Plot overall performance summary."""
    
    # Calculate average performance for each checkpoint
    metrics_summary = {
        'QA - Avg BERT F1': [],
        'QA - Avg ROUGE': [],
        'AST - Avg BLEU': [],
        'ASR - Avg WER': []
    }
    
    for ckpt in checkpoints:
        # BLEU average
        bleu_scores = []
        bert_scores = []
        rouge_scores = []
        wer_scores = []
        
        for dataset, metrics in results[ckpt].items():
            # Prioritize metric types explicitly
            if 'bleu' in metrics and not 'bert_f1' in metrics:
                bleu_scores.append(metrics['bleu'])
            elif 'bert_f1' in metrics:
                bert_scores.append(metrics['bert_f1'])
                rouge_scores.append(metrics['rougeL'])
            elif 'wer' in metrics:
                wer_scores.append(metrics['wer'])

        
        metrics_summary['AST - Avg BLEU'].append(np.mean(bleu_scores) if bleu_scores else 0)
        metrics_summary['QA - Avg BERT F1'].append(np.mean(bert_scores) if bert_scores else 0)
        metrics_summary['ASR - Avg WER'].append(np.mean(wer_scores) if wer_scores else 0)
        metrics_summary['QA - Avg ROUGE'].append(np.mean(rouge_scores) if rouge_scores else 0)
    
    add_bars(metrics_summary, checkpoints, list(metrics_summary.keys()), ax, colors)
    
    ax.set_xlabel('Metric Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Summary', fontsize=14, fontweight='bold')
    ax.set_xticklabels(list(metrics_summary.keys()))
    ax.set_ylim(ylim)

def select_data(results, data_type=None, lang=None):
    """Select data based on metric type."""
    if data_type is not None and isinstance(data_type, str):
        data_type = [data_type]
    if lang is not None and isinstance(lang, str):
        lang = [lang]
    data = {}
    for ckpt, datasets in results.items():
        for dataset, metrics in datasets.items():
            if (data_type is None or metrics["data_type"] in data_type) and (lang is None or metrics["lang"] in lang):
                if not ckpt in data:
                    data[ckpt] = dict()
                data[ckpt][dataset] = metrics
    return data

def create_simple_comparison(results):
    """Create simple, clear comparison plots."""
    
    # Create a large figure with clear subplots
    fig, axes = plt.subplots(3, 2, figsize=(22, 14))
    fig.suptitle('Model Performance Comparison', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    checkpoints = results.keys()
    colors = ['#3498db', '#e74c3c', '#f1c40f', '#2ecc71']  # Blue, Red, Green
    
    ast_data = select_data(results, "ast")
    qa_data = select_data(results, "qa")
    asr_data = select_data(results, "asr")
    summary_data = select_data(results, lang=["en", "fr", "fr-en"])
    
    # 2. BERT F1 Scores Comparison (Top Right)
    plot_comparison(qa_data, colors, axes[0, 0], "bert_f1", "BERT F1")
    plot_comparison(qa_data, colors, axes[0, 1], "rougeL", "Rouge L")
    
    # 1. BLEU Scores Comparison (Top Left)
    plot_comparison(ast_data, colors, axes[1, 0], "bleu", ylim=(0, 60))
    plot_comparison(ast_data, colors, axes[1, 1], "wer", ylim=(50, 100))

    # 3. WER Comparison (Bottom Left)
    plot_comparison(asr_data, colors, axes[2, 0], "wer")
    
    # 4. Overall Performance Summary (Bottom Right)
    plot_summary_comparison(summary_data, checkpoints, colors, axes[2, 1], ylim=(0, 80))
    
    plt.tight_layout()
    plt.savefig('simple_checkpoint_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_simple_summary(results):
    """Print a simple comparison summary."""
    print("\n" + "="*80)
    print("📊 SIMPLE CHECKPOINT COMPARISON SUMMARY")
    print("="*80)
    
    checkpoints = list(results.keys())
    
    print("\n🏆 BEST PERFORMING CHECKPOINT BY DATASET:")
    print("-" * 50)
    
    for dataset, metrics in results[checkpoints[0]].items():
        print(f"\n{dataset}:")
        
        for metric in metrics.keys():
            values = {}
            for ckpt in checkpoints:
                values[ckpt] = results[ckpt][dataset][metric]
            
            # Determine best (lower for WER/error metrics, higher for others)
            if metric in ['wer', 'nins', 'ndel', 'nsub']:
                best_ckpt = min(values, key=values.get)
                symbol = "📉"
            elif metric in ['lang', 'data_type']:
                continue
            else:
                best_ckpt = max(values, key=values.get)
                symbol = "📈"
            
            print(f"  {symbol} {metric}: {best_ckpt}  ({values[best_ckpt]:.3f})")

def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="comparison.yaml",
        help="Path to the experiment folder containing results from the evaluation script."
    )
    args = parser.parse_args()    
    print("Creating simple and clear checkpoint comparisons...")
    results = load_results(args.model_file)
    
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