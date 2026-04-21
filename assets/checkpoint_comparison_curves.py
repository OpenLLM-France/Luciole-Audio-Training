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
import argparse

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(experiment_folder):
    """Load results from all checkpoint directories."""
    
    checkpoints_folder = Path(experiment_folder)
    results_files = checkpoints_folder.rglob('results.json')
    checkpoints = {int(p.parent.name.strip("-last").split('=')[1]): str(p) for p in results_files}
    checkpoints = dict(sorted(checkpoints.items()))
    
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

def plot_task_summary(results, asr_datasets, ast_datasets, qa_datasets, steps, ax):
    """Plot one averaged curve per task as % change from the first checkpoint.

    WER is a lower-is-better metric, so its sign is flipped: a WER drop shows as a
    positive % change. All three curves start at 0% by construction.
    """
    tasks = [
        ('ASR (WER)',       asr_datasets, 'wer',        True),
        ('AST (METEOR)',    ast_datasets, 'meteor',     False),
        ('QA (flow_judge)', qa_datasets,  'flow_judge', False),
    ]
    for label, datasets, metric, lower_better in tasks:
        avg = []
        for s in steps:
            vals = [results[s][d][metric]
                    for d in datasets
                    if d in results[s] and metric in results[s][d]]
            avg.append(np.mean(vals) if vals else np.nan)
        arr = np.array(avg, dtype=float)
        if np.all(np.isnan(arr)):
            continue
        baseline = next((v for v in arr if not np.isnan(v)), np.nan)
        if np.isnan(baseline) or baseline == 0:
            continue
        pct = (arr - baseline) / baseline * 100.0
        if lower_better:
            pct = -pct
        ax.plot(steps, pct, marker='o', linewidth=2.5, markersize=8, label=label)

    ax.axhline(0.0, color='gray', linewidth=1, linestyle='--', alpha=0.6)
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('% change from first checkpoint (higher = better)', fontsize=12, fontweight='bold')
    ax.set_title('Task Summary — Relative Improvement', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

def analyze_trends(results):
    """Analyze and print trends across checkpoints."""
    steps = sorted(results.keys())
    
    print("\n" + "="*80)
    print("📈 TRAINING TRENDS ANALYSIS")
    print("="*80)
    
    # Get all datasets across all checkpoints
    datasets = sorted({d for step in steps for d in results[step].keys()})

    for dataset in datasets:
        print(f"\n{dataset}:")
        metrics = set()
        for step in steps:
            if dataset in results[step]:
                metrics.update(results[step][dataset].keys())
        metrics = [m for m in metrics if m not in ['data_type', 'lang']]
        
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

def _classify_datasets(results):
    """Group datasets by `data_type` field: ASR, AST, QA (qa/mqa/aqa)."""
    asr, ast, qa = [], [], []
    all_datasets = sorted({d for r in results.values() for d in r})
    for dataset in all_datasets:
        for step in results:
            entry = results[step].get(dataset)
            if entry and 'data_type' in entry:
                dt = entry['data_type']
                if dt == 'asr':
                    asr.append(dataset)
                elif dt == 'ast':
                    ast.append(dataset)
                elif dt in ('qa', 'mqa', 'aqa'):
                    qa.append(dataset)
                break
    return asr, ast, qa


def create_curve_plots(results, plot_folder="plots", plot_name="checkpoint_curves.png"):
    """Create a 2x2 curve plot: ASR/WER, AST/METEOR, QA/flow_judge, and a normalized task summary."""

    plot_folder = Path(plot_folder)
    plot_folder.mkdir(exist_ok=True, parents=True)
    if not plot_name.endswith(".png"):
        plot_name = f"{plot_name}.png"

    steps = list(results.keys())
    asr_datasets, ast_datasets, qa_datasets = _classify_datasets(results)

    fig = plt.figure(figsize=(16, 10))

    ax_asr = plt.subplot(2, 2, 1)
    plot_metric_curves(results, asr_datasets, 'wer', 'ASR — WER', steps, ax_asr, lower_is_better=True)

    ax_ast = plt.subplot(2, 2, 2)
    plot_metric_curves(results, ast_datasets, 'meteor', 'AST — METEOR', steps, ax_ast)

    ax_qa = plt.subplot(2, 2, 3)
    plot_metric_curves(results, qa_datasets, 'flow_judge', 'QA — Flow Judge', steps, ax_qa)

    ax_sum = plt.subplot(2, 2, 4)
    plot_task_summary(results, asr_datasets, ast_datasets, qa_datasets, steps, ax_sum)

    plt.tight_layout(pad=3.0)
    plt.savefig(plot_folder / Path(plot_name), dpi=300, bbox_inches='tight')
    plt.close()

def _is_experiment_folder(path: Path) -> bool:
    """True if `path` directly contains at least one `step=*/results.json`."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and child.name.startswith("step=") and (child / "results.json").exists():
            return True
    return False


def load_concatenated_results(paths):
    """Load and chain results from multiple experiments.

    Step numbers of each subsequent experiment are offset by the max step of
    the previous one, so the whole curriculum renders as one continuous curve.
    """
    combined = {}
    offset = 0
    for path in paths:
        if not _is_experiment_folder(Path(path)):
            print(f"⚠ Skipping '{path}': no step=*/results.json found.")
            continue
        results = load_results(experiment_folder=str(path))
        if not results:
            print(f"⚠ Skipping '{path}': no results loaded.")
            continue
        for step, data in results.items():
            combined[offset + step] = data
        offset += max(results.keys())
    return dict(sorted(combined.items()))


def main():
    """Main function."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_folders",
        type=str,
        nargs='+',
        help=(
            "One or more paths. A single path can be an experiment folder or a parent "
            "folder containing many experiments. Multiple paths are chained into a "
            "single curriculum curve (e.g. path/to/exp1 path/to/exp2)."
        )
    )
    parser.add_argument(
        "--plot_folder",
        type=str,
        default="plots",
        help="Path to save the plots."
    )
    args = parser.parse_args()

    if len(args.experiment_folders) > 1:
        paths = [Path(p) for p in args.experiment_folders]
        name = '+'.join(p.name for p in paths)
        print("\n" + "=" * 80)
        print(f"Chained experiment: {name}")
        for p in paths:
            print(f"  - {p}")
        print("=" * 80)
        print("Loading and concatenating evaluation results...")
        results = load_concatenated_results(paths)
        if not results:
            print("No results found for chained experiments.")
            return
        print("Creating curve plots...")
        create_curve_plots(results, plot_folder=args.plot_folder, plot_name=name)
        print("Analyzing trends...")
        analyze_trends(results)
        print(f"\n✅ Analysis complete!")
        print(f"📊 Curve plots saved under '{args.plot_folder}/'")
        return

    root = Path(args.experiment_folders[0])
    if _is_experiment_folder(root):
        experiments = [root]
    else:
        experiments = sorted(p for p in root.iterdir() if _is_experiment_folder(p))
        if not experiments:
            print(f"No experiment folders (step=*/results.json) found under {root}")
            return
        print(f"Found {len(experiments)} experiments under {root}")

    for exp in experiments:
        print("\n" + "=" * 80)
        print(f"Experiment: {exp.name}")
        print("=" * 80)
        print("Loading evaluation results for curve analysis...")
        results = load_results(experiment_folder=str(exp))

        print("Creating curve plots...")
        create_curve_plots(results, plot_folder=args.plot_folder, plot_name=exp.name)

        print("Analyzing trends...")
        analyze_trends(results)

    print(f"\n✅ Analysis complete!")
    print(f"📊 Curve plots saved under '{args.plot_folder}/'")

if __name__ == "__main__":
    main()