#!/usr/bin/env python3
"""
Create focused plots showing Audio Segment Duration Ranges for each task type

This script generates separate duration range visualizations for each task in your dataset,
helping to understand task-specific patterns in audio segment lengths.

Usage:
    python plot_duration_ranges_by_task.py

Generates:
    - duration_ranges_{task_name}.png for each task
    - Shows min/max ranges with proper time units (seconds, minutes, hours)
    - Color-codes by language when multiple languages exist
    - Includes task-specific statistics
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# Resolve paths properly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root of project
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PLOTS_DIR = os.path.join(ASSETS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)  # Ensure folder exists
os.makedirs(os.path.join(PLOTS_DIR, "task_duration_ranges"), exist_ok=True)

def format_duration_ticks(ax, axis='x'):
    """Format duration axis with appropriate time units (seconds, minutes, hours)"""
    if axis == 'x':
        ticks = ax.get_xticks()
    else:
        ticks = ax.get_yticks()
    
    # Create formatted labels
    labels = []
    for tick in ticks:
        if tick <= 0:
            labels.append('')
            continue
            
        minutes = tick
        seconds = minutes * 60
        hours = minutes / 60
        
        if minutes < 1:
            if seconds < 1:
                labels.append(f"{seconds:.2f}s")
            elif seconds < 10:
                labels.append(f"{seconds:.1f}s")
            else:
                labels.append(f"{seconds:.0f}s")
        elif minutes < 60:
            if minutes < 10:
                labels.append(f"{minutes:.1f}min")
            else:
                labels.append(f"{minutes:.0f}min")
        else:
            labels.append(f"{hours:.1f}h")
    
    if axis == 'x':
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    
    return ax

def load_data():
    """Load the dataset metadata"""
    # Try multiple possible locations for the metadata file
    possible_paths = [
        "datasets_metadata.csv",
        "assets/datasets_metadata.csv", 
        f"{ASSETS_DIR}/datasets_metadata.csv",
        "data/datasets_metadata.csv"
    ]
    
    for data_path in possible_paths:
        if os.path.exists(data_path):
            print(f"📂 Loading data from: {data_path}")
            break
    else:
        raise FileNotFoundError("Could not find datasets_metadata.csv in any expected location")
    
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    # Strip whitespace from all string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    df['total_duration_hours'] = df['total_duration_sec'] / 3600
    
    print(f"✅ Loaded {len(df)} dataset entries")
    return df

def create_task_duration_ranges(df, task_name, save_path):
    """Create focused plot showing duration ranges for a specific task"""
    
    # Filter data for specific task
    task_data = df[df['sub_task'] == task_name].copy()
    
    if len(task_data) == 0:
        print(f"⚠️  No data found for task: {task_name}")
        return None
    
    # Calculate min/max segment durations for each dataset
    dataset_ranges = task_data.groupby(['dataset_name', 'language']).agg({
        'min_segment_duration_sec': 'min',
        'max_segment_duration_sec': 'max',
        'total_duration_hours': 'sum',
        'num_audio_segments': 'sum'
    }).reset_index()
    
    # Convert to minutes for plotting
    dataset_ranges['min_duration_min'] = dataset_ranges['min_segment_duration_sec'] / 60
    dataset_ranges['max_duration_min'] = dataset_ranges['max_segment_duration_sec'] / 60
    dataset_ranges['range_min'] = dataset_ranges['max_duration_min'] - dataset_ranges['min_duration_min']
    
    # Create better labels
    dataset_ranges['short_label'] = dataset_ranges['dataset_name'].str[:25]
    dataset_ranges['full_label'] = (dataset_ranges['dataset_name'].str[:25] + 
                                   ' (' + dataset_ranges['language'] + ')')
    
    # Sort by total duration
    dataset_ranges = dataset_ranges.sort_values('total_duration_hours', ascending=True)
    
    # Take top 15 datasets for better readability
    top_datasets = dataset_ranges.tail(min(15, len(dataset_ranges)))
    
    # Create single large plot
    fig, ax = plt.subplots(1, 1, figsize=(14, max(8, len(top_datasets) * 0.6)))
    
    y_pos = range(len(top_datasets))
    
    # Define colors based on language
    color_map = {
        'en': '#1f77b4',      # Blue for English
        'fr': '#ff7f0e',      # Orange for French  
        'fr-en': '#2ca02c',   # Green for French-English
        '': '#d62728',        # Red for unspecified
        'mixed': '#9467bd'    # Purple for mixed
    }
    
    # Create range bars with language-based coloring
    for i, (_, row) in enumerate(top_datasets.iterrows()):
        bar_color = color_map.get(row['language'], '#9467bd')
        
        # Draw range bar
        ax.barh(i, row['range_min'], left=row['min_duration_min'], 
                color=bar_color, alpha=0.7, height=0.7, 
                edgecolor='darkblue', linewidth=0.8)
        
        # Add min marker with enhanced styling
        ax.scatter(row['min_duration_min'], i, color='darkgreen', s=80, 
                   marker='|', linewidth=4, alpha=0.95, zorder=5)
        
        # Add max marker with enhanced styling
        ax.scatter(row['max_duration_min'], i, color='darkred', s=80, 
                   marker='|', linewidth=4, alpha=0.95, zorder=5)
        
        # Add duration range text annotation for significant ranges
        if row['range_min'] > 0.05:  # Show for ranges > 3 seconds
            range_center = row['min_duration_min'] + row['range_min'] / 2
            if row['range_min'] < 60:
                range_text = f"{row['range_min']:.1f}min" if row['range_min'] >= 1 else f"{row['range_min']*60:.0f}s"
            else:
                range_text = f"{row['range_min']/60:.1f}h"
            
            ax.text(range_center, i, range_text, ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='navy', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Add min/max duration values below markers
        if row['min_duration_min'] < 1:
            min_text = f"{row['min_duration_min']*60:.0f}s"
        elif row['min_duration_min'] < 60:
            min_text = f"{row['min_duration_min']:.1f}min"
        else:
            min_text = f"{row['min_duration_min']/60:.1f}h"
            
        if row['max_duration_min'] < 60:
            max_text = f"{row['max_duration_min']:.1f}min"
        else:
            max_text = f"{row['max_duration_min']/60:.1f}h"
        
        # Add small text labels for min/max values
        ax.text(row['min_duration_min'], i-0.25, min_text, ha='center', va='top', 
                fontsize=7, color='darkgreen', fontweight='bold')
        ax.text(row['max_duration_min'], i-0.25, max_text, ha='center', va='top', 
                fontsize=7, color='darkred', fontweight='bold')
    
    # Enhance y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([row['full_label'] for _, row in top_datasets.iterrows()], fontsize=10)
    
    # Enhanced labeling and styling
    task_display = task_name.replace('_', ' ').replace('-', ' ').title()
    ax.set_xlabel('Audio Segment Duration', fontsize=14, fontweight='bold')
    ax.set_title(f'Audio Segment Duration Ranges - {task_display}\n'
                f'({len(top_datasets)} datasets, {task_data["total_duration_hours"].sum():.0f} total hours)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.5, linestyle='--', linewidth=1)
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Format the x-axis with proper time units
    format_duration_ticks(ax, 'x')
    
    # Enhanced legend
    legend_elements = [
        ax.scatter([], [], color='darkgreen', marker='|', linewidth=4, s=80, label='Minimum Duration'),
        ax.scatter([], [], color='darkred', marker='|', linewidth=4, s=80, label='Maximum Duration'),
    ]
    
    # Add language legend if multiple languages exist
    unique_langs = top_datasets['language'].unique()
    if len(unique_langs) > 1:
        for lang in sorted(unique_langs):
            if lang:  # Skip empty language
                lang_label = lang.upper() if lang else 'Unspecified'
                legend_elements.append(
                    ax.bar([], [], color=color_map.get(lang, '#9467bd'), alpha=0.7, 
                          label=f'{lang_label} Language')
                )
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)
    
    # Add comprehensive task-specific statistics
    stats_text = f"""Task Statistics:
    • Datasets: {len(top_datasets)}
    • Shortest segment: {top_datasets['min_duration_min'].min():.3f} min
    • Longest segment: {top_datasets['max_duration_min'].max():.1f} min  
    • Average range: {top_datasets['range_min'].mean():.2f} min
    • Total duration: {task_data['total_duration_hours'].sum():.0f} hours"""
        
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved {task_display} duration ranges: {save_path}")
    
    return fig

def create_combined_task_group_plot(df, task_patterns, group_name, save_path):
    """Create combined plot for a group of related tasks"""
    
    # Filter data for tasks matching any of the patterns
    matching_tasks = []
    for task in df['sub_task'].unique():
        for pattern in task_patterns:
            if pattern.lower() in task.lower():
                matching_tasks.append(task)
                break
    
    if not matching_tasks:
        print(f"⚠️  No tasks found matching patterns: {task_patterns}")
        return None
    
    print(f"📊 Found {group_name} tasks: {matching_tasks}")
    
    # Filter data
    group_data = df[df['sub_task'].isin(matching_tasks)].copy()
    
    # Calculate min/max segment durations for each dataset
    dataset_ranges = group_data.groupby(['dataset_name', 'sub_task', 'language']).agg({
        'min_segment_duration_sec': 'min',
        'max_segment_duration_sec': 'max',
        'total_duration_hours': 'sum',
        'num_audio_segments': 'sum'
    }).reset_index()
    
    # Convert to minutes for plotting
    dataset_ranges['min_duration_min'] = dataset_ranges['min_segment_duration_sec'] / 60
    dataset_ranges['max_duration_min'] = dataset_ranges['max_segment_duration_sec'] / 60
    dataset_ranges['range_min'] = dataset_ranges['max_duration_min'] - dataset_ranges['min_duration_min']
    
    # Create better labels with task info
    dataset_ranges['task_short'] = dataset_ranges['sub_task'].str.replace('_', ' ').str.title()
    dataset_ranges['full_label'] = (dataset_ranges['dataset_name'].str[:20] + 
                                   '\n[' + dataset_ranges['task_short'] + ', ' + dataset_ranges['language'] + ']')
    
    # Sort by task type first, then by total duration
    task_order = {task: i for i, task in enumerate(matching_tasks)}
    dataset_ranges['task_order'] = dataset_ranges['sub_task'].map(task_order)
    dataset_ranges = dataset_ranges.sort_values(['task_order', 'total_duration_hours'], ascending=[True, True])
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(16, max(10, len(dataset_ranges) * 0.5)))
    
    y_pos = range(len(dataset_ranges))
    
    # Define colors based on task type with consistent mapping
    task_color_map = {
        'asr': '#FFD700',  # Gold/Yellow for ASR
        'ast': '#87CEEB',  # Light Blue for AST (Translation)
        'qa_audio_context_text_question': '#1f77b4',  # Blue
        'qa_audio_question_text_context': '#ff7f0e',   # Orange
        'qa_audio_question_only': '#2ca02c',           # Green
    }
    
    # Assign colors based on task patterns (handle actual task name formats)
    task_colors = {}
    for task in matching_tasks:
        task_clean = task.strip().lower()
        if 'asr' in task_clean:
            task_colors[task] = task_color_map['asr']
        elif 'ast' in task_clean or 'translation' in task_clean:
            task_colors[task] = task_color_map['ast']
        elif 'qa_audio-context-text-question' in task_clean or 'context-text-question' in task_clean:
            task_colors[task] = task_color_map['qa_audio_context_text_question']
        elif 'qa_audio-question-text-context' in task_clean or 'question-text-context' in task_clean:
            task_colors[task] = task_color_map['qa_audio_question_text_context']
        elif 'qa_audio-question-only' in task_clean or 'question-only' in task_clean:
            task_colors[task] = task_color_map['qa_audio_question_only']
        else:
            # Fallback colors for any unexpected tasks
            task_colors[task] = plt.cm.Set3(len(task_colors) % 12)
    
    # Create range bars with task-based coloring
    for i, (_, row) in enumerate(dataset_ranges.iterrows()):
        bar_color = task_colors.get(row['sub_task'], '#d62728')
        
        # Draw range bar
        ax.barh(i, row['range_min'], left=row['min_duration_min'], 
                color=bar_color, alpha=0.7, height=0.7, 
                edgecolor='darkblue', linewidth=0.8)
        
        # Add min/max markers
        ax.scatter(row['min_duration_min'], i, color='darkgreen', s=80, 
                   marker='|', linewidth=4, alpha=0.95, zorder=5)
        ax.scatter(row['max_duration_min'], i, color='darkred', s=80, 
                   marker='|', linewidth=4, alpha=0.95, zorder=5)
        
        # Add duration range text annotation
        if row['range_min'] > 0.05:  # Show for ranges > 3 seconds
            range_center = row['min_duration_min'] + row['range_min'] / 2
            if row['range_min'] < 60:
                range_text = f"{row['range_min']:.1f}min" if row['range_min'] >= 1 else f"{row['range_min']*60:.0f}s"
            else:
                range_text = f"{row['range_min']/60:.1f}h"
            
            ax.text(range_center, i, range_text, ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='navy', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Add min/max values below markers
        if row['min_duration_min'] < 1:
            min_text = f"{row['min_duration_min']*60:.0f}s"
        elif row['min_duration_min'] < 60:
            min_text = f"{row['min_duration_min']:.1f}min"
        else:
            min_text = f"{row['min_duration_min']/60:.1f}h"
            
        if row['max_duration_min'] < 60:
            max_text = f"{row['max_duration_min']:.1f}min"
        else:
            max_text = f"{row['max_duration_min']/60:.1f}h"
        
        ax.text(row['min_duration_min'], i-0.25, min_text, ha='center', va='top', 
                fontsize=7, color='darkgreen', fontweight='bold')
        ax.text(row['max_duration_min'], i-0.25, max_text, ha='center', va='top', 
                fontsize=7, color='darkred', fontweight='bold')
    
    # Enhance y-axis labels  
    ax.set_yticks(y_pos)
    ax.set_yticklabels([row['full_label'] for _, row in dataset_ranges.iterrows()], fontsize=9)
    
    # Enhanced labeling
    total_hours = group_data['total_duration_hours'].sum()
    ax.set_xlabel('Audio Segment Duration', fontsize=14, fontweight='bold')
    ax.set_title(f'Audio Segment Duration Ranges - {group_name}\n'
                f'({len(dataset_ranges)} datasets, {total_hours:.0f} total hours)', 
                 fontsize=18, fontweight='bold', pad=25)
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.5, linestyle='--', linewidth=1)
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Format the x-axis
    format_duration_ticks(ax, 'x')
    
    # Enhanced legend
    legend_elements = [
        ax.scatter([], [], color='darkgreen', marker='|', linewidth=4, s=80, label='Minimum Duration'),
        ax.scatter([], [], color='darkred', marker='|', linewidth=4, s=80, label='Maximum Duration'),
    ]
    
    # Add task type legend with proper display names
    task_display_names = {
        'asr': 'ASR (Speech Recognition)',
        'ast': 'AST (Speech Translation)',  
        'qa_audio_context_text_question': 'QA: Audio Context + Text Question',
        'qa_audio_question_text_context': 'QA: Audio Question + Text Context',
        'qa_audio_question_only': 'QA: Audio Question Only'
    }
    
    # Create legend entries based on the actual colors used in the plot
    # Get unique task types and their colors from the data
    unique_task_colors = {}
    for task in matching_tasks:
        if task in dataset_ranges['sub_task'].values:
            task_clean = task.strip().lower()
            if 'asr' in task_clean:
                task_type = 'asr'
                display_name = 'ASR (Speech Recognition)'
            elif 'ast' in task_clean or 'translation' in task_clean:
                task_type = 'ast'
                display_name = 'AST (Speech Translation)'
            elif 'qa_audio-context-text-question' in task_clean or 'context-text-question' in task_clean:
                task_type = 'qa_audio_context_text_question'
                display_name = 'QA: Audio Context + Text Question'
            elif 'qa_audio-question-text-context' in task_clean or 'question-text-context' in task_clean:
                task_type = 'qa_audio_question_text_context'
                display_name = 'QA: Audio Question + Text Context'
            elif 'qa_audio-question-only' in task_clean or 'question-only' in task_clean:
                task_type = 'qa_audio_question_only'
                display_name = 'QA: Audio Question Only'
            else:
                task_type = task
                display_name = task.replace('_', ' ').replace('-', ' ').title().strip()
            
            # Store the actual color used for this task type
            if task_type not in unique_task_colors:
                unique_task_colors[task_type] = {
                    'color': task_colors[task],
                    'display_name': display_name
                }
    
    # Add legend entries using the exact colors from the bars with patches
    from matplotlib.patches import Patch
    for task_type, info in unique_task_colors.items():
        legend_elements.append(
            Patch(facecolor=info['color'], alpha=0.7, label=info['display_name'])
        )
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)
    
    # Add statistics
    stats_by_task = []
    for task in matching_tasks:
        task_data = dataset_ranges[dataset_ranges['sub_task'] == task]
        if len(task_data) > 0:
            task_short = task.replace('_', ' ').title()
            stats_by_task.append(f"• {task_short}: {len(task_data)} datasets")
    
    stats_text = f"""{group_name} Statistics:
    {chr(10).join(stats_by_task)}
    • Total datasets: {len(dataset_ranges)}
    • Shortest segment: {dataset_ranges['min_duration_min'].min():.3f} min
    • Longest segment: {dataset_ranges['max_duration_min'].max():.1f} min  
    • Average range: {dataset_ranges['range_min'].mean():.2f} min
    • Total duration: {total_hours:.0f} hours"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='lightcyan', alpha=0.9))
    
    # Add task separators
    current_task = None
    for i, (_, row) in enumerate(dataset_ranges.iterrows()):
        if current_task != row['sub_task']:
            if current_task is not None:
                ax.axhline(y=i-0.5, color='gray', linestyle='-', alpha=0.3, linewidth=1)
            current_task = row['sub_task']
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved combined {group_name} duration ranges: {save_path}")
    
    return fig

def create_all_task_plots(df):
    """Create duration range plots for all tasks and combined task groups"""
    
    # Get unique tasks
    tasks = df['sub_task'].unique()
    
    print(f"\n🎯 Creating duration range plots for {len(tasks)} tasks...")
    print("=" * 60)
    
    generated_files = []
    
    # Create individual task plots
    for task in sorted(tasks):
        # Create clean filename
        clean_task = task.replace(' ', '_').replace('-', '_').lower()
        save_path = os.path.join(PLOTS_DIR, "task_duration_ranges", f"duration_ranges_{clean_task}.png")
        
        try:
            create_task_duration_ranges(df, task, save_path)
            generated_files.append(save_path)
        except Exception as e:
            print(f"❌ Error creating plot for {task}: {e}")
    
    # Create combined plots for task groups
    print(f"\n🎯 Creating combined task group plots...")
    print("=" * 60)
    
    task_groups = [
        (['qa'], 'All QA Tasks Combined', os.path.join(PLOTS_DIR, 'duration_ranges_all_qa_combined.png')),
        (['asr', 'ast'], 'ASR and Translation Tasks Combined', os.path.join(PLOTS_DIR, 'duration_ranges_asr_ast_combined.png')),
        (['transcription', 'translation'], 'Speech Tasks Combined', os.path.join(PLOTS_DIR, 'duration_ranges_speech_tasks_combined.png')),
    ]
    
    for patterns, group_name, filename in task_groups:
        try:
            create_combined_task_group_plot(df, patterns, group_name, filename)
            generated_files.append(filename)
        except Exception as e:
            print(f"❌ Error creating combined plot for {group_name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Successfully created {len(generated_files)} plots!")
    print("\n📁 Generated files:")
    for file in generated_files:
        print(f"  • {file}")
    
    # Print summary of tasks analyzed
    print(f"\n📊 Task Summary:")
    for task in sorted(tasks):
        task_data = df[df['sub_task'] == task]
        print(f"  • {task}: {len(task_data)} entries, {task_data['total_duration_sec'].sum()/3600:.0f} hours")

def create_distribution_pies(df):
    """Create the three distribution pie charts"""
    
    # Calculate summary statistics
    task_stats = df.groupby('task_type').agg({
        'total_duration_hours': 'sum'
    })
    
    lang_stats = df.groupby('language').agg({
        'total_duration_hours': 'sum'
    })
    
    # Create figure with three pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # 1. Task Type Distribution
    colors1 = plt.cm.Set3(np.linspace(0, 1, len(task_stats)))
    wedges1, texts1, autotexts1 = ax1.pie(task_stats['total_duration_hours'].values, 
                                          labels=task_stats.index, 
                                          autopct='%1.1f%%',
                                          colors=colors1, 
                                          startangle=90)
    ax1.set_title('Distribution by Task Type\n(Total Hours)', fontweight='bold', fontsize=14)
    
    # 2. Language Distribution
    colors2 = plt.cm.Pastel1(np.linspace(0, 1, len(lang_stats)))
    wedges2, texts2, autotexts2 = ax2.pie(lang_stats['total_duration_hours'].values,
                                          labels=lang_stats.index,
                                          autopct='%1.1f%%',
                                          colors=colors2,
                                          startangle=90,
                                          textprops={'fontsize': 12})
    ax2.set_title('Distribution by Language\n(Total Hours)', fontweight='bold', fontsize=14, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(PLOTS_DIR, "distribution_pie_charts.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Generated plots/distribution_pie_charts.png")

def create_dataset_composition_by_subtask(df, save_path=os.path.join(PLOTS_DIR,"dataset_composition_by_subtask.png")):
    """Create plots showing dataset composition for French ASR and English Question Answering specifically"""
    
    # Filter for specific tasks and languages (training data only)
    # French ASR: ASR task + French language + training split
    french_asr_data = df[(df['sub_task'] == 'asr') & 
                        (df['language'] == 'fr') &
                        (df['split'] == 'train')]
    
    # English Question Answering: QA tasks + English language + training split
    english_qa_data = df[(df['sub_task'].str.contains('qa_audio', case=False, na=False)) & 
                        (df['language'] == 'en') &
                        (df['split'] == 'train')]
    
    # Check if we have data for both categories
    if len(french_asr_data) == 0:
        print("⚠️ No French ASR training data found")
    if len(english_qa_data) == 0:
        print("⚠️ No English QA training data found")
        
    if len(french_asr_data) == 0 and len(english_qa_data) == 0:
        print("❌ No training data found for French ASR or English QA tasks")
        return None
    
    # Create figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: French ASR
    if len(french_asr_data) > 0:
        ax1 = axes[0]
        
        # Group by dataset, sum durations
        french_asr_stats = french_asr_data.groupby(['dataset_name', 'sub_task']).agg({
            'total_duration_hours': 'sum',
            'num_samples': 'sum'
        }).reset_index()
        
        # Sort by duration
        french_asr_stats = french_asr_stats.sort_values('total_duration_hours', ascending=True)
        
        # Create horizontal bar plot
        bars1 = ax1.barh(range(len(french_asr_stats)), french_asr_stats['total_duration_hours'], 
                        color='#d62728', alpha=0.8, edgecolor='darkred', linewidth=1)
        
        # Customize subplot
        ax1.set_yticks(range(len(french_asr_stats)))
        ax1.set_yticklabels([name[:25] + '...' if len(name) > 25 else name 
                           for name in french_asr_stats['dataset_name']], fontsize=12)
        ax1.set_xlabel('Duration (Hours)', fontsize=12)
        ax1.set_title('French ASR Training Datasets\n(Automatic Speech Recognition)', 
                     fontsize=14, fontweight='bold', color='#d62728')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add duration labels
        for bar, duration in zip(bars1, french_asr_stats['total_duration_hours']):
            if duration > 0.5:  # Only label if more than 0.5 hour
                ax1.text(duration + duration*0.02, bar.get_y() + bar.get_height()/2,
                       f'{duration:.1f}h', va='center', fontsize=9, fontweight='bold')
        
        # Add statistics text
        total_french_duration = french_asr_stats['total_duration_hours'].sum()
        total_french_samples = french_asr_stats['num_samples'].sum()
        ax1.text(0.02, 0.98, f'Total: {total_french_duration:.1f}h\nDatasets: {len(french_asr_stats)}\nSamples: {total_french_samples:,}', 
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    else:
        axes[0].text(0.5, 0.5, 'No French ASR training data found', ha='center', va='center', 
                    fontsize=12, transform=axes[0].transAxes)
        axes[0].set_title('French ASR Training Datasets', fontsize=14, fontweight='bold')
    
    # Plot 2: English QA
    if len(english_qa_data) > 0:
        ax2 = axes[1]
        
        # Group by dataset and QA subtask, sum durations
        english_qa_stats = english_qa_data.groupby(['dataset_name', 'sub_task']).agg({
            'total_duration_hours': 'sum',
            'num_samples': 'sum'
        }).reset_index()
        
        # Create labels with QA subtask info
        english_qa_stats['label'] = english_qa_stats['dataset_name'] + '\n[' + \
                                   english_qa_stats['sub_task'].str.replace('qa_audio-', '').str.replace('-', ' ').str.title() + ']'
        
        # Sort by duration
        english_qa_stats = english_qa_stats.sort_values('total_duration_hours', ascending=True)
        
        # Color mapping for different QA subtasks
        qa_colors = {
            'qa_audio-context-text-question': '#1f77b4',  # Blue
            'qa_audio-question-text-context': '#2ca02c',   # Green
            'qa_audio-question-only': '#d62728',           # Red
            'audio-question-answering': '#9467bd'          # Purple (fallback)
        }
        
        # Assign colors based on subtask
        colors = [qa_colors.get(subtask, '#9467bd') for subtask in english_qa_stats['sub_task']]
        
        # Create horizontal bar plot
        bars2 = ax2.barh(range(len(english_qa_stats)), english_qa_stats['total_duration_hours'], 
                        color=colors, alpha=0.8, edgecolor='darkblue', linewidth=1)
        
        # Customize subplot
        ax2.set_yticks(range(len(english_qa_stats)))
        ax2.set_yticklabels([name[:25] + '...' if len(name) > 25 else name 
                           for name in english_qa_stats['dataset_name']], fontsize=12)
        ax2.set_xlabel('Duration (Hours)', fontsize=12)
        ax2.set_title('English Question Answering Training Datasets\n(Audio-based QA Tasks)', 
                     fontsize=14, fontweight='bold', color='#1f77b4')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add duration labels
        for bar, duration in zip(bars2, english_qa_stats['total_duration_hours']):
            if duration > 0.5:  # Only label if more than 0.5 hour
                ax2.text(duration + duration*0.02, bar.get_y() + bar.get_height()/2,
                       f'{duration:.1f}h', va='center', fontsize=9, fontweight='bold')
        
        # Add statistics text
        total_english_duration = english_qa_stats['total_duration_hours'].sum()
        total_english_samples = english_qa_stats['num_samples'].sum()
        ax2.text(0.02, 0.98, f'Total: {total_english_duration:.1f}h\nDatasets: {len(english_qa_stats)}\nSamples: {total_english_samples:,}', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
        
        # Add legend for QA subtasks if multiple types exist
        unique_subtasks = english_qa_stats['sub_task'].unique()
        if len(unique_subtasks) > 1:
            from matplotlib.patches import Patch
            legend_elements = []
            for subtask in unique_subtasks:
                color = qa_colors.get(subtask, '#9467bd')
                label = subtask.replace('qa_audio-', '').replace('-', ' ').title()
                legend_elements.append(Patch(facecolor=color, alpha=0.8, label=label))
            ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'No English QA training data found', ha='center', va='center', 
                    fontsize=12, transform=axes[1].transAxes)
        axes[1].set_title('English Question Answering Training Datasets', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved French ASR and English QA composition plots: {save_path}")
    return fig

def create_language_task_heatmap(df, save_path=os.path.join(PLOTS_DIR,"language_task_heatmap.png")):
    """Create a heatmap showing dataset count and total duration by language and task"""
    
    # Create pivot tables
    duration_pivot = df.pivot_table(values='total_duration_hours', 
                                   index='sub_task', columns='language', 
                                   aggfunc='sum', fill_value=0)
    
    count_pivot = df.pivot_table(values='dataset_name', 
                                index='sub_task', columns='language', 
                                aggfunc='count', fill_value=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Duration heatmap
    sns.heatmap(duration_pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                ax=ax1, cbar_kws={'label': 'Total Duration (Hours)'})
    ax1.set_title('Total Duration by Task and Language', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Language', fontsize=12)
    ax1.set_ylabel('Task Type', fontsize=12)
    
    # Dataset count heatmap
    sns.heatmap(count_pivot, annot=True, fmt='d', cmap='Blues', 
                ax=ax2, cbar_kws={'label': 'Number of Datasets'})
    ax2.set_title('Dataset Count by Task and Language', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Language', fontsize=12)
    ax2.set_ylabel('Task Type', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved language-task heatmap: {save_path}")
    return fig    

def main():
    """Main function to run the script"""
    print("🎵 Audio Segment Duration Range Analysis by Task")
    print("=" * 60)
    
    try:
        # Load data
        df = load_data()
        
        # Create all plots
        # create_all_task_plots(df)
        # create_distribution_pies(df)
        create_dataset_composition_by_subtask(df)
        # create_language_task_heatmap(df)
        print(f"\n🎉 Analysis complete! Check the generated PNG files.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())