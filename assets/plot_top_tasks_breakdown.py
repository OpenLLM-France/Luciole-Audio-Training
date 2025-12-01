#!/usr/bin/env python3
"""
Script to plot detailed task breakdown for training split only - Top 15 by duration
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root of project
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PLOTS_DIR = os.path.join(ASSETS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)  # Ensure folder exists
os.makedirs(os.path.join(PLOTS_DIR, "task_duration_ranges"), exist_ok=True)

DATA_PATH = os.path.join(ASSETS_DIR, "datasets_metadata.csv")
OUTPUT_PATH = os.path.join(PLOTS_DIR, "top_15_tasks_breakdown_train.png")

def load_and_process_data(data_path):
    """Load and process the dataset metadata"""
    print(f"📂 Loading data from: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from all string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # Convert duration to hours
    df['total_duration_hours'] = df['total_duration_sec'] / 3600
    
    print(f"✅ Loaded {len(df)} dataset entries")
    return df

def create_top_tasks_breakdown(df, output_path, top_n=15):
    """Create two detailed task breakdown visualizations for top N tasks by duration (train split only)"""
    
    # Filter for training data only
    train_data = df[df['split'] == 'train'].copy()
    print(f"📊 Filtered to {len(train_data)} training entries")
    
    if len(train_data) == 0:
        print("❌ No training data found!")
        return None
    
    # Group by task characteristics and sum durations
    task_stats = train_data.groupby(['sub_task', 'language']).agg({
        'total_duration_hours': 'sum',
        'num_samples': 'sum',
        'dataset_name': 'nunique'  # Number of unique datasets
    }).reset_index()
    
    # Create detailed task labels like in the reference image
    def format_task_label(row):
        task = row['sub_task']
        lang = row['language'].upper()
        
        if task == 'asr':
            return f"ASR\n({lang})"
        elif task == 'ast':
            return f"AST\n({lang})"
        elif task == 'qa_audio-context-text-question':
            return f"Audio Context + Text Question\n({lang})"
        elif task == 'qa_audio-question-only':
            return f"AQ Only\n({lang})"
        elif task == 'qa_audio-question-text-context':
            return f"AQ + Text Context\n({lang})"
        elif task == 'audio-question-answering':
            return f"AQA\n({lang})"
        else:
            return f"{task.replace('_', ' ').replace('-', ' ').title()}\n({lang})"
    
    task_stats['task_label'] = task_stats.apply(format_task_label, axis=1)
    
    # Calculate percentages
    total_duration = task_stats['total_duration_hours'].sum()
    task_stats['percentage'] = (task_stats['total_duration_hours'] / total_duration) * 100
    
    # Sort by duration and get top N
    task_stats = task_stats.sort_values('total_duration_hours', ascending=False).head(top_n)
    
    print(f"🔝 Showing top {len(task_stats)} tasks by duration")
    
    # Create color map based on task type - matching reference image colors
    task_colors = {
        'asr': ['#3182bd', '#ff7f0e'],      # Blue for EN, Orange for FR
        'ast': '#2ca02c',                   # Green  
        'qa_audio-context-text-question': ['#9467bd', '#8c564b'],  # Purple/Brown
        'qa_audio-question-text-context': '#d62728',   # Red    
        'qa_audio-question-only': ['#e377c2', '#bcbd22'],  # Pink/Olive
        'audio-question-answering': '#17becf',  # Cyan
    }
    
    def get_color(row):
        task = row['sub_task']
        lang = row['language']
        
        if task in task_colors:
            if isinstance(task_colors[task], list):
                # Use first color for English, second for French/others
                return task_colors[task][0] if lang == 'en' else task_colors[task][1]
            else:
                return task_colors[task]
        else:
            return '#7f7f7f'  # Default gray
    
    # Split data into two groups
    top_5 = task_stats.head(5)
    remaining = task_stats.iloc[5:]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plt.style.use('seaborn-v0_8-whitegrid')  # Light gray background like reference
    
    # Plot 1: Top 5 tasks
    if len(top_5) > 0:
        plot_data1 = top_5.sort_values('total_duration_hours', ascending=True)
        colors1 = [get_color(row) for _, row in plot_data1.iterrows()]
        
        bars1 = ax1.barh(range(len(plot_data1)), plot_data1['total_duration_hours'], 
                        color=colors1, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Customize first subplot
        ax1.set_yticks(range(len(plot_data1)))
        ax1.set_yticklabels(plot_data1['task_label'], fontsize=13)
        ax1.set_xlabel('Duration (Hours)', fontsize=12)
        ax1.set_title('Top 5 Tasks by Duration', fontsize=14, fontweight='bold', pad=20)
        
        # Style first subplot
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.grid(axis='x', alpha=0.3, color='white', linewidth=1)
        ax1.set_axisbelow(True)
        ax1.set_facecolor('#f5f5f5')
        
        # Add labels to first subplot
        for bar, duration, percentage in zip(bars1, plot_data1['total_duration_hours'], plot_data1['percentage']):
            label = f"{percentage:.1f}% ({duration:.1f}h)"
            x_pos = bar.get_width() + total_duration * 0.01
            y_pos = bar.get_y() + bar.get_height()/2
            ax1.text(x_pos, y_pos, label, va='center', ha='left', 
                    fontsize=12, fontweight='bold', color='black')
    
    # Plot 2: Remaining tasks
    if len(remaining) > 0:
        plot_data2 = remaining.sort_values('total_duration_hours', ascending=True)
        colors2 = [get_color(row) for _, row in plot_data2.iterrows()]
        
        bars2 = ax2.barh(range(len(plot_data2)), plot_data2['total_duration_hours'], 
                        color=colors2, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Customize second subplot
        ax2.set_yticks(range(len(plot_data2)))
        ax2.set_yticklabels(plot_data2['task_label'], fontsize=13)
        ax2.set_xlabel('Duration (Hours)', fontsize=12)
        ax2.set_title('Remaining Tasks', fontsize=14, fontweight='bold', pad=20)
        
        # Style second subplot
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.grid(axis='x', alpha=0.3, color='white', linewidth=1)
        ax2.set_axisbelow(True)
        ax2.set_facecolor('#f5f5f5')
        
        # Add labels to second subplot
        max_remaining_duration = plot_data2['total_duration_hours'].max()
        for bar, duration, percentage in zip(bars2, plot_data2['total_duration_hours'], plot_data2['percentage']):
            label = f"{percentage:.1f}% ({duration:.1f}h)"
            # Use relative offset based on the max value in this subplot for better positioning
            x_pos = bar.get_width() + max_remaining_duration * 0.05
            y_pos = bar.get_y() + bar.get_height()/2
            ax2.text(x_pos, y_pos, label, va='center', ha='left', 
                    fontsize=12, fontweight='bold', color='black')
    else:
        ax2.text(0.5, 0.5, 'No additional tasks', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Remaining Tasks', fontsize=14, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('Detailed Task Breakdown - Train Split', fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.15, right=0.95, wspace=0.3)
    
    # Save the combined plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved combined tasks breakdown plot: {output_path}")
    plt.close()
    
    # Print detailed breakdown
    print("\n📋 Detailed Task Breakdown (All Tasks):")
    print("=" * 80)
    print("TOP 5 TASKS:")
    for i, row in top_5.iterrows():
        print(f"{row['task_label'].replace(chr(10), ' '):50} | {row['total_duration_hours']:8.1f}h | "
              f"{row['percentage']:5.1f}% | {row['num_samples']:8,} samples")
    
    if len(remaining) > 0:
        print("\nREMAINING TASKS:")
        for i, row in remaining.iterrows():
            print(f"{row['task_label'].replace(chr(10), ' '):50} | {row['total_duration_hours']:8.1f}h | "
                  f"{row['percentage']:5.1f}% | {row['num_samples']:8,} samples")
    
    return True

def main():
    """Main function"""
    print("🎯 Top 15 Tasks Breakdown Analysis (Training Data Only)")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    try:
        # Load and process data
        df = load_and_process_data(DATA_PATH)
        
        # Create the breakdown plot
        fig = create_top_tasks_breakdown(df, OUTPUT_PATH, top_n=15)
        
        if fig is not None:
            print(f"\n🎉 Analysis complete! Check the generated plot at: {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())