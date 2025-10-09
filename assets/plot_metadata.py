import os
import pandas as pd
import matplotlib.pyplot as plt

# Resolve paths properly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root of project
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)  # Ensure folder exists

# Load CSV (expand ~ manually)
data_path = os.path.join(ASSETS_DIR, "datasets_metadata.csv")
df = pd.read_csv(data_path, sep=r'\s*,\s*', engine='python')

for split in "train", "test":

    # Copy dataframe
    qa_df = df.copy()

    # Filter by split
    qa_df = qa_df[qa_df['split'] == split]

    # Compute duration in hours
    qa_df['total_duration_hours'] = qa_df['total_duration_sec'] / 3600

    # Rename sub_task labels
    qa_df['sub_task'] = qa_df['sub_task'].replace({
        'asr': 'ASR',
        'ast': 'AST',
        'qa_audio-context-text-question': 'Audio Context + Text Question',
        'qa_audio-question-only': 'Audio Question Only',
        'qa_audio-question-text-context': 'Audio Question + Text Context',
        'sqa': 'Audio Question + Text Context', # 'Spoken Question Answering',
        'pqa': 'Paralinguistic Question Answering',
        'sds': 'Summarization', # 'Spoken Dialogue Summarization'
    })

    # Aggregate by sub_task and language
    grouped = qa_df.groupby(['sub_task', 'language'])['total_duration_hours'].sum().reset_index()

    # Sort by total duration
    grouped = grouped.sort_values(by='total_duration_hours', ascending=False)

    # Combine sub_task and language for labels
    labels = grouped.apply(
        lambda row: f"{row['sub_task']} ({row['language']})" if row['language'] else row['sub_task'],
        axis=1
    )

    # Function for pie chart labels
    def absolute_value(val):
        total_hours = grouped['total_duration_hours'].sum()
        hours_value = round(val / 100 * total_hours, 0)
        sep = "\n" if val > 1.4 else " "
        if hours_value >= 1:
            return f'{val:.1f}%{sep}({hours_value:.0f} hr)'
        
    ratio = grouped["total_duration_hours"] / grouped["total_duration_hours"].sum()
    explode = []
    current_explode = 0.1
    for r in ratio[::-1]:
        if r < 0.02:
            explode.append(current_explode)
            current_explode /= 2
        else:
            explode.append(0)
    explode = explode[::-1]

    # Donut-style pie chart
    fig, ax = plt.subplots(figsize=(15, 10))
    wedges, texts, autotexts = ax.pie(
        grouped['total_duration_hours'],
        labels=labels,
        autopct=absolute_value,
        startangle=180,
        pctdistance=0.85,
        explode=explode,
        wedgeprops={'width': 0.4, 'edgecolor': 'black', 'linewidth': 0.3},
        textprops={'fontsize': 10}
    )

    # Set title and layout
    ax.set_title(f"Split: {split}", fontsize=14)
    ax.axis('equal')
    plt.tight_layout()

    # ✅ Always save to absolute path
    output_path = os.path.join(ASSETS_DIR, f"subtasks_{split}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"✅ Donut chart saved at: {output_path}")

plt.show()
