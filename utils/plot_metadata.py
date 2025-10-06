import os
import pandas as pd
import matplotlib.pyplot as plt

# Resolve paths properly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root of project
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)  # Ensure folder exists

# Load CSV (expand ~ manually)
data_path = os.path.expanduser("~/Audio_Adapter_Training/assets/datasets_metadata.csv")
df = pd.read_csv(data_path, sep=r',\s*', engine='python')
df.columns = df.columns.str.strip()

# Copy dataframe
qa_df = df.copy()

# Compute duration in hours
qa_df['total_duration_hours'] = qa_df['total_duration_sec'] / 3600

# Rename sub_task labels
qa_df['sub_task'] = qa_df['sub_task'].replace({
    'qa_audio-context-text-question': 'Audio Context + Text Question',
    'qa_audio-question-only        ': 'Audio Question Only',
    'qa_audio-question-text-context': 'Audio Question + Text Context',
    'asr                           ': 'ASR',
    'ast                           ': 'AST',
    'pqa                           ': 'Paralinguistic Question Answering',
    'sqa                           ': 'Spoken Question Answering',
    'sds                           ': 'Spoken Dialogue Summarization'
})

# Aggregate by sub_task and language
grouped = qa_df.groupby(['sub_task', 'language'])['total_duration_hours'].sum().reset_index()

# Combine sub_task and language for labels
labels = grouped.apply(
    lambda row: f"{row['sub_task']} ({row['language'].strip()})" if row['language'] else row['sub_task'],
    axis=1
)

explode = (0, 0, 0, 0.16, 0, 0.1, 0.04, 0, 0, 0, 0)

# Function for pie chart labels
def absolute_value(val):
    total_hours = grouped['total_duration_hours'].sum()
    hours_value = round(val / 100 * total_hours, 0)
    if hours_value >= 1:
        return f'{val:.1f}%\n({hours_value:.0f} hr)'

# Donut-style pie chart
fig, ax = plt.subplots(figsize=(15, 10))
wedges, texts, autotexts = ax.pie(
    grouped['total_duration_hours'],
    labels=labels,
    autopct=absolute_value,
    startangle=90,
    pctdistance=0.85,
    explode=explode,
    wedgeprops={'edgecolor': 'black', 'linewidth': 0.3},
    textprops={'fontsize': 10}
)

# Draw a circle in the middle to make it a donut
centre_circle = plt.Circle((0, 0.1), 0.50, fc='white')
fig.gca().add_artist(centre_circle)

# Set title and layout
ax.set_title("All Sub-Task Durations (in %)", fontsize=14)
ax.axis('equal')
plt.tight_layout()

# ✅ Always save to absolute path
output_path = os.path.join(ASSETS_DIR, "all_subtask_donut_chart_exploded.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"✅ Donut chart saved at: {output_path}")
