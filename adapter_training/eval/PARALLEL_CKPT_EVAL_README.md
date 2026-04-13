# Parallel Checkpoint Evaluation Guide

## Quick Start

Use `run_eval_checkpoints.slurm` to evaluate multiple model checkpoints in parallel (one checkpoint per GPU).

```bash
# 1. Set your model name
export XP=Canary-Qwen/SpeechLM2-multitasks_Qwen3-8B

# 2. Configure filtering (Optional)
export CKPT_START=5000   # Start at step 5000
export CKPT_STEP=3000    # Eval every 3000 steps (5000, 8000, 11000...)
# Note: The *last* checkpoint is always included automatically.

# 3. Submit job with 4 GPUs (or more)
sbatch --gres=gpu:4 run_eval_checkpoints.slurm
```

---

## Detailed Usage

### Basic Command

To evaluate **all** checkpoints (excluding headers, unless specified) with default settings (every checkpoint found):

```bash
export XP=your_model_name
export CKPT_START=5000
export CKPT_STEP=3000

# Optional: Filter specific datasets (e.g., only ASR)
export DATASET_FILTER="FLEURS_FR" 

sbatch --gres=gpu:8 run_eval_checkpoints.slurm
```
*(Requesting 8 GPUs here for faster processing)*

### Filtering Datasets

You can restrict evaluation to specific datasets to save time.

**Variable:** `DATASET_FILTER`

**Usage:**
- **Single Dataset**: `export DATASET_FILTER="FLEURS_FR"`
- **By Index**: `export DATASET_FILTER=0`
- **List (requires python dict format)**: `export DATASET_FILTER="['FLEURS_FR', 'VoxPopuli_EN']"`

This is useful if you want to quickly check performance on a single validation set across all checkpoints. Checkpoints
You can control which checkpoints are evaluated using environment variables.

### Advanced Usage: Filtering Checkpoints

**Variables:**
- `CKPT_START`: Step number to start from (Default: 0)
- `CKPT_STEP`: Interval between footsteps (Default: 5000)

**Logic:**
The script evaluates a checkpoint if:
1.  `step >= CKPT_START`
2.  `(step - CKPT_START) % CKPT_STEP == 0`
3.  OR if the checkpoint is the **last** checkpoint (always included).

## What Happens

1.  **Job starts** with 4 GPUs (or however many you requested).
2.  **Filtering**: The script finds all checkpoints matching your criteria.
3.  **Parallel Execution**:
    *   **GPU 0** → Checkpoint A
    *   **GPU 1** → Checkpoint B
    *   **GPU 2** → Checkpoint C
    *   **GPU 3** → Checkpoint D
4.  **Auto-Balancing**: As soon as a GPU finishes its checkpoint, it picks up the next one from the queue.
5.  **Completion**: Job runs until all selected checkpoints are evaluated.

## Monitoring

**View Queue:**
```bash
squeue -u $USER
```

**View Main Log:**
```bash
tail -f log/eval_ckpt_<job_id>.out
```
*This log shows which GPU is assigned to which checkpoint.*

**View Individual Checkpoint Log:**
```bash
tail -f log/eval_<checkpoint_name>_gpu*.log
```

## Results Format

Results are saved to: `evaluations/<model_name>/<checkpoint_name>/results.json`

Example structure:
```json
{
    "FLEURS_FR": {
        "data_type": "asr",
        "lang": "fr",
        "wer": 12.34
    },
    ...
}
```

## Technical Details

- **File Locking**: The script uses file locking to safely write to `results.json` even when multiple checkpoints finish at the same time.
- **Auto-Recovery**: If a GPU finishes early, it immediately picks up the next available checkpoint from the queue.
