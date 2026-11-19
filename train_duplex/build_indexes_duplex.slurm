#!/bin/bash
#SBATCH --job-name=build_indexes_duplex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --hint=nomultithread
#SBATCH --account=qgz@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev

# Build the .idx sidecars of the duplex input_cfgs under the duplex indexes_root (read from
# conf/duplex_luciole.yaml), apart from the SALM mirror ($ALL_CCFRSCRATCH/audio/idx). Without them the
# training job builds the missing ones on the fly, from every rank at startup.
#
# build_indexes.py skips `conversation_as_duplex` entries (unknown type), so it is fed copies where that
# type reads `multimodal_conversation`: same manifest_filepath, and an .idx only holds the line offsets
# of the jsonl, whatever parses it. Already built indexes are skipped.
#
# Submit from train_duplex/ (needs a logs/ dir):
#   mkdir -p logs && sbatch build_indexes_duplex.slurm                                    # both mixes
#   sbatch build_indexes_duplex.slurm conf/data/input_cfg_duplex_fr_en.yaml               # one mix

set -euo pipefail

NEMO_DIR="${NEMO_DIR:-$HOME/NeMo_Duplex}"
CONFIG="${CONFIG:-conf/duplex_luciole.yaml}"
NEMO_MODULE="${NEMO_MODULE:-nemo/2.7.3}"
CONDA_ENV="${CONDA_ENV:-$SCRATCH/speechlm/envs/nemo27-salm_automodel_3}"
INPUT_CFGS=("$@")
(( ${#INPUT_CFGS[@]} )) || INPUT_CFGS=(conf/data/input_cfg_duplex_fr_en.yaml)

module purge
module load arch/h100 "$NEMO_MODULE"
conda activate "$CONDA_ENV"
export PYTHONPATH="$NEMO_DIR${PYTHONPATH:+:$PYTHONPATH}"
# $HOME/NeMo (installed editable) serves a running training: never use it here.
nemo_file="$(python -c "import nemo; print(nemo.__file__)")"
[[ "$nemo_file" == "$NEMO_DIR"/* ]] || { echo "❌ nemo imported from $nemo_file, not $NEMO_DIR" >&2; exit 1; }
export DATA_FOLDER="$ALL_CCFRSCRATCH/audio"  # interpolated in the input_cfgs' manifest_filepath
export TMPDIR="${JOBSCRATCH:-/tmp}"

INDEXES_ROOT="$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG').data.train_ds.indexes_root)" | tail -1)"
[[ "$INDEXES_ROOT" == /* ]] || { echo "❌ could not read data.train_ds.indexes_root from $CONFIG: '$INDEXES_ROOT'" >&2; exit 1; }
[[ "$INDEXES_ROOT" != "$ALL_CCFRSCRATCH/audio/idx" ]] || { echo "❌ $CONFIG points at the SALM idx mirror" >&2; exit 1; }

umask 007  # shared project space: group qgz only (parent dirs are setgid)
mkdir -p "$INDEXES_ROOT"

as_salm=()
for cfg in "${INPUT_CFGS[@]}"; do
    out="$TMPDIR/$(basename "$cfg")"
    sed 's/type: conversation_as_duplex/type: multimodal_conversation/' "$cfg" > "$out"
    as_salm+=("$out")
done

echo "indexes_root=$INDEXES_ROOT input_cfgs=${INPUT_CFGS[*]} nemo=$NEMO_DIR@$(cut -d' ' -f2 "$NEMO_DIR/GIT_REVISION")"
python "$NEMO_DIR/scripts/dataloading/build_indexes.py" \
    --indexes-root "$INDEXES_ROOT" \
    --workers "$SLURM_CPUS_PER_TASK" \
    "${as_salm[@]}"
