#!/bin/bash
#SBATCH --job-name=build_indexes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --hint=nomultithread
#SBATCH --account=qgz@cpu
#SBATCH --partition=cpu_p1
#
# ============================================================
# Build .idx sidecars for the v4 curriculum blends (CPU-only, no GPU
# needed). Wraps build-indexes-cmd.sh with a proper SLURM/module/conda
# context so it can be sbatch'd directly instead of run by hand.
#
# PREREQUISITE: the *_sharded.yaml blends this indexes must already exist --
# shard them first (see the PREREQUISITE note in
# training/conf/run/xp/v4/launch_curriculum_v4_8b.sh):
#   python $HOME/ssak/tools/nemo/shard_manifest.py <blend>_weighted_randomorder.yaml
# also run as its own SLURM CPU job (same account/partition as this one).
#
# Submit (from this directory, needs a logs/ subfolder next to it):
#   mkdir -p logs && sbatch build-indexes.slurm
#
# Re-run whenever a blend's source manifests change, or a new blend/
# data-version is added to build-indexes-cmd.sh's BLENDS array.
# ============================================================

set -euo pipefail

module purge
module load arch/h100 nemo/2.7.3
# CPU-only job: no arch/h100 needed. Match the conda env actually used for
# v4/8B training (the one built against the indexed-data-capable NeMo3 fork)
# so `import nemo` resolves the right package version -- see the NEMO_FORK
# note in build-indexes-cmd.sh for why $HOME/NeMo3 (not $HOME/NeMo) matters.
CONDA_ENV="${CONDA_ENV:-$SCRATCH/speechlm/envs/nemo27-salm_automodel_3}"
echo "Conda env: $CONDA_ENV"
conda activate "$CONDA_ENV"

python - <<EOF
import nemo
print(f"NeMo version: {nemo.__version__} (from {nemo.__file__})")
EOF

export TMPDIR="${JOBSCRATCH:-/tmp}"

bash "$(dirname "${BASH_SOURCE[0]}")/build-indexes-cmd.sh"
