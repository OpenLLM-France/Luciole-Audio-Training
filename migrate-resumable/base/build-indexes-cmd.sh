#!/bin/bash
# Build .idx sidecars for the JZ speechlm2 training blends, mirrored under
# $ALL_CCFRSCRATCH/audio/idx/base (matches data.train_ds.indexes_root /
# data.validation_ds.indexes_root in training/conf/base.yaml).
#
# Run on a JZ partition with network+CPU (prepost/compil), not the login node:
# this walks every manifest referenced by the blends below and can take a
# while for the full v4 mixture (dozens of languages/datasets, many sharded
# via __OP_..._CL_ patterns).
#
# Re-run whenever a blend's source manifests change (new/edited jsonl) or a
# blend/data-version not listed here is used (--data-version on
# slurm_launcher.py / DATA_VERSION on run_train.slurm).
#
# For an sbatch-ready wrapper (SLURM headers, module/conda setup, all v4
# curriculum blends), see build-indexes.slurm next to this file.
set -euo pipefail

# NEMO_FORK must be the indexed-data-capable fork: $HOME/NeMo (the branch
# training normally runs against, luciole_speech.2.8.0-rc0) does NOT carry
# scripts/dataloading/ or the indexed/indexes_root plumbing at all -- that
# only exists on $HOME/NeMo3 (luciole_speech.3.1.0-rc0), confirmed present on
# JZ. Do not default this to $HOME/NeMo, it will fail with "No such file".
NEMO_FORK="${NEMO_FORK:-$HOME/NeMo}"
CONF="$(cd "$HOME/speechlm/Luciole-Audio-Training/training/conf" && pwd)"
INDEXES_ROOT="${ALL_CCFRSCRATCH}/audio/idx"

# Required: the input_cfg YAMLs interpolate ${oc.env:DATA_FOLDER} for every
# manifest_filepath. OmegaConf resolves that lazily as build_indexes.py reads
# each entry, so this must be exported before it runs (mirrors DATA_FOLDER in
# training/run_train.slurm) -- otherwise this dies on the first resolve.
export DATA_FOLDER="${DATA_FOLDER:-${ALL_CCFRSCRATCH}/audio}"

# $ALL_CCFRSCRATCH is shared group (qgz) space: keep new dirs/files
# group-writable (drwxrws--- / rw-rw---- -- audio/ is already setgid, so
# umask 007 is the only thing needed for subdirs to inherit it correctly).
umask 007

# One blend per curriculum stage that actually gets launched (see
# training/conf/run/xp/v4/launch_curriculum_v4_8b.sh) plus validation. Each
# is an independent blend file, not derived from the others, so all need
# their own indexes.
BLENDS=(
    "${CONF}/data/v4/input_cfg_train_weighted_randomorder_sharded.yaml"          # single-phase / stage2_from_lora-style full mix
    # "${CONF}/data/v4/input_cfg_train_stage1_weighted_randomorder_sharded.yaml"   # curriculum stage 1
    # "${CONF}/data/v4/input_cfg_train_stage2_v2_weighted_randomorder_sharded.yaml" # curriculum stage 2 (v2 rebalance)
    # "${CONF}/data/v4/input_cfg_train_stage3_weighted_randomorder_sharded.yaml"   # curriculum stage 3
    "${CONF}/data/val/v3_2.yaml"
)

python "${NEMO_FORK}/scripts/dataloading/build_indexes.py" \
    --indexes-root "${INDEXES_ROOT}" \
    --workers "${SLURM_CPUS_PER_TASK:-4}" \
    "${BLENDS[@]}"

# If yet another data-version / blend file is used later, build indexes for
# it too the same way -- e.g.:
#
#   python "${NEMO_FORK}/scripts/dataloading/build_indexes.py" \
#       --indexes-root "${INDEXES_ROOT}" --workers "${SLURM_CPUS_PER_TASK:-8}" \
#       "${CONF}/data/v4/input_cfg_train.yaml"

echo "Indexes built under: ${INDEXES_ROOT}"
echo "NOTE: no dataset-level .idxpack step here -- the v4 blend nests many"
echo "small per-language/per-source manifests rather than one large sharded"
echo "outer dataset, so loose sidecars are the right granularity. Revisit"
echo "with convert_indexes_to_idxpack.py only if startup ends up opening an"
echo "impractical number of loose .idx files."
