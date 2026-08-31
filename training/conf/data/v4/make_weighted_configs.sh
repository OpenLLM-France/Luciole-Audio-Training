set -e

# Run from the folder holding this script, whatever the caller's cwd
cd "$(dirname "$0")"

SRC_DIR="../../../../assets/"

# Step 1 — parse YAML → CSV  (slow, reads all JSONL files)
# metadata.csv is seeded from v3; only the six manifests new in v4 are parsed
# (existing rows are kept, and task_type/sub_task are refreshed from the YAML).
# Delete metadata.csv (or pass --force_overwrite) to recompute everything.
python $SRC_DIR/generate_csv_metadata.py  input_cfg_train.yaml --data_root /data-server/datasets/audio --output_dir . --workers 24

# Step 2 — CSV + original YAML → suggested YAML  (instant)
# Step 3 — same weights, pointing at the shuffled manifests
for NAME in input_cfg_train input_cfg_train_stage1 input_cfg_train_stage2 input_cfg_train_stage3; do

    python $SRC_DIR/suggest_yaml_cfg.py metadata.csv --metric samples --temperature 2 \
        --input_weights $NAME.yaml --max_passes 20 --output ${NAME}_weighted.yaml | tee ${NAME}_weighted_notes.txt

    sed 's/.jsonl/_randomorder.jsonl/g' ${NAME}_weighted.yaml > ${NAME}_weighted_randomorder.yaml

done

# Lastly — CSV + any YAML → plots  (a bit long)
python $SRC_DIR/plot_composition.py metadata.csv --yaml input_cfg_train_weighted.yaml --metric samples --output ./
python $SRC_DIR/plot_composition.py metadata.csv --yaml input_cfg_train_weighted.yaml --metric duration --output ./
