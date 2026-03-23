# Step 1 — parse YAML(s) → CSV  (slow, reads all JSONL files)
```bash
python generate_csv_metadata.py \
    ../adapter_training/speechlm2/conf/input_cfg_train.yaml \
    --data_root /data-server/datasets/audio --workers 8 \
    --output_dir dataset_analysis 
```

# Step 2 — CSV + optional original YAML → suggested YAML  (instant)
```bash
python suggest_yaml_cfg.py dataset_analysis/metadata.csv --metric duration --temperature 2 \
    --input_weights ../adapter_training/speechlm2/conf/input_cfg_train.yaml \
    --output dataset_analysis/suggested.yaml | tee dataset_analysis/suggested_notes.txt
```
Note: the analysis can be re-run with:
```bash
python analyze_yaml_cfg.py dataset_analysis/metadata.csv dataset_analysis/suggested.yaml
```

# Step 3 — CSV + any YAML → plots  (instant, iterate freely)
```bash
python plot_composition.py dataset_analysis/metadata.csv --yaml dataset_analysis/suggested.yaml --output ./dataset_analysis
```
