set -e

CURRENT_DIR=$(dirname "$0")
cd "$CURRENT_DIR"

for YAML in dataset_analysis/suggested_v*.yaml; do
    echo "Processing $YAML"
    python plot_composition.py dataset_analysis/metadata.csv --yaml "$YAML" --metric duration
    python plot_composition.py dataset_analysis/metadata.csv --yaml "$YAML" --metric samples
done