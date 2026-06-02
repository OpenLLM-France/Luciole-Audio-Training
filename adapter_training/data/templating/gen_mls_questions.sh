#!/usr/bin/env bash
# Generate timestamped-transcription question/answer data from the aligned MLS
# French jsonl splits (train/dev/test, each with custom_metadata.word2time).
#
# Uses template_timestamped_transcription.py, which — unlike the SLU variant
# generator — does NOT require `answer_spans`; every record with a word2time
# table and an audio turn yields one (question, audio, answer) triple.
#
# Emits BOTH English- and French-language prompts (spoken content stays French
# either way) into per-language subfolders, so you get both datasets.
#
# Usage:
#   ./gen_mls_questions.sh [INPUT_DIR] [OUTPUT_DIR] [MIX...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"   # flat imports (from template_qa_common import ...) need cwd here

INPUT_DIR="${1:-/data-server/datasets/audio/raw/transcript/multilang/MultilingualLibriSpeech/mls_french/concatenated/aligned}"
OUTPUT_DIR="${2:-$INPUT_DIR/questions}"
shift || true; shift || true
MIX=("${@:-mixed}")   # mixed | no_json | full_json (one subfolder per preset)

echo "Input:   $INPUT_DIR"
echo "Output:  $OUTPUT_DIR/{en,fr}"
echo "Mix:     ${MIX[*]}"

for LANG in en fr; do
    echo "===== language: $LANG ====="
    python template_timestamped_transcription.py "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR/$LANG" \
        --mix "${MIX[@]}" \
        --language "$LANG" \
        --seed 0
done
