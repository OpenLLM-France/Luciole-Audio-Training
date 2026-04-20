#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_docker.sh — Launch the WebRTC SALM app container
#
# Usage:
#   ./run_docker.sh [MODEL_DIR] [HOST_PORT] [IMAGE_TAG]
#
#   MODEL_DIR  : Absolute path to your local SALM model directory
#                (default: /home/hnaouara/salm_models/linagora_canary_luciole-1B-SFT-1.1_s082829)
#   HOST_PORT  : Host port to expose the web UI on  (default: 9009)
#   IMAGE_TAG  : Docker image tag to run            (default: salm-webrtc-app:latest)
#
# Examples:
#   ./run_docker.sh
#   ./run_docker.sh /data/models/MyModel 8080
#   ./run_docker.sh /data/models/MyModel 8080 salm-webrtc-app:v2
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Load .env file safely if it exists
if [ -f .env ]; then
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Skip comments and empty lines
        if [[ ! "$key" =~ ^# ]] && [ -n "$key" ]; then
            # Strip potential surrounding quotes
            value="${value%\"}"
            value="${value#\"}"
            export "$key"="$value"
        fi
    done < .env
fi

# Use arguments if provided, otherwise fallback to .env vars or defaults
MODEL_DIR="${1:-${MODEL_PATH:-/home/hnaouara/salm_models/linagora_canary_luciole-1B-SFT-1.1_s082829}}"
HOST_PORT="${2:-${PORT:-9009}}"
IMAGE_TAG="${3:-salm-webrtc-app:v2}"
CONTAINER_NAME="salm-webrtc-app"

# Check if model directory exists and has files. If not, download the model!
if [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A "${MODEL_DIR}")" ]; then
    echo "Model directory '${MODEL_DIR}' is missing or empty."
    if [ -n "${HF_TOKEN:-}" ] && [ -n "${BASE_MODEL:-}" ]; then
        echo "Downloading base model '${BASE_MODEL}' from Hugging Face to '${MODEL_DIR}'..."
        mkdir -p "${MODEL_DIR}"
        # Use a temporary python container with huggingface-cli to download the model to the host folder
        docker run --rm -it \
            -v "${MODEL_DIR}:/model_data" \
            python:3.9-slim bash -c "pip install -q huggingface_hub && huggingface-cli login --token ${HF_TOKEN} && huggingface-cli download ${BASE_MODEL} --local-dir /model_data"
        echo "✅ Model downloaded successfully."
    else
        echo "❌ Error: HF_TOKEN and BASE_MODEL must be defined in .env to auto-download the model."
        exit 1
    fi
fi

# Stop & remove any existing container with the same name
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container '${CONTAINER_NAME}'..."
    docker rm -f "${CONTAINER_NAME}"
fi

echo "Starting container '${CONTAINER_NAME}' on port ${HOST_PORT}..."
echo "  Model : ${MODEL_DIR}"
echo "  Image : ${IMAGE_TAG}"

docker run --init -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -p "${HOST_PORT}":9009 \
    -p 3478:3478/udp \
    -p 50000-50100:50000-50100/udp \
    -v "${MODEL_DIR}":/app/model:ro \
    -v "$(pwd)":/app \
    -e MODEL_PATH=/app/model \
    "${IMAGE_TAG}"

echo ""
echo "✅ Container started. Open http://localhost:${HOST_PORT} in your browser."
echo "   To follow logs : docker logs -f ${CONTAINER_NAME}"
echo "   To stop        : docker stop ${CONTAINER_NAME}"
