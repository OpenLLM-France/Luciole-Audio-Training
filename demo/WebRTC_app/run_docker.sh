#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_docker.sh — Lance la démo à la main : un serveur vLLM par modèle, puis l'app.
#
# Le chemin normal est le DAG Airflow `demo_on_dgx`, qui fait la même chose en
# résolvant le checkpoint, en rsyncant depuis le data-server et en posant un chien
# de garde. Ce script est là pour un poste sans Airflow, ou pour déboguer.
#
# Usage:
#   ./run_docker.sh MODEL_DIR[:NOM] [MODEL_DIR[:NOM] ...]
#
#   Un argument par modèle. Le nom (après ':') est celui affiché dans le sélecteur
#   du chat ; sans lui, on prend le nom du dossier. Deux modèles ou plus font
#   apparaître le sélecteur et la case « comparer ».
#
#   IMAGE      : tag de l'image                     (défaut: salm-demo:vllm)
#   HOST_PORT  : port de l'app                      (défaut: 9009 ; serveurs sur 9010, 9011…)
#   KV_GIB     : cache KV par serveur, en Gio       (défaut: 6)
#   GPU_FRAC   : gpu-memory-utilization par serveur (défaut: 0.35)
#
# Exemples:
#   ./run_docker.sh /home/abert/models/salm/Luciole-1B/xp/hf_checkpoints/step_100000
#   ./run_docker.sh /path/ckpt1:Luciole-1B /path/ckpt2:Luciole-8B
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ "$#" -lt 1 ]; then
    sed -n '3,23p' "$0"
    exit 1
fi

IMAGE="${IMAGE:-salm-demo:vllm}"
HOST_PORT="${HOST_PORT:-9009}"
KV_GIB="${KV_GIB:-6}"
GPU_FRAC="${GPU_FRAC:-0.35}"
APP_NAME="salm-webrtc-app"
SRV_PREFIX="salm-webrtc-vllm"

# Plafond mémoire à 90 % de la RAM. Sur GB10 la mémoire GPU EST la RAM système : un
# conteneur qui déborde emmène la machine entière, pas seulement lui-même.
MEM=$(awk '/MemTotal/{printf "%d", $2*0.9/1024}' /proc/meminfo)

# Ménage : sans ça les anciens conteneurs tiennent le GPU et les ports.
docker rm -f "${APP_NAME}" >/dev/null 2>&1 || true
OLD=$(docker ps -aq --filter "name=^${SRV_PREFIX}" 2>/dev/null || true)
if [ -n "${OLD}" ]; then docker rm -f ${OLD} >/dev/null 2>&1 || true; fi

ENDPOINTS=""
INDEX=0
for SPEC in "$@"; do
    INDEX=$((INDEX + 1))
    PORT=$((HOST_PORT + INDEX))
    DIR="${SPEC%%:*}"
    NAME="${SPEC#*:}"
    if [ "${NAME}" = "${SPEC}" ]; then NAME="$(basename "${DIR}")"; fi

    if [ ! -d "${DIR}" ] || [ -z "$(ls -A "${DIR}")" ]; then
        echo "❌ Dossier de modèle vide ou absent : ${DIR}" >&2
        exit 1
    fi

    echo "Serveur vLLM '${NAME}' sur le port ${PORT}…"
    # serve_salm.py, pas `vllm serve` : le plugin s'enregistre par l'entry point
    # vllm.general_plugins, que vLLM ne charge que dans le processus moteur, alors que
    # le front-end de l'API valide la config AVANT, dans le processus principal.
    # --kv-cache-memory-bytes : sur GB10 le profilage mémoire de vLLM échoue dès qu'un
    # autre conteneur bouge (le « libre » remonte pendant la mesure). Le fixer court-
    # circuite tout le profilage.
    docker run --init -d --name "${SRV_PREFIX}${INDEX}" \
        --network host --gpus all --shm-size=8g \
        --memory="${MEM}m" --memory-swap="${MEM}m" \
        -v "${DIR}":/app/model:ro \
        -v "$(pwd)":/app \
        --entrypoint python "${IMAGE}" /app/serve_salm.py serve /app/model \
        --served-model-name "${NAME}" --port "${PORT}" \
        --enforce-eager --dtype bfloat16 --max-model-len 16384 \
        --limit-mm-per-prompt '{"audio":1}' \
        --gpu-memory-utilization "${GPU_FRAC}" \
        --kv-cache-memory-bytes $((KV_GIB * 1073741824)) >/dev/null

    if [ -n "${ENDPOINTS}" ]; then ENDPOINTS="${ENDPOINTS},"; fi
    ENDPOINTS="${ENDPOINTS}${NAME}=http://localhost:${PORT}"
done

# Un 8B met ~2 min à charger, et l'app refuse un serveur qui ne répond pas encore.
INDEX=0
for SPEC in "$@"; do
    INDEX=$((INDEX + 1))
    PORT=$((HOST_PORT + INDEX))
    echo -n "Attente du serveur sur ${PORT} "
    for _ in $(seq 1 90); do
        if curl -sf -m 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then break; fi
        echo -n "."; sleep 5
    done
    if curl -sf -m 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "→ prêt"
    else
        echo ""
        echo "❌ Le serveur sur ${PORT} n'a pas répondu. Logs :" >&2
        docker logs --tail 40 "${SRV_PREFIX}${INDEX}" >&2
        exit 1
    fi
done

# --network host : indispensable pour le chat TEXTE. Sans permission micro, le navigateur
# n'annonce que des candidats ICE mDNS « .local », qu'aiortc résout par une requête
# multicast — laquelle ne sort pas d'un conteneur en bridge.
echo "Démarrage de l'app sur le port ${HOST_PORT}…"
docker run --init -d --name "${APP_NAME}" \
    --network host \
    -v "$(pwd)":/app \
    -e MODEL_ENDPOINTS="${ENDPOINTS}" \
    -e PORT="${HOST_PORT}" \
    "${IMAGE}" >/dev/null

echo ""
echo "✅ Démo lancée. Ouvre http://localhost:${HOST_PORT}"
echo "   Modèles : ${ENDPOINTS}"
echo "   Logs    : docker logs -f ${APP_NAME}"
echo "   Arrêt   : docker rm -f ${APP_NAME} \$(docker ps -aq --filter name=^${SRV_PREFIX})"
