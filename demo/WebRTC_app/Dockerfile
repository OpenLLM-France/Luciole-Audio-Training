# Démo WebRTC SALM, pour les DGX Spark (arm64).
#
#   docker build --build-context nemo=/home/abert/NeMo -t salm-demo:latest .
#
# --build-context REQUIS : le NeMo de la base (2.7.1) ignore llm_backbone/ et aucun
# checkpoint récent ne charge. Le fork apporte la redirection + 18 commits d'inférence.
# Ni secret ni modèle dans l'image : le checkpoint est monté au lancement.

ARG BASE_IMAGE=nvcr.io/nvidia/nemo:26.02
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libsndfile1/ffmpeg : décodage audio (librosa, soundfile).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# `python -m pip`, jamais `pip` : ici pip est rattaché à l'interpréteur système et écrit
# dans un site-packages que /opt/venv/bin/python ne lit pas (build vert, effet nul).
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Le fork remplace le NeMo de la base. Le pip install ne pose pas les fichiers (le finder
# editable mappe déjà `nemo` -> /opt/NeMo/nemo) mais corrige les métadonnées de version.
# --no-deps : sinon pip réinstalle torch/transformers depuis PyPI par-dessus les roues NVIDIA.
RUN rm -rf /opt/NeMo
COPY --from=nemo . /opt/NeMo
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/NeMo

# Les 4 dépendances que le fork déclare et que la base ne satisfait pas. Bornées : un `>=`
# ouvert tire la dernière version publiée (datasets 5.0.0, fsspec +2 ans).
RUN python -m pip install --no-cache-dir --no-deps \
        "lhotse>=1.33.0,<1.34" \
        "datasets>=3.2.0,<4" \
        "fsspec>=2024.12.0,<2025" \
        "aistore"

# Point de montage du checkpoint (-v <step_XXXXXX>:/app/model:ro), jamais copié.
ENV MODEL_PATH=/app/model
RUN mkdir -p /app/model

WORKDIR /app
COPY . /app

# 9009 = app, 3478/udp = STUN, 50000-50100/udp = plage média aiortc.
EXPOSE 9009/tcp
EXPOSE 3478/udp
EXPOSE 50000-50100/udp

CMD ["python", "app.py"]
