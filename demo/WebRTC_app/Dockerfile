# Démo WebRTC SALM, pour les DGX Spark (arm64).
#
#   Luciole-1B (backend SALM, transformers 4.x de la base) :
#   docker build --build-context nemo=/home/abert/NeMo -t salm-demo:latest .
#
#   Luciole-8B (backend SALMAutomodel, Nemotron-H hybride Mamba2/attention) :
#   docker build --build-context nemo=/home/abert/NeMo --build-arg AUTOMODEL=1 \
#                -t salm-demo:8b .
#
# DEUX images, pas une : le 8B impose transformers 5.x et le 1B tourne sur les 4.57 de
# la base. Recette identique par ailleurs, d'où le build-arg.
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

# AUTOMODEL=1 : couche pour les checkpoints SALMAutomodel (backbones Nemotron-H,
# Luciole-8B/23B). Ce que la base ne fournit pas :
#   - nemo_automodel : construit le LLM (pretrained.py::load_pretrained_automodel_llm).
#     --no-deps, sinon pip réinstalle torch/transformers par-dessus les roues NVIDIA.
#   - transformers 5.6 : le nemotron_h built-in ne parse le hybrid_override_pattern dense
#     ('-' = MLP) que depuis la 5.6 (KeyError: '-' sur 5.3-5.5). Le code remote n'est pas
#     une échappatoire : son prepare_inputs_for_generation découpe input_ids, None sur le
#     chemin inputs_embeds-only (audio) de SALM — d'où trust_remote_code=false à l'export.
#   - peft/accelerate : le config du checkpoint porte un bloc lora, donc
#     maybe_install_lora() s'exécute au chargement.
# Versions relevées sur l'env JZ de référence (nemo27-salm_automodel), pas devinées.
ARG AUTOMODEL=0
RUN if [ "$AUTOMODEL" = "1" ]; then \
        python -m pip install --no-cache-dir \
            "transformers==5.6.0" \
            "peft==0.19.1" \
            "accelerate==1.14.0" && \
        python -m pip install --no-cache-dir --no-deps "nemo-automodel==0.4.0" ; \
    fi

# mamba-ssm recompilé POUR L'ARCHI DE LA MACHINE. La base livre un `selective_scan_cuda`
# x86 : dans un conteneur arm64 `import mamba_ssm` échoue, transformers met alors ses cinq
# noyaux à None et exécute le torch_forward NAÏF de Mamba2, qui matérialise les états SSM.
# Sur Nemotron-H (~50 couches Mamba2) c'est lent ET gourmand — un audio long a rempli les
# 119 Go de spark-906c et fait tomber la machine.
#
# Les trois symboles réclamés par nemotron_h sont pourtant des noyaux TRITON, sans
# extension compilée : c'est le `__init__.py` du paquet qui importe `selective_scan_cuda`
# et fait tout échouer. On recompile donc uniquement pour rendre le paquet importable.
#
# MAMBA_FORCE_BUILD : sans lui pip récupère la roue précompilée (et le problème revient).
# TORCH_CUDA_ARCH_LIST=12.1 : GB10 (Blackwell) seule, pas les dix archis.
# MAX_JOBS bridé : nvcc sature la RAM sinon.
# --no-deps : mamba-ssm réclame `triton`, absent de PyPI en aarch64 ; la base le fournit
# sous le nom `pytorch-triton`, que pip ne sait pas rapprocher. Toutes les dépendances
# réelles sont déjà dans l'image.
RUN if [ "$AUTOMODEL" = "1" ]; then \
        python -m pip uninstall -y mamba-ssm >/dev/null 2>&1 || true; \
        MAMBA_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="12.1" MAX_JOBS=8 \
        python -m pip install --no-cache-dir --no-build-isolation --no-deps \
            "mamba-ssm==2.3.1" && \
        python -c "import mamba_ssm; from mamba_ssm.ops.triton.selective_state_update import selective_state_update; print('mamba_ssm OK', mamba_ssm.__version__)" ; \
    fi

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
