# Image de la démo : serveurs vLLM + client. UNE image, UN mode.
#
#   docker build --build-context nemo=/home/abert/NeMo -t salm-demo:vllm .
#
# Elle sert deux rôles à partir du même contenu :
#   - les serveurs (`python /app/serve_salm.py serve <ckpt> ...`), un par modèle ;
#   - la démo elle-même, qui n'est plus qu'un client HTTP et ne charge aucun poids.
#
# CE QU'ELLE N'A PAS, par rapport à salm-demo:8b, et pourquoi :
#
#   - nemo_automodel. Le plugin vLLM ne l'importe JAMAIS (vérifié : deux mentions dans le
#     dépôt, toutes deux en commentaire). Il construit le backbone avec le
#     NemotronHForCausalLM de vLLM et n'emprunte à NeMo que le module de perception.
#     C'est tout le chemin SALMAutomodel qui devient sans objet ici.
#
#   - la gymnastique PEFT / torchao. Elle n'existait que pour le chemin in-process, où
#     Luciole-1B passe par HF PEFT : PEFT 0.19 lève sur le torchao 0.14 de la base, et
#     torchao ne peut ni monter (il importe un torch plus récent) ni disparaître
#     (nemo_automodel l'importait). Sans nemo_automodel ni PEFT en jeu, plus de contrainte :
#     le plugin fusionne les adaptateurs LoRA lui-même, en arithmétique tensorielle pure
#     (backends.py::_merge_lora_weights).
#
#   - la recompilation de mamba-ssm pour aarch64. vLLM apporte ses propres noyaux Mamba2
#     (vllm/model_executor/layers/mamba/), il ne passe pas par le paquet mamba_ssm dont la
#     base livre un binaire x86. C'est l'hypothèse la plus fragile des trois : si le 8B
#     s'avère lent ou refuse de démarrer, c'est ici qu'il faut regarder en premier — le bloc
#     à réintroduire est dans le Dockerfile principal.
#
# CE QU'ELLE GARDE :
#   - transformers 5.6. Non négociable pour le 8B : la 4.57 de la base ne connaît pas le
#     model_type `nemotron_h` (« Transformers does not recognize this architecture »), et
#     vLLM 0.14 n'a pas de classe de config de repli pour ce type dans son _CONFIG_REGISTRY.
#     Le 1B, lui, passerait en 4.57 — mais une image par modèle n'aurait aucun sens.
#   - le fork NeMo, pour AudioPerceptionModule et le plugin lui-même.

ARG BASE_IMAGE=nvcr.io/nvidia/nemo:26.02
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libsndfile1/ffmpeg : décodage audio (librosa, soundfile), utilisé des deux côtés — la démo
# rééchantillonne avant d'envoyer, le plugin relit le signal côté serveur.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# `python -m pip`, jamais `pip` : ici pip est rattaché à l'interpréteur système et écrit dans
# un site-packages que /opt/venv/bin/python ne lit pas (build vert, effet nul).
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Le fork remplace le NeMo de la base : c'est lui qui porte le plugin vLLM
# (nemo/collections/speechlm2/vllm/) et l'entry point vllm.general_plugins.
# --no-deps : sinon pip réinstalle torch/transformers depuis PyPI par-dessus les roues NVIDIA.
RUN rm -rf /opt/NeMo
COPY --from=nemo . /opt/NeMo
#
# Un seul RUN pour l'installation du fork, ses 4 dépendances non satisfaites (bornées : un
# `>=` ouvert tirerait datasets 5.0.0 et fsspec +2 ans) et transformers 5.6 + accelerate.
# La base NeMo arrive déjà à ~120 couches et overlay2 en refuse plus de 127 — au-delà, le
# build passe mais tout `docker run` échoue sur "max depth exceeded".
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/NeMo && \
    python -m pip install --no-cache-dir --no-deps \
        "lhotse>=1.33.0,<1.34" \
        "datasets>=3.2.0,<4" \
        "fsspec>=2024.12.0,<2025" \
        "aistore" && \
    python -m pip install --no-cache-dir "transformers==5.6.0" "accelerate==1.14.0" && \
    python -c "import transformers; from transformers.models.nemotron_h import configuration_nemotron_h; print('transformers', transformers.__version__, '+ nemotron_h OK')"

# Point de montage du checkpoint. En mode serveur c'est /app/model (un checkpoint par
# conteneur) ; la démo, elle, ne monte rien puisqu'elle ne charge pas de poids.
ENV MODEL_PATH=/app/model
RUN mkdir -p /app/model

WORKDIR /app
COPY . /app

# 9009 = démo, 9010+ = serveurs vLLM (un par modèle), 3478/udp = STUN,
# 50000-50100/udp = plage média aiortc.
EXPOSE 9009/tcp
EXPOSE 9010-9019/tcp
EXPOSE 3478/udp
EXPOSE 50000-50100/udp

CMD ["python", "app.py"]
