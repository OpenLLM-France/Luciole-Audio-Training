#!/bin/bash
# ============================================================================
# Construit de bout en bout l'env conda des runs SALMAutomodel (Luciole 8B/23B).
# Fusionne l'ancien speechLM/tools/install.slurm (base) et install_automodel.sh
# (paquets + patches), qui vivaient dans deux depots differents.
#
#   sbatch training/xp_commands/install_env.slurm
#
# Phases, desactivables individuellement :
#   SKIP_BASE=1     ne pas recopier l'env de base (reprise sur un DST existant)
#   SKIP_PIP=1      ne pas (re)installer les paquets
#   SKIP_PATCHES=1  ne pas reappliquer les patches (ils sont idempotents)
#
# La phase pip a besoin du proxy IDRIS ; il vient du profil de login et Slurm
# le propage (--export=ALL par defaut). Repris en dur plus bas au cas ou.
# ============================================================================

#SBATCH --job-name=install_salm_automodel
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --partition=compil_h100
#SBATCH --hint=nomultithread
#SBATCH --account=qgz@cpu

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NEMO_MODULE=${NEMO_MODULE:-nemo/2.4.0}
DST=${DST:-$SCRATCH/speechlm/envs/nemo24-salm_automodel}
# SRC (env de base) n'est pas ecrit en dur : il est deduit du module, apres son
# chargement. Surchargeable si besoin.

# Fork NeMo : epingle au commit, contrairement a l'ancien `pip install -e $HOME/NeMo`
# qui installait un dossier non versionne au contenu indetermine.
NEMO_FORK=${NEMO_FORK:-$HOME/NeMo}
NEMO_FORK_REPO=${NEMO_FORK_REPO:-https://github.com/linagora-labs/NeMo.git}
NEMO_FORK_BRANCH=${NEMO_FORK_BRANCH:-luciole_speech.2.8.0-rc0}
NEMO_FORK_COMMIT=${NEMO_FORK_COMMIT:-e7e44dc92}

AUTOMODEL_COMMIT=9eccbb6102a260efd7cbdffa890fc57b94f94528   # nemo_automodel 0.4.0

# 5.5.0 = la version reellement presente dans l'env qui a produit les checkpoints
# du 8B. L'ancien install_automodel.sh epinglait 5.6.0, jamais teste sur ces runs.
# A 5.5.0 le patch (2) plus bas est NECESSAIRE ; a partir de 5.6.0 il se saute seul.
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-5.5.0}

# Figes sur les versions observees dans l'env de production (avant : non epinglees,
# donc un reinstall les faisait deriver).
PEFT_VERSION=0.19.1
MLFLOW_VERSION=3.14.0
LATEX2SYMPY2_VERSION=1.9.1
LHOTSE_VERSION=1.33.0

# Le proxy vient du profil de login et Slurm le propage. On ne le code pas en dur :
# sans lui les pip echoueraient de toute facon, autant le dire tout de suite.
if [ "${SKIP_PIP:-0}" != "1" ] && [ -z "${https_proxy:-}" ]; then
    echo "https_proxy absent : les pip n'auront pas de reseau. Soumettre depuis un" >&2
    echo "shell de login (le job herite de son environnement) ou l'exporter." >&2
    exit 1
fi

module purge
module load arch/h100 "$NEMO_MODULE"

# Env de base = celui que le module vient de selectionner. conda le designe d'un '*'
# dans `conda env list` ; a defaut on le retrouve sous la racine conda a partir du
# nom du module (nemo/2.4.0 -> envs/nemo-2.4.0+py*). Aucun chemin absolu ecrit ici.
if [ -z "${SRC:-}" ]; then
    SRC=$(conda env list 2>/dev/null | awk '$2 == "*" {print $3}')
    [ -n "$SRC" ] || SRC=$(ls -d "$(conda info --base)/envs/nemo-${NEMO_MODULE#nemo/}"+py* 2>/dev/null | head -1)
fi
[ -n "${SRC:-}" ] && [ -x "$SRC/bin/python" ] || {
    echo "env de base introuvable pour $NEMO_MODULE (SRC='${SRC:-}')" >&2; exit 1; }

echo "=== module   : $NEMO_MODULE"
echo "=== base     : $SRC"
echo "=== cible    : $DST"
echo "=== fork NeMo: $NEMO_FORK @ $NEMO_FORK_COMMIT"

# ---------------------------------------------------------------------------
# 1. Env de base : copie du module partage
# ---------------------------------------------------------------------------
# `conda create --clone` echoue sur cet env : tout y est en pypi_0 (le clone ne
# sait pas les reproduire) et il tente d'ecrire dans la source en lecture seule.
# La copie directe le reproduit exactement.
if [ "${SKIP_BASE:-0}" = "1" ]; then
    echo "--- [1/4] base : saute (SKIP_BASE=1)"
else
    echo "--- [1/4] base : copie de 12 Go, compter ~15 min sur Lustre"
    # Si DST existe, cp -a copierait SRC *dedans*. Echouer plutot qu'abimer l'env vivant.
    [ -e "$DST" ] && { echo "DST existe deja, le supprimer ou passer SKIP_BASE=1 : $DST" >&2; exit 1; }
    mkdir -p "$(dirname "$DST")"
    cp -a "$SRC" "$DST"

    # Reecrire le prefixe source code en dur dans les console scripts.
    grep -rIl "$SRC" "$DST/bin" 2>/dev/null | xargs -r sed -i "s|$SRC|$DST|g"

    # Idem dans les .pth. Ils contiennent des chemins auto-referents (setuptools.pth)
    # qui, non reecrits, reinjectent le site-packages de l'env partage en lecture seule
    # dans sys.path et masquent celui-ci — notamment le fork NeMo installe en editable.
    grep -rIl "$SRC" "$DST"/lib/python*/site-packages/*.pth 2>/dev/null \
        | xargs -r sed -i "s|$SRC|$DST|g"
fi

conda activate "$DST"
echo "--- python   : $(python -V 2>&1) ($(which python))"

# ---------------------------------------------------------------------------
# 2. Paquets
# ---------------------------------------------------------------------------
if [ "${SKIP_PIP:-0}" = "1" ]; then
    echo "--- [2/4] paquets : saute (SKIP_PIP=1)"
else
    echo "--- [2/4] paquets"
    pip install --no-deps \
        "nemo_automodel @ git+https://github.com/NVIDIA-NeMo/Automodel.git@${AUTOMODEL_COMMIT}"
    pip install "transformers==${TRANSFORMERS_VERSION}"
    pip install "peft==${PEFT_VERSION}" \
                "mlflow==${MLFLOW_VERSION}" \
                "latex2sympy2==${LATEX2SYMPY2_VERSION}" \
                "lhotse==${LHOTSE_VERSION}" \
                "kaldialign==0.9.1"
    pip install --no-deps "torchcodec==0.7.0"
    pip install "nv-one-logger-core==2.3.1" \
                "nv-one-logger-pytorch-lightning-integration==2.3.1" \
                "nv-one-logger-training-telemetry==2.3.1"

    # Fork NeMo au commit epingle.
    if [ ! -d "$NEMO_FORK/.git" ]; then
        if [ -e "$NEMO_FORK" ]; then
            echo "$NEMO_FORK existe mais n'est pas un depot git — le deplacer ou fixer NEMO_FORK" >&2
            exit 1
        fi
        git clone --branch "$NEMO_FORK_BRANCH" "$NEMO_FORK_REPO" "$NEMO_FORK"
    fi
    git -C "$NEMO_FORK" fetch --quiet origin "$NEMO_FORK_BRANCH"
    git -C "$NEMO_FORK" checkout --quiet "$NEMO_FORK_COMMIT"
    echo "--- fork NeMo: $(git -C "$NEMO_FORK" log --oneline -1)"
    pip install --no-deps -e "$NEMO_FORK"
fi

# ===========================================================================
# 3. Patches post-install — necessaires pour charger le Nemotron-H dense
#    (Luciole-8B) sous transformers 5.x avec le modeling INTEGRE
#    (trust_remote_code off ; cf. conf/base_automodel.yaml). Tous idempotents.
#    (3) et (4) sont ce qui rend le modeling integre utilisable en multi-GPU.
# ===========================================================================
if [ "${SKIP_PATCHES:-0}" = "1" ]; then
    echo "--- [3/4] patches : saute (SKIP_PATCHES=1)"
else
echo "--- [3/4] patches"

# (1) ESSENTIEL. mamba-ssm 2.2.5 importe en dur GreedySearchDecoderOnlyOutput /
#     SampleDecoderOnlyOutput depuis transformers.generation, supprimes en 5.x.
#     Le modeling nemotron_h INTEGRE charge mamba_ssm (lazy_load_kernel ->
#     ops.triton.*, ce qui execute mamba_ssm/__init__.py -> mixer_seq_simple ->
#     utils.generation) et meurt sur l'ImportError. On alias les classes
#     supprimees vers GenerateDecoderOnlyOutput ; elles ne servent qu'au decode()
#     de mamba_ssm, que l'entrainement Nemotron-H n'appelle jamais.
python - <<'PY'
import io, importlib.util
path = importlib.util.find_spec("mamba_ssm.utils.generation").origin
src = io.open(path, encoding="utf-8").read()
old = "from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer"
new = (
    "from transformers.generation import TextStreamer\n"
    "try:  # transformers<5\n"
    "    from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput\n"
    "except ImportError:  # transformers>=5 removed these legacy output classes\n"
    "    try:\n"
    "        from transformers.generation import GenerateDecoderOnlyOutput as _GenOut\n"
    "    except Exception:\n"
    "        _GenOut = dict\n"
    "    GreedySearchDecoderOnlyOutput = SampleDecoderOnlyOutput = _GenOut"
)
if old in src:
    io.open(path, "w", encoding="utf-8").write(src.replace(old, new))
    print("patched mamba_ssm:", path)
else:
    print("mamba_ssm: nothing to patch (already patched or layout changed):", path)
PY

# (2) Auto-detectant. transformers 5.3-5.5 a perdu le token dense '-'=MLP dans le
#     parseur de pattern du nemotron_h integre (KeyError: '-'). Comme
#     trust_remote_code est OFF, c'est l'integre qui est charge : a 5.5.0 ce patch
#     est REQUIS. A partir de 5.6.0 le test ci-dessous echoue et le patch se saute.
if ! python -c "import io,importlib.util; p=importlib.util.find_spec('transformers.models.nemotron_h.configuration_nemotron_h').origin; import sys; sys.exit(0 if '\"-\": \"mlp\"' in io.open(p).read() else 3)"; then
  echo "Overlay du nemotron_h de transformers 5.6.0 (l'integre n'a pas '-'=mlp)…"
  TMP=$(mktemp -d)
  pip download --no-deps -q "transformers==5.6.0" -d "$TMP"
  ( cd "$TMP" && unzip -oq transformers-*.whl 'transformers/models/nemotron_h/*' )
  DEST=$(python -c "import os,transformers.models.nemotron_h as m; print(os.path.dirname(m.__file__))")
  cp -f "$TMP"/transformers/models/nemotron_h/*.py "$DEST/"
  rm -rf "$TMP"
  echo "overlaid nemotron_h into $DEST"
fi

# (3) ESSENTIEL en multi-GPU. Le parallelizer de nemo_automodel code en dur le nom
#     de tronc BUNDLED `backbone` a deux endroits, alors que le NemotronHForCausalLM
#     integre nomme le sien `model` (modeling_nemotron_h.py: self.model = NemotronHModel).
#     Les deux dispatchent sur le nom de classe, commun aux deux implementations :
#     avec trust_remote_code off, l'integre atteint ce code et casse sur le
#     `.backbone` manquant. On fait accepter les DEUX noms (no-op sur le bundled) :
#       - LLM_MODEL_CLS_TO_LAYERS (wrapping FSDP) : ajouter "model.layers" a cote de
#         "backbone.layers" ; _reduce_attrs ignore silencieusement le chemin absent.
#       - NemotronHParallelizationStrategy (chemin TP, seulement si tp_size>1) :
#         resoudre le tronc en backbone-ou-model avant de lire `.layers`.
python - <<'PY'
import io, os, sysconfig
# Localiser le fichier SANS importer nemo_automodel : son __init__ distribue importe
# torch de facon eager, ce qui exige des MKL/module-loads qu'un shell nu peut ne pas
# avoir. Un patch purement textuel ne doit pas en dependre.
path = os.path.join(sysconfig.get_paths()["purelib"],
                    "nemo_automodel", "components", "distributed", "parallelizer.py")
if not os.path.isfile(path):
    print("nemo_automodel parallelizer not found (layout changed?):", path)
else:
    src = io.open(path, encoding="utf-8").read()
    orig = src
    src = src.replace(
        '"NemotronHForCausalLM": ["backbone.layers"],',
        '"NemotronHForCausalLM": ["backbone.layers", "model.layers"],',
    )
    src = src.replace(
        "model.backbone.layers",
        '(model.backbone if hasattr(model, "backbone") else model.model).layers',
    )
    if src != orig:
        io.open(path, "w", encoding="utf-8").write(src)
        print("patched nemo_automodel parallelizer (backbone/model trunk):", path)
    else:
        print("nemo_automodel parallelizer: nothing to patch (already patched or layout changed):", path)
PY

# (4) ESSENTIEL en multi-GPU. Le checkpointer de nemo_automodel saute le
#     initialize_weights() de transformers pour Nemotron-H HF, parce que
#     HF _init_weights reinitialise les A_log/dt_bias Mamba via un copy_() in-place
#     qui casse sur des DTensors. Mais sa detection a un trou qui touche exactement
#     un Nemotron-H DENSE charge avec le code integre (trust_remote_code off) :
#       - is_nemotron_v2 exige `not n_routed_experts`, or la config dense porte un
#         n_routed_experts=8 parasite (truthy) -> le rate.
#       - is_nemotron_v3_hf exige hasattr(model, "backbone"), or le tronc integre
#         est `.model` -> le rate.
#     -> initialize_weights() s'execute et casse. On ajoute is_nemotron_hf : sauter
#     pour TOUT NemotronHForCausalLM adosse a transformers (`.model` integre OU
#     `.backbone` remote-code), mais PAS l'implementation native automodel (module
#     sous nemo_automodel.*, qui surcharge initialize_weights sans risque). Le
#     checkpoint de base est charge ensuite (le remap backbone.*->model.* vient
#     gratuitement du conversion_mapping nemotron_h de transformers), donc sauter
#     l'init from-scratch est correct.
python - <<'PY'
import io, os, sysconfig
path = os.path.join(sysconfig.get_paths()["purelib"],
                    "nemo_automodel", "components", "checkpoint", "checkpointing.py")
if not os.path.isfile(path):
    print("nemo_automodel checkpointing not found (layout changed?):", path)
else:
    src = io.open(path, encoding="utf-8").read()
    orig = src
    anchor = (
        '        is_nemotron_v3_hf = (\n'
        '            model_class == "NemotronHForCausalLM"\n'
        '            and getattr(model.config, "n_routed_experts", None)  # is Nemotron V3\n'
        '            and hasattr(model, "backbone")  # is HF remote code\n'
        '        )\n'
    )
    addition = (
        '        # Any transformers-backed NemotronH (built-in `.model` trunk OR remote-code\n'
        '        # `.backbone`): HF _init_weights re-inits Mamba A_log/dt_bias with in-place\n'
        '        # copy_, which fails on DTensors. This ALSO catches dense Nemotron-H loaded\n'
        '        # with trust_remote_code=false (built-in): is_nemotron_v2 misses it (stray\n'
        '        # default n_routed_experts=8 is truthy) and is_nemotron_v3_hf misses it (no\n'
        '        # `.backbone`). Native automodel impls (nemo_automodel.*) override\n'
        '        # initialize_weights safely, so restrict the skip to transformers-backed classes.\n'
        '        is_nemotron_hf = model_class == "NemotronHForCausalLM" and not type(\n'
        '            model\n'
        '        ).__module__.startswith("nemo_automodel")\n'
    )
    skip_anchor = (
        '            or is_nemotron_v2\n'
        '            or is_nemotron_v3_hf\n'
    )
    skip_new = (
        '            or is_nemotron_v2\n'
        '            or is_nemotron_v3_hf\n'
        '            or is_nemotron_hf\n'
    )
    if 'is_nemotron_hf' not in src and anchor in src and skip_anchor in src:
        src = src.replace(anchor, anchor + addition)
        src = src.replace(skip_anchor, skip_new)
    if src != orig:
        io.open(path, "w", encoding="utf-8").write(src)
        print("patched nemo_automodel checkpointing (skip HF NemotronH init):", path)
    elif 'is_nemotron_hf' in orig:
        print("nemo_automodel checkpointing: already patched:", path)
    else:
        print("nemo_automodel checkpointing: nothing to patch (layout changed?):", path)
PY
fi

# ---------------------------------------------------------------------------
# 4. Verification
# ---------------------------------------------------------------------------
echo "--- [4/4] verification"
python - <<'PY'
import importlib.metadata as md, importlib.util, io, os, sysconfig, sys

expected = {"nemo_automodel": "0.4.0+9eccbb61", "transformers": "5.5.0",
            "torchcodec": "0.7.0", "kaldialign": "0.9.1", "lhotse": "1.33.0",
            "peft": "0.19.1", "mlflow": "3.14.0", "latex2sympy2": "1.9.1",
            "nv-one-logger-core": "2.3.1"}
bad = []
for pkg, want in expected.items():
    try:
        got = md.version(pkg)
    except md.PackageNotFoundError:
        got = "ABSENT"
    flag = "ok " if got == want else "!! "
    if got != want:
        bad.append(pkg)
    print(f"  {flag}{pkg:<28} {got}  (attendu {want})")

purelib = sysconfig.get_paths()["purelib"]
checks = [
    ("patch 1 mamba_ssm", importlib.util.find_spec("mamba_ssm.utils.generation").origin,
     "transformers>=5 removed these legacy output classes"),
    ("patch 2 nemotron_h",
     importlib.util.find_spec("transformers.models.nemotron_h.configuration_nemotron_h").origin,
     '"-": "mlp"'),
    ("patch 3 parallelizer",
     os.path.join(purelib, "nemo_automodel/components/distributed/parallelizer.py"),
     '"backbone.layers", "model.layers"'),
    ("patch 4 checkpointing",
     os.path.join(purelib, "nemo_automodel/components/checkpoint/checkpointing.py"),
     "is_nemotron_hf"),
]
for name, path, needle in checks:
    ok = os.path.isfile(path) and needle in io.open(path, encoding="utf-8").read()
    if not ok:
        bad.append(name)
    print(f"  {'ok ' if ok else '!! '}{name}")

print("\nENV INCOMPLET: " + ", ".join(bad) if bad else "\nENV CONFORME")
sys.exit(1 if bad else 0)
PY

echo "=== termine : $DST"
