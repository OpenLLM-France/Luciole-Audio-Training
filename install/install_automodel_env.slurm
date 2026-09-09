#!/bin/bash
# ============================================================================
# Installs the conda env for SALM automodel training (branch
# luciole_speech.3.1.0-rc0, NeMo fork under $HOME/NeMo), based on the
# nemo/2.7.3 module (torch 2.12.1 : needed for
# torch.distributed.pipelining.microbatch.BlockMask, imported by
# nemo_automodel and required as soon as create_device_mesh() runs, so for
# every 8B/23B run in FSDP2 even without pipeline parallelism active).
#
# Pinned versions:
#   - AUTOMODEL_COMMIT: pinned to the ACTUAL commit from NeMo/pyproject.toml
#     (uv sources). This is the commit that exposes
#     DistributedSetup/FSDP2Config/ParallelismSizes used by
#     nemo/collections/speechlm2/parts/parallel.py::create_device_mesh.
#   - LHOTSE_VERSION: NeMo/nemo/collections/speechlm2/data/*_dataset.py calls
#     AudioSamples(..., ais_force_individual=...), a kwarg absent from
#     lhotse<2.0 -> TypeError (blocking for every model size). Pinned to
#     2.0.0a3 like NeMo/pyproject.toml.
#   - TRANSFORMERS_VERSION: pinned to 5.12.1 like NeMo/pyproject.toml. Patch
#     (2) below auto-skips as soon as the built-in nemotron_h already has
#     '"-": "mlp"' (checked at runtime).
#
# Why --no-deps for nemo_automodel (do NOT remove): a full `pip install`
# (without --no-deps) of this commit pulls in quack-kernels ->
# nvidia-cutlass-dsl -> cuda-python/nvidia-cuda-nvdisasm (CUDA 13 "nvidia-*"
# packages), which turns the `nvidia` namespace package (previously absent
# from the env) into a BROKEN one (nvidia/cu13/ with no __init__.py ->
# nvidia.__file__ is None) and crashes
# transformer_engine._nvidia_cudart_include_dir (TypeError on Path(None)).
# The same chain also installs torchao 0.18.0, incompatible with torch<2.9
# (torch.nn.functional.ScalingType absent). --no-deps + explicit pins below
# avoids all of this.
#
# Prerequisite: NeMo (~137 MB, no .git) already pushed under $HOME (rsync
# from this workstation, checkout luciole_speech.3.1.0-rc0).
#
# Phases, individually disableable:
#   SKIP_BASE=1     don't re-copy the base env (resume on an existing DST)
#   SKIP_PIP=1      don't (re)install packages
#   SKIP_PATCHES=1  don't reapply patches (they're idempotent)
#
# The pip phase needs the IDRIS proxy; it comes from the login profile and
# Slurm propagates it (--export=ALL by default). Hardcoded fallback below
# just in case.
# ============================================================================

#SBATCH --job-name=install_salm_automodel_3.1.0rc0
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
NEMO_MODULE=${NEMO_MODULE:-nemo/2.7.3}
DST=${DST:-$SCRATCH/speechlm/envs/nemo27-salm_automodel_3}
# SRC (base env) is not hardcoded: it's derived from the module, after it's
# loaded. Overridable if needed.

# NeMo fork: $HOME/NeMo, a snapshot rsynced from this workstation (checkout
# luciole_speech.3.1.0-rc0 WITHOUT .git). NEMO_FORK_BRANCH/NEMO_FORK_COMMIT
# only apply to an alternative NEMO_FORK pointed at a real clone.
NEMO_FORK=${NEMO_FORK:-$HOME/NeMo}
NEMO_FORK_REPO=${NEMO_FORK_REPO:-https://github.com/linagora-labs/NeMo.git}
NEMO_FORK_BRANCH=${NEMO_FORK_BRANCH:-luciole_speech.3.1.0-rc0}
NEMO_FORK_COMMIT=${NEMO_FORK_COMMIT:-}

# Commit actually pinned in luciole_speech.3.1.0-rc0's pyproject.toml. This is
# the commit that exposes DistributedSetup / FSDP2Config / ParallelismSizes
# used by nemo/collections/speechlm2/parts/parallel.py::create_device_mesh.
AUTOMODEL_COMMIT=${AUTOMODEL_COMMIT:-7b15c9bfbc620bec4dd3b819fce2e48726a7ddeb}

# Pinned to the actual pin in NeMo/pyproject.toml.
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-5.12.1}

PEFT_VERSION=0.19.1
MLFLOW_VERSION=3.14.0
LATEX2SYMPY2_VERSION=1.9.1
# Pinned to the actual pin in NeMo/pyproject.toml: AudioSamples there is
# missing the ais_force_individual kwarg called by
# nemo/collections/speechlm2/data/*_dataset.py on lhotse<2.0 -> TypeError.
LHOTSE_VERSION=${LHOTSE_VERSION:-2.0.0a3}

# The proxy comes from the login profile and Slurm propagates it. Not
# hardcoded here: without it the pips would fail anyway, so fail fast.
if [ "${SKIP_PIP:-0}" != "1" ] && [ -z "${https_proxy:-}" ]; then
    echo "https_proxy absent: pip won't have network access. Submit from a" >&2
    echo "login shell (the job inherits its environment) or export it." >&2
    exit 1
fi

module purge
module load arch/h100 "$NEMO_MODULE"

# Base env = the one the module just selected. conda flags it with a '*' in
# `conda env list`; failing that, we derive it from the module name under the
# conda root (nemo/2.7.3 -> envs/nemo-2.7.3+py*). No absolute path hardcoded
# here.
if [ -z "${SRC:-}" ]; then
    # $2 == "*" alone also matches the legend line "# *  -> active" that this
    # conda (miniforge 25.9.1) always prints -> SRC ended up corrupted
    # ("-> /path..."). We exclude comment lines and stop at the first real
    # match.
    SRC=$(conda env list 2>/dev/null | awk '$1 !~ /^#/ && $2 == "*" {print $3; exit}')
    [ -n "$SRC" ] || SRC=$(ls -d "$(conda info --base)/envs/nemo-${NEMO_MODULE#nemo/}"+py* 2>/dev/null | head -1)
fi
[ -n "${SRC:-}" ] && [ -x "$SRC/bin/python" ] || {
    echo "base env not found for $NEMO_MODULE (SRC='${SRC:-}')" >&2; exit 1; }

echo "=== module   : $NEMO_MODULE"
echo "=== base     : $SRC"
echo "=== target   : $DST"
echo "=== NeMo fork: $NEMO_FORK"

# ---------------------------------------------------------------------------
# 1. Base env: copy of the shared module
# ---------------------------------------------------------------------------
# `conda create --clone` fails on this env: everything is pypi_0 (the clone
# can't reproduce them) and it tries to write into the read-only source. A
# direct copy reproduces it exactly.
if [ "${SKIP_BASE:-0}" = "1" ]; then
    echo "--- [1/4] base: skipped (SKIP_BASE=1)"
else
    echo "--- [1/4] base: copying 12 GB, expect ~15 min on Lustre"
    # If DST exists, cp -a would copy SRC *into* it. Fail instead of damaging
    # a live env.
    [ -e "$DST" ] && { echo "DST already exists, remove it or pass SKIP_BASE=1: $DST" >&2; exit 1; }
    mkdir -p "$(dirname "$DST")"
    cp -a "$SRC" "$DST"

    # Rewrite the hardcoded source prefix in the console scripts.
    grep -rIl "$SRC" "$DST/bin" 2>/dev/null | xargs -r sed -i "s|$SRC|$DST|g"

    # Same for the .pth files. They contain self-referencing paths
    # (setuptools.pth) that, left unrewritten, re-inject the read-only shared
    # env's site-packages into sys.path and shadow this one — notably the
    # NeMo fork installed in editable mode.
    grep -rIl "$SRC" "$DST"/lib/python*/site-packages/*.pth 2>/dev/null \
        | xargs -r sed -i "s|$SRC|$DST|g"
fi

conda activate "$DST"
echo "--- python   : $(python -V 2>&1) ($(which python))"

# ---------------------------------------------------------------------------
# 2. Packages
# ---------------------------------------------------------------------------
if [ "${SKIP_PIP:-0}" = "1" ]; then
    echo "--- [2/4] packages: skipped (SKIP_PIP=1)"
else
    echo "--- [2/4] packages"
    # --no-deps: see banner at the top. Without it, this commit pulls in
    # quack-kernels -> nvidia-cutlass-dsl -> cuda-python (CUDA 13 "nvidia-*"
    # packages), which breaks the `nvidia` namespace package
    # (transformer_engine._nvidia_cudart_include_dir crashes on Path(None))
    # and installs torchao 0.18.0 (incompatible with torch<2.9).
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
    pip install torchdata
    # NeMo fork. Two cases:
    #   - NEMO_FORK is already a snapshot without .git (the default case,
    #     rsynced from this workstation — see banner at the top): install it
    #     as-is, no git operation is possible on it (NEMO_FORK_BRANCH/COMMIT
    #     ignored).
    #   - NEMO_FORK is (or should become) a real git clone: original
    #     behavior, clone if absent then fetch + checkout at the desired
    #     commit (latest on the branch by default, see NEMO_FORK_COMMIT
    #     above).
    if [ -e "$NEMO_FORK" ] && [ ! -d "$NEMO_FORK/.git" ]; then
        echo "--- NeMo fork: $NEMO_FORK (snapshot without .git — no git checkout, NEMO_FORK_BRANCH/COMMIT ignored)"
    else
        if [ ! -d "$NEMO_FORK/.git" ]; then
            git clone --branch "$NEMO_FORK_BRANCH" "$NEMO_FORK_REPO" "$NEMO_FORK"
        fi
        git -C "$NEMO_FORK" fetch --quiet origin "$NEMO_FORK_BRANCH"
        # NEMO_FORK_COMMIT empty (default) -> latest commit fetched on the branch.
        [ -n "$NEMO_FORK_COMMIT" ] || NEMO_FORK_COMMIT=$(git -C "$NEMO_FORK" rev-parse FETCH_HEAD)
        git -C "$NEMO_FORK" checkout --quiet "$NEMO_FORK_COMMIT"
        echo "--- NeMo fork: $(git -C "$NEMO_FORK" log --oneline -1)"
    fi
    pip install --no-deps -e "$NEMO_FORK"
fi

# ===========================================================================
# 3. Post-install patches — needed to load the dense Nemotron-H (Luciole-8B)
#    under transformers 5.x with the BUILT-IN modeling (trust_remote_code
#    off; cf. conf/base_automodel.yaml). All idempotent, self-detecting
#    (silently skip if already satisfied).
# ===========================================================================
if [ "${SKIP_PATCHES:-0}" = "1" ]; then
    echo "--- [3/4] patches: skipped (SKIP_PATCHES=1)"
else
echo "--- [3/4] patches"

# (1) ESSENTIAL. mamba-ssm hardcodes an import of GreedySearchDecoderOnlyOutput
#     / SampleDecoderOnlyOutput from transformers.generation, removed in 5.x.
#     The BUILT-IN nemotron_h modeling loads mamba_ssm (lazy_load_kernel ->
#     ops.triton.*, which runs mamba_ssm/__init__.py -> mixer_seq_simple ->
#     utils.generation) and dies on the ImportError. We alias the removed
#     classes to GenerateDecoderOnlyOutput; they're only used by mamba_ssm's
#     decode(), which Nemotron-H training never calls.
python - <<'PY'
import io, os, sysconfig
path = os.path.join(sysconfig.get_paths()["purelib"], "mamba_ssm", "utils", "generation.py")
if not os.path.isfile(path):
    print("mamba_ssm generation.py not found (layout changed?):", path)
else:
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

# (2) Self-detecting. transformers 5.3-5.5 lost the dense '-'=MLP token in the
#     built-in nemotron_h's pattern parser (KeyError: '-'). Since
#     trust_remote_code is OFF, it's the built-in that's loaded. On 5.12.1
#     this is normally already fixed upstream: the check below verifies it at
#     runtime rather than assuming it, and the overlay only triggers if
#     needed.
if ! python -c "import io,importlib.util; p=importlib.util.find_spec('transformers.models.nemotron_h.configuration_nemotron_h').origin; import sys; sys.exit(0 if '\"-\": \"mlp\"' in io.open(p).read() else 3)"; then
  echo "Overlaying nemotron_h from transformers 5.6.0 (built-in is missing '-'=mlp)…"
  TMP=$(mktemp -d)
  pip download --no-deps -q "transformers==5.6.0" -d "$TMP"
  ( cd "$TMP" && unzip -oq transformers-*.whl 'transformers/models/nemotron_h/*' )
  DEST=$(python -c "import os,transformers.models.nemotron_h as m; print(os.path.dirname(m.__file__))")
  cp -f "$TMP"/transformers/models/nemotron_h/*.py "$DEST/"
  rm -rf "$TMP"
  echo "overlaid nemotron_h into $DEST"
fi

# (3) ESSENTIAL in multi-GPU. nemo_automodel's parallelizer hardcodes the
#     BUNDLED trunk name `backbone` in two places, whereas the built-in
#     NemotronHForCausalLM names its own `model`
#     (modeling_nemotron_h.py: self.model = NemotronHModel). Both dispatch on
#     the class name, common to both implementations: with trust_remote_code
#     off, the built-in hits this code and breaks on the missing `.backbone`.
#     We make it accept BOTH names (no-op on the bundled impl):
#       - LLM_MODEL_CLS_TO_LAYERS (FSDP wrapping): add "model.layers" next to
#         "backbone.layers"; _reduce_attrs silently ignores the absent path.
#       - NemotronHParallelizationStrategy (TP path, only if tp_size>1):
#         resolve the trunk as backbone-or-model before reading `.layers`.
python - <<'PY'
import io, os, sysconfig
# Locate the file WITHOUT importing nemo_automodel: its __init__ eagerly
# imports torch, which requires MKL/module-loads a bare shell may not have.
# A purely textual patch shouldn't depend on that.
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

# (4) ESSENTIAL in multi-GPU. nemo_automodel's checkpointer skips
#     transformers' initialize_weights() for Nemotron-H HF, because HF
#     _init_weights re-initializes the Mamba A_log/dt_bias via an in-place
#     copy_() that breaks on DTensors. But its detection has a gap that hits
#     exactly a DENSE Nemotron-H loaded with the built-in code
#     (trust_remote_code off):
#       - is_nemotron_v2 requires `not n_routed_experts`, but the dense
#         config carries a stray n_routed_experts=8 (truthy) -> misses it.
#       - is_nemotron_v3_hf requires hasattr(model, "backbone"), but the
#         built-in trunk is `.model` -> misses it.
#     -> initialize_weights() runs and breaks. We add is_nemotron_hf: skip
#     for ANY NemotronHForCausalLM backed by transformers (built-in `.model`
#     OR remote-code `.backbone`), but NOT the native automodel
#     implementation (module under nemo_automodel.*, which overrides
#     initialize_weights safely). The base checkpoint is loaded afterwards
#     (the backbone.*->model.* remap comes for free from transformers'
#     nemotron_h conversion_mapping), so skipping the from-scratch init is
#     correct.
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
python - <<PY
import importlib.metadata as md, importlib.util, io, os, sysconfig, sys

expected = {"transformers": "${TRANSFORMERS_VERSION}",
            "torchcodec": "0.7.0", "kaldialign": "0.9.1", "lhotse": "${LHOTSE_VERSION}",
            "peft": "${PEFT_VERSION}", "mlflow": "${MLFLOW_VERSION}",
            "latex2sympy2": "${LATEX2SYMPY2_VERSION}", "nv-one-logger-core": "2.3.1"}
bad = []
for pkg, want in expected.items():
    try:
        got = md.version(pkg)
    except md.PackageNotFoundError:
        got = "ABSENT"
    flag = "ok " if got == want else "!! "
    if got != want:
        bad.append(pkg)
    print(f"  {flag}{pkg:<28} {got}  (expected {want})")

try:
    automodel_got = md.version("nemo_automodel")
    print(f"  -   {'nemo_automodel':<28} {automodel_got}  (expected commit ${AUTOMODEL_COMMIT})")
except md.PackageNotFoundError:
    print("  !! nemo_automodel ABSENT")
    bad.append("nemo_automodel")

import torch
torch_ok = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 7)
print(f"  {'ok ' if torch_ok else '!! '}{'torch':<28} {torch.__version__}  (expected >=2.7.0)")
if not torch_ok:
    bad.append("torch")

blockmask_ok = importlib.util.find_spec("torch.distributed.pipelining.microbatch") is not None
if blockmask_ok:
    try:
        from torch.distributed.pipelining.microbatch import BlockMask  # noqa: F401
    except ImportError:
        blockmask_ok = False
print(f"  {'ok ' if blockmask_ok else '!! '}{'torch pipelining.BlockMask':<28} {'present' if blockmask_ok else 'ABSENT'}  (expected present, required by nemo_automodel autopipeline.py)")
if not blockmask_ok:
    bad.append("torch.distributed.pipelining.BlockMask")

nvidia_ok = True
try:
    import nvidia
    nvidia_ok = getattr(nvidia, "__file__", None) is not None
except ModuleNotFoundError:
    pass  # absent = ok, the expected state without quack-kernels/cuda-python
print(f"  {'ok ' if nvidia_ok else '!! '}{'nvidia namespace package':<28} {'ok (absent or regular)' if nvidia_ok else 'BROKEN (namespace without __file__)'}")
if not nvidia_ok:
    bad.append("nvidia namespace package")

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

print("\nENV INCOMPLETE: " + ", ".join(bad) if bad else "\nENV CONFORMANT")
sys.exit(1 if bad else 0)
PY

echo "=== done: $DST"
