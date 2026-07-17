#!/usr/bin/env python3
"""Submit run_train.slurm by materializing a self-contained copy into SAVE_DIR.

    sbatch <save_dir>/<job_name>_<jobid>.slurm

Examples:
  python slurm_launcher.py
  python slurm_launcher.py --data-version data_v3 --gpus 4 --time 02:00:00
  python slurm_launcher.py --config foo.yaml --qos t3 --nodes 2
  python slurm_launcher.py --llm-model /path/to/model --prompt-format llama3 --dry-run
  python slurm_launcher.py --config run/xp/automodel_8b --conda-env $SCRATCH/speechlm/envs/salm_automodel
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SLURM_SCRIPT = Path(__file__).resolve().parent / "run_train.slurm"

QOS_MAP = {
    "dev": "qos_gpu_h100-dev",
    "t3": "qos_gpu_h100-t3",
}

CPUS_PER_GPU = 24

# exp_manager.max_time_per_run = SLURM --time minus this many minutes, clamped to >= MIN
MAX_TIME_MARGIN_MIN = 15
MAX_TIME_FLOOR_MIN = 30


def parse_slurm_time_to_minutes(s):
    """Parse a SLURM --time string (HH:MM:SS, MM:SS, MM, or DD-HH:MM:SS) to minutes."""
    days = 0
    if "-" in s:
        days_str, s = s.split("-", 1)
        days = int(days_str)
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = (int(x) for x in parts)
    elif len(parts) == 2:
        h, (m, sec) = 0, (int(parts[0]), int(parts[1]))
    elif len(parts) == 1:
        h, m, sec = 0, int(parts[0]), 0
    else:
        raise ValueError(f"Unsupported SLURM time format: {s!r}")
    return days * 24 * 60 + h * 60 + m + (sec // 60)


def minutes_to_lightning_time(mins):
    """DD:HH:MM:SS for Lightning's exp_manager.max_time_per_run."""
    d, rem = divmod(mins, 24 * 60)
    h, m = divmod(rem, 60)
    return f"{d:02d}:{h:02d}:{m:02d}:00"


def parse_slurm_time_from_template(script_path):
    m = re.search(r"^#SBATCH\s+--time=(\S+)", script_path.read_text(), re.MULTILINE)
    return m.group(1) if m else None


def parse_job_name(script_path):
    m = re.search(r"^#SBATCH\s+--job-name=(\S+)", script_path.read_text(), re.MULTILINE)
    if not m:
        sys.exit(f"Could not find '#SBATCH --job-name=...' in {script_path}")
    return m.group(1)


def rewrite_sbatch(lines, key, value):
    """Replace the value of `#SBATCH --<key>=...`, or insert after last #SBATCH if missing."""
    pattern = re.compile(rf"^(#SBATCH\s+--{re.escape(key)}=)\S+.*$")
    new_line = f"#SBATCH --{key}={value}\n"
    for i, line in enumerate(lines):
        if pattern.match(line.rstrip("\n")):
            lines[i] = new_line
            return
    last_sbatch = max(
        (i for i, l in enumerate(lines) if l.startswith("#SBATCH")),
        default=0,
    )
    lines.insert(last_sbatch + 1, new_line)


def inject_exports(lines, env):
    """Insert an `export` block right after the last #SBATCH directive."""
    last_sbatch = max(i for i, l in enumerate(lines) if l.startswith("#SBATCH"))
    block = ["\n# ---- Overrides injected by slurm_launcher.py ----\n",
             "# To rerun this exact job: `sbatch <this-file>`\n",
             "# (NOTE: env vars NOT injected here, e.g. CHECKPOINT for curriculum\n",
             "#  warm-starts, must be re-exported in the shell before re-running.)\n"]
    for k, v in env.items():
        block.append(f"export {k}={shlex.quote(v)}\n")
    block.append("# ---- End overrides ----\n")
    lines[last_sbatch + 1:last_sbatch + 1] = block


def materialize(template, out_path, job_name, save_dir, sbatch_overrides, env_overrides):
    lines = template.read_text().splitlines(keepends=True)

    # These #SBATCH directives are always set by the launcher
    forced = {
        "job-name": job_name,
        "output":   f"{save_dir}/{job_name}_%j.out",
        "error":    f"{save_dir}/{job_name}_%j.err",
    }
    for key, value in {**forced, **sbatch_overrides}.items():
        rewrite_sbatch(lines, key, value)

    inject_exports(lines, env_overrides)

    out_path.write_text("".join(lines))
    out_path.chmod(0o755)


def resolve_input_cfg(config_path, data_version):
    """Mirror run_train.slurm's INPUT_CFG resolution from DATA_VERSION."""
    if data_version.endswith(".yaml"):
        return config_path / "data" / data_version
    return (config_path / "data" / data_version /
            "input_cfg_train_weighted_randomorder_sharded.yaml")


def copy_configs(save_dir_path, job_name, config_name, data_version, suffix):
    """Replicate the `cp` steps run_train.slurm does at runtime (config + input_cfg yamls).

    Returns the list of destination paths. Used for --dry-run, where the slurm
    script never executes and so would otherwise not produce these snapshots.
    """
    config_path = SLURM_SCRIPT.parent / "conf"
    config_yaml = config_path / f"{config_name}.yaml"
    input_cfg = resolve_input_cfg(config_path, data_version)

    copied = []
    for src, dst in [
        (config_yaml, save_dir_path / f"{job_name}_{suffix}.yaml"),
        (input_cfg,   save_dir_path / f"{job_name}_{suffix}.input_cfg.yaml"),
    ]:
        if not src.exists():
            sys.exit(f"Config to copy does not exist: {src}")
        shutil.copyfile(src, dst)
        copied.append(dst)
    return copied


def config_needs_checkpoint(config_name):
    """Return True if the composed Hydra config references ${oc.env:CHECKPOINT}.

    Curriculum warm-start stages set `init_from_checkpoint: ${oc.env:CHECKPOINT}`
    (directly or via their defaults list, e.g. phase/curriculum_encoder.yaml), so
    the job fails at config-resolve time if CHECKPOINT is unset. We compose the
    config the same way salm_train.py does and inspect it *without* resolving, so
    the missing env var doesn't blow up the check itself.
    """
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError:
        return False  # can't introspect here; let the job decide at runtime

    conf_dir = str(SLURM_SCRIPT.parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name=config_name)
    return "oc.env:CHECKPOINT" in OmegaConf.to_yaml(cfg, resolve=False)


def parse_job_id(sbatch_stdout):
    m = re.search(r"Submitted batch job (\d+)", sbatch_stdout)
    if not m:
        sys.exit(f"Could not parse job id from sbatch output: {sbatch_stdout!r}")
    return m.group(1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g_train = parser.add_argument_group("training knobs (env vars)")
    g_train.add_argument("--config", default="run/luciole",
                         help="Hydra config name under conf/ (default: %(default)s). "
                              "Use 'run/<name>' to pick a composed run file.")
    g_train.add_argument("--data-version", default="v2",
                         help="Subfolder under conf/data/ (default: %(default)s)")
    g_train.add_argument("--experiment-folder", default=None,
                         help="Override EXPERIMENT_FOLDER (NeMo exp_manager output root)")
    g_train.add_argument("--llm-model", default=None, help="Override LLM_MODEL path")
    g_train.add_argument("--prompt-format", default=None, help="Override PROMPT_FORMAT")
    g_train.add_argument("--speech-encoder-model", default=None,
                         help="Override SPEECH_ENCODER_MODEL path")
    g_train.add_argument("--nemo-fork", default=None, help="Override NEMO_FORK path")
    g_train.add_argument("--conda-env", default=None,
                         help="Override CONDA_ENV, the conda env activated by the job "
                              "(default: $SCRATCH/speechlm/envs/nemo24-salm). Automodel "
                              "runs need a dedicated env, e.g. .../envs/salm_automodel")
    g_train.add_argument("--set", dest="overrides", action="append", default=[],
                         metavar="KEY=VAL",
                         help="Extra Hydra override appended to the torchrun command "
                              "(repeatable), e.g. --set ++model.optimizer.lr=1e-6")
    g_train.add_argument("--overwrite", action="store_true",
                         help="Wipe existing experiment folder before launch")
    g_train.add_argument("--resume", action="store_true",
                         help="Keep an existing experiment folder and resume from its "
                              "last checkpoint (exp_manager resume_if_exists)")
    g_train.add_argument("--overwrite-log", action="store_true",
                         help="Clear the log/save directory before launch "
                              "(removes previous .slurm/.out/.err snapshots)")
    g_train.add_argument("--save-dir", default=None,
                         help="Log + snapshot directory "
                              "(default: $ALL_CCFRSCRATCH/audio/training/logs/<job_name>)")

    g_slurm = parser.add_argument_group("SLURM directives")
    g_slurm.add_argument("--gpus", type=int, default=None,
                         help=f"GPUs per node (sets --gres=gpu:N and --cpus-per-task={CPUS_PER_GPU}*N)")
    g_slurm.add_argument("--qos", choices=list(QOS_MAP), default=None,
                         help="SLURM QoS shorthand (maps to qos_gpu_h100-<name>)")
    g_slurm.add_argument("--time", default=None, help="SLURM time limit (e.g. 02:00:00)")
    g_slurm.add_argument("--nodes", type=int, default=None, help="Number of nodes")
    g_slurm.add_argument("--job-name", default=None,
                         help="Override SLURM --job-name (default: from run_train.slurm)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Materialize the slurm file and copy the config + input_cfg "
                             "yamls (as run_train.slurm would) but don't submit; leave them "
                             "at <save_dir>/<job_name>_dryrun.* for inspection")
    args = parser.parse_args()

    if args.resume and args.overwrite:
        sys.exit("--resume and --overwrite are mutually exclusive "
                 "(one keeps the folder to resume, the other wipes it).")

    job_name = args.job_name or parse_job_name(SLURM_SCRIPT)

    save_dir_raw = args.save_dir or f"$ALL_CCFRSCRATCH/audio/training/logs/{job_name}"
    save_dir = os.path.expandvars(save_dir_raw)
    if "$" in save_dir:
        raise Exception(f"save-dir still contains unexpanded vars: {save_dir!r} "
                 f"(source your env or pass --save-dir explicitly)")
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    if args.overwrite_log:
        print(f"Clearing log directory contents: {save_dir_path}")
        # Delete the *contents*, not the dir itself, so shells cd'd into it (and
        # their relative paths) keep working.
        for entry in save_dir_path.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    env_overrides = {
        "CONFIG_NAME":  args.config,
        "DATA_VERSION": args.data_version,
        "OVERWRITE":    "true" if args.overwrite else "false",
        "RESUME":       "true" if args.resume else "false",
        "SAVE_DIR":     save_dir,
        "LOCAL_FOLDER": str(SLURM_SCRIPT.parent),
    }
    for name, value in [
        ("EXPERIMENT_FOLDER",     args.experiment_folder),
        ("LLM_MODEL",             args.llm_model),
        ("PROMPT_FORMAT",         args.prompt_format),
        ("SPEECH_ENCODER_MODEL",  args.speech_encoder_model),
        ("NEMO_FORK",             args.nemo_fork),
        ("CONDA_ENV",             args.conda_env),
    ]:
        if value is not None:
            # An empty string almost always means an undefined shell variable was
            # passed (e.g. --speech-encoder-model "$PARAKEET" with $PARAKEET unset).
            # Injecting `export X=''` would silently trip run_train.slurm's
            # `${X:-<default>}` fallback, running with the WRONG model. Fail loud.
            if value == "":
                cli_flag = "--" + name.lower().replace("_", "-")
                sys.exit(f"Empty value for {cli_flag} (probably an undefined shell "
                         f"variable). Pass a real path or omit the flag to use the "
                         f"default.")
            env_overrides[name] = value

    if args.overrides:
        for o in args.overrides:
            if "=" not in o:
                sys.exit(f"--set expects KEY=VAL, got: {o!r}")
        env_overrides["EXTRA_OVERRIDES"] = " ".join(args.overrides)

    sbatch_overrides = {}
    if args.gpus is not None:
        sbatch_overrides["gres"]          = f"gpu:{args.gpus}"
        sbatch_overrides["cpus-per-task"] = str(CPUS_PER_GPU * args.gpus)
    if args.qos is not None:
        sbatch_overrides["qos"]   = QOS_MAP[args.qos]
    if args.time is not None:
        sbatch_overrides["time"]  = args.time
    if args.nodes is not None:
        sbatch_overrides["nodes"] = str(args.nodes)

    # Derive exp_manager.max_time_per_run = SLURM time - 15min, min 30min.
    slurm_time_str = args.time or parse_slurm_time_from_template(SLURM_SCRIPT)
    if slurm_time_str is not None:
        slurm_mins = parse_slurm_time_to_minutes(slurm_time_str)
        max_time_mins = max(slurm_mins - MAX_TIME_MARGIN_MIN, MAX_TIME_FLOOR_MIN)
        env_overrides["MAX_TIME_PER_RUN"] = minutes_to_lightning_time(max_time_mins)

    submitted = save_dir_path / f"{job_name}_submitted.slurm"
    materialize(SLURM_SCRIPT, submitted, job_name, save_dir, sbatch_overrides, env_overrides)

    # sanity-check the materialized script parses
    check = subprocess.run(["bash", "-n", str(submitted)], capture_output=True, text=True)
    if check.returncode != 0:
        sys.exit(f"Materialized slurm file has syntax errors:\n{check.stderr}\nPath: {submitted}")

    print(f"Materialized: {submitted}")
    print("Env overrides:")
    for k, v in env_overrides.items():
        print(f"  {k}={v}")
    if sbatch_overrides:
        print("SBATCH overrides:")
        for k, v in sbatch_overrides.items():
            print(f"  --{k}={v}")

    if args.dry_run:
        dryrun_path = save_dir_path / f"{job_name}_dryrun.slurm"
        submitted.rename(dryrun_path)
        copied = copy_configs(save_dir_path, job_name, args.config,
                              args.data_version, "dryrun")
        print(f"(dry run — not submitted; inspect: {dryrun_path})")
        for dst in copied:
            print(f"  config: {dst}")
        return

    if config_needs_checkpoint(args.config) and not os.environ.get("CHECKPOINT"):
        submitted.unlink(missing_ok=True)
        sys.exit(
            f"Config '{args.config}' warm-starts from ${{oc.env:CHECKPOINT}} but "
            f"CHECKPOINT is not set in the environment.\n"
            f"Export it before launching, e.g.:\n"
            f'  export CHECKPOINT="$(ls -t "$EXP"/<stage>/checkpoints/*-last.ckpt | head -1)"'
        )

    result = subprocess.run(["sbatch", "--export=ALL", str(submitted)], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        submitted.unlink(missing_ok=True)
        sys.exit(result.returncode)

    job_id = parse_job_id(result.stdout)
    final_path = save_dir_path / f"{job_name}_{job_id}.slurm"
    submitted.rename(final_path)
    print(f"Snapshot: {final_path}")


if __name__ == "__main__":
    main()
