#!/usr/bin/env python3
"""Submit run_train.slurm by materializing a self-contained copy into SAVE_DIR.

    sbatch <save_dir>/<job_name>_<jobid>.slurm

Examples:
  python slurm_launcher.py
  python slurm_launcher.py --data-version data_v3 --gpus 4 --time 02:00:00
  python slurm_launcher.py --config foo.yaml --qos t3 --nodes 2
  python slurm_launcher.py --llm-model /path/to/model --prompt-format llama3 --dry-run
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

SLURM_SCRIPT = Path(__file__).resolve().parent / "run_train.slurm"

QOS_MAP = {
    "dev": "qos_gpu_h100-dev",
    "t3": "qos_gpu_h100-t3",
}

CPUS_PER_GPU = 24


def parse_job_name(script_path: Path) -> str:
    m = re.search(r"^#SBATCH\s+--job-name=(\S+)", script_path.read_text(), re.MULTILINE)
    if not m:
        sys.exit(f"Could not find '#SBATCH --job-name=...' in {script_path}")
    return m.group(1)


def rewrite_sbatch(lines: list[str], key: str, value: str) -> None:
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


def inject_exports(lines: list[str], env: dict[str, str]) -> None:
    """Insert an `export` block right after the last #SBATCH directive."""
    last_sbatch = max(i for i, l in enumerate(lines) if l.startswith("#SBATCH"))
    block = ["\n# ---- Overrides injected by slurm_launcher.py ----\n",
             "# To rerun this exact job: `sbatch <this-file>`\n"]
    for k, v in env.items():
        block.append(f"export {k}={shlex.quote(v)}\n")
    block.append("# ---- End overrides ----\n")
    lines[last_sbatch + 1:last_sbatch + 1] = block


def materialize(template: Path, out_path: Path, job_name: str, save_dir: str,
                sbatch_overrides: dict[str, str], env_overrides: dict[str, str]) -> None:
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


def parse_job_id(sbatch_stdout: str) -> str:
    m = re.search(r"Submitted batch job (\d+)", sbatch_stdout)
    if not m:
        sys.exit(f"Could not parse job id from sbatch output: {sbatch_stdout!r}")
    return m.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g_train = parser.add_argument_group("training knobs (env vars)")
    g_train.add_argument("--config", default="config_multitask-v2.yaml",
                         help="Hydra config filename under conf/ (default: %(default)s)")
    g_train.add_argument("--data-version", default="data_v2",
                         help="Subfolder under conf/data/ (default: %(default)s)")
    g_train.add_argument("--experiment-folder", default=None,
                         help="Override EXPERIMENT_FOLDER (NeMo exp_manager output root)")
    g_train.add_argument("--llm-model", default=None, help="Override LLM_MODEL path")
    g_train.add_argument("--prompt-format", default=None, help="Override PROMPT_FORMAT")
    g_train.add_argument("--speech-encoder-model", default=None,
                         help="Override SPEECH_ENCODER_MODEL path")
    g_train.add_argument("--nemo-fork", default=None, help="Override NEMO_FORK path")
    g_train.add_argument("--overwrite", action="store_true",
                         help="Wipe existing experiment folder before launch")
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
                        help="Materialize the slurm file but don't submit; leave it at "
                             "<save_dir>/<job_name>_dryrun.slurm for inspection")
    args = parser.parse_args()

    job_name = args.job_name or parse_job_name(SLURM_SCRIPT)

    save_dir_raw = args.save_dir or f"$ALL_CCFRSCRATCH/audio/training/logs/{job_name}"
    save_dir = os.path.expandvars(save_dir_raw)
    if "$" in save_dir:
        raise Exception(f"save-dir still contains unexpanded vars: {save_dir!r} "
                 f"(source your env or pass --save-dir explicitly)")
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    env_overrides: dict[str, str] = {
        "CONFIG_NAME":  args.config,
        "DATA_VERSION": args.data_version,
        "OVERWRITE":    "true" if args.overwrite else "false",
        "SAVE_DIR":     save_dir,
        "LOCAL_FOLDER": str(SLURM_SCRIPT.parent),
    }
    for name, value in [
        ("EXPERIMENT_FOLDER",     args.experiment_folder),
        ("LLM_MODEL",             args.llm_model),
        ("PROMPT_FORMAT",         args.prompt_format),
        ("SPEECH_ENCODER_MODEL",  args.speech_encoder_model),
        ("NEMO_FORK",             args.nemo_fork),
    ]:
        if value is not None:
            env_overrides[name] = value

    sbatch_overrides: dict[str, str] = {}
    if args.gpus is not None:
        sbatch_overrides["gres"]          = f"gpu:{args.gpus}"
        sbatch_overrides["cpus-per-task"] = str(CPUS_PER_GPU * args.gpus)
    if args.qos is not None:
        sbatch_overrides["qos"]   = QOS_MAP[args.qos]
    if args.time is not None:
        sbatch_overrides["time"]  = args.time
    if args.nodes is not None:
        sbatch_overrides["nodes"] = str(args.nodes)

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
        print(f"(dry run — not submitted; inspect: {dryrun_path})")
        return

    result = subprocess.run(["sbatch", str(submitted)], capture_output=True, text=True)
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
