#!/usr/bin/env python3
"""Submit run_duplex.slurm with one folder per experiment, logs included:

    <root>/<job_name>/                 root = $ALL_CCFRSCRATCH/audio/training/duplex
        <job_name>_<jobid>.slurm       materialized copy of run_duplex.slurm (rerun: sbatch it)
        <job_name>_<jobid>.out/.err    Slurm logs
        <job_name>_<jobid>.yaml        config + input_cfg snapshots (copied by the job)
        checkpoints/, TensorBoard, ... NeMo exp_manager (exp_dir=<root>, name=<job_name>)

Slurm opens --output before the job starts, so the folder must exist at submission time:
this script creates it, like ../training/slurm_launcher.py does for the SALM runs (whose helpers it reuses).

Examples:
  python duplex_launcher.py --job-name smoke --qos dev --gpus 1 --time 00:30:00 \
      --set ++trainer.max_steps=50 --set ++trainer.val_check_interval=25
  python duplex_launcher.py --model qa --job-name duplex_test_v1
  python duplex_launcher.py --model transcribe --job-name transcribe_v1
  python duplex_launcher.py --job-name duplex_fr_v1 --input-cfg conf/data/input_cfg_duplex_fr.yaml
  python duplex_launcher.py --job-name duplex_test_v1 --resume    # continue from its last checkpoint
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "training"))
from slurm_launcher import (  # noqa: E402
    CPUS_PER_GPU,
    MAX_TIME_FLOOR_MIN,
    MAX_TIME_MARGIN_MIN,
    QOS_MAP,
    expand_or_die,
    materialize,
    minutes_to_lightning_time,
    parse_job_id,
    parse_job_name,
    parse_slurm_time_from_template,
    parse_slurm_time_to_minutes,
)

SLURM_SCRIPT = HERE / "run_duplex.slurm"
DEFAULT_ROOT = "$ALL_CCFRSCRATCH/audio/training/duplex"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-name", default=None,
                        help="Experiment name = Slurm job name = folder name "
                             "(default: run_duplex.slurm's --job-name)")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="Folder holding one subfolder per experiment (default: %(default)s)")
    parser.add_argument("--model", choices=["qa", "transcribe"], default=None,
                        help="qa: ASR/QA, 2 heads (v4 logic); transcribe: Luciole-Audio-Transcribe streaming, "
                             "1 head (v5 logic). Picks conf/duplex_<model>.yaml, conf/data/input_cfg_<model>.yaml "
                             "and conf/data/val_cfg_<model>.yaml (default in run_duplex.slurm: qa)")
    parser.add_argument("--config", default=None, help="CONFIG_NAME under conf/ (default: duplex_<model>)")
    parser.add_argument("--input-cfg", default=None,
                        help="INPUT_CFG, relative to train_duplex/ or absolute "
                             "(default: conf/data/input_cfg_<model>.yaml)")
    parser.add_argument("--val-cfg", default=None,
                        help="VAL_CFG (validation sets), relative to train_duplex/ or absolute "
                             "(default: conf/data/val_cfg_<model>.yaml)")
    parser.add_argument("--llm-model", default=None, help="Override LLM_MODEL")
    parser.add_argument("--nemo-dir", default=None, help="Override NEMO_DIR (default: $HOME/NeMo_Duplex)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL",
                        help="Extra Hydra override (repeatable), e.g. --set ++trainer.max_steps=50")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse an existing experiment folder; exp_manager resumes from its "
                             "last checkpoint")
    parser.add_argument("--gpus", type=int, default=None,
                        help=f"GPUs per node (sets --gres=gpu:N and --cpus-per-task={CPUS_PER_GPU}*N)")
    parser.add_argument("--qos", choices=list(QOS_MAP), default=None, help="QoS shorthand")
    parser.add_argument("--time", default=None, help="Slurm time limit (e.g. 00:30:00)")
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--after", metavar="JOBID", default=None,
                        help="Submit with --dependency=afterok:JOBID")
    parser.add_argument("--dry-run", action="store_true",
                        help="Create the folder and the materialized .slurm, but don't submit")
    args = parser.parse_args()

    job_name = args.job_name or parse_job_name(SLURM_SCRIPT)
    root = expand_or_die(args.root, "--root")
    xp_dir = Path(root) / job_name
    # A dry run leaves only its _dryrun.slurm behind: that alone doesn't make it an existing run.
    leftovers = [f for f in xp_dir.iterdir() if not f.name.endswith("_dryrun.slurm")] if xp_dir.exists() else []
    if leftovers and not args.resume:
        sys.exit(f"Experiment folder already exists: {xp_dir}\n"
                 f"Pass --resume to continue it, or pick another --job-name.")

    # Shared project space: dirs 2770 / files 660, group qgz. The parent folders are setgid,
    # so new subfolders inherit the group and the setgid bit; the umask drops access for others.
    os.umask(0o007)
    xp_dir.mkdir(parents=True, exist_ok=True)

    env_overrides = {
        "EXPERIMENT_FOLDER": root,
        "SAVE_DIR": str(xp_dir),
        "LOCAL_FOLDER": str(HERE),
    }
    for name, value in [
        ("MODEL", args.model),
        ("CONFIG_NAME", args.config),
        ("INPUT_CFG", args.input_cfg),
        ("VAL_CFG", args.val_cfg),
        ("LLM_MODEL", args.llm_model),
        ("NEMO_DIR", args.nemo_dir),
    ]:
        if value is not None:
            if value == "":
                sys.exit(f"Empty value for {name} (probably an undefined shell variable).")
            env_overrides[name] = value
    for o in args.overrides:
        if "=" not in o:
            sys.exit(f"--set expects KEY=VAL, got: {o!r}")
    if args.overrides:
        env_overrides["EXTRA_OVERRIDES"] = " ".join(args.overrides)

    sbatch_overrides = {}
    if args.gpus is not None:
        sbatch_overrides["gres"] = f"gpu:{args.gpus}"
        sbatch_overrides["cpus-per-task"] = str(CPUS_PER_GPU * args.gpus)
    if args.qos is not None:
        sbatch_overrides["qos"] = QOS_MAP[args.qos]
    if args.time is not None:
        sbatch_overrides["time"] = args.time
    if args.nodes is not None:
        sbatch_overrides["nodes"] = str(args.nodes)

    # exp_manager.max_time_per_run = Slurm time minus a margin, so the -last.ckpt gets written.
    slurm_time = args.time or parse_slurm_time_from_template(SLURM_SCRIPT)
    if slurm_time is not None:
        mins = max(parse_slurm_time_to_minutes(slurm_time) - MAX_TIME_MARGIN_MIN, MAX_TIME_FLOOR_MIN)
        env_overrides["MAX_TIME_PER_RUN"] = minutes_to_lightning_time(mins)

    submitted = xp_dir / f"{job_name}_submitted.slurm"
    materialize(SLURM_SCRIPT, submitted, job_name, str(xp_dir), sbatch_overrides, env_overrides)
    submitted.chmod(0o750)  # materialize() makes it 755: no access for others in the shared space
    check = subprocess.run(["bash", "-n", str(submitted)], capture_output=True, text=True)
    if check.returncode != 0:
        sys.exit(f"Materialized slurm file has syntax errors:\n{check.stderr}\nPath: {submitted}")

    print(f"Experiment:   {xp_dir}")
    print(f"Materialized: {submitted}")
    for k, v in env_overrides.items():
        print(f"  {k}={v}")
    for k, v in sbatch_overrides.items():
        print(f"  --{k}={v}")

    if args.dry_run:
        dryrun_path = xp_dir / f"{job_name}_dryrun.slurm"
        submitted.rename(dryrun_path)
        print(f"(dry run — not submitted; inspect: {dryrun_path})")
        return

    sbatch_cmd = ["sbatch", "--export=ALL"]
    if args.after is not None:
        sbatch_cmd.append(f"--dependency=afterok:{args.after}")
    sbatch_cmd.append(str(submitted))
    # Run from train_duplex/ so the job's working directory is the same as with a plain sbatch.
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True, cwd=HERE)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        submitted.unlink(missing_ok=True)
        sys.exit(result.returncode)

    final_path = xp_dir / f"{job_name}_{parse_job_id(result.stdout)}.slurm"
    submitted.rename(final_path)
    print(f"Snapshot: {final_path}")


if __name__ == "__main__":
    main()
