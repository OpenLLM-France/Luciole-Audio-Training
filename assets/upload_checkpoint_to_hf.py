import argparse
import os
from pathlib import Path

# Disable xet storage backend which can cause "MerkleDB Shard error" on some systems
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import huggingface_hub  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Upload a checkpoint folder (hf_checkpoints/step_XXXXX) to a HuggingFace Hub model repo")
    parser.add_argument("checkpoint_dir", type=Path, help="Path to the checkpoint folder to upload, e.g. .../hf_checkpoints/step_100000")
    parser.add_argument("repo_id", help="Target HuggingFace repo id, e.g. OpenLLM-France/Luciole-8B")
    parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist yet")
    parser.add_argument("--revision", default=None, help="Branch to upload to (default: main)")
    parser.add_argument("--path-in-repo", default=None, help="Subfolder in the repo to upload into (default: repo root)")
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit message (default: 'Upload checkpoint <checkpoint_dir.name>')",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if not _is_hf_logged_in():
        huggingface_hub.login()

    api = huggingface_hub.HfApi()

    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    commit_message = args.commit_message or f"Upload checkpoint {checkpoint_dir.name}"

    print(f"Uploading {checkpoint_dir} to {args.repo_id} (revision={args.revision or 'main'})")

    future = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(checkpoint_dir),
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=commit_message,
        run_as_future=False,
    )

    print(f"Done: https://huggingface.co/{args.repo_id}")
    return future


def _is_hf_logged_in():
    try:
        huggingface_hub.HfApi().whoami()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
