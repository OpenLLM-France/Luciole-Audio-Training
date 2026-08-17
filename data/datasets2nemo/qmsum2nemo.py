import argparse
import json
import logging
import os
import re
import subprocess
from pathlib import Path

from tqdm import tqdm
from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "QMSum"
QMSUM_REPO = "https://github.com/Yale-LILY/QMSum.git"
# QMSum layout: data/{ALL,Academic,Committee,Product}/{train,val,test}/*.json
SUBSETS = ["ALL", "Academic", "Committee", "Product"]
SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}


def sanitize(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)


def ensure_repo(repo_dir: Path) -> None:
    if repo_dir.exists() and (repo_dir / "data").is_dir():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cloning {QMSUM_REPO} → {repo_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", QMSUM_REPO, str(repo_dir)],
        check=True,
    )


def format_transcript(meeting_transcripts: list[dict]) -> str:
    lines = []
    for t in meeting_transcripts:
        speaker = t.get("speaker", "").strip()
        content = t.get("content", "").strip()
        lines.append(f"{speaker}: {content}" if speaker else content)
    return "\n".join(lines)


def extract_span(meeting_transcripts: list[dict], spans) -> str:
    """spans is a list like [['0', '15'], ['37', '42']] (inclusive ranges of turn indices)."""
    if not spans:
        return ""
    picked = []
    n = len(meeting_transcripts)
    for span in spans:
        try:
            start, end = int(span[0]), int(span[1])
        except (ValueError, TypeError, IndexError):
            continue
        start = max(0, start)
        end = min(n - 1, end)
        for i in range(start, end + 1):
            picked.append(meeting_transcripts[i])
    return format_transcript(picked)


def build_rows(sample: dict, file_uid: str, split: str, subset: str):
    transcript_text = format_transcript(sample.get("meeting_transcripts", []))
    topics = sample.get("topic_list") or []
    topic_text = "\n".join(
        f"- {t.get('topic', '')}" for t in topics if t.get("topic")
    )

    rows = []

    def make_row(query_id: str, query: str, answer: str, query_kind: str,
                 relevant_span_text: str | None, relevant_span_raw):
        if not query or not answer:
            return None
        user_value = (
            "Meeting transcript:\n"
            f"{transcript_text}\n\n"
            f"Query: {query}"
        )
        user_turn = NemoTurn(role="User", value=user_value, turn_type="text")
        assistant_turn = NemoTurn(role="Assistant", value=answer, turn_type="text")
        uid = sanitize(f"{file_uid}_{query_kind}_{query_id}")
        return NemoDatasetRow(
            id=uid,
            dataset_name=DATASET_NAME,
            split=split,
            language="en",
            turns=[user_turn, assistant_turn],
            custom_metadata={
                "source_file": file_uid,
                "subset": subset,
                "query_kind": query_kind,
                "query": query,
                "answer": answer,
                "topic_list": topic_text,
                "relevant_text_span": relevant_span_raw,
                "relevant_text": relevant_span_text,
            },
        )

    for i, q in enumerate(sample.get("general_query_list") or []):
        r = make_row(
            query_id=str(i),
            query=q.get("query", ""),
            answer=q.get("answer", ""),
            query_kind="general",
            relevant_span_text=None,
            relevant_span_raw=None,
        )
        if r is not None:
            rows.append(r)

    for i, q in enumerate(sample.get("specific_query_list") or []):
        spans = q.get("relevant_text_span")
        span_text = extract_span(sample.get("meeting_transcripts", []), spans)
        r = make_row(
            query_id=str(i),
            query=q.get("query", ""),
            answer=q.get("answer", ""),
            query_kind="specific",
            relevant_span_text=span_text or None,
            relevant_span_raw=spans,
        )
        if r is not None:
            rows.append(r)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Convert QMSum to NeMo manifest")
    parser.add_argument("--repo-dir", type=str, default=None,
                        help="Path to an existing QMSum git clone. If missing, the repo is cloned here.")
    parser.add_argument("--subset", type=str, default="ALL", choices=SUBSETS,
                        help="Which QMSum subset folder under data/ to convert.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing manifest .jsonl files instead of skipping them.")
    parser.add_argument("--raw-manifest-path", type=str, default=None,
                        help="Override raw manifest output folder (with custom_metadata).")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Override the metadata-free manifest output folder.")
    args = parser.parse_args()

    data_dir = os.environ.get("DATA_FOLDER")
    if args.repo_dir is None:
        if not data_dir:
            parser.error("--repo-dir not set and DATA_FOLDER env var is missing")
        args.repo_dir = f"{data_dir}/raw/summary/en/qmsum/QMSum"
    if args.raw_manifest_path is None:
        args.raw_manifest_path = f"{data_dir}/raw/summary/en/qmsum"
    if args.manifest_path is None:
        args.manifest_path = f"{data_dir}/nemo/summary/en/qmsum"

    repo_dir = Path(args.repo_dir)
    ensure_repo(repo_dir)

    subset_dir = repo_dir / "data" / args.subset
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"QMSum subset not found: {subset_dir}")

    raw_manifest_root = Path(args.raw_manifest_path) / args.subset
    manifest_root = Path(args.manifest_path) / args.subset
    raw_manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    for src_split, out_split in SPLIT_MAP.items():
        split_dir = subset_dir / src_split
        if not split_dir.is_dir():
            logger.info(f"Split dir missing, skipping: {split_dir}")
            continue

        manifest_file_raw = raw_manifest_root / f"{out_split}.jsonl"
        manifest_file = manifest_root / f"{out_split}.jsonl"
        if manifest_file_raw.exists() and manifest_file.exists() and not args.force:
            logger.info(f"[{out_split}] manifests already exist, skipping: {manifest_file_raw}, {manifest_file}")
            continue

        files = sorted(split_dir.glob("*.json"))
        logger.info(f"[{out_split}] {len(files)} source files in {split_dir}")

        out_raw = NemoDataset(name=DATASET_NAME)
        out = NemoDataset(name=DATASET_NAME)

        for fp in tqdm(files, desc=out_split):
            file_uid = sanitize(fp.stem)
            try:
                with fp.open("r", encoding="utf-8") as f:
                    sample = json.load(f)
            except Exception as e:
                logger.warning(f"[{out_split}] {fp}: failed to load ({e}), skipping")
                continue

            rows = build_rows(sample, file_uid=file_uid, split=out_split, subset=args.subset)
            for r in rows:
                out_raw.append(r)
                stripped = NemoDatasetRow(
                    id=r.id,
                    dataset_name=r.dataset_name,
                    split=r.split,
                    language=r.language,
                    turns=r.turns,
                )
                out.append(stripped)

        out_raw.save(manifest_file_raw)
        out.save(manifest_file)
        logger.info(f"[{out_split}] wrote {len(out_raw)} rows → {manifest_file_raw}")
        logger.info(f"[{out_split}] wrote {len(out)} rows → {manifest_file}")


if __name__ == "__main__":
    main()
