#!/usr/bin/env python3
# Copyright (c) LINAGORA / OpenLLM-France
# SPDX-License-Identifier: AGPL-3.0-only
"""
Export a NeMo-style ``input_cfg_*.yaml`` training mix into a HuggingFace-ready
dataset laid out like https://huggingface.co/datasets/nvidia/AF-Chat.

What it produces under ``OUTPUT_ROOT`` (e.g.
``/data-server/public_future/datasets/OpenLLM-France/Luciole-Audio-Training-Dataset``):

    data/<domain>/<task>/<Split>.jsonl   the conversations, text + metadata only,
                                         with audio paths rewritten to *relative* ones
    audio/<domain>/<task>/<Split>/...    hard links (default; or --copy / --symlinks) to the
                                         real audio files (NOT meant to be pushed to the Hub
                                         — too heavy; served separately e.g. over HTTP)
    README.md                            dataset card: the template at ``--card-dir/README.md``
                                         with ``configs:``/``language:``/``size_categories:``
                                         and the license + audio-availability tables injected
                                         on the fly from the license registry (--licenses)
    dataset_stats.csv                    per-split #examples / #audio / hours / license / hosted
    dataset_index.json                   split -> source dataset, license, provenance

Per-dataset redistribution is governed by ``--licenses`` (dataset_licenses.yaml): datasets
whose text is not redistributable (NoDerivatives, proprietary) are OMITTED; datasets whose
audio is not redistributable keep their text but their audio is not linked/hosted.

The conversation schema is kept identical to the source manifests (turns with
``from``/``value``/``type`` and ``duration``/``offset``); only the absolute audio
paths inside ``type: audio`` turns are rewritten to be relative to ``OUTPUT_ROOT``
(``audio/<domain>/<task>/<Split>/<sub-path>``).

Splits are one-per-leaf-manifest, named ``<DatasetName>_<language>``. Full datasets
are exported (no sub-sampling); the YAML ``weight`` fields are ignored.

Usage
-----
    python assets/export_hf_dataset.py \
        training/conf/data/v3/input_cfg_train.yaml \
        /data-server/public_future/datasets/OpenLLM-France/Luciole-Audio-Training-Dataset \
        --data-root /data-server/datasets/audio

    # quick test without touching anything:
    python assets/export_hf_dataset.py INPUT.yaml OUTPUT_ROOT --dry-run --limit-rows 5

    # also create a private HF dataset repo and upload everything but the audio:
    python assets/export_hf_dataset.py INPUT.yaml OUTPUT_ROOT \
        --hf_repo OpenLLM-France/Luciole-Audio-Training-Dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import errno
import os
import re
import shutil
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

import yaml

# Default location of the dataset-card sources (README.md template + LICENSES.md),
# copied/expanded into the output root at export time.
DEFAULT_CARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_dataset_card")

# Where the audio is served (the `audio/` tree, not on the Hub). Used to build README links.
AUDIO_BASE_URL = ("https://dl.labs.linagora.com/files/datasets/OpenLLM-France/"
                  "Luciole-Audio-Training-Dataset/audio/")

# --------------------------------------------------------------------------- #
# Taxonomy: how (task, sub_task) tags map to the two-level <domain>/<task>
# folder structure (speech/asr, speech/qa, music/qa, sound/qa, ...).
# --------------------------------------------------------------------------- #

# First path component (right after ``nemo/``) that only describes the task root.
TASK_ROOTS = {
    "asr", "ast", "question-answering", "summary", "music", "sounds",
    "misc", "temporal", "diarization",
}
# Components that describe the *format/method*, already captured by the task folder.
METHOD_TOKENS = {
    "audio-question-answering", "audio-captioning",
    "music-question-answering", "music-captioning",
}
# Purely structural directory names that carry no dataset identity.
STRUCTURAL_TOKENS = {"context", "content_level", "audios", "mixed", "chat", "summary"}
# Language codes that may appear as path components.
LANG_TOKENS = {"en", "fr", "de", "es", "it", "nl", "pt", "ar", "multilang", "mixed"}


def map_domain_task(tags: dict) -> tuple[str, str]:
    """Return (domain, task_dir) for the two-level folder from the merged tags."""
    task = tags.get("task")
    sub_task = tags.get("sub_task")

    if task == "music":
        # music_qa -> qa, music_captioning -> captioning
        leaf = (sub_task or "qa").replace("music_", "")
        return "music", leaf
    if task == "sound":
        leaf = (sub_task or "qa").replace("sound_", "")
        return "sound", leaf
    if task in ("asr", "ast", "qa"):
        # Honor an explicit sub_task (e.g. qa + summarization -> speech/summarization).
        return "speech", (sub_task or task)
    if task == "task_switching":
        return "speech", "task_switching"
    if task == "other" or (task is None and sub_task):
        return "speech", sub_task or "misc"
    if task == "aqa":
        # Should have been overridden by a sound/music subgroup; fall back gracefully.
        return ("sound" if sub_task and "sound" in sub_task else "speech"), (sub_task or "qa")
    # Unknown -> keep something deterministic.
    return "speech", (task or sub_task or "misc")


def detect_language(tags: dict, parts: list[str]) -> str | None:
    """Language suffix for the split name (tags first, then path)."""
    src, tgt = tags.get("source_lang"), tags.get("target_lang")
    if src and tgt:
        return f"{src}-{tgt}"
    if tags.get("lang"):
        return tags["lang"]
    for p in parts:
        if p in LANG_TOKENS:
            return p
    return None


def meaningful_parts(rel_parts: list[str]) -> list[str]:
    """Drop task-root / method / structural / language tokens, keep dataset identity.

    ``rel_parts`` is the manifest path relative to the data root, e.g.
    ``["nemo", "asr", "fr", "context", "FLEURS", "train.jsonl"]``.
    """
    middle = rel_parts[:-1]  # drop filename
    if middle and middle[0] == "nemo":
        middle = middle[1:]
    kept = []
    for i, p in enumerate(middle):
        if i == 0 and p in TASK_ROOTS:
            continue
        if p in METHOD_TOKENS or p.startswith("qa_audio"):
            continue
        if p in STRUCTURAL_TOKENS or p in LANG_TOKENS:
            continue
        kept.append(p)
    return kept or [middle[-1]] if middle else ["dataset"]


def sanitize(name: str) -> str:
    """HF split/config names: keep [A-Za-z0-9_] only."""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def config_name(domain: str, task: str) -> str:
    """HF config identifier ``<domain>.<task>`` (e.g. ``speech.qa``). A dot is a
    valid config-name char (unlike ``/``, which the datasets library forbids as a
    cache-path separator); each part is sanitized to [A-Za-z0-9_]."""
    return f"{sanitize(domain)}.{sanitize(task)}"


def normalize_base(base: str) -> str:
    """Strip a trailing translation-direction marker (e.g. CommonVoiceEN2FR -> CommonVoice)."""
    m = re.match(r"^(.*?)[_-]?([A-Za-z]{2}2[A-Za-z]{2})$", base)
    return m.group(1) if (m and m.group(1)) else base


def compose_split_name(raw_base: str, lang: str | None) -> str:
    """Build a split name: normalized base + language suffix, without duplicating the language.

    Single languages use ``_xx`` and translation pairs ``_xx_yy`` (underscores only, so the
    name stays a valid HF split name). If the base already ends with the language (e.g. SIFT's
    ``common_voice_fr``), it is not repeated.
    """
    base = sanitize(normalize_base(raw_base))
    if not lang:
        return base
    lang_s = sanitize(lang)  # "en-fr" -> "en_fr", "fr" -> "fr"
    if not lang_s or base.lower() == lang_s or base.lower().endswith("_" + lang_s):
        return base
    return f"{base}_{lang_s}"


# --------------------------------------------------------------------------- #
# YAML parsing: walk the tree, inherit tags, collect leaf manifests.
# --------------------------------------------------------------------------- #


@dataclass
class Leaf:
    manifest: str               # absolute path on disk
    rel_parts: list[str]        # path relative to data root, split into components
    tags: dict                  # merged tags (ancestors + leaf)


def resolve_env(value: str, env: dict) -> str:
    """Substitute ${oc.env:VAR} (and ${oc.env:VAR,default}) references."""
    def repl(m):
        body = m.group(1)
        name, _, default = body.partition(",")
        name = name.strip()
        if name in env:
            return env[name]
        if name in os.environ:
            return os.environ[name]
        if default:
            return default.strip()
        raise KeyError(f"Environment variable {name!r} is not set (referenced in YAML)")

    return re.sub(r"\$\{oc\.env:([^}]*)\}", repl, value)


def collect_leaves(node, inherited: dict, env: dict, data_root: str, out: list[Leaf]):
    """Recursively walk an input_cfg node, accumulating tags down to the leaves."""
    if isinstance(node, list):
        for item in node:
            collect_leaves(item, inherited, env, data_root, out)
        return
    if not isinstance(node, dict):
        return

    tags = dict(inherited)
    tags.update(node.get("tags") or {})

    ntype = node.get("type")
    if ntype == "group" or "input_cfg" in node:
        collect_leaves(node.get("input_cfg", []), tags, env, data_root, out)
    elif ntype == "multimodal_conversation" or "manifest_filepath" in node:
        manifest = resolve_env(node["manifest_filepath"], env)
        rel = os.path.relpath(manifest, data_root) if manifest.startswith(data_root) else manifest
        out.append(Leaf(manifest=manifest, rel_parts=rel.split(os.sep), tags=tags))


# --------------------------------------------------------------------------- #
# Split assignment with collision-resistant naming.
# --------------------------------------------------------------------------- #


@dataclass
class Split:
    domain: str
    task: str
    name: str
    language: str | None
    leaves: list[Leaf] = field(default_factory=list)

    @property
    def key(self):
        return (self.domain, self.task, self.name)

    @property
    def rel_dir(self):
        return os.path.join(self.domain, self.task)


def assign_splits(leaves: list[Leaf]) -> "OrderedDict[tuple, Split]":
    """Group leaves into named splits (one per source directory + language).

    Manifests merge into the same split only when they live in the same source
    directory (e.g. YouTubeFr's ``train_split0..6`` shards, MusicQA's subsets).
    Distinct source directories always become distinct splits, even if their
    folder name coincides; such name clashes are resolved by prepending more
    path context, then a numeric suffix as a last resort.
    """
    # Primary grouping: (domain, task, source_dir, language) -> merged leaves.
    groups: "OrderedDict[tuple, list[Leaf]]" = OrderedDict()
    for leaf in leaves:
        domain, task = map_domain_task(leaf.tags)
        lang = detect_language(leaf.tags, leaf.rel_parts)
        source_dir = os.sep.join(leaf.rel_parts[:-1])
        groups.setdefault((domain, task, source_dir, lang), []).append(leaf)

    # used[(domain, task)] -> set of split names already taken in that task folder
    used: dict[tuple, set] = {}
    splits: "OrderedDict[tuple, Split]" = OrderedDict()

    for (domain, task, _source_dir, lang), gleaves in groups.items():
        parts = meaningful_parts(gleaves[0].rel_parts)
        taken = used.setdefault((domain, task), set())

        # Start from the most specific component, prepend more on collision.
        depth = 1
        while True:
            name = compose_split_name("_".join(parts[-depth:]), lang)
            if name not in taken:
                break
            if depth < len(parts):
                depth += 1
            else:
                n = 2
                while f"{name}_{n}" in taken:
                    n += 1
                name = f"{name}_{n}"
                break

        taken.add(name)
        splits[(domain, task, name)] = Split(domain, task, name, lang, gleaves)

    return splits


# --------------------------------------------------------------------------- #
# Audio path handling.
# --------------------------------------------------------------------------- #


def iter_audio_values(record: dict):
    """Yield each ``type: audio`` turn dict of a record (so callers can mutate value)."""
    for turn in record.get("conversations", []):
        if isinstance(turn, dict) and turn.get("type") == "audio" and turn.get("value"):
            yield turn


def normalize_language(lang) -> str:
    """Record language: keep codes ('fr') and translation pairs ('en-fr'); 'mixed' -> 'multilingual'."""
    low = str(lang).lower()
    return "multilingual" if low in ("mixed", "multilang", "multilingual") else str(lang)


def fmt_duration(sec) -> str:
    """Compact audio duration for the card. Minutes are shown only when they matter
    (< 10 h); above that they are noise, so round to the nearest hour:
    '8801h16' -> '8801h', '45h50' -> '46h', but '1h30' and '8h45' are kept as-is."""
    sec = int(round(float(sec or 0)))
    h, rem = divmod(sec, 3600)
    m = rem // 60
    if h >= 10:
        return f"{round(sec / 3600)}h"
    if h:
        return f"{h}h{m:02d}" if m else f"{h}h"
    if m:
        return f"{m}min"
    return f"{sec}s"


def disp_task(domain: str, task: str) -> str:
    """Table label for a task: drop the dominant 'speech/' prefix, render as 'domain.task'."""
    t = f"{domain}/{task}"
    t = t[len("speech/"):] if t.startswith("speech/") else t
    return t.replace("/", ".")


# OpenLLM-France curation marks, shown per (task, language) line in the Content column:
#   🎙️ = audio synthesized by OpenLLM-France; ✍️ = text (Q&A / translations / captions)
#   generated or modified by OpenLLM-France.
CURATED_MARKS = {"audio": "🎙️", "text": "✍️"}


def normalize_curated(spec):
    """Normalize a registry `curated` value into a list of rules ``{task, kind, lang?}``.

    Accepts: None -> []; a scalar ``"text"``/``"audio"`` -> one wildcard rule covering
    the whole dataset; a ``{task: kind}`` mapping; or an explicit list of rule dicts.
    In a rule, ``task == "*"`` matches any task and a missing/None ``lang`` matches any
    language, so curation can be pinned to a single task line (e.g. only ``temporal``) or
    even a single language of a task (e.g. MusicCaps ``music.captioning`` in French only)."""
    if not spec:
        return []
    if isinstance(spec, str):
        return [{"task": "*", "kind": spec}]
    if isinstance(spec, dict):
        return [{"task": k, "kind": v} for k, v in spec.items()]
    return list(spec)


def curated_kind(rules, task_label, lang):
    """Return 'audio'/'text' if a curation rule matches this (task, language) line, else None."""
    for r in rules or []:
        if r.get("task") in ("*", task_label) and r.get("lang") in (None, lang):
            return r.get("kind")
    return None


def load_stats_csv(path: str, data_root: str) -> dict:
    """Precomputed per-manifest stats: abspath -> (num_samples, duration_sec).

    Keyed on ``raw_manifest_path`` from a metadata.csv; ``${oc.env:DATA_FOLDER}`` and
    ``$DATA_FOLDER`` are resolved against ``data_root``."""
    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    table = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("raw_manifest_path") or "").strip()
            if not raw:
                continue
            p = raw.replace("${oc.env:DATA_FOLDER}", data_root).replace("$DATA_FOLDER", data_root)
            table[os.path.normpath(os.path.abspath(p))] = (
                int(_f(row.get("num_samples"))), _f(row.get("total_duration_sec")))
    return table


def split_stats(present, table):
    """Sum precomputed (samples, duration_sec) over a split's manifests, or None if the table
    is absent or does not cover *every* manifest of the split (so the caller can recompute)."""
    if not table:
        return None
    s, d = 0, 0.0
    for m in present:
        stat = table.get(os.path.normpath(os.path.abspath(m)))
        if stat is None:
            return None
        s += stat[0]
        d += stat[1]
    return (s, d)


def uniform_record(rec: dict, language: str, fallback_id: str) -> dict:
    """Project any source record onto the uniform schema: id, conversations, language, extra.

    Everything that is not id/conversations/language is folded into ``extra`` (a JSON string;
    the contents of a ``meta`` dict are surfaced into it). Empty extra serializes to "{}".
    """
    rid = rec.get("id")
    rid = str(rid) if rid not in (None, "") else fallback_id
    extra = {k: v for k, v in rec.items() if k not in ("id", "conversations", "language")}
    meta = extra.pop("meta", None)
    if isinstance(meta, dict):
        extra = {**meta, **extra}  # surface meta contents; other keys win on collision
    elif meta is not None:
        extra["meta"] = meta
    return {
        "id": rid,
        "conversations": rec.get("conversations", []),
        "language": language,
        "extra": json.dumps(extra, ensure_ascii=False),
    }


def common_audio_prefix(audio_paths: list[str]) -> str:
    """Longest common *directory* of all audio files in a split."""
    dirs = sorted({os.path.dirname(p) for p in audio_paths})
    if not dirs:
        return ""
    if len(dirs) == 1:
        return dirs[0]
    try:
        return os.path.commonpath(dirs)
    except ValueError:
        return ""  # different roots (shouldn't happen)


# --------------------------------------------------------------------------- #
# Main export.
# --------------------------------------------------------------------------- #


def _make_link(src: str, link_path: str, args):
    """Materialise ``src`` at ``link_path`` as a hard link (default), copy, or symlink.

    Hard links are self-contained (the content is reachable at the published path
    regardless of what's mounted), which is what an HTTP file server needs; they
    require the same filesystem AND permission to link the source (you own it or it
    is writable). When that's not possible, --copy is the robust HTTP-servable
    fallback and --symlinks the cheapest.
    """
    if args.copy:
        shutil.copy2(src, link_path)
        return
    if args.symlinks:
        target = os.path.relpath(src, os.path.dirname(link_path)) if args.relative_symlinks else src
        os.symlink(target, link_path)
        return
    try:
        os.link(src, link_path)
    except OSError as e:
        if e.errno in (errno.EXDEV, errno.EPERM, errno.EACCES):
            reason = ("the source and the output root are on different filesystems"
                      if e.errno == errno.EXDEV
                      else "you don't own the source files / they aren't writable by you")
            raise SystemExit(
                f"Hard-link failed ({os.strerror(e.errno)}) — {reason}:\n"
                f"  src:  {src}\n  dest: {link_path}\n\n"
                "Hard links need the same filesystem AND permission to link the source. Fixes:\n"
                f"  - run as the files' owner, e.g.:  sudo -u ubuntu python {os.path.basename(sys.argv[0])} ...\n"
                "  - or  --copy      (real copies; uses disk but always works and is HTTP-servable)\n"
                "  - or  --symlinks  (cheapest, but may not be served by HTTP servers / containers)"
            ) from e
        raise


def _configs_blocks(configs: "OrderedDict[str, list]") -> list:
    """Turn the {config_name: [(split, path), …]} mapping into HF ``configs:`` blocks."""
    return [
        {
            "config_name": cfg,
            "data_files": [{"split": sp, "path": path} for sp, path in entries],
        }
        for cfg, entries in configs.items()
    ]


# Explicit record schema, declared in the card's ``dataset_info`` so ``datasets`` uses a known
# type for every config instead of the dataset-server's ``Json`` inference (which older/most
# ``datasets`` versions cannot load). ``conversations`` is a list<struct>; turns that omit
# ``duration``/``offset`` are read as ``null`` — the stored JSONL is left untouched.
_RECORD_FEATURES = [
    {"name": "id", "dtype": "string"},
    {"name": "conversations", "list": [
        {"name": "from", "dtype": "string"},
        {"name": "value", "dtype": "string"},
        {"name": "type", "dtype": "string"},
        {"name": "duration", "dtype": "float64"},
        {"name": "offset", "dtype": "float64"},
    ]},
    {"name": "language", "dtype": "string"},
    {"name": "extra", "dtype": "string"},
]


def _dataset_info_blocks(configs: "OrderedDict[str, list]") -> list:
    """One ``dataset_info`` entry per config, all sharing the fixed record schema."""
    return [{"config_name": cfg, "features": _RECORD_FEATURES} for cfg in configs]


def _size_category(n: int) -> str:
    """HF ``size_categories`` bucket for ``n`` examples."""
    buckets = [
        (1e3, "n<1K"), (1e4, "1K<n<10K"), (1e5, "10K<n<100K"), (1e6, "100K<n<1M"),
        (1e7, "1M<n<10M"), (1e8, "10M<n<100M"), (1e9, "100M<n<1B"),
    ]
    for hi, label in buckets:
        if n < hi:
            return label
    return "n>1B"


def hf_prepare(repo_id):
    """Verify HF auth/connection and create the (private) dataset repo if missing.

    Called at the START of the run so a bad token / network / name fails fast, before
    the long export. An existing repo is kept as-is (privacy/contents untouched).
    Returns an authenticated ``HfApi``.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("--hf_repo needs huggingface_hub:  pip install huggingface_hub")

    api = HfApi()
    try:
        who = api.whoami()  # fails fast if not logged in / no token
    except Exception as e:
        raise SystemExit(
            f"Not authenticated to the HF Hub ({e}).\n"
            "Run `huggingface-cli login` or set HF_TOKEN before using --hf_repo."
        ) from e
    try:
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        raise SystemExit(f"Could not create/access HF dataset repo '{repo_id}': {e}") from e
    print(f"HF connection OK (user: {who.get('name', '?')}). "
          f"Repo ready: https://huggingface.co/datasets/{repo_id}", flush=True)
    return api


def hf_upload(api, output_root, repo_id):
    """Upload everything but the audio: README + side files it links, and all data/**.jsonl."""
    print(f"\nUploading to HF dataset repo '{repo_id}'…", flush=True)

    # Top-level files (README.md, dataset_stats.csv, dataset_index.json, …) — never dirs,
    # so the audio/ and data/ folders are skipped here.
    top_files = sorted(e for e in os.listdir(output_root)
                       if os.path.isfile(os.path.join(output_root, e)))
    for entry in top_files:
        api.upload_file(path_or_fileobj=os.path.join(output_root, entry), path_in_repo=entry,
                        repo_id=repo_id, repo_type="dataset")
    print(f"  uploaded: {', '.join(top_files)}")

    # The conversations — only *.jsonl under data/ (explicitly never audio/).
    data_dir = os.path.join(output_root, "data")
    if os.path.isdir(data_dir):
        api.upload_folder(folder_path=data_dir, path_in_repo="data", repo_id=repo_id,
                          repo_type="dataset", allow_patterns=["*.jsonl"])
        print("  uploaded: data/ (*.jsonl)")
    print(f"  done -> https://huggingface.co/datasets/{repo_id}")


def load_registry(path):
    """Load the license/redistribution registry. Returns (entries, default, collection_license)."""
    data = yaml.safe_load(open(path)) if os.path.isfile(path) else {}
    entries = data.get("datasets", []) or []
    for e in entries:
        e["_m"] = str(e.get("match", "")).lower()
    default = data.get("default") or {
        "name": "UNMATCHED", "license": "Unknown",
        "redistribute_text": True, "redistribute_audio": True,
        "audio_source_url": "", "notes": "",
    }
    return entries, default, data.get("collection_license")


def policy_for(rel_path, entries, default):
    """Longest case-insensitive ``match`` substring of ``rel_path`` wins; else default."""
    p = rel_path.lower()
    best = None
    for e in entries:
        if e["_m"] and e["_m"] in p and (best is None or len(e["_m"]) > len(best["_m"])):
            best = e
    return best or default


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def write_dataset_card(output_root, configs, stats_rows, datasets_used, collection_license, card_dir):
    """Build README.md from the template: inject front-matter fields + dataset/audio tables.

    ``datasets_used`` maps a source-dataset name -> dict(license, text_ok, audio_ok, url,
    notes, splits[]). ``LICENSES.md`` is NOT copied; its content is generated here.
    """
    tmpl = os.path.join(card_dir, "README.md")
    if not os.path.isfile(tmpl):
        print(f"  [WARN] no README.md template in {card_dir} — README not generated", file=sys.stderr)
        return
    text = open(tmpl).read()
    if text.startswith("---"):
        _, front_matter, body = text.split("---", 2)
        meta = yaml.safe_load(front_matter) or {}
    else:
        meta, body = {}, "\n" + text

    # ---- front-matter fields constructed on the fly ----
    # HF only accepts ISO 639 codes (2-3 letters) or "multilingual"; map translation
    # pairs ("ar-en") and "mixed" to "multilingual".
    def _norm_lang(v):
        v = str(v).lower()
        return v if re.fullmatch(r"[a-z]{2,3}", v) else "multilingual"

    # Order languages by occurrence (total published examples), not alphabetically.
    comp_counts, fm_counts = Counter(), Counter()
    for r in stats_rows:
        lv = str(r["language"])
        fm_counts[_norm_lang(lv)] += r["examples"]
        for code in re.split(r"[^a-z]+", lv.lower()):
            if code:
                comp_counts[code] += r["examples"]

    # Front-matter order: single codes by their component count (so 'en'/'de'/… agree with the
    # table column), 'multilingual' by the volume of translation-pair/mixed data.
    def _fm_key(lang):
        return fm_counts["multilingual"] if lang == "multilingual" else comp_counts[lang]

    langs = sorted(fm_counts, key=lambda lang: (-_fm_key(lang), lang))
    if langs:
        meta["language"] = langs
    meta["size_categories"] = [_size_category(sum(r["examples"] for r in stats_rows))]
    if collection_license:
        meta["license"] = collection_license.lower()
    meta["configs"] = _configs_blocks(configs)
    meta["dataset_info"] = _dataset_info_blocks(configs)

    # ---- generated markdown blocks ----
    def nbsp(s):
        # Non-breaking spaces keep the phrase on one line, but a "(" may still wrap to the next.
        return s.replace(" ", " ").replace(" (", " (")

    def status(d):
        if not d["text_ok"]:
            return nbsp("✗ excluded (external only)")
        return nbsp("✓ text + audio" if d["audio_ok"] else "✓ text only")

    def short_license(s):
        # Table cells want the canonical token(s) only; the audit stores long
        # "(justification)" clauses in license_text/audio — drop them (and any
        # trailing ";"-separated caveat) so the column stays narrow. Full text
        # is preserved in the registry / index / notes.
        s = re.sub(r"\s*\([^)]*\)", "", str(s))       # strip "(…)" clauses
        s = re.split(r"\s+[—–]\s+", s)[0]              # drop em/en-dash caveat
        s = s.split(";")[0]                            # keep first alt before ";"
        return re.sub(r"\s+", " ", s).strip(" /") or "Unknown"

    def license_str(d, short=False):
        lt, la = d["license_text"], d["license_audio"]
        if short:
            lt, la = short_license(lt), short_license(la)
            if lt == la:
                return lt
            # separate lines so the column wraps instead of running wide
            return f"{lt} *(text)*<br>{la} *(audio)*"
        if lt == la:
            return lt
        return f"{lt} (text), {la} (audio)"

    def content_cell(d):
        # One "Content" column: total samples · duration, then a per-(task, language)
        # breakdown. Single group collapses to one line; groups with unknown counts
        # (e.g. excluded datasets exported without --stats-csv) show the label only.
        # OpenLLM-France curation marks (🎙️/✍️) hug the specific task line(s) concerned.
        groups = d.get("groups") or {}
        if not groups:
            return "—"
        rules = d.get("curated_rules")
        items = sorted(groups.items(), key=lambda kv: (-kv[1]["samples"], kv[0]))
        known = [g for _, g in items if g["known"]]
        total_s = sum(g["samples"] for g in known)
        total_d = sum(g["duration"] for g in known)

        def qty(g, word=False):
            n = f"{g['samples']:,} samples" if word else f"{g['samples']:,}"
            return f"{n} · {fmt_duration(g['duration'])}" if g["duration"] else n

        def mark(task, lang):
            k = curated_kind(rules, task, lang)
            return f"{CURATED_MARKS[k]} " if k else ""

        if len(items) == 1:
            (task, lang), g = items[0]
            m = mark(task, lang)
            return f"{qty(g, word=True)} — {m}{task}, {lang}" if g["known"] else f"{m}{task}, {lang}"

        head = f"**{total_s:,} samples · {fmt_duration(total_d)}**" if total_d else f"**{total_s:,} samples**"
        lines = [head]
        for (task, lang), g in items:
            m = mark(task, lang)
            lines.append(f"• {m}{task}, {lang} — {qty(g)}" if g["known"] else f"• {m}{task}, {lang}")
        return "<br>".join(lines)

    def hosted_url(d):
        dirs = sorted(d["audio_dirs"])
        if not dirs:
            return AUDIO_BASE_URL
        common = os.path.commonpath(dirs) if len(dirs) > 1 else dirs[0]
        sub = common[len("audio/"):] if common.startswith("audio/") else ""
        return AUDIO_BASE_URL + sub

    def audio_cell(d):
        if d["text_ok"] and d["audio_ok"]:
            return f"[hosted]({hosted_url(d)})"
        return f"[source]({d['url']})" if d["url"] else "—"

    def dataset_label(name, d, always_bold=False):
        # Curation is flagged per task line in the Content column (see content_cell),
        # not on the whole dataset — so the dataset name carries no mark here.
        return f"**{name}**" if always_bold else name

    rows = sorted(datasets_used.items(), key=lambda kv: kv[0].lower())
    table_rows = [
        [dataset_label(name, d), license_str(d, short=True), status(d), audio_cell(d), content_cell(d)]
        for name, d in rows
    ]
    instr_note = ("The **instruction prompts** (the user-turn wording that states each task) were written "
                  "by OpenLLM-France for every task **except question answering** (`qa`).\n\n")
    marks_legend = ("Marks in the **Content** column flag data produced by OpenLLM-France, on the specific "
                    "task line(s) concerned: **🎙️ = audio synthesized by OpenLLM-France** · **✍️ = text "
                    "(questions/answers, translations, or captions) generated or modified by OpenLLM-France** "
                    "— see each dataset's note below for details.\n\n") if any(d.get("curated_rules") for _, d in rows) else ""
    legend = instr_note + marks_legend
    dataset_table = legend + _md_table(
        ["Dataset", "License", "In this release", "Audio", "Content"], table_rows
    )

    def render_source(s):
        # Typed sources: "[audio] Clotho (Freesound) — CC-BY-NC-4.0 — <url>".
        # The "both" type (text + audio) is the default, so it is left implicit;
        # only "audio"/"text" are shown as a "[type]" tag.
        bits = []
        if s.get("type") and s["type"] != "both":
            bits.append(f"[{s['type']}]")
        label = " — ".join(p for p in (s.get("name") or s.get("desc"), s.get("license")) if p)
        if label:
            bits.append(label)
        prefix = " ".join(bits)
        return f"{prefix + ' — ' if prefix else ''}{s.get('url', '')}"

    def sources_sublist(d):
        # Explicit multi-source list, or fall back to the single audio_source_url.
        srcs = d.get("sources") or ([{"url": d["url"]}] if d["url"] else [])
        if not srcs:
            return ""
        if len(srcs) == 1:
            return f"  \n  Source: {render_source(srcs[0])}"
        return "  \n  Sources:\n" + "\n".join(f"    - {render_source(s)}" for s in srcs)

    def bullet(name, d):
        link = (f"[hosted]({hosted_url(d)})" if (d["text_ok"] and d["audio_ok"])
                else (f"[source]({d['url']})" if d["url"] else ""))
        line = f"- {dataset_label(name, d, always_bold=True)} — {license_str(d)}" + (f" — {link}" if link else "")
        if d.get("notes"):
            line += f"  \n  {d['notes']}"
        return line + sources_sublist(d)

    audio_link = f"[`{AUDIO_BASE_URL}`]({AUDIO_BASE_URL})"
    groups = {"hosted": [], "external": [], "excluded": []}
    for name, d in datasets_used.items():
        key = "excluded" if not d["text_ok"] else ("hosted" if d["audio_ok"] else "external")
        groups[key].append(name)
    for g in groups.values():
        g.sort(key=str.lower)

    def group_block(names):
        return "\n".join(bullet(n, datasets_used[n]) for n in names) or "_none_"

    avail = []
    # Part 1 — text published + audio hosted.
    avail.append("### Text and audio\n\n")
    avail.append(f"Both the conversations and the audio are published. The audio is hosted at "
                 f"{audio_link}; each record's audio path is relative to that folder.\n\n")
    avail.append(group_block(groups["hosted"]))
    # Part 2 — text published, audio not (fetch from source).
    avail.append("\n\n### Text only\n\n")
    avail.append("The conversations are published here, but the audio is **not** redistributable "
                 "by us — download it from the original source(s):\n\n")
    avail.append(group_block(groups["external"]))
    avail.append("\n\n*For YouTube-sourced audio* (e.g. MusicCaps, YouTubeFr): the original "
                 "video/clip ID is preserved in the audio file name (and in each record's "
                 "`extra` field). Fetch the audio from YouTube with the source dataset's tooling "
                 "or a downloader such as [`yt-dlp`](https://github.com/yt-dlp/yt-dlp), then cut "
                 "each segment using the `offset` / `duration` of the corresponding audio turn.")
    # Part 3 — neither text nor audio published.
    avail.append("\n\n### Not included\n\n")
    avail.append("Neither the audio nor the conversations are published (the license forbids "
                 "redistribution) — get the whole dataset from the source(s):\n\n")
    avail.append(group_block(groups["excluded"]))
    audio_availability = "".join(avail)

    # Attribution: every CC-BY* source (text and/or audio) we actually publish must be
    # credited (excluded datasets are not distributed here).
    incl = {n: d for n, d in datasets_used.items() if d["text_ok"]}

    def needs_by(d):
        return ("cc-by" in d["license_text"].lower()
                or (d["audio_ok"] and "cc-by" in d["license_audio"].lower()))

    by = sorted((kv for kv in incl.items() if needs_by(kv[1])), key=lambda kv: kv[0].lower())
    other_attr = sorted((kv for kv in incl.items()
                         if kv[1]["license_text"].lower().startswith(("apache", "mit", "cdla"))),
                        key=lambda kv: kv[0].lower())
    attr = ["The following datasets carry a **CC-BY / attribution** license on their text "
            "and/or audio — you **must credit** them (and cite their papers) when you use this "
            "dataset:\n"]
    attr.append("\n".join(f"- **{n}** ({license_str(d)}) — [source]({d['url']})" for n, d in by)
                or "_none_")
    if other_attr:
        attr.append("\n\nThese also require keeping their text license/attribution notice "
                    "(Apache-2.0 / MIT / CDLA):\n")
        attr.append("\n".join(f"- **{n}** ({d['license_text']}) — [source]({d['url']})"
                              for n, d in other_attr))
    attribution = "".join(attr)

    body = body.replace("<!-- AUTOGEN:DATASET_TABLE -->", dataset_table)
    body = body.replace("<!-- AUTOGEN:AUDIO_AVAILABILITY -->", audio_availability)
    body = body.replace("<!-- AUTOGEN:ATTRIBUTION -->", attribution)
    # Drop the maintainer-only template note so it doesn't ship in the published README.
    body = re.sub(r"[ \t]*<!--\s*TEMPLATE-NOTE:.*?-->[ \t]*\n?", "", body, flags=re.S)

    front_matter_out = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    with open(os.path.join(output_root, "README.md"), "w") as f:
        f.write("---\n" + front_matter_out + "---" + body)


def export(args):
    env = {"DATA_FOLDER": args.data_root}
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        env[k] = v

    with open(args.input_yaml) as f:
        cfg = yaml.safe_load(f)

    leaves: list[Leaf] = []
    collect_leaves(cfg, {}, env, args.data_root, leaves)
    print(f"Found {len(leaves)} leaf manifests in {args.input_yaml}")

    splits = assign_splits(leaves)
    if args.filter:
        rx = re.compile(args.filter)
        splits = OrderedDict(
            (k, v) for k, v in splits.items() if rx.search(f"{v.domain}/{v.task}/{v.name}")
        )
        print(f"Filter {args.filter!r} -> {len(splits)} splits kept", flush=True)
    print(f"Grouped into {len(splits)} splits across "
          f"{len({(s.domain, s.task) for s in splits.values()})} <domain>/<task> configs",
          flush=True)

    if args.plan:
        # Fast structural preview: do not read manifest contents at all.
        last_cfg = None
        for (domain, task, name), split in splits.items():
            header = f"{domain}/{task}"
            if header != last_cfg:
                print(f"\n[{header}]  (config: {config_name(domain, task)})")
                last_cfg = header
            srcs = ", ".join(os.path.relpath(lf.manifest, args.data_root) for lf in split.leaves)
            print(f"  - {name}  <-  {srcs}")
        print(f"\nPlan: {len(splits)} splits. (--plan: no manifests read, nothing written)", flush=True)
        return

    # Test the HF connection and create the repo up front, so a bad token / name fails
    # fast — before the long export — rather than after it.
    hf_api = hf_prepare(args.hf_repo) if (args.hf_repo and not args.dry_run) else None

    data_dir = os.path.join(args.output_root, "data")
    if not args.dry_run:
        os.makedirs(data_dir, exist_ok=True)
        if not args.nolinks:
            os.makedirs(os.path.join(args.output_root, "audio"), exist_ok=True)

    entries, default, collection_license = load_registry(args.licenses)

    stats_table = load_stats_csv(args.stats_csv, args.data_root) if args.stats_csv else None
    if stats_table is not None:
        print(f"Loaded precomputed stats for {len(stats_table)} manifests from {args.stats_csv}")

    stats_rows = []
    index = {}
    configs: "OrderedDict[str, list]" = OrderedDict()
    datasets_used: "OrderedDict[str, dict]" = OrderedDict()
    n_missing_manifests = 0
    n_missing_audio = 0
    n_excluded = 0
    n_stats_seen = 0
    n_stats_hit = 0

    for (domain, task, name), split in splits.items():
        manifests = [lf.manifest for lf in split.leaves]
        present = [m for m in manifests if os.path.isfile(m)]
        for m in manifests:
            if not os.path.isfile(m):
                n_missing_manifests += 1
                print(f"  [WARN] missing manifest: {m}", file=sys.stderr)
        if not present:
            continue

        # ---- License / redistribution policy for this split ----
        rel_manifest = os.path.relpath(split.leaves[0].manifest, args.data_root)
        policy = policy_for(rel_manifest, entries, default)
        if policy is default:
            print(f"  [WARN] no license registry match for {rel_manifest} — using default", file=sys.stderr)
        text_ok = policy.get("redistribute_text", True)
        audio_ok = policy.get("redistribute_audio", True)
        # license_text / license_audio fall back to a single `license` when not split out.
        lic_text = policy.get("license_text", policy.get("license", "Unknown"))
        lic_audio = policy.get("license_audio", policy.get("license", "Unknown"))
        du = datasets_used.setdefault(policy["name"], {
            "license_text": lic_text, "license_audio": lic_audio,
            "text_ok": text_ok, "audio_ok": audio_ok,
            "url": policy.get("audio_source_url", ""),
            "notes": policy.get("notes", ""),
            "sources": policy.get("sources", []),
            "curated_rules": normalize_curated(policy.get("curated")),  # [{task, kind, lang?}]
            "splits": [], "tasks": set(), "langs": set(), "audio_dirs": set(), "groups": {},
        })
        du["splits"].append(f"{domain}/{task}/{name}")
        du["tasks"].add(f"{domain}/{task}")
        for code in re.split(r"[^a-z]+", (split.language or "").lower()):
            if code:
                du["langs"].add(code)

        # Per-(task, language) content group for the card. Precomputed stats (--stats-csv)
        # let us fill counts even for excluded datasets (whose manifests we don't read).
        csv_split = split_stats(present, stats_table)
        if stats_table is not None:
            n_stats_seen += 1
            if csv_split:
                n_stats_hit += 1
            elif args.verbose:
                miss = [os.path.relpath(m, args.data_root) for m in present
                        if os.path.normpath(os.path.abspath(m)) not in stats_table]
                print(f"  [stats-csv] no match for {domain}/{task}/{name}: {miss}", file=sys.stderr)
        grp = du["groups"].setdefault(
            (disp_task(domain, task), normalize_language(split.language) or "?"),
            {"samples": 0, "duration": 0.0, "known": False})

        if not text_ok:
            n_excluded += 1
            if csv_split:
                grp["samples"] += csv_split[0]
                grp["duration"] += csv_split[1]
                grp["known"] = True
            print(f"  {domain}/{task}/{name}: EXCLUDED (text: {lic_text}) — {policy['name']}")
            continue

        # ---- Pass 1: collect every audio absolute path to compute a common prefix.
        audio_paths = []
        seen = 0
        for m in present:
            with open(m) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    for turn in iter_audio_values(rec):
                        audio_paths.append(turn["value"])
                    seen += 1
                    if args.limit_rows and seen >= args.limit_rows:
                        break
            if args.limit_rows and seen >= args.limit_rows:
                break
        prefix = common_audio_prefix(audio_paths)

        # ---- Pass 2: write jsonl with relative paths + create symlinks.
        rel_audio_root = os.path.join("audio", split.rel_dir, name)
        if audio_ok:
            du["audio_dirs"].add(rel_audio_root)
        out_jsonl = os.path.join(data_dir, split.rel_dir, f"{name}.jsonl")
        rec_language = normalize_language(split.language)
        n_rows = 0
        n_audio = 0
        total_dur = 0.0
        linked = set()

        if not args.dry_run:
            os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
        writer = open(out_jsonl, "w") if not args.dry_run else None
        try:
            for m in present:
                with open(m) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        for turn in iter_audio_values(rec):
                            src = turn["value"]
                            rel_in_split = os.path.relpath(src, prefix) if prefix else os.path.basename(src)
                            rel_ref = os.path.join(rel_audio_root, rel_in_split)
                            turn["value"] = rel_ref
                            n_audio += 1
                            if isinstance(turn.get("duration"), (int, float)):
                                total_dur += turn["duration"]
                            # link the file (once per file) unless linking is disabled
                            # or this dataset's audio is not redistributable
                            if rel_ref not in linked:
                                linked.add(rel_ref)
                                if not os.path.exists(src):
                                    n_missing_audio += 1
                                    if args.verbose:
                                        print(f"  [WARN] missing audio: {src}", file=sys.stderr)
                                elif audio_ok and not args.dry_run and not args.nolinks:
                                    link_path = os.path.join(args.output_root, rel_ref)
                                    os.makedirs(os.path.dirname(link_path), exist_ok=True)
                                    if not os.path.lexists(link_path):
                                        _make_link(src, link_path, args)
                        if writer:
                            out_rec = uniform_record(rec, rec_language, f"{name}_{n_rows}")
                            writer.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                        n_rows += 1
                        if args.limit_rows and n_rows >= args.limit_rows:
                            break
                if args.limit_rows and n_rows >= args.limit_rows:
                    break
        finally:
            if writer:
                writer.close()

        cfg_name = config_name(domain, task)
        rel_jsonl = os.path.relpath(out_jsonl, args.output_root)
        configs.setdefault(cfg_name, []).append((name, rel_jsonl))

        # Sample count / duration: precomputed CSV wins (authoritative, unaffected by
        # --limit-rows); otherwise use what we just counted while writing.
        samples = csv_split[0] if csv_split else n_rows
        dur_sec = csv_split[1] if csv_split else total_dur
        grp["samples"] += samples
        grp["duration"] += dur_sec
        grp["known"] = True

        stats_rows.append({
            "domain": domain, "task": task, "split": name,
            "language": split.language or "",
            "examples": samples, "audio_files": len(linked),
            "audio_refs": n_audio, "duration_hours": round(dur_sec / 3600.0, 2),
            "audio_hosted": "yes" if audio_ok else "no",
            "license_text": du["license_text"], "license_audio": du["license_audio"],
            "source_dataset": policy["name"], "data_file": rel_jsonl,
        })
        index[f"{domain}/{task}/{name}"] = {
            "config": cfg_name,
            "data_file": rel_jsonl,                          # relative to the repo root
            "audio_dir": rel_audio_root if audio_ok else None,  # relative; None when not hosted
            "language": split.language,
            "examples": samples,
            "audio_files": len(linked),
            "audio_hosted": audio_ok,
            "license_text": du["license_text"],
            "license_audio": du["license_audio"],
            "source_dataset": policy["name"],
            "audio_source_url": du["url"],
            "sources": du["sources"] or ([{"url": du["url"]}] if du["url"] else []),
        }
        print(f"  {domain}/{task}/{name}: {samples} ex, {len(linked)} audio "
              f"({round(dur_sec/3600,1)} h){'' if audio_ok else '  [audio NOT hosted]'}")

    # ---- Write side files (stats, index) + the dataset card (README + LICENSES).
    if not args.dry_run:
        with open(os.path.join(args.output_root, "dataset_stats.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()) if stats_rows else
                               ["domain", "task", "split", "language", "examples", "audio_files",
                                "audio_refs", "duration_hours", "audio_hosted", "license_text",
                                "license_audio", "source_dataset", "data_file"])
            w.writeheader()
            w.writerows(stats_rows)

        with open(os.path.join(args.output_root, "dataset_index.json"), "w") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        # README.md (front-matter + configs + license/audio tables, all built on the fly).
        write_dataset_card(args.output_root, configs, stats_rows, datasets_used,
                           collection_license, args.card_dir)

    print(f"\nDone. {len(stats_rows)} splits written, {n_excluded} excluded (non-redistributable).")
    if stats_table is not None:
        print(f"  stats-csv: {n_stats_hit}/{n_stats_seen} splits sourced from {args.stats_csv} "
              f"({n_stats_seen - n_stats_hit} recomputed).")
    if n_missing_manifests:
        print(f"  {n_missing_manifests} manifest(s) listed in the YAML were missing on disk.")
    if n_missing_audio:
        print(f"  {n_missing_audio} audio file(s) referenced were missing (not linked).")
    if args.dry_run:
        print("  (dry-run: nothing was written)")
        return
    if args.nolinks:
        print(f"  Output: {args.output_root}  (--nolinks: jsonl + card written, no audio linked)")
    else:
        print(f"  Output: {args.output_root}")

    if hf_api is not None:
        hf_upload(hf_api, args.output_root, args.hf_repo)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_yaml", help="input_cfg_*.yaml training mix")
    p.add_argument("output_root", help="HF dataset root (e.g. .../Luciole-Audio-Training-Dataset)")
    p.add_argument("--data-root", default=os.environ.get("DATA_FOLDER", "/data-server/datasets/audio"),
                   help="Value substituted for ${oc.env:DATA_FOLDER} (default: $DATA_FOLDER or /data-server/datasets/audio)")
    p.add_argument("--env", action="append", metavar="VAR=VALUE",
                   help="Extra ${oc.env:VAR} substitutions (repeatable)")
    p.add_argument("--hf_repo", default=None, metavar="OWNER/NAME",
                   help="After export, create this HF dataset repo (private, if it does not "
                        "exist) and upload README + side files + all jsonl (never the audio)")
    p.add_argument("--symlinks", action="store_true",
                   help="Use symbolic links for audio instead of hard links (default: hard links, "
                        "so the files are self-contained and can be served directly over HTTP)")
    p.add_argument("--copy", action="store_true",
                   help="Copy audio files instead of linking (robust when you can't hard-link "
                        "files you don't own; uses disk space but is HTTP-servable)")
    p.add_argument("--relative-symlinks", action="store_true",
                   help="With --symlinks, make them relative to the link location instead of "
                        "absolute targets (ignored for hard links)")
    p.add_argument("--nolinks", action="store_true",
                   help="Do not create any audio links — only write the jsonl + dataset card "
                        "(fast, for testing)")
    p.add_argument("--card-dir", default=DEFAULT_CARD_DIR,
                   help=f"Folder holding the README.md template (default: {DEFAULT_CARD_DIR})")
    p.add_argument("--licenses", default=os.path.join(DEFAULT_CARD_DIR, "dataset_licenses.yaml"),
                   help="License/redistribution registry that decides which text/audio is "
                        "published (default: <card-dir>/dataset_licenses.yaml)")
    p.add_argument("--limit-rows", type=int, default=0, help="Cap rows per split (for testing)")
    p.add_argument("--stats-csv", default=None, metavar="PATH",
                   help="Precomputed metadata.csv (num_samples / total_duration_sec keyed by "
                        "raw_manifest_path) used for sample counts & durations instead of "
                        "recomputing them. Any manifest not fully covered falls back to recompute.")
    p.add_argument("--dry-run", action="store_true",
                   help="Read manifests and report counts, but write nothing")
    p.add_argument("--plan", action="store_true",
                   help="Fast: print the split/config structure only (no manifest reads)")
    p.add_argument("--filter", default=None,
                   help="Only process splits whose '<domain>/<task>/<name>' matches this regex")
    p.add_argument("--verbose", action="store_true", help="Warn on every missing audio file")
    args = p.parse_args()
    export(args)


if __name__ == "__main__":
    main()
