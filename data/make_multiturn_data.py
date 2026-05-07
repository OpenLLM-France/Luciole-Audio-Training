"""
make_multiturn_data.py
======================
Generate synthetic multi-turn conversations by concatenating single
conversations sampled from NeMo-format JSONL manifests, weighted by a
hierarchical YAML config.
"""

import argparse
import json
import os
import random
from pathlib import Path

import yaml
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("yaml", help="Input YAML config with hierarchical dataset weights")
    p.add_argument("--output_file", default="multiturn_data.jsonl", help="Output JSONL file")
    p.add_argument("--num_samples", type=int, default=100_000, help="Number of conversations to generate")
    p.add_argument("--max_duration", type=float, default=180.0, help="Maximum conversation duration in seconds")
    p.add_argument("--max_num_turns", type=int, default=10, help="Maximum assistant turns per conversation")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--data_root",
        default=os.environ.get("DATA_FOLDER"),
        help="Overrides ${DATA_FOLDER} in manifest paths (falls back to $DATA_FOLDER env var)",
    )
    p.add_argument(
        "--max_lines_per_jsonl",
        type=int,
        default=0,
        help="Cap the number of lines considered per JSONL manifest; 0 or negative means no limit",
    )
    p.add_argument(
        "--id_strategy",
        choices=["folders", "ids"],
        default="folders",
        help="How to build per-sample IDs: 'folders' = <manifest-folder-name><line-number> "
        "(e.g. Europarl107); 'ids' = use the sample's 'id' field from the JSONL",
    )
    p.add_argument(
        "--proba_lang_switch",
        type=float,
        default=0.05,
        help="Probability of switching language between two consecutive turns in a conversation",
    )
    p.add_argument(
        "--first_language",
        nargs="+",
        default=["fr", "en"],
        help="Restrict the language(s) allowed for the first turn of a new conversation",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print each generated conversation turn by turn (long text is truncated to 150 chars)",
    )
    p.add_argument(
        "--proba_avoid_instruct_repetition",
        type=float,
        default=0.8,
        help="Probability of dropping each redundant user text instruction (consecutive same-lang "
        "ASR/AST turns, or the generic 'Listen to the audio and answer the question.' prompt). "
        "0 disables the behaviour, 1 always drops.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# YAML → per-manifest probability
# ──────────────────────────────────────────────────────────────────────────────

_LANG_CODES = ("en", "fr", "es", "it", "de", "pt", "nl", "ar")


def _infer_lang_from_path(path):
    """Return a language code if a `/<code>/` segment is found in *path*, else None.

    Iterates path segments from leaf to root, so a language code closer to the
    file name takes precedence over one higher up the directory tree.
    """
    codes = set(_LANG_CODES)
    for part in reversed(path.split("/")):
        if part in codes:
            return part
    return None


def compute_manifest_probs(yaml_path, data_root):
    """Parse YAML and return a list of (manifest_path, probability, langs, task).

    Weights are normalised among siblings at each level, then multiplied down
    the hierarchy to give a probability per leaf manifest. All probabilities
    sum to 1. *langs* is a tuple of language codes: a single element for
    monolingual manifests, or both ``(source_lang, target_lang)`` for AST
    manifests (one of them is randomly chosen at sample time). *task* comes
    from the nearest ancestor's ``tags.task``. Raises if any leaf has no
    language assigned after those rules and a path-based fallback.
    """
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if isinstance(cfg, dict):
        root = cfg.get("input_cfg", [])
    elif isinstance(cfg, list):
        root = cfg
    else:
        raise ValueError(f"Unexpected YAML root type: {type(cfg).__name__}")

    manifests = []

    def _recurse(nodes, prob_prefix, lang, task):
        if not nodes:
            return
        weights = [float(n.get("weight", 1.0) or 1.0) for n in nodes]
        total = sum(weights)
        assert total > 0, f"Total weight must be positive, got {total}"
        for node, w in zip(nodes, weights):
            tags = node.get("tags") or {}
            if tags.get("lang"):
                node_lang = tags["lang"]
            elif tags.get("source_lang") and tags.get("target_lang"):
                node_lang = (tags["source_lang"], tags["target_lang"])
            else:
                node_lang = lang
            node_task = tags.get("task") or task
            if w == 1.0 and len(weights) > 1:
                print(
                    f"WARNING: missing weight for node with manifest {node.get('manifest_filepath', '[no manifest]')} (treating as 1.0)"
                )
            p_here = prob_prefix * (w / total)
            if "manifest_filepath" in node:
                path = node["manifest_filepath"]
                if data_root is not None:
                    path = path.replace("${oc.env:DATA_FOLDER}", data_root)
                node_lang_check = _infer_lang_from_path(path)
                if not node_lang:
                    node_lang = node_lang_check
                elif isinstance(node_lang, str) and node_lang_check and node_lang_check != node_lang:
                    raise RuntimeError(
                        f"WARNING: inferred language {node_lang_check} from path {path} "
                        f"does not match node language {node_lang}"
                    )
                if isinstance(node_lang, str):
                    langs_tuple = (node_lang,)
                elif isinstance(node_lang, tuple) and node_lang and all(isinstance(x, str) and x for x in node_lang):
                    langs_tuple = node_lang
                else:
                    raise ValueError(
                        f"Manifest has no language (lang/source_lang tag) in any ancestor "
                        f"and none could be inferred from the path: {path}"
                    )
                manifests.append((path, p_here, langs_tuple, node_task))
            elif "input_cfg" in node:
                _recurse(node["input_cfg"], p_here, node_lang, node_task)
            else:
                raise ValueError(f"Node has no 'manifest_filepath' or 'input_cfg': {node}")

    _recurse(root, 1.0, None, None)
    if not manifests:
        raise ValueError(f"No manifests found in {yaml_path}")
    return manifests


# ──────────────────────────────────────────────────────────────────────────────
# Truncated geometric PMF
# ──────────────────────────────────────────────────────────────────────────────


def truncated_geom_pmf(p, n_min, n_max):
    """Truncated geometric on {n_min, ..., n_max}: P(k) ∝ (1-p)^(k-1) · p."""
    if n_min > n_max:
        raise ValueError(f"n_min ({n_min}) > n_max ({n_max})")
    vals = list(range(n_min, n_max + 1))
    probs = [(1.0 - p) ** (k - 1) * p for k in vals]
    total = sum(probs)
    probs = [x / total for x in probs]
    return vals, probs


# ──────────────────────────────────────────────────────────────────────────────
# Manifest loading and sample validation
# ──────────────────────────────────────────────────────────────────────────────


def get_manifest_offsets(path, cache, max_lines=0):
    """Return the list of byte offsets for non-empty lines in *path*.

    The file is scanned once (on first access) to build the index; subsequent
    calls reuse the cached offsets. This avoids loading entire manifests into
    memory, which matters for very large JSONL files.

    If *max_lines* is > 0, only the first *max_lines* non-empty lines are
    indexed and the file scan stops there.
    """
    if path not in cache:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Manifest not found: {path}")
        offsets = []
        with open(path, "rb") as f:
            offset = 0
            for line in f:
                if line.strip():
                    offsets.append(offset)
                    if max_lines > 0 and len(offsets) >= max_lines:
                        break
                offset += len(line)
        if not offsets:
            raise ValueError(f"Empty manifest: {path}")
        cache[path] = offsets
    return cache[path]


def read_random_line(path, cache, max_lines=0):
    """Return (line_text, line_number) for a random non-empty line in *path*.

    *line_number* is 1-based over the non-empty lines that were indexed.
    """
    offsets = get_manifest_offsets(path, cache, max_lines=max_lines)
    idx = random.randrange(len(offsets))
    offset = offsets[idx]
    with open(path, "rb") as f:
        f.seek(offset)
        line = f.readline()
    return line.decode("utf-8"), idx + 1


def validate_conversation(sub_conv, manifest_path):
    """Sanity checks on a sampled conversation. Returns n_assistant_turns."""
    if not isinstance(sub_conv, list) or not sub_conv:
        raise ValueError(f"'conversations' missing or empty (manifest: {manifest_path})")
    for i, t in enumerate(sub_conv):
        if not isinstance(t, dict):
            raise ValueError(f"Turn {i} is not a dict (manifest: {manifest_path})")
        frm = t.get("from")
        if frm not in ("User", "Assistant"):
            raise ValueError(
                f"Turn {i} has invalid 'from'={frm!r}; expected 'User' or 'Assistant' (manifest: {manifest_path})"
            )
    n_assist = sum(1 for t in sub_conv if t["from"] == "Assistant")
    if n_assist == 0:
        raise ValueError(f"No Assistant turn in conversation (manifest: {manifest_path})")
    if sub_conv[0]["from"] != "User":
        raise ValueError(f"Conversation does not start with User (manifest: {manifest_path})")
    if sub_conv[-1]["from"] != "Assistant":
        raise ValueError(f"Conversation does not end with Assistant (manifest: {manifest_path})")
    return n_assist


def _format_turn(turn, max_chars=150):
    speaker = turn.get("from", "?")
    if turn.get("type") == "audio":
        return f"{speaker}: <audio>"
    s = str(turn.get("value", "")).replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\r")
    if len(s) > max_chars:
        s = s[:max_chars] + "…"
    return f"{speaker}: {s}"


def audio_user_duration(sub_conv):
    """Sum of 'duration' over turns with from='User' and type='audio'."""
    return sum(
        float(t.get("duration", 0) or 0) for t in sub_conv if t.get("from") == "User" and t.get("type") == "audio"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.data_root is None:
        raise ValueError("DATA_FOLDER env var is not set and --data_root was not provided")
    if args.max_num_turns < 2:
        raise ValueError(f"--max_num_turns must be >= 2 (got {args.max_num_turns})")

    manifests = compute_manifest_probs(args.yaml, args.data_root)
    paths = [m[0] for m in manifests]
    probs = [m[1] for m in manifests]
    langs = [m[2] for m in manifests]
    tasks = [m[3] for m in manifests]
    all_langs = sorted({lg for tup in langs for lg in tup})
    all_tasks = sorted({t for t in tasks if t})
    print(f"Loaded {len(paths)} manifests from {args.yaml} (languages: {all_langs}, tasks: {all_tasks})")

    same_lang_dist = {}
    other_lang_dist = {}
    for lg in all_langs:
        same_idxs = [i for i, tup in enumerate(langs) if lg in tup]
        other_idxs = [i for i, tup in enumerate(langs) if lg not in tup]
        same_lang_dist[lg] = (same_idxs, [probs[i] for i in same_idxs])
        other_lang_dist[lg] = (other_idxs, [probs[i] for i in other_idxs])

    first_lang_set = set(args.first_language)
    unknown = first_lang_set - set(all_langs)
    if unknown:
        raise ValueError(
            f"--first_language contains language(s) not present in the YAML: {sorted(unknown)} (available: {all_langs})"
        )
    first_lang_idxs = [i for i, tup in enumerate(langs) if any(lg in first_lang_set for lg in tup)]
    if not first_lang_idxs:
        raise ValueError(f"No manifest matches --first_language={args.first_language}")
    first_lang_probs = [probs[i] for i in first_lang_idxs]

    n_vals, n_probs = truncated_geom_pmf(p=0.1, n_min=2, n_max=args.max_num_turns)

    cache = {}
    turn_counts = []
    durations = []
    n_restarts = 0
    n_skipped_only_asr_ast = 0
    n_attempts = 0

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=args.num_samples, desc="Generating conversations")
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        while n_written < args.num_samples:
            # init
            conv = []
            ids = []
            durs = []
            # target number of turns
            target_n = random.choices(n_vals, weights=n_probs, k=1)[0]

            conv_assist_count = 0
            exceeded = False
            current_lang = None
            current_langs = None
            current_task = None
            conv_tasks = set()
            verbose_lines = []
            while conv_assist_count < target_n:
                pbar.set_postfix_str(f"collecting turn {min(conv_assist_count + 1, target_n)}/{target_n}")
                n_attempts += 1
                # sample manifest (first turn uses --first_language; turns 2+ are conditioned on current_lang)
                if current_lang is None:
                    sel_idx = random.choices(first_lang_idxs, weights=first_lang_probs, k=1)[0]
                elif not same_lang_dist.get(current_lang, ([], []))[0]:
                    sel_idx = random.choices(range(len(paths)), weights=probs, k=1)[0]
                else:
                    switch = random.random() < args.proba_lang_switch
                    pool_idxs, pool_probs = other_lang_dist[current_lang] if switch else same_lang_dist[current_lang]
                    if not pool_idxs:
                        pool_idxs, pool_probs = same_lang_dist[current_lang]
                    sel_idx = random.choices(pool_idxs, weights=pool_probs, k=1)[0]
                mpath = paths[sel_idx]
                sel_langs = langs[sel_idx]
                sel_lang_options = sel_langs
                # for the first turn, restrict AST source/target to languages in --first_language
                if current_lang is None:
                    allowed = [lg for lg in sel_lang_options if lg in first_lang_set]
                    if allowed:
                        sel_lang_options = tuple(allowed)
                sel_lang = sel_lang_options[0] if len(sel_lang_options) == 1 else random.choice(sel_lang_options)
                sel_task = tasks[sel_idx]
                # sample random line and validate
                line, line_no = read_random_line(mpath, cache, max_lines=args.max_lines_per_jsonl)
                sample = json.loads(line)
                sub_conv = sample.get("conversations")
                n_assist = validate_conversation(sub_conv, mpath)
                if args.id_strategy == "ids":
                    if "id" not in sample:
                        raise ValueError(f"Sample missing 'id' field (manifest: {mpath})")
                    turn_id = str(sample["id"])
                else:
                    turn_id = f"{Path(mpath).parent.name}{line_no}"
                # drop redundant user text instructions
                skipped_instructions = []
                if args.proba_avoid_instruct_repetition > 0:
                    if sel_task == "ast":
                        # for AST, the (source, target) pair must match the previous AST turn
                        same_lang_ctx = current_langs is not None and tuple(sel_langs) == tuple(current_langs)
                    else:
                        same_lang_ctx = sel_lang == current_lang
                    strip_all_text = sel_task in ("asr", "ast") and sel_task == current_task and same_lang_ctx
                    kept = []
                    for t in sub_conv:
                        is_instruction = (
                            t.get("from") == "User"
                            and t.get("type") == "text"
                            and (strip_all_text or t.get("value") == "Listen to the audio and answer the question.")
                        )
                        if is_instruction and random.random() < args.proba_avoid_instruct_repetition:
                            if args.verbose:
                                instr = str(t.get("value", "")).replace("\r\n", "\\n").replace("\n", "\\n")
                                if len(instr) > 150:
                                    instr = instr[:150] + "…"
                                skipped_instructions.append(instr)
                            continue
                        kept.append(t)
                    sub_conv = kept
                # compute duration
                d = audio_user_duration(sub_conv)
                # reject if total would exceed max_duration
                if sum(durs) + d > args.max_duration:
                    exceeded = True
                    break
                if args.verbose:
                    if current_lang is not None and sel_lang != current_lang:
                        verbose_lines.append("[language switch]")
                    dataset_name = Path(mpath).parent.name
                    verbose_lines.append(f"📜 {dataset_name} (task={sel_task}, lang={sel_lang})")
                    for instr in skipped_instructions:
                        verbose_lines.append(f"[skip instruction {{{instr}}}]")
                    for t in sub_conv:
                        verbose_lines.append(f"— {_format_turn(t)}")
                conv.extend(sub_conv)
                ids.append(turn_id)
                durs.append(d)
                conv_assist_count += n_assist
                conv_tasks.add(sel_task)
                current_lang = sel_lang
                current_langs = sel_langs
                current_task = sel_task

            if exceeded:
                if args.verbose:
                    pbar.write("[skipped a conversation because max_duration was exceeded]")
                n_restarts += 1
                continue

            # skip conversations that only involve asr and/or ast
            if conv_tasks and conv_tasks.issubset({"asr", "ast"}):
                if args.verbose:
                    pbar.write(f"[skipped a conversation because only {sorted(conv_tasks)} tasks]")
                n_skipped_only_asr_ast += 1
                continue

            # write and move on
            record = {"id": "--".join(ids), "conversations": conv}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
            turn_counts.append(conv_assist_count)
            durations.append(sum(durs))
            pbar.update(1)
            if args.verbose:
                for line in verbose_lines:
                    pbar.write(line)
                pbar.write("═" * 80)

    pbar.close()

    n = len(turn_counts)
    avg_turns = sum(turn_counts) / n if n else 0.0
    avg_dur = sum(durations) / n if n else 0.0
    print(f"\nGenerated {n} conversations → {out_path}")
    print(f"  avg assistant turns / conversation : {avg_turns:.2f}")
    print(f"  avg duration                       : {avg_dur:.2f} s")
    print(f"  min / max duration                 : {min(durations):.2f} / {max(durations):.2f} s")
    print(f"  min / max turns                    : {min(turn_counts)} / {max(turn_counts)}")
    print(f"  total sample draws                 : {n_attempts}")
    print(f"  restarts (over max_duration)       : {n_restarts}")
    print(f"  skipped (only asr/ast tasks)       : {n_skipped_only_asr_ast}")


if __name__ == "__main__":
    main()
