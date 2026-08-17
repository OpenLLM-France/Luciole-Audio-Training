"""Turn any single-turn QA manifest into multi-turn conversations on one audio.

No LLM: the questions and answers are the manifest's own, only the grouping and
the connectives are added.

Many of our QA manifests already ask SEVERAL questions about the SAME audio, one
per line -- clotho_aqa 4.9 per audio on 100% of its clips, Nvidia LongAudio/meld
24.9, slue-sqa5 3.2 on 59% of them. Emitted as separate samples they teach
single-turn QA; grouped, they are the shape our mix lacks: the audio arrives once
and every later turn is text-only, so answering requires still having it.

What it generates
-----------------
Three files, one per --mode:

* ``sequential`` -- one question per turn, audio in turn 1 only.
* ``compound``   -- 2-3 questions bundled in a single turn, answered in one go.
* ``mixed``      -- 40% sequential, 40% hybrid (one bundled turn among single
  ones), 20% compound alone.

Questions are cut into DISJOINT chunks per audio, so an audio carrying 25
questions yields several conversations that never repeat a question. Chunks are
2-5 questions (TURN_DIST); with --multi-audio-rate a conversation chains chunks
from 2-3 different audios, each introduced explicitly ("Passons a un autre
extrait."). Unlike the translation generator there is no elliptic switch: a
question about one sound has no annotated answer for another.

Proportions are inherited from template_commonvoice_multiturn (TURN_DIST,
CONT_TURN_DIST, EXTRA_CLIP_DIST, bundle position, 40/40/20) so both generators
produce the same conversation statistics.

Usage
-----
    python template_qa_multiturn.py \\
        --manifests $DATA_FOLDER/nemo/sounds/audio-question-answering/en/context/clotho_aqa/train.jsonl \\
        --lang en --output-dir out/ --mode all
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from template_commonvoice_multiturn import (  # noqa: E402
    ALL_MODES, BUNDLE_SIZES, CONTINUATION_EXPLICIT, CONT_TURN_DIST,
    EXTRA_CLIP_DIST, MIXED_COMPOUND_OF_REST, MIXED_HYBRID_SHARE, TURN_DIST,
    assistant, count_audios, draw_compound_position, draw_turns, oxford_join,
    user_audio, user_text,
)

# Light connectives in front of a follow-up question. Half are empty: the questions
# are self-contained, and a marker on every turn would be learned as required.
#
# The last group re-anchors the question on the audio explicitly. A mix of both is
# the point: a follow-up that says nothing forces the model to keep the audio live
# by itself, which is the capability we are short of, but never re-anchoring is
# just as much an artefact -- a real user does say "and on the same clip". Turns
# that mention the audio go from 37% to 48% with these, and none of that 37% was
# coming from the connective before.
FOLLOWUP_LEADS = {
    "fr": [""] * 8 + ["Autre question : ", "Et aussi : ", "Une autre : ", "Ensuite : "]
          + ["Toujours sur cet extrait : ", "Sur le même enregistrement : ",
             "Encore une chose à propos de cet audio : "],
    "en": [""] * 8 + ["Another question: ", "And also: ", "One more: ", "Next: "]
          + ["Still on this clip: ", "About the same recording: ",
             "One more thing about this audio: "],
}

# Compound frames, split on whether they commit to a count. A bare `{tasks}` sits
# in both pools: questions concatenated with no frame at all is how anyone
# actually asks two things at once, and with a frame on every bundled turn the
# frame becomes the cue for "answer several things" instead of the questions.
COMPOUND_FRAMES_ANY = {
    "fr": ["{tasks}", "{tasks}",
           "Sur cet extrait, réponds à ces questions : {tasks}",
           "J'ai plusieurs questions sur cet audio : {tasks}",
           "À propos de cet enregistrement : {tasks}",
           "Écoute et réponds : {tasks}",
           "Quelques questions sur ce son : {tasks}",
           "Dis-moi, pour cet enregistrement : {tasks}",
           "J'aimerais savoir, sur cet audio : {tasks}",
           "Plusieurs choses à propos de cet extrait : {tasks}"],
    "en": ["{tasks}", "{tasks}",
           "On this clip, answer these questions: {tasks}",
           "I have several questions about this audio: {tasks}",
           "About this recording: {tasks}",
           "Listen and answer: {tasks}",
           "A few questions about this sound: {tasks}",
           "Tell me, for this recording: {tasks}",
           "I would like to know, about this audio: {tasks}",
           "Several things about this clip: {tasks}"],
}
COMPOUND_FRAMES_TWO = {
    "fr": ["Deux questions sur cet extrait : {tasks}",
           "Réponds à ces deux questions sur cet audio : {tasks}",
           "Deux choses à propos de ce son : {tasks}",
           "J'ai deux questions : {tasks}"],
    "en": ["Two questions about this clip: {tasks}",
           "Answer these two questions about this audio: {tasks}",
           "Two things about this sound: {tasks}",
           "I have two questions: {tasks}"],
}

# Some manifests pack a LIST of valid answers into one pipe-delimited string --
# MLS word2sentence does it on 13% of its lines, when the target word occurs in
# several sentences. Copied through verbatim, the bar becomes a format the model
# learns to emit. Enumerated here instead.
#
# This only applies when the parts are ALTERNATIVES. Where the bar separates
# different FIELDS (SLUE answer_with_time is "three|10.2s", an answer and its
# timestamp), enumerating them would be nonsense: use --multi-answer keep, and
# split those into separate turns instead.
MULTI_ANSWER = {
    "fr": ["Il y en a {n} : {items}.", "On en trouve {n} : {items}.",
           "{n} passages correspondent : {items}."],
    "en": ["There are {n}: {items}.", "{n} of them match: {items}.",
           "{n} passages match: {items}."],
}

ORDINALS = {
    "fr": ["Pour la première", "pour la deuxième", "pour la troisième", "pour la quatrième"],
    "en": ["For the first", "for the second", "for the third", "for the fourth"],
}
RESTATE = {"fr": "Réponse : ", "en": "Answer: "}

ANSWER_STYLES = ("numbered", "restated", "prose")


def expand_multi_answer(answer, lang, policy):
    """Turn a pipe-delimited list of alternative answers into one readable answer."""
    if policy == "keep" or "|" not in answer:
        return answer
    parts = [p.strip() for p in answer.split("|") if p.strip()]
    if len(parts) < 2:
        return parts[0] if parts else answer
    if policy == "first":
        return parts[0]
    quote = "« {} »" if lang == "fr" else '"{}"'
    items = oxford_join([quote.format(p.rstrip(".")) for p in parts], lang)
    return random.choice(MULTI_ANSWER[lang]).format(n=len(parts), items=items)


def capitalise(text):
    """Uppercase the first letter, leaving the rest alone."""
    return text[0].upper() + text[1:] if text else text


def uncapitalise(text):
    """Lowercase a leading capital unless it opens an acronym or a proper noun.

    Only used to graft an answer into a longer sentence. The second-letter test
    keeps "DVD" and "Paris" intact while lowering "Yes" and "The".
    """
    if len(text) > 1 and text[1].isupper():
        return text
    return text[0].lower() + text[1:] if text else text


def render_bundle_answer(pairs, lang, style):
    """Compose one answer covering 2-3 questions, in one of three shapes."""
    if style == "numbered":
        return "\n".join(f"{i}. {a}" for i, (_, a) in enumerate(pairs, 1))
    if style == "restated":
        # The question is restated on its own line: appending the answer straight
        # after it gives "Is this a storm?: yes".
        return "\n".join(f"{q}\n{RESTATE[lang]}{a}" for q, a in pairs)
    # Ordinal prose. The answers become clauses of one sentence, so they are
    # lowercased and joined with semicolons: "For the first, Yes, there is a
    # vehicle" is not a sentence, and neither is joining them with "and".
    return "; ".join(f"{ORDINALS[lang][i]}, {uncapitalise(a.rstrip('.'))}"
                     for i, (_, a) in enumerate(pairs)) + "."


def bundle_turn(pairs, lang, audio=None, trace=None):
    """One user turn asking 2-3 questions, plus its composed answer.

    Falls back to a plain single turn on one question: a continuation segment can
    legitimately be one question long, and "I have several questions" in front of
    exactly one of them is the kind of incoherence a model reproduces happily.
    """
    if len(pairs) < 2:
        return single_turn(pairs[0][0], pairs[0][1], lang, True, audio, trace)
    if trace is not None:
        trace.append(len(pairs))
    # Several manifests store questions uncapitalised, which shows as soon as one
    # is not the first thing in the turn.
    pairs = [(capitalise(q), a) for q, a in pairs]
    tasks = " ".join(q for q, _ in pairs)
    pool = list(COMPOUND_FRAMES_ANY[lang])
    if len(pairs) == 2:
        pool += COMPOUND_FRAMES_TWO[lang]
    turns = [user_text(random.choice(pool).format(tasks=tasks))]
    if audio is not None:
        turns.append(audio)
    turns.append(assistant(render_bundle_answer(pairs, lang, random.choice(ANSWER_STYLES))))
    return turns


def single_turn(question, answer, lang, first, audio=None, trace=None):
    if trace is not None:
        trace.append(1)
    lead = "" if first else random.choice(FOLLOWUP_LEADS[lang])
    turns = [user_text(lead + capitalise(question))]
    if audio is not None:
        turns.append(audio)
    turns.append(assistant(answer))
    return turns


def audio_turn(record):
    turn = user_audio(record["audio"], record["duration"])
    if record["offset"]:
        turn["offset"] = record["offset"]
    return turn


def build_sequential(chunk, lang, audio, max_turns=None, trace=None):
    """One question per turn; the audio only rides the first."""
    pairs = chunk["pairs"][:max_turns] if max_turns else chunk["pairs"]
    turns = []
    for i, (question, answer) in enumerate(pairs):
        turns += single_turn(question, answer, lang, i == 0,
                             audio if i == 0 else None, trace)
    return turns


def build_compound(chunk, lang, audio, max_turns=None, trace=None):
    """A single bundled turn and nothing else; questions past the bundle are dropped."""
    return bundle_turn(chunk["pairs"][:random.choice(BUNDLE_SIZES)], lang, audio, trace)


def build_hybrid(chunk, lang, audio, max_turns=None, trace=None):
    """Single-question turns with exactly one bundled turn among them.

    The bundle's position is drawn over ALL turns, so "several questions at once"
    is never learned as meaning "first turn".
    """
    pairs = list(chunk["pairs"])
    if max_turns:
        pairs = pairs[:max_turns + max(BUNDLE_SIZES) - 1]
    bundle_size = 2 if len(pairs) < 4 else random.choice(BUNDLE_SIZES)
    n_turns = len(pairs) - bundle_size + 1
    if max_turns:
        n_turns = min(n_turns, max_turns)
    if n_turns < 2:
        return None
    at = draw_compound_position(n_turns)

    turns, cursor = [], 0
    for i in range(n_turns):
        take = bundle_size if i == at else 1
        group = pairs[cursor:cursor + take]
        cursor += take
        clip = audio if i == 0 else None
        if take == 1:
            turns += single_turn(group[0][0], group[0][1], lang, i == 0, clip, trace)
        else:
            turns += bundle_turn(group, lang, clip, trace)
    return turns


def build_mixed(chunk, lang, audio, max_turns=None, trace=None,
                p_hybrid=MIXED_HYBRID_SHARE):
    """Draw one shape per conversation, so one file carries all three."""
    if len(chunk["pairs"]) >= 3 and random.random() < p_hybrid:
        turns = build_hybrid(chunk, lang, audio, max_turns, trace)
        if turns is not None:
            return turns
        del trace[:]
    if random.random() < MIXED_COMPOUND_OF_REST:
        return build_compound(chunk, lang, audio, max_turns, trace)
    return build_sequential(chunk, lang, audio, max_turns, trace)


BUILDERS = {"sequential": build_sequential, "compound": build_compound,
            "mixed": build_mixed}


def load_manifest(path, lang, multi_answer):
    """Group a QA manifest by audio: (audio, offset, duration) -> [(question, answer)].

    Duplicate questions on one audio are dropped: several manifests ship the same
    question under different prompt phrasings, and two of those in one
    conversation would be a contradiction waiting to happen.
    """
    groups = defaultdict(list)
    seen = defaultdict(set)
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            audio = offset = duration = question = answer = None
            for turn in record.get("conversations") or []:
                if turn.get("type") == "audio":
                    audio = turn.get("value")
                    duration = turn.get("duration")
                    offset = turn.get("offset") or 0.0
                elif turn.get("from") == "User":
                    question = (turn.get("value") or "").strip()
                elif turn.get("from") == "Assistant":
                    answer = (turn.get("value") or "").strip()
            if not (audio and question and answer):
                skipped += 1
                continue
            key = (audio, offset, duration)
            if question in seen[key]:
                continue
            seen[key].add(question)
            groups[key].append((question, expand_multi_answer(answer, lang, multi_answer)))
    return groups, skipped


# Largest chunk TURN_DIST can draw; a redistributed tail must not exceed it.
MAX_CHUNK = max(n for n, _ in TURN_DIST)


def draw_chunk(buckets, take):
    """Take `take` questions, each from a different answer bucket when possible.

    Clotho-AQA answers are 57% "yes" and "no", so a plain shuffle put every
    question of a conversation on the same answer 11% of the time -- teaching the
    answer rather than the listening. Only 2.1% of its audios are genuinely
    single-answer, so nearly all of that was the cutting, not the corpus.
    Draining the largest buckets first spreads the dominant answer instead of
    leaving it to pile up in the last chunk.
    """
    chunk = []
    for bucket in sorted(buckets, key=len, reverse=True):
        if len(chunk) == take:
            break
        if bucket:
            chunk.append(bucket.pop())
    while len(chunk) < take:
        largest = max(buckets, key=len, default=None)
        if not largest:
            break
        chunk.append(largest.pop())
    buckets[:] = [b for b in buckets if b]
    return chunk


def cut_chunks(groups, min_turns):
    """Cut each audio's questions into disjoint chunks of TURN_DIST length.

    Disjoint, so an audio with 25 questions yields several conversations without
    ever repeating one. Leftovers too short to open a conversation are kept aside
    as continuation material, where a single question is legitimate.
    """
    chunks, singles = [], []
    for (audio, offset, duration), pairs in groups.items():
        buckets = defaultdict(list)
        for pair in pairs:
            buckets[pair[1].strip().lower()].append(pair)
        for bucket in buckets.values():
            random.shuffle(bucket)
        buckets = [b for b in buckets.values()]
        record = {"audio": audio, "offset": offset, "duration": duration}
        built = []
        while buckets:
            left = sum(len(b) for b in buckets)
            take = draw_turns(left, TURN_DIST) or left
            group = draw_chunk(buckets, take)
            if not group:
                break
            random.shuffle(group)
            built.append(group)

        # Spreading the answers empties the small buckets first, so what is left
        # at the end of an audio is the dominant answer on its own: a whole
        # conversation answering "yes" every turn, which teaches the answer
        # rather than the listening. Fixed by SWAPPING with another group of the
        # same audio, never by moving or dropping -- group sizes stay put, so
        # every question still reaches a conversation.
        def answers(group):
            return {p[1].strip().lower() for p in group}

        for group in built:
            if len(group) < 2 or len(answers(group)) > 1:
                continue
            mine = next(iter(answers(group)))
            for other in built:
                if other is group or len(other) < 2:
                    continue
                j = next((k for k, p in enumerate(other)
                          if p[1].strip().lower() != mine
                          and len(answers(other) - {p[1].strip().lower()}) > 1), None)
                if j is None:
                    continue
                group[0], other[j] = other[j], group[0]
                break

        for group in built:
            entry = dict(record, pairs=group)
            (chunks if len(group) >= min_turns else singles).append(entry)
    random.shuffle(chunks)
    random.shuffle(singles)
    return chunks, singles


def generate(args):
    lang = args.lang
    groups, skipped = {}, 0
    for path in args.manifests:
        loaded, bad = load_manifest(path, lang, args.multi_answer)
        for key, pairs in loaded.items():
            groups.setdefault(key, []).extend(pairs)
        skipped += bad
        print(f"[{path}] {len(loaded)} audios, "
              f"{sum(len(v) for v in loaded.values())} questions, {bad} unusable lines",
              file=sys.stderr)

    usable = {k: v for k, v in groups.items() if len(v) >= args.min_turns}
    print(f"{len(usable)}/{len(groups)} audios carry at least {args.min_turns} questions",
          file=sys.stderr)
    if not usable:
        print("nothing to generate", file=sys.stderr)
        return

    chunks, singles = cut_chunks(usable, args.min_turns)
    capable = sum(1 for c in chunks if len(c["pairs"]) >= 3) / max(1, len(chunks))
    p_hybrid = min(1.0, MIXED_HYBRID_SHARE / capable) if capable else 0.0
    print(f"{len(chunks)} chunks ({len(singles)} single-question leftovers), "
          f"{capable * 100:.0f}% can carry a bundle -> hybrid drawn at "
          f"{p_hybrid * 100:.0f}%", file=sys.stderr)

    modes = ALL_MODES if args.mode == "all" else [args.mode]
    out_root = Path(args.output_dir)
    writers, counters = {}, {}
    for mode in modes:
        path = out_root / mode / f"{args.split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        writers[mode] = path.open("w", encoding="utf-8")
        counters[mode] = 0

    audio_counts = defaultdict(Counter)
    length_counts = defaultdict(Counter)
    position_counts = defaultdict(Counter)
    shape_counts = defaultdict(Counter)
    single_cursor = 0
    dropped_tail = 0

    for index, chunk in enumerate(chunks):
        if all(counters[m] >= args.max_samples for m in modes):
            break
        group = [chunk]
        if random.random() < args.multi_audio_rate:
            n_extra = random.choices([n for n, _ in EXTRA_CLIP_DIST],
                                     weights=[w for _, w in EXTRA_CLIP_DIST], k=1)[0]
            for _ in range(n_extra):
                extra = None
                # Leftover singles first: that is what they are for, and it keeps
                # a continuation segment short without truncating a full chunk.
                while single_cursor < len(singles):
                    candidate = singles[single_cursor]
                    single_cursor += 1
                    if candidate["audio"] not in {g["audio"] for g in group}:
                        extra = candidate
                        break
                if extra is None:
                    for candidate in chunks[index + 1:index + 6]:
                        if candidate["audio"] not in {g["audio"] for g in group}:
                            extra = candidate
                            break
                if extra is None:
                    break
                group.append(extra)

        for mode in modes:
            if counters[mode] >= args.max_samples:
                continue
            turns, used, traces = [], 0, []
            used_leads = set()
            for position, segment in enumerate(group):
                clip = audio_turn(segment)
                trace = []
                if position == 0:
                    built = BUILDERS[mode](segment, lang, clip, None, trace) \
                        if mode != "mixed" else \
                        build_mixed(segment, lang, clip, None, trace, p_hybrid)
                else:
                    # A continuation segment is short: the point is the switch.
                    cap = draw_turns(len(segment["pairs"]), CONT_TURN_DIST) or 1
                    if len(segment["pairs"]) > cap:
                        dropped_tail += len(segment["pairs"]) - cap
                    built = BUILDERS[mode](segment, lang, clip, cap, trace) \
                        if mode != "mixed" else \
                        build_mixed(segment, lang, clip, cap, trace, p_hybrid)
                if built is None:
                    if position == 0:
                        break
                    continue
                traces.append(trace)
                if position > 0:
                    # Avoid repeating the same lead inside one conversation:
                    # "Autre audio maintenant." twice in four turns reads badly.
                    pool = [c for c in CONTINUATION_EXPLICIT[lang] if c not in used_leads] \
                        or CONTINUATION_EXPLICIT[lang]
                    lead = random.choice(pool)
                    used_leads.add(lead)
                    # Some manifests store questions uncapitalised, which reads as a
                    # broken sentence once a lead sits in front of them.
                    body = built[0]["value"]
                    built[0]["value"] = f"{lead} {body[0].upper()}{body[1:]}"
                turns += built
                used = position + 1
            if not turns:
                continue

            stem = Path(group[0]["audio"]).stem
            writers[mode].write(json.dumps(
                {"id": f"{stem}_qachain_{counters[mode]}_{mode}", "conversations": turns},
                ensure_ascii=False) + "\n")
            counters[mode] += 1

            audio_counts[mode][count_audios(turns)] += 1
            n_asks = sum(len(t) for t in traces)
            if n_asks > 1:
                length_counts[mode][min(n_asks, 6)] += 1
            # Shape and bundle position come from the OPENING SEGMENT's trace: past
            # a switch the turn index mixes position with the previous segment's
            # length, and a frame-matching heuristic cannot see a bundle that was
            # rendered with the bare `{tasks}` frame at all.
            opening = traces[0]
            bundled = [i for i, size in enumerate(opening) if size > 1]
            if not bundled:
                shape_counts[mode]["sequential"] += 1
            elif len(opening) == 1:
                shape_counts[mode]["compound"] += 1
            else:
                shape_counts[mode]["hybrid"] += 1
                position_counts[mode][min(bundled[0] + 1, 3)] += 1

    for w in writers.values():
        w.close()

    def pct(counter, denom):
        return "  ".join(f"{k}:{v / denom * 100:.0f}%" for k, v in sorted(counter.items()))

    if dropped_tail:
        print(f"{dropped_tail} questions dropped truncating continuation segments",
              file=sys.stderr)
    for mode in modes:
        total = counters[mode]
        print(f"  {mode}: {total} conversations -> {out_root / mode / f'{args.split}.jsonl'}",
              file=sys.stderr)
        if not total:
            continue
        if len(shape_counts[mode]) > 1:
            print("    shapes: " + "  ".join(f"{k} {v / total * 100:.0f}%"
                                             for k, v in shape_counts[mode].most_common()),
                  file=sys.stderr)
        print(f"    audios per conversation: {pct(audio_counts[mode], total)}", file=sys.stderr)
        n_multi = sum(length_counts[mode].values())
        if n_multi:
            print(f"    question turns ({n_multi} multi-turn): "
                  f"{pct(length_counts[mode], n_multi)}", file=sys.stderr)
        n_hybrid = sum(position_counts[mode].values())
        if n_hybrid:
            print(f"    bundle at turn ({n_hybrid} hybrids): "
                  f"{pct(position_counts[mode], n_hybrid)}   (3 = 3rd or later)",
                  file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifests", nargs="+", required=True,
                   help="Single-turn QA manifests in multimodal_conversation format. Several "
                        "are pooled, so pass only manifests in the same language.")
    p.add_argument("--lang", default="en", choices=["fr", "en"],
                   help="Language of the manifest's questions; picks the connectives and "
                        "compound frames.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default=None,
                   help="Output file name (default: the first manifest's stem)")
    p.add_argument("--mode", default="all", choices=ALL_MODES + ["all"])
    p.add_argument("--min-turns", type=int, default=2,
                   help="Audios with fewer questions than this are skipped: one question is "
                        "the single-turn dataset we already have.")
    p.add_argument("--multi-audio-rate", type=float, default=0.15,
                   help="Share of conversations chaining chunks from 2-3 different audios, "
                        "each introduced explicitly.")
    p.add_argument("--multi-answer", default="enumerate",
                   choices=["enumerate", "first", "keep"],
                   help="What to do with a pipe-delimited answer, i.e. a manifest packing "
                        "several ALTERNATIVE answers into one string. 'keep' is required when "
                        "the bar separates different fields rather than alternatives.")
    p.add_argument("--max-samples", type=int, default=1_000_000, help="Per mode")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.split is None:
        args.split = Path(args.manifests[0]).stem
    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    generate(args)
