"""Convert VoxLingua107 (spoken language identification) to NeMo manifests with en and fr prompts.

VoxLingua107 is YouTube speech labelled by the language of the video title (one folder per
language code, wav 16 kHz). The label is noisy for close languages (some be is Russian, some gl is
Spanish), no filter is applied here.

Classes: one per folder, except no + nn merged into Norwegian (Bokmål/Nynorsk are written norms,
not audible). Clips per class in train:
    MAIN_LANGS     no cap
    BIG_LANGS      BIG_CAP (> 100 M speakers)
    others         OTHER_CAP
Each prompt language (en, fr) gets the whole cap of every class. The clips of a class are picked
round-robin over its videos (shuffled), so a cap keeps as many different videos as possible. en and
fr get different clips when the class has 2 x cap clips, otherwise they share the fewest possible
(2 x cap - available); uncapped classes have the same clips in both. dev (1609 clips validated by
people, 33 languages, el only there) is kept whole, in both prompt languages.

The answer is the bare language: the ISO code by default ("Which language is this?" -> "pt"), the
name when the prompt asks for it (NAME_FRACTION of the lines, "Name the language..." -> "Portuguese").
Codes are ISO 639-1 (iw -> he, jw -> jv), ISO 639-3 for ceb, haw, sco, war, which have none.

Output: $DATA_FOLDER/nemo/misc/{en,fr}/clean/VoxLingua_lang_identification/{train,dev}.jsonl

Usage:
    python data/datasets2nemo/multilang/voxlingua2nemo.py
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voxlingua107_languages import LANGUAGES  # noqa: E402

RAW = "raw/language/identification/VoxLingua107"
OUT = "nemo/misc/{lang}/clean/VoxLingua_lang_identification"

MAIN_LANGS = {"fr", "en", "de", "it", "es", "nl", "ar", "pt"}
BIG_LANGS = {"zh", "hi", "bn", "ru", "ur", "id", "ja"}
BIG_CAP = 10000
OTHER_CAP = 1000
MERGED = {"nn": "no"}
MERGED_NAMES = {"no": ("no", "Norwegian", "norvégien")}

# "code": answered by the ISO code (plain questions = the default); "name": asks for the name
PROMPTS = {
    "en": {
        "code": [
            "Which language is this?",
            "What language is spoken?",
            "Identify the language.",
            "Identify the spoken language.",
            "Give the language code of this audio.",
            "Give the ISO code of the spoken language.",
            "What is the language code?",
            "Language code of this audio?",
        ],
        "name": [
            "Give the name of the spoken language.",
            "Name the language of this audio.",
            "What is the name of the language spoken?",
            "Language name of this audio?",
            "Which language is this? Name only.",
            "Identify the language, give its name.",
        ],
    },
    "fr": {
        "code": [
            "Quelle est cette langue ?",
            "Quelle langue est parlée dans l'audio ?",
            "Identifiez la langue.",
            "Identifiez la langue parlée.",
            "Donnez le code langue de l'audio.",
            "Donnez le code ISO de la langue parlée.",
            "Quel est le code de langue ?",
            "Code de langue de cet audio ?",
        ],
        "name": [
            "Donnez le nom de la langue parlée.",
            "Nommez la langue de cet audio.",
            "Quel est le nom de la langue parlée ?",
            "Nom de la langue de cet audio ?",
            "Quelle langue est parlée ? Le nom seulement.",
            "Identifiez la langue, donnez son nom.",
        ],
    },
}
NAME_FRACTION = 1 / 3
FILE_RE = re.compile(r"^(?P<video>.+)__U__S\d+---\d+\.\d+-\d+\.\d+\.wav$")


def label(folder):
    """Folder code -> (class key, ISO code, English name, French name)."""
    key = MERGED.get(folder, folder)
    iso, en, fr = MERGED_NAMES.get(key) or LANGUAGES[key]
    return key, iso, en, fr


def cap_of(key):
    if key in MAIN_LANGS:
        return None
    return BIG_CAP if key in BIG_LANGS else OTHER_CAP


def list_split(root, split):
    """class key -> sorted list of (folder, file)."""
    clips = defaultdict(list)
    for folder in sorted(os.listdir(root / split)):
        if not (root / split / folder).is_dir():
            continue
        assert folder in LANGUAGES, folder
        for name in sorted(os.listdir(root / split / folder)):
            if name.endswith(".wav"):
                assert FILE_RE.match(name), name
                clips[label(folder)[0]].append((folder, name))
    return clips


def pick(clips, n, rng):
    """n clips (all if fewer), in round-robin order over the shuffled videos."""
    n = min(n, len(clips))
    by_video = defaultdict(list)
    for c in clips:
        by_video[FILE_RE.match(c[1])["video"]].append(c)
    queues = list(by_video.values())
    rng.shuffle(queues)
    for q in queues:
        rng.shuffle(q)
    picked = []
    while len(picked) < n:
        queues = [q for q in queues if q]
        for q in queues:
            picked.append(q.pop())
            if len(picked) == n:
                break
    return picked


def split_en_fr(clips, cap, rng):
    """cap clips for en and cap for fr, disjoint when possible."""
    cap = min(cap or len(clips), len(clips))
    picked = pick(clips, 2 * cap, rng)
    return picked[:cap], picked[len(picked) - cap:]


def make_line(prompt_lang, root, split, folder, name, duration, rng):
    _, iso, en, fr = label(folder)
    mode = "name" if rng.random() < NAME_FRACTION else "code"
    question = rng.choice(PROMPTS[prompt_lang][mode])
    lang_name = en if prompt_lang == "en" else fr
    answer = lang_name[0].upper() + lang_name[1:] if mode == "name" else iso
    return {
        "id": f"{folder}--{name[:-4]}",
        "conversations": [
            {"from": "User", "value": question, "type": "text"},
            {"from": "User", "value": str(root / split / folder / name), "type": "audio", "duration": duration},
            {"from": "Assistant", "value": answer, "type": "text"},
        ],
        "dataset_name": "VoxLingua107",
        "Task": "Spoken language identification",
        "custom": {"answer_mode": mode, "label": iso},
    }


def durations(root, split, clips, jobs, known):
    def one(c):
        path = str(root / split / c[0] / c[1])
        return known[path] if path in known else round(sf.info(path).duration, 3)

    with ThreadPoolExecutor(jobs) as ex:
        return list(ex.map(one, clips, chunksize=64))


def write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=32, help="threads reading the wav headers")
    parser.add_argument("--reuse-durations", action="store_true",
                        help="take the durations of the clips already in the output manifests (no wav read)")
    args = parser.parse_args()
    data_folder = Path(os.environ["DATA_FOLDER"])
    root = data_folder / RAW
    rng = random.Random(args.seed)

    # train: cap clips per class for each prompt language
    train = list_split(root, "train")
    halves = {"en": [], "fr": []}
    print("class\tavailable\tper_prompt_lang\tshared_en_fr")
    for key in sorted(train, key=lambda k: -len(train[k])):
        en, fr = split_en_fr(train[key], cap_of(key), rng)
        halves["en"] += en
        halves["fr"] += fr
        print(f"{key}\t{len(train[key])}\t{len(en)}\t{len(set(en) & set(fr))}")
    shared = len(set(halves["en"]) & set(halves["fr"]))
    print("train classes", len(train), "lines per prompt lang", len(halves["en"]), "shared clips", shared)

    dev_clips = list_split(root, "dev")
    dev = [c for key in sorted(dev_clips) for c in dev_clips[key]]
    print("dev classes", len(dev_clips), "clips", len(dev))

    known = {}
    if args.reuse_durations:
        for prompt_lang in ["en", "fr"]:
            for split in ["train", "dev"]:
                path = data_folder / OUT.format(lang=prompt_lang) / f"{split}.jsonl"
                if path.exists():
                    for line in open(path, encoding="utf-8"):
                        audio = json.loads(line)["conversations"][1]
                        known[audio["value"]] = audio["duration"]
    dur = {}
    for split, clips in [("train", sorted(set(halves["en"]) | set(halves["fr"]))), ("dev", dev)]:
        dur[split] = dict(zip(clips, durations(root, split, clips, args.jobs, known)))

    for prompt_lang in ["en", "fr"]:
        out = data_folder / OUT.format(lang=prompt_lang)
        for split, clips in [("train", halves[prompt_lang]), ("dev", dev)]:
            clips = sorted(clips)
            durs = [dur[split][c] for c in clips]
            lines = [make_line(prompt_lang, root, split, f, n, d, rng) for (f, n), d in zip(clips, durs)]
            write(out / f"{split}.jsonl", lines)
            modes = Counter(line["custom"]["answer_mode"] for line in lines)
            hours = sum(durs) / 3600
            print(f"{out / split}.jsonl\t{len(lines)} lines\t{hours:.1f} h\t{dict(modes)}")


if __name__ == "__main__":
    main()
