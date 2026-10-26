"""OFROM: current text kept, with the words of the raw TextGrid intervals (raw/transcript/fr/OFROM/<REC>.TextGrid,
first tier, the interval starting at each segment's offset in the context/ manifests).

The raw text has no punctuation; transfer_format takes the raw words (truncation dashes
"ve-", false starts "t/t'as" -> "t- t'as", proper noun case) and the case/punctuation
of the current re-punctuated text.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import first_tier, interval_at, manifest_windows  # noqa: E402
from cuts import run  # noqa: E402

RAW = "/data-server/datasets/audio/raw/transcript/fr/OFROM"
FILES = ["train_recasepunc.jsonl"]  # max30 only for LesVocaux, LINAGORA_Meetings, CFPP2000


def clean(t):
    t = t.replace("’", "'").replace("–", " ")
    t = re.sub(r"(?<=\S)/(?=\S)", "- ", t)        # "t/t'as" false start
    t = re.sub(r"(?<!\S)/|_", " ", t)
    t = re.sub(r"(?<=\w)/(?!\S)", "-", t)
    t = re.sub(r"(?<!\S)[#@%](?!\S)", " ", t)
    return " ".join(t.split())


def main():
    grids = {os.path.basename(p)[:-9]: first_tier(p) for p in glob.glob(f"{RAW}/*.TextGrid")}
    windows = manifest_windows("OFROM", FILES)

    def raw_of(seg_id):
        if seg_id not in windows:
            return None, "no_raw"
        rec, start, _ = windows[seg_id]
        t = interval_at(grids.get(rec, {}), start)
        return (None, "no_raw") if t is None else clean(t)

    return {name: run("OFROM", name, raw_of) for name in FILES}


if __name__ == "__main__":
    main()
