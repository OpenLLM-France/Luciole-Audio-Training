"""SimSamu: current text kept, with the words of the raw srt (raw/transcript/fr/SimSamu/annotations/normalized/text_raw).

transfer_format brings back the truncation dashes ("para- paracétamol") and the
digits of the srt; case and punctuation stay those of the current text. Names written
in capitals in the srt ("docteur LAKHLIFI") take the current case unless the current
text also has them in capitals (acronyms: SAMU, COVID). Untranscribed "<NA>" segments
are dropped. "heu" is left for clean_asr.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONTEXT, assistant_text  # noqa: E402
from cuts import run  # noqa: E402

RAW = "/data-server/datasets/audio/raw/transcript/fr/SimSamu/annotations/normalized/text_raw"


def main():
    raw = {}
    for line in open(RAW, encoding="utf-8"):
        u, _, t = line.rstrip("\n").partition(" ")
        raw[u] = " ".join(t.replace("’", "'").split())
    current = {}
    for name in ["train.jsonl"]:
        for line in open(os.path.join(CONTEXT, "SimSamu", name), encoding="utf-8"):
            rec = json.loads(line)
            current[rec["id"]] = assistant_text(rec)["value"]

    def raw_of(seg_id):
        t = raw.get(seg_id)
        if t is None:
            return None, "no_raw"
        if t.strip() == "<NA>":
            return None, "untranscribed_NA"
        caps = set(re.findall(r"\b[A-ZÀ-Ý]{2,}\b", current.get(seg_id, "")))
        return re.sub(r"\b[A-ZÀ-Ý]{2,}\b", lambda m: m[0] if m[0] in caps else m[0].lower(), t)

    return {name: run("SimSamu", name, raw_of)
            for name in ["train.jsonl"]}


if __name__ == "__main__":
    main()
