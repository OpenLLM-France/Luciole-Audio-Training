"""VoxForge: text rebuilt from the read prompts raw/transcript/fr/VOXFORGE/VoxForge_FR_40H/<session>/etc/prompts-original.

prompts-original has the sentences as written, with their case and punctuation
("Il m’a conduit, ensuite, à l’établissement des chanteuses."); the context/ text was
lowercased and re-punctuated by the recase model ("Il m'a conduit ensuite à ..."). The line
of a segment is found by its session (wav directory) and prompt id (wav name). ’ -> '.
Sessions without prompts-original (only the uppercase PROMPTS) keep their context/ text.
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import assistant_text, audio_turn, rebuild_manifest  # noqa: E402

RAW = "/data-server/datasets/audio/raw/transcript/fr/VOXFORGE/VoxForge_FR_40H"
FILES = ["train_recasepunc.jsonl"]  # max30 only for LesVocaux, LINAGORA_Meetings, CFPP2000


def load_prompts(session, cache={}):
    if session not in cache:
        path = os.path.join(RAW, session, "etc", "prompts-original")
        prompts = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    pid, _, text = line.strip().partition(" ")
                    if text:
                        prompts[pid] = " ".join(text.replace("’", "'").split())
        cache[session] = prompts
    return cache[session]


def main():
    for name in FILES:
        stats = collections.Counter()

        def fn(rec):
            wav = audio_turn(rec)["value"]
            session = os.path.basename(os.path.dirname(os.path.dirname(wav)))
            text = load_prompts(session).get(os.path.splitext(os.path.basename(wav))[0])
            if text is None:
                stats["no_prompt_kept_current"] += 1
                return assistant_text(rec)["value"]
            return text
        rebuild_manifest("VoxForge", name, fn)
        print(f"  {dict(stats)}")


if __name__ == "__main__":
    main()
