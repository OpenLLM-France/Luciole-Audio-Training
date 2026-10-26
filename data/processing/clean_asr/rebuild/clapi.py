"""Rebuild CLAPI text from raw/transcript/fr/CLAPI/clapi/2/CorpusComplet/*/<REC>_TEI.xml.

The TEI files are parsed again exactly like ssak clapi2kaldi.py (same row ids
<REC>_<SPK>_<n>, same flushes at the "#T" anchors), and each flush group (the rows
sharing one audio window [start anchor, end anchor]) gets the text of everything
transcribed in the window, in file order and for all speakers: the old conversion
gave each row of a group the same window with only its own words (36 % of the lines).
The lowest row of a group keeps the merged text, the others are dropped as
duplicate_window.
Kept from raw, beyond the <w> words: <unclear> and "xx"/"xxx" -> [inaudible],
<vocal> rires / <seg> "en riant" -> [laughter], the CHAT utterance terminators of
histoire_grenouille ("?", "!", "." in <choice><orig>, or glued to the last word).
Removed: pauses, overlap marks (<space>, "<" ">"), CHAT/ICOR symbols (= ^ ` * + "),
notes; ".tsk" -> [noise]; TEI tokens glued again ("s' y" -> "s'y", "est- ce" -> "est-ce"
when the compound is in the TCOF vocabulary). "ï¿½" (aperitif_rupture) is restored from the CLAPI (then TCOF) vocabulary, else -> "é", "\\x9c" -> "œ".
Case and the remaining punctuation via transfer_format.
"""
import collections
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tcof import BAD, compounds, repair  # noqa: E402
from transcriber import noise_tag  # noqa: E402
from windows import run  # noqa: E402

ROOT = "/data-server/datasets/audio/raw/transcript/fr/CLAPI/clapi/2/CorpusComplet"
NS = "{http://www.tei-c.org/ns/1.0}"
TERMINATORS = {"?": "?", "!": "!", ".": ".", "+..?": "?", "+/?": "?", "+/.": ".", "+...": "."}


def name(el):
    return el.tag.replace(NS, "")


def all_text(el):
    return "".join(el.itertext())


class Parser:
    """clapi2kaldi.Xml2Kaldi.parse_file, keeping for every row its flush group."""

    def __init__(self, file_id):
        self.file_id = file_id
        self.rows = []           # (row id, plain text)
        self.group_of = {}       # row id -> group index
        self.groups = []         # rich tokens of each group window
        self.tmp, self.rich, self.ct = [], [], 0

    def flush(self):
        # flush_rows: the rows of the window [start, this anchor] get their end; a new window starts
        if self.tmp:
            g = len(self.groups)
            for rid, _ in self.tmp:
                self.group_of[rid] = g
            self.groups.append(self.rich)
            self.rows.extend(self.tmp)
        else:
            self.groups.append(self.rich)
        self.tmp, self.rich = [], []

    def add_row(self, speaker, text):
        self.tmp.append((f"{self.file_id}_{speaker}_{self.ct}", text.strip()))
        self.ct += 1

    def turn(self, u, speaker):
        text = ""
        for el in u:
            n = name(el)
            if n == "anchor":
                synch = el.get("synch")
                if synch is not None and synch.startswith("#T") and len(text) > 1:
                    self.add_row(speaker, text)
                    text = ""
                if synch is not None and synch.startswith("#T") and self.tmp:
                    self.flush()
            elif n == "w":
                text += " " + all_text(el)
                self.rich.append(word(all_text(el)))
            elif n == "choice":
                reg = el.find(f".//{NS}reg")
                reg = all_text(reg)
                text += " " + reg
                orig = el.find(f".//{NS}orig")
                o = " ".join(all_text(orig).split()) if orig is not None else ""
                self.rich.append(word(reg) if reg.strip() else TERMINATORS.get(o.split(" ")[0], ""))
            elif n == "seg":
                rend = el.get("rend") or ""
                if re.search(r"rire|riant|rit\b|rient", rend):
                    self.rich.append("[laughter]")
                self.turn(el, speaker)       # a nested <seg> makes its own row
            elif n == "unclear":
                self.rich.append("[inaudible]")
            elif n == "vocal":
                self.rich.append(noise_tag(all_text(el)))
        if len(text) > 1:
            self.add_row(speaker, text)

    def parse(self, root):
        body = root.find(f".//{NS}body")
        if body is None:
            return self
        for el in body:
            t = all_text(el)
            if t and t.isspace():
                continue
            if name(el) == "anchor":
                self.flush()
            elif name(el) == "u":
                self.turn(el, el.get("who").strip("#"))
        return self


def word(w):
    w = w.replace("\x9c", "œ")
    w = re.sub(r"[=^`*+\"<>/\[\]]", " ", w)
    w = re.sub(r"^\.?tsk$", "[noise]", w.strip())                    # ICOR tongue click
    w = re.sub(r"([^\s.?!])([.?!])(?=\s|$)", r"\1 \2", w)          # CHAT terminator glued to the word
    return re.sub(r"(?<!\S)[xX]{2,}(?!\S)", " [inaudible] ", w)


_index, _vocab, _compounds = {}, collections.Counter(), set()


def load():
    roots = {}
    for f in sorted(glob.glob(ROOT + "/**/*_TEI.xml", recursive=True)):
        file_id = "_".join(os.path.splitext(os.path.basename(f))[0].split("_")[0:-1])
        with open(f, encoding="ISO-8859-1") as fh:
            roots[file_id] = ET.fromstring(fh.read().split("?>", 1)[1])
        for w in roots[file_id].iter(f"{NS}w"):
            if BAD not in all_text(w):
                _vocab.update(re.findall(r"[\w'-]+", all_text(w)))
    _compounds.update(compounds())
    for file_id, root in roots.items():
        p = Parser(file_id).parse(root)
        for rid, _ in p.rows:
            _index[rid] = p.groups[p.group_of[rid]]


def finish(tokens):
    out = []
    for t in " ".join(tokens).split():
        if BAD in t:
            pre, core, suf = re.match(r"^(\W*)(.*?)([^\w\uFFFD¿½]*)$", t).groups()
            t = pre + repair(core, _vocab) + suf
        out.append(t)
    text = re.sub(r"(\w['’]) (?=\w)", r"\1", " ".join(out))       # TEI tokens "s' y" -> "s'y"
    text = re.sub(r"(?<!\S)(\S+)- (\w+)", lambda m: m[1] + "-" + m[2] if (m[1].split("'")[-1] + "-" + m[2]).lower()
                  in _compounds or m[1].lower() in {"saint", "sainte"} else m[0], text)   # "est- ce" -> "est-ce", not "ch- chat"
    text = re.sub(r"(\[[a-z]+\])( \1)+", r"\1", text)
    return re.sub(r"^[.?!]( |$)", "", text).strip()


def raw_text(audio, start, end, seg_id):
    if not _index:
        load()
    if seg_id not in _index:
        return None, "no_raw"
    return finish(_index[seg_id]) or (None, "empty")


def row_order(seg_id):
    m = re.match(r"^(.*)_[^_]+_(\d+)((?:_cut\d+)?)$", seg_id)
    return (m[1], int(m[2]), m[3]) if m else (seg_id, 0, "")


if __name__ == "__main__":
    run("CLAPI", raw_text, order=row_order)
