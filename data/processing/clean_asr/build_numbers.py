"""Versions of the train lines of the clean/ base outputs where no line mixes numbers in digits and
in words (NumberFormat in clean_asr.py tells them apart).

    clean/<Dataset>/train.jsonl -> clean/<Dataset>/numbers/none/train.jsonl     lines with no number
                                   clean/<Dataset>/numbers/digits/train.jsonl   lines with numbers, all in digits
                                   clean/<Dataset>/numbers/words/train.jsonl    lines with numbers, all in words
                                   clean/<Dataset>/numbers/mixed/train.jsonl    one line per base line: its none
                                       version, or its digits or words version at random

none + digits + words, or mixed alone, replace the base output.

The base files are read, never written. A line with numbers is in the version of its format as it
is, and in the other version converted; a mixed line is in each version whose conversion gets it
right. A conversion that cannot get a line right leaves it out of that version (`failed` in the
stats); `lost`: the mixed lines in no version, kept as they are in mixed/ (with a digits prompt). Arabic is not handled (no `numbers` rules,
conf/rules.yaml). The prompts follow ChoosePrompt's `numbers` block (`numbers_prompts` and
explicit_fraction of the config): always digits in digits/, words in some of words/, a format at
random in some of none/.

Digits -> words (to_words, num2words): integers, thousands separators ("20 000", "1,000",
"1.000"), decimals, percentages, hours ("14h30"), ordinals ("1er", "11th", "2º", "XIXe"), Roman
numerals ("Louis XIV" -> "Louis quatorze"), decades ("50s"), English and German years. Left out: digit
ranges and other glued signs ("1939-1945", "1/2", "14:30"), leading zeros ("007"), numbers whose
gender the next word decides ("21 personnes": vingt et un / une; es/it/pt uno/una, um/uma,
dois/duas, doscientos/doscientas; de ein/eine), German ordinals ("2." and a sentence end look alike).

Words -> digits (to_digits, text2num's alpha2digit): every number but the isolated "un", "une",
"one"... (threshold 1.5: they are determiners or pronouns as often as numbers); the scale words stay
("2,7 milliards"), "pour cent" -> "%", "a hundred" -> "100", French "2ème" -> "2e", English
years ("nineteen ninety-eight" -> "1998"). Words that look like numbers are kept: "second" (a
second), Italian "sei" after a pronoun (tu sei), French "neuf" as an adjective (un livre neuf).

The prompts are chosen again from the final text, as clean_asr.py chooses them for the base output
(ChoosePrompt / AstPrompt, the dataset's config), with the format of the numbers.

Usage:
    python data/processing/clean_asr/build_numbers.py data/processing/clean_asr/conf/fr.yaml --datasets FLEURS --limit 2000 --output-dir tmp/numbers
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal

from tqdm import tqdm

from clean_asr import (AstPrompt, ChoosePrompt, Disfluencies, NumberFormat, OUTPUTS, REPO_ROOT, Sample, load_config,
                       record_rng)
from rules import rules as language_rules

SKIP = ("_contextual_biasing", "_nonspeech", "_silence", "_max30")
VERSIONS = ("none", "digits", "words", "mixed")
LETTER = r"[^\W\d_]"

# ---------------------------------------------------------------------------------- digits -> words
DECIMAL_WORD = {"fr": "virgule", "en": "point", "es": "coma", "it": "virgola", "pt": "vírgula", "de": "Komma",
                "nl": "komma"}
PERCENT = {"fr": "pour cent", "en": "percent", "es": "por ciento", "it": "per cento", "pt": "por cento",
           "de": "Prozent", "nl": "procent"}
CURRENCY_AFTER = {"€": {"fr": "euros", "en": "euros", "es": "euros", "it": "euro", "pt": "euros", "de": "Euro", "nl": "euro"},
                  "$": {"fr": "dollars", "en": "dollars", "es": "dólares", "it": "dollari", "pt": "dólares",
                        "de": "Dollar", "nl": "dollar"},
                  "£": {"fr": "livres", "en": "pounds", "es": "libras", "it": "sterline", "pt": "libras", "de": "Pfund",
                        "nl": "pond"}}
DECIMAL_SEP = {"en": "."}                       # the others: ","
# a cardinal whose form depends on the gender of the noun after it: its words end with these
GENDERED_END = {"fr": r"(?:^|[ -])(?:un)$", "es": r"(?:uno|cientos)$", "it": r"uno$", "pt": r"(?:um|dois|entos)$",
                "de": r"eins$"}
ORDINAL_SUFFIX = {"fr": r"er|re|ère|ème|eme|e|es|èmes", "en": r"st|nd|rd|th", "es": r"º|ª|°|\.º|\.ª",
                  "it": r"º|ª|°|\.º|\.ª", "pt": r"º|ª|°|\.º|\.ª", "nl": r"ste|de|e"}
ROMAN = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}


class Unconvertible(Exception):
    pass


def roman_value(s):
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = ROMAN[a]
        total += -v if b != " " and ROMAN[b] > v else v
    return total


class ToWords:
    def __init__(self, lang):
        from num2words import num2words
        self.lang = lang
        n2w_lang = {"pt": "pt_BR"}.get(lang, lang)        # the Portuguese corpora are mostly Brazilian
        self.n2w = lambda n, **kw: num2words(n, lang=n2w_lang, **kw).replace(",", "")   # en: "one thousand, nine..."
        self.fmt = NumberFormat(lang)
        dec = re.escape(DECIMAL_SEP.get(lang, ","))
        thou = r"," if lang == "en" else r"[.   ]" if lang != "fr" else r"[   .]"
        self.group_re = re.compile(rf"(?<![\d,.])(\d{{1,3}})((?:{thou}\d{{3}})+)(?![\d]|[.,]\d)")
        self.decimal_re = re.compile(rf"(?<![\d.,])(\d+){dec}(\d+)(?![\d]|[.,]\d)")
        self.percent_re = re.compile(r"(\S+?)\s?%")
        suffix = ORDINAL_SUFFIX.get(lang)
        # not the digits of a name ("COVID-19", "HJR-3")
        self.ordinal_re = re.compile(rf"(?<![\w.,])(?<!\w-)(\d+)({suffix})(?!{LETTER}|\d|-\w)") if suffix else None
        self.int_re = re.compile(rf"(?<![\w.,])(?<!\w-)(\d+)(?![\w]|[.,]\d|-\w)")
        self.gendered = re.compile(GENDERED_END[lang]) if lang in GENDERED_END else None

    def cardinal(self, digits, year=False):
        if len(digits) > 1 and digits.startswith("0"):
            raise Unconvertible("leading zero")
        n = int(digits)
        if n >= 10 ** 12:
            raise Unconvertible("too long")
        if year and self.lang in ("en", "de", "nl") and 1100 <= n <= 2099 and len(digits) == 4:
            words = self.n2w(n, to="year")
            return words.replace(" ", "") if self.lang == "nl" else words
        return self.n2w(n)

    def ordinal(self, n, suffix):
        if self.lang == "fr" and n == 1:
            return "premier" if suffix in ("er", "") else "première" if suffix in ("re", "ère") else "premier"
        words = self.n2w(n, to="ordinal")
        if suffix.endswith("ª"):
            words = re.sub(r"o\b", "a", words)
        if self.lang == "fr" and suffix in ("es", "èmes"):
            words += "s"
        return words

    def __call__(self, text):
        lang = self.lang
        if re.search(r"\d\s?[-–/:×x+=]\s?\d|\d[/:]|[/:]\d", text):
            raise Unconvertible("range or sign between digits")
        if lang == "de" and re.search(r"(?<![\w.,])\d+\.(?=\s*[^\W\d_])", text):
            raise Unconvertible("German ordinal")
        out = text
        # Roman numerals
        def roman(m):
            token = m.group(0)
            letters = re.match(r"[MDCLXVI]+", token).group(0)
            suffix = token[len(letters):]
            n = roman_value(letters)
            return self.ordinal(n, suffix) if suffix else self.n2w(n)
        out = self.fmt.roman_re.sub(roman, out)
        out = self.group_re.sub(lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), out)
        # currencies: "$5" -> "5 $", then "5 €" -> "cinq euros"
        out = re.sub(r"([$£€])\s?(\d[\d.,]*)", r"\2 \1", out)
        if lang == "fr":
            out = re.sub(r"(?<![\w.,])(\d{1,2}) ?h ?(\d{2})?(?![\w])",
                         lambda m: ("une heure" if m.group(1) in ("1", "01") else self.cardinal(m.group(1).lstrip("0") or "0")
                                    + " heures") + (" " + self.cardinal(m.group(2).lstrip("0") or "0")
                                                    if m.group(2) and int(m.group(2)) else ""), out)
        if lang == "en":       # decades: 50s, 1990s; dates: October 7 -> October seventh
            def decade(m):
                n = m.group(1)
                words = self.cardinal(n, year=len(n) == 4)
                return (words[:-1] + "ies") if words.endswith("y") else words + "s"
            out = re.sub(r"(?<![\w.,'])'?(\d{1,3}0)s\b", decade, out)
            out = re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
                         r" (\d{1,2})(?![\d.,]\d|\w)", lambda m: f"{m.group(1)} {self.n2w(int(m.group(2)), to='ordinal')}", out)
        if lang == "de":       # 1960er Jahre -> neunzehnhundertsechziger Jahre
            out = re.sub(r"(?<![\w.,])(\d{1,3}0)er\b", lambda m: self.cardinal(m.group(1), year=True) + "er", out)
        if self.ordinal_re:
            out = self.ordinal_re.sub(lambda m: self.ordinal(int(m.group(1)), m.group(2)), out)
        out = self.decimal_re.sub(lambda m: f"{self.cardinal(m.group(1))} {DECIMAL_WORD[lang]} "
                                            f"{' '.join(self.n2w(int(c)) for c in m.group(2)) if m.group(2).startswith('0') else self.cardinal(m.group(2))}",
                                  out)
        out = re.sub(r"(\d)\s?%", rf"\1 {PERCENT[lang]}", out)
        for sign, names in CURRENCY_AFTER.items():
            out = re.sub(rf"(\d)\s?{re.escape(sign)}", rf"\1 {names[lang]}", out)

        def integer(m):
            words = self.cardinal(m.group(1), year=True)
            after = out[m.end():]
            if self.gendered and self.gendered.search(words) and re.match(rf"\s+{LETTER}", after):
                raise Unconvertible("gender")
            return words
        out = self.int_re.sub(integer, out)
        # a text with no punctuation (TEDLIUM, MLS) writes "thirty six", not "thirty-six"
        if "-" not in text and not re.search(r"[.,;:!?…]", text):
            out = out.replace("-", " ")
        # a sentence starting with a number: capitalized like the text
        if re.search(r"[A-ZÀ-Ý]", text):
            out = re.sub(r"(^|[.!?…]\s+)([a-zà-ÿ])", lambda m: m.group(1) + m.group(2).upper(), out) \
                if re.match(r"\s*\d", text) or re.search(r"[.!?…]\s+\d", text) else out
        return re.sub(r"\s{2,}", " ", out)


# ---------------------------------------------------------------------------------- words -> digits
SCALE = {"fr": "millions? milliards?", "en": "millions? billions?", "es": "millón millones",
         "it": "milion[ei] miliard[oi]", "pt": "milhão milhões bilhão bilhões", "de": "(?i:millionen|million|milliarden|milliarde)",
         "nl": "miljoen(?:en)? miljard(?:en)?"}
PERCENT_WORDS = {"fr": r"pour cent", "en": r"per cent|percent", "es": r"por ciento", "it": r"per cento",
                 "pt": r"por cento", "de": r"Prozent", "nl": r"procent"}
# kept as words: they read as numbers to alpha2digit but are not here
PROTECT = {
    "fr": [r"\bsecondes?\b", r"\bseconds?\b", r"\bneuves?\b", r"\bneufs\b",
           r"(?i)\b(?:tout|flambant|à|de|du|comme) neuf\b(?!\s+[^\W\d_])",
           # pronouns, written in words even among digits: "toutes deux", "tous les trois"
           r"(?i)\b(?:tous|toutes)(?: les)? (?:deux|trois|quatre|cinq)\b",
           r"\bneuf(?=\s*(?:[,.;:!?…»)]|$|et\b|ou\b|mais\b|pour\b|que\b|qui\b|comme\b))"],
    "en": [r"\bseconds?\b", r"(?i)\ball (?:two|three|four|five)\b"],
    "es": [r"\bsegund[oa]s?\b"],
    "it": [r"\bsecond[oaie]\b", r"(?i)\b(?:tu|non|ci|ne|lo|la|li|gli|mi|ti|se|che|dove|come|chi|cosa) sei\b",
           r"(?i)\bsei(?= (?:stat[oaie]|già|proprio|davvero|veramente|così|sempre|mai|ancora|qui|lì|là)\b)"],
    "pt": [r"\bsegund[oa]s?\b"],
    "de": [r"(?i)\b(?:außer|in|gib|gebt|geben) acht\b"],
    "nl": [],
}
EN_TEEN = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
EN_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
EN_UNITS = "one two three four five six seven eight nine".split()


def en_two_digits(words):
    """'ninety-eight' -> 98, 'twenty' -> 20, 'fifteen' -> 15, or None."""
    w = words.lower().replace(" ", "-")
    if w in EN_TEEN:
        return 10 + EN_TEEN.index(w)
    parts = w.split("-")
    if parts[0] in EN_TENS:
        tens = 20 + 10 * EN_TENS.index(parts[0])
        if len(parts) == 1:
            return tens
        if len(parts) == 2 and parts[1] in EN_UNITS:
            return tens + 1 + EN_UNITS.index(parts[1])
    return None


TENS_PLURAL = {t[:-1] + "ies": t for t in EN_TENS}
EN_YEAR_RE = re.compile(
    r"(?i)\b(eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+"
    r"(hundred|oh[ -](?:one|two|three|four|five|six|seven|eight|nine)|"
    r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:-(?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"(?:twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)\b")


def en_years(text):
    def year(m):
        century = en_two_digits(m.group(1))
        rest = m.group(2).lower()
        if rest == "hundred":
            return str(century * 100)
        if rest.startswith("oh"):
            return str(century * 100 + 1 + EN_UNITS.index(rest[3:]))
        if rest in TENS_PLURAL:
            return f"{century * 100 + en_two_digits(TENS_PLURAL[rest])}s"
        return str(century * 100 + en_two_digits(rest))
    return EN_YEAR_RE.sub(year, text)


class ToDigits:
    def __init__(self, lang):
        from text_to_num import alpha2digit
        self.lang, self.a2d = lang, alpha2digit
        self.scale_re = re.compile(rf"\b(?:{'|'.join(SCALE[lang].split())})\b")
        self.percent_re = re.compile(rf"\b(?:{PERCENT_WORDS[lang]})\b")
        self.protect = [re.compile(p) for p in PROTECT[lang]]

    def __call__(self, text):
        kept = []

        def hold(m):
            kept.append(m.group(0))
            return f"zzkeep{len(kept) - 1}zz"
        out = text
        for p in self.protect:
            out = p.sub(hold, out)
        out = self.scale_re.sub(hold, out)
        out = self.percent_re.sub("zzpercentzz", out)
        if self.lang == "en":
            out = re.sub(r"(?i)\ba (hundred|thousand)\b", r"one \1", out)
            out = en_years(out)
        if self.lang == "es":
            out = re.sub(r"\bveintiún\b", "veintiuno", out)
        out = re.sub(r"(?i)\b(?:zéro|zero|cero)\b", "0", out)
        out = self.a2d(out, self.lang, threshold=1.5)
        out = re.sub(r"(\d)\s*zzpercentzz", r"\1 %" if self.lang == "fr" else r"\1%", out)
        if "zzpercentzz" in out:
            raise Unconvertible("percent without a number")
        if self.lang == "fr":
            out = re.sub(r"(\d)ème(s?)\b", lambda m: m.group(1) + ("es" if m.group(2) else "e"), out)
        out = re.sub(r"zzkeep(\d+)zz", lambda m: kept[int(m.group(1))], out)
        return out


# ---------------------------------------------------------------------------------- the files
def build_prompts(cfg, name, dcfg, prompts):
    """The prompt of the base output of the dataset, as clean_asr.py main builds it."""
    explicit = cfg.get("explicit_fraction", {"default": 0.05, "with_output": 0.15})
    if prompts == "ast":
        return AstPrompt(name, (), {"words": explicit["default"], "without": explicit["default"]})
    ctx = {"composer": cfg["_composer"]}
    outputs = dcfg.get("outputs") or ["base"]
    favored = {m for o in outputs for m in OUTPUTS[o]} | (
        {"casepunc"} if dcfg.get("nocasepunc", False) is not False else set())
    return ChoosePrompt(ctx, OUTPUTS["base"], explicit=explicit, favored=favored, nocasepunc=dcfg.get("nocasepunc", False),
                        lowercase=dcfg.get("lowercase", False), numbers=True)


def convert_file(path, outs, lang, prompt, seed, limit, examples):
    """Each line of `path` into the versions: none (no number), digits and words (every line with
    numbers in the version of its format, and in the other one when the conversion gets it right).
    stats["lost"]: the lines in no version (mixed lines whose two conversions failed). mixed: the
    none version of each line, or its digits or words version at random, or the lost line as it is."""
    to = {"words": ToWords(lang), "digits": ToDigits(lang)}
    fmt = NumberFormat(lang)
    stats = Counter()
    ex = defaultdict(list)
    with open(path, encoding="utf-8") as fin:
        for lineno, line in enumerate(tqdm(fin, desc=os.path.basename(path), unit=" lines", leave=False)):
            if limit and lineno >= limit:
                break
            stats["lines"] += 1
            text = Sample(json.loads(line), None).text
            digits, words = fmt.analyse(text)
            stats["mixed input"] += digits and words
            texts = {}
            if not (digits or words):
                texts["none"] = text
            for version in ("digits", "words") if digits or words else ():
                if (version == "words" and not digits) or (version == "digits" and not words):
                    texts[version] = text             # already in that format
                    continue
                try:
                    new = to[version](text)
                except Unconvertible as e:
                    stats[f"{version} failed: {e}"] += 1
                    continue
                d, w = fmt.analyse(new)
                # "un", "premier" are no number to NumberFormat: a words line may show none
                if new == text or (version == "words" and d) or (version == "digits" and (w or not d)):
                    stats[f"{version} failed: still mixed"] += 1
                    if len(ex[f"{version} failed"]) < examples:
                        ex[f"{version} failed"].append((text, new))
                    continue
                texts[version] = new
                stats[f"{version} converted"] += 1
            lost = not texts
            if lost:                                  # in mixed/ only, as it is: it has digits, so asks for digits
                stats["lost"] += 1
                if len(ex["lost"]) < examples:
                    ex["lost"].append((text, ""))
                texts = {"digits": text}
            written = {}
            for version, new in texts.items():
                rec = json.loads(line)
                s = Sample(rec, record_rng(f"{seed}:numbers_{version}", rec, lineno), prompt.nocasepunc, prompt.lowercase)
                s.numbers_rng = record_rng(f"{seed}:numbers_{version}:numbers", rec, lineno)
                s.text = new
                if not prompt(s):
                    stats[f"{version} no prompt"] += 1
                    continue
                if lost:
                    written[version] = json.dumps(rec, ensure_ascii=False) + "\n"
                    continue
                stats[version] += 1
                if new != text and len(ex[version]) < examples:
                    ex[version].append((text, f"{s.text}   <prompt: {s.prompt}>"))
                written[version] = json.dumps(rec, ensure_ascii=False) + "\n"
                if outs[version] is not None:
                    outs[version].write(written[version])
            if written:
                choice = "none" if "none" in written else sorted(written)[0] if len(written) == 1 else \
                    random.Random(f"{seed}:numbers_mixed:{lineno}:{text}").choice(sorted(written))
                stats["mixed"] += 1
                stats[f"mixed {choice}"] += 1
                if outs["mixed"] is not None:
                    outs["mixed"].write(written[choice])
    return stats, ex


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="the clean_asr.py config of the datasets (conf/fr.yaml, conf/ast.yaml...)")
    p.add_argument("--datasets", nargs="+", help="only these datasets from the config")
    p.add_argument("--limit", type=int, default=0, help="only the first N lines of each file")
    p.add_argument("--examples", type=int, default=5, help="examples printed per version and file")
    p.add_argument("--output-dir", help="write <dir>/<dataset>/numbers/ instead of the dataset's output (tests)")
    p.add_argument("--dry-run", action="store_true", help="print stats and examples, write nothing")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    prompts = cfg.get("prompts", "fixed")
    root, seed = os.path.expandvars(cfg["root"]), cfg.get("seed", 0)
    if not cfg.get("numbers_prompts"):
        raise SystemExit(f"{args.config}: numbers_prompts is not on")
    sys.path.insert(0, os.path.join(REPO_ROOT, "data", "contexts"))
    lang = None
    if prompts == "compositional":
        from compose_asr_prompts import PromptComposer
        comp = dict(cfg["compositional"])
        lang = comp["audio_language"]
        hesitations = [form for form, _ in Disfluencies.forms(lang)]
        cfg["_composer"] = PromptComposer(os.path.join(REPO_ROOT, comp.pop("dir")), hesitations=hesitations, **comp)
    elif prompts != "ast":
        raise SystemExit(f"prompts: {prompts!r}: only compositional and ast")

    datasets = cfg["datasets"]
    if args.datasets:
        missing = set(args.datasets) - set(datasets)
        if missing:
            raise SystemExit(f"not in config: {sorted(missing)}")
        datasets = {k: datasets[k] for k in args.datasets}
    for name, dcfg in datasets.items():
        target = name.split("/")[0].split("-")[1] if prompts == "ast" else lang
        if not language_rules(target).get("numbers"):
            print(f"[{name}] no `numbers` rules for {target}: skipped")
            continue
        out_dir = os.path.join(root, dcfg["output"])
        inputs = dcfg["input"] if isinstance(dcfg["input"], list) else [dcfg["input"]]
        names = sorted({os.path.basename(f) for pattern in inputs for f in glob.glob(os.path.join(root, pattern))})
        names = [n for n in names if n.startswith("train") and not any(s in n for s in SKIP)]
        prompt = build_prompts(cfg, name, dcfg, prompts)
        dest = os.path.join(args.output_dir, name) if args.output_dir else out_dir
        for base in names:
            path = os.path.join(out_dir, base)
            if not os.path.exists(path):
                print(f"[{name}] {path} missing: skipped")
                continue
            targets = {v: os.path.join(dest, "numbers", v, base) for v in VERSIONS}
            if not args.dry_run and not args.overwrite and any(os.path.exists(t) for t in targets.values()):
                print(f"[{name}] {base}: numbers/ exists (use --overwrite)")
                continue
            outs = {v: None for v in VERSIONS}
            if not args.dry_run:
                for v, t in targets.items():
                    os.makedirs(os.path.dirname(t), exist_ok=True)
                    outs[v] = open(t + ".tmp", "w", encoding="utf-8")
            try:
                stats, ex = convert_file(path, outs, target, prompt, seed, args.limit, args.examples)
            finally:
                for f in outs.values():
                    if f is not None:
                        f.close()
            if not args.dry_run:
                for v, t in targets.items():
                    if stats[v]:
                        os.replace(t + ".tmp", t)
                    else:
                        os.remove(t + ".tmp")
                        if not os.listdir(os.path.dirname(t)):
                            os.rmdir(os.path.dirname(t))
            print(f"[{name}] {base}: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
            for key, pairs in ex.items():
                print(f"  -- {key}")
                for before, after in pairs:
                    print(f"     {before[:160]}\n  -> {after[:300]}")


if __name__ == "__main__":
    main()
