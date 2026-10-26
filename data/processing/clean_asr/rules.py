"""The language rules of conf/rules.yaml: `common` merged with each language's section."""
import functools
import os

import yaml

RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf", "rules.yaml")


def _merge(base, extra):
    """Dictionaries merged key by key, lists appended, anything else replaced."""
    if isinstance(base, dict) and isinstance(extra, dict):
        return {**base, **{k: _merge(base[k], v) if k in base else v for k, v in extra.items()}}
    if isinstance(base, list) and isinstance(extra, list):
        return base + extra
    return extra


@functools.lru_cache(maxsize=None)
def _load():
    with open(RULES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=None)
def rules(lang):
    """The rules of a language: its section of conf/rules.yaml on top of `common`."""
    all_rules = _load()
    if lang not in all_rules or lang == "common":
        raise ValueError(f"no rules for lang {lang!r} in {RULES_PATH}: {sorted(set(all_rules) - {'common'})}")
    return _merge(all_rules.get("common") or {}, all_rules[lang] or {})


def as_words(value):
    """A word list of the rules (a list, or words separated by spaces) as a frozenset."""
    if value is None:
        return frozenset()
    return frozenset(value.split() if isinstance(value, str) else value)


def words(lang, *keys):
    """The word list at rules(lang)[keys[0]][keys[1]]..., empty when missing."""
    value = rules(lang)
    for key in keys:
        value = (value or {}).get(key)
    return as_words(value)
