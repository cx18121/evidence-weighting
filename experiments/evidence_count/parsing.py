"""Strict readout parsers (spec v3.1). Used by run.py for live progress and by analyze.py, which re-parses
every value from the raw logged output.

Categories: valid | ambiguous_scale (counted as a parsed value, flagged) | truncated | attempted_reasoning |
refusal | invalid | empty | api_error.  "Non-valid" = everything except valid and ambiguous_scale.
"""
import math
import re

NUM = re.compile(r"^[+]?(?P<num>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<pct>%?)$")
INT = re.compile(r"^(?P<num>\d+)$")
REFUSAL = re.compile(r"\b(cannot|can't|sorry|refuse|unable|not able|won't)\b", re.I)
TRUNC = {"length", "max_tokens"}


def _clean(t):
    t = t.strip()
    t = re.sub(r"^[*_`\"']+|[*_`\"']+$", "", t).strip()   # **76.9**, `76`, "76"
    t = re.sub(r"\.$", "", t).strip()                        # trailing full stop
    return t


def _fail(t):
    if REFUSAL.search(t):
        return "refusal"
    if re.search(r"[A-Za-z]{3,}", t) or "\n" in t or "=" in t:
        return "attempted_reasoning"
    return "invalid"


def parse_prob(text, finish=None):
    """Stated probability on the 0-100 scale -> (p in [0,1] or None, category, alt_p).
    alt_p: the 0-1 reading for ambiguous_scale replies (value strictly between 0 and 1 written as a decimal)."""
    if text is None or not str(text).strip():
        return None, ("truncated" if finish in TRUNC else "empty"), None
    if finish in TRUNC:
        return None, "truncated", None
    t = _clean(str(text))
    m = NUM.match(t)
    if not m:
        return None, _fail(t), None
    v = float(m.group("num"))
    if not math.isfinite(v) or v > 100:
        return None, "invalid", None
    if 0 < v < 1 and not m.group("pct"):
        return v / 100, "ambiguous_scale", v
    return v / 100, "valid", None


def parse_answer_line(text, finish=None):
    """Sonnet reasoning reference: last line 'Answer: N' (0-100)."""
    if finish in TRUNC:
        return None, "truncated", None
    if not text:
        return None, "empty", None
    hits = re.findall(r"Answer\s*[:=]\s*\**\s*([0-9.eE+-]+)\s*%?\s*\**", text)
    if not hits:
        return None, "invalid", None
    return parse_prob(hits[-1], None)


def parse_count(text, n, finish=None):
    if text is None or not str(text).strip():
        return None, ("truncated" if finish in TRUNC else "empty")
    if finish in TRUNC:
        return None, "truncated"
    t = _clean(str(text))
    m = INT.match(t)
    if not m:
        return None, _fail(t)
    v = int(m.group("num"))
    return (v, "valid") if v <= n else (None, "invalid")


def parse_noul(resp, key):
    """Jev: resp['answers'][key]['noul'] in [0, 1]."""
    try:
        v = resp["answers"][key]["noul"]
    except (KeyError, TypeError):
        return None, "empty"
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
        return None, "invalid"
    return float(v), "valid"


def is_parsed(cat):
    return cat in ("valid", "ambiguous_scale")


def parse_yesno(text, finish=None):
    if finish in TRUNC:
        return None, "truncated"
    if text is None or not str(text).strip():
        return None, "empty"
    t = _clean(str(text)).lower().rstrip(".!")
    if t in ("yes", "y"):
        return 1, "valid"
    if t in ("no", "n"):
        return 0, "valid"
    return None, _fail(t)


def parse_choice(resp, key, yes_label="YES"):
    """Jev Choice: resp['answers'][key]['probabilities'][yes_label]."""
    try:
        v = resp["answers"][key]["probabilities"][yes_label]
    except (KeyError, TypeError):
        return None, "empty"
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
        return None, "invalid"
    return float(v), "valid"
