"""Analysis for E2 (pairs), spec v3.1.

Reads items.jsonl and results/raw/<cond>.jsonl, re-parses every stored raw LLM output with the v3.1
parser, and writes results/summary_tables.md, results/tables/*.csv and Figure 1b (results/fig1b.pdf/.png).

Declared rules
- Primary conditions: jev, llama70b_pin (Together), mistral24b_pin (DeepInfra), gemma27b (DeepInfra),
  deepseekv32 (DeepInfra), haiku; reference: sonnet5_reason; control: jev_holonly.
  Superseded (reported separately): llama70b and mistral24b (unpinned OpenRouter), sonnet5_direct (smoke only).
- AUROC (positives vs matched hard negatives; Mann-Whitney, ties 1/2) is the primary holistic metric.
- Holistic decisions: Jev Noul P >= 0.5; LLM stated probability > 50 (primary) and >= 50 (sensitivity);
  LLM Yes/No answers directly. Per-piece answers: P >= 0.5 (Jev) or "Yes" (LLM) counts as yes.
- v3.1 parser: one number (integer, decimal, scientific, optional %, optional ** markdown); a truncated reply
  (stop reason max_tokens / length, or for OpenRouter completion tokens >= max_tokens, inferred because the
  wrapper does not return finish_reason) is invalid. Values strictly between 0 and 1 are flagged
  ambiguous_scale. Categories: valid / refusal / attempted-reasoning / invalid.
- Non-valid outputs are excluded in the main tables (counts reported) and counted as wrong in a sensitivity table.
- Per-row OR remedy: yes if any of the 12 per-row answers is yes. Per-part join: parts answered yes form the
  present set; yes if both members of any rule are present. Remedies use items whose relevant pieces
  (12 rows, or 24 parts) are all valid. Nothing is fitted on any model output for any remedy.
- Within-size slope of false alarms on the number of unpaired listed parts: descriptive logistic regression on
  hard negatives, decision ~ set-size fixed effects + slope * a (a = unpaired listed parts), with a weak L2
  penalty (lambda = 0.1 on all coefficients) so estimates stay finite under separation; also the linear
  probability slope with the same fixed effects. This is a description of the outputs, not a model used for
  any decision. 95% CIs: item bootstrap (1000 resamples, items resampled within set size).
- 95% CIs: percentile bootstrap (2000 resamples) with items resampled within class (item = cluster); for
  proportions whose estimate is exactly 0 or 1, the Wilson score interval is used instead.
"""
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "raw")
TAB = os.path.join(HERE, "results", "tables")
SPEC = "v3.1"
B = 2000
RNG_SEED = 12345
PRIMARY_LLM = ["llama70b_pin", "mistral24b_pin", "gemma27b", "deepseekv32", "haiku"]
PRIMARY = ["jev"] + PRIMARY_LLM
SUPERSEDED = ["llama70b", "mistral24b", "sonnet5_direct"]
ALL = PRIMARY + ["sonnet5_reason", "jev_holonly"] + SUPERSEDED
LABEL = {"jev": "Jev", "llama70b_pin": "Llama-3.3-70B", "mistral24b_pin": "Mistral-Small-3.2-24B",
         "gemma27b": "Gemma-3-27B", "deepseekv32": "DeepSeek-V3.2", "haiku": "Claude Haiku 4.5",
         "sonnet5_reason": "Sonnet 5 (reasoning ref.)", "jev_holonly": "Jev (holistic-only call)",
         "llama70b": "Llama-3.3-70B UNPINNED (superseded)", "mistral24b": "Mistral-24B UNPINNED (superseded)",
         "sonnet5_direct": "Sonnet 5 direct (smoke; dropped)"}
MAXTOK = {"llama70b": 10, "mistral24b": 10, "llama70b_pin": 10, "mistral24b_pin": 10, "gemma27b": 10,
          "deepseekv32": 10, "haiku": 10, "sonnet5_direct": 4000, "sonnet5_reason": 6000}

# ------------------------------------------------------------------ v3.1 parsing (independent of run.py)
PROB_RE = re.compile(r"^\s*\**\s*(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|\.\d+)\s*%?\s*\**\s*\.?\s*$")
YN_RE = re.compile(r"^\s*\**\s*(yes|no)\s*\**\s*\.?\s*$", re.IGNORECASE)
FINAL_RE = re.compile(r"(?:Final )?Answer:\s*\**\s*(\d+(?:\.\d+)?)\s*%?\s*\**\s*\.?\s*$", re.IGNORECASE)
REFUSAL_RE = re.compile(r"\b(I can(?:no|')t|I am unable|I'm unable|I'm not able|I won't|sorry)\b", re.IGNORECASE)


def truncated(r, cond):
    sr = r.get("stop_reason")
    if sr in ("max_tokens", "length", "length(inferred)"):
        return True
    u = r.get("usage") or {}
    ct = u.get("completion_tokens")
    return ct is not None and cond in MAXTOK and ct >= MAXTOK[cond]


def reparse(r, cond):
    """Return (parsed value or None, category, ambiguous_scale)."""
    text = r.get("raw")
    if not isinstance(text, str):
        text = ""
    kind = "yn" if (r["task"] == "yn" or ":" in r["task"]) else ("reason" if cond == "sonnet5_reason" else "prob")
    if kind == "yn":
        mt = YN_RE.match(text)
        v = None if not mt else int(mt.group(1).lower() == "yes")
    elif kind == "reason":
        mt = FINAL_RE.search(text.strip())
        v = float(mt.group(1)) if mt else None
        v = v if v is not None and 0 <= v <= 100 else None
    else:
        mt = PROB_RE.match(text)
        v = float(mt.group(1)) if mt else None
        v = v if v is not None and 0 <= v <= 100 else None
    if truncated(r, cond):
        v = None
    if v is not None:
        cat = "valid"
    elif REFUSAL_RE.search(text):
        cat = "refusal"
    elif truncated(r, cond) or len(text.split()) >= 4:
        cat = "attempted-reasoning"
    else:
        cat = "invalid"
    return v, cat, bool(kind != "yn" and v is not None and 0 < v < 1)


# ------------------------------------------------------------------ loading
def load_items():
    return {it["item_id"]: it for it in map(json.loads, open(os.path.join(HERE, "items.jsonl")))}


def load_para_items():
    p = os.path.join(HERE, "items_para.jsonl")
    return {it["item_id"]: it for it in map(json.loads, open(p))} if os.path.exists(p) else {}


def load_cond(cond, suffix=""):
    path = os.path.join(RAW, f"{cond}{suffix}.jsonl")
    if not os.path.exists(path):
        return {}
    recs = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error"):
            continue
        if not cond.startswith("jev"):
            r["parsed"], r["category"], r["ambiguous_scale"] = reparse(r, cond)
            r["valid"] = r["parsed"] is not None
        prev = recs.get(r["key"])
        # latest successful record wins, except that a valid record is never replaced by an invalid one
        if prev is None or r.get("valid") or not prev.get("valid"):
            recs[r["key"]] = r
    return recs


# ------------------------------------------------------------------ statistics
def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def boot(stat, groups, b=B, seed=RNG_SEED, clusters=None, prop_n=None):
    """Percentile bootstrap, resampling within each group (optionally by cluster).
    prop_n: if the statistic is a proportion over prop_n Bernoulli units, use Wilson at 0 or 1."""
    est = stat(groups)
    if prop_n and not np.isnan(est) and est in (0.0, 1.0):
        lo, hi = wilson(round(est * prop_n), prop_n)
        return est, lo, hi
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        g2 = {}
        for k, v in groups.items():
            if not v:
                g2[k] = v
                continue
            if clusters is None:
                idx = rng.integers(0, len(v), len(v))
                g2[k] = [v[i] for i in idx]
            else:
                cl = clusters[k]
                ids = sorted(set(cl))
                by = defaultdict(list)
                for i, c in enumerate(cl):
                    by[c].append(i)
                pick = rng.integers(0, len(ids), len(ids))
                g2[k] = [v[i] for p in pick for i in by[ids[p]]]
        vals.append(stat(g2))
    vals = np.array(vals, float)
    vals = vals[~np.isnan(vals)]
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan))
    return est, lo, hi


def fmt(t, d=2):
    e, lo, hi = t
    if e is None or (isinstance(e, float) and np.isnan(e)):
        return "n/a"
    return f"{e:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def mean(x):
    return float(np.mean(x)) if len(x) else float("nan")


def prop(rows, key="dec"):
    return lambda d: mean([r[key] for r in d["x"]])


# ------------------------------------------------------------------ holistic readouts
def holistic_rows(cond, recs, items, task="prob", fifty_yes=False, key_cond=None):
    kc = key_cond or cond
    out = []
    for it in items.values():
        if kc in ("jev", "jev_holonly", "jev_choice"):
            r = recs.get(f"{kc}|{it['item_id']}|full")
            if r is None:
                continue
            p = r["parsed"].get("choice" if kc == "jev_choice" else "holistic")
            valid = isinstance(p, (int, float))
            out.append(dict(it=it, valid=valid, cat="valid" if valid else "invalid", score=p if valid else None,
                            dec=(int(p >= 0.5) if valid else None), exact50=False))
        else:
            r = recs.get(f"{kc}|{it['item_id']}|{task}")
            if r is None:
                continue
            v = r["parsed"]
            if task == "yn":
                out.append(dict(it=it, valid=v is not None, cat=r["category"], score=v, dec=v, exact50=False))
            else:
                dec = None if v is None else (int(v >= 50) if fifty_yes else int(v > 50))
                out.append(dict(it=it, valid=v is not None, cat=r["category"], score=(v / 100 if v is not None else None),
                                dec=dec, exact50=(v == 50), amb=r.get("ambiguous_scale", False)))
    return out


def class_metrics(rows_, pooled=False, nonvalid_wrong=False, impute=None):
    """impute: None (exclude non-valid), 'half' (score 0.5; decision no, as a stated 50), 'normative' (truth),
    or nonvalid_wrong=True (counted as wrong)."""
    rr = []
    for r in rows_:
        if r["valid"]:
            rr.append(r)
        elif nonvalid_wrong:
            wrong = 0 if r["it"]["cls"] == "pos" else 1
            rr.append(dict(r, dec=wrong, score=float(wrong), valid=True))
        elif impute == "half":
            rr.append(dict(r, dec=0, score=0.5, valid=True))
        elif impute == "normative":
            t = 1 if r["it"]["cls"] == "pos" else 0
            rr.append(dict(r, dec=t, score=float(t), valid=True))
    g = {c: [r for r in rr if r["it"]["cls"] == c] for c in ("pos", "hard", "easy")}
    cl = (lambda ks: {k: [r["it"]["base_id"] for r in g[k]] for k in ks}) if pooled else (lambda ks: None)
    res = {"n_pos": len(g["pos"]), "n_hard": len(g["hard"]), "n_easy": len(g["easy"]),
           "n_nonvalid": sum(1 for r in rows_ if not r["valid"]),
           "cats": Counter(r["cat"] for r in rows_ if not r["valid"]),
           "n_exact50": sum(1 for r in rows_ if r.get("exact50")),
           "n_amb": sum(1 for r in rows_ if r.get("amb"))}
    res["AUROC"] = boot(lambda d: auroc([r["score"] for r in d["pos"]], [r["score"] for r in d["hard"]]),
                        {"pos": g["pos"], "hard": g["hard"]}, clusters=cl(["pos", "hard"]))
    res["TPR"] = boot(lambda d: mean([r["dec"] for r in d["pos"]]), {"pos": g["pos"]}, clusters=cl(["pos"]),
                      prop_n=len(g["pos"]))
    res["FPR_hard"] = boot(lambda d: mean([r["dec"] for r in d["hard"]]), {"hard": g["hard"]},
                           clusters=cl(["hard"]), prop_n=len(g["hard"]))
    res["acc_matched"] = boot(lambda d: (mean([r["dec"] for r in d["pos"]]) + 1 - mean([r["dec"] for r in d["hard"]])) / 2,
                              {"pos": g["pos"], "hard": g["hard"]}, clusters=cl(["pos", "hard"]),
                              prop_n=len(g["pos"]) + len(g["hard"]))
    if g["easy"]:
        res["easy_acc"] = boot(lambda d: 1 - mean([r["dec"] for r in d["easy"]]), {"easy": g["easy"]},
                               clusters=cl(["easy"]), prop_n=len(g["easy"]))
        res["acc_easy_test"] = boot(lambda d: (mean([r["dec"] for r in d["pos"]]) + 1 - mean([r["dec"] for r in d["easy"]])) / 2,
                                    {"pos": g["pos"], "easy": g["easy"]}, clusters=cl(["pos", "easy"]),
                                    prop_n=len(g["pos"]) + len(g["easy"]))
    return res


def readouts_for(cond):
    if cond.startswith("jev"):
        return [("prob", "Noul P(yes)")]
    if cond.startswith("sonnet"):
        return [("prob", "stated prob")]
    return [("prob", "stated prob"), ("yn", "yes/no")]


# ------------------------------------------------------------------ pieces
def row_answers(cond, recs, it):
    if cond == "jev":
        r = recs.get(f"jev|{it['item_id']}|full")
        try:
            return [float(r["parsed"][x["key"]]) for x in it["rows"]]
        except (KeyError, TypeError):
            return None
    out = []
    for x in it["rows"]:
        r = recs.get(f"{cond}|{it['item_id']}|row:{x['key']}")
        if r is None or r["parsed"] is None:
            return None
        out.append(float(r["parsed"]))
    return out


def part_answers(cond, recs, it):
    if cond == "jev":
        r = recs.get(f"jev|{it['item_id']}|full")
        try:
            return {x["x"]: float(r["parsed"][x["key"]]) for x in it["parts"]}
        except (KeyError, TypeError):
            return None
    out = {}
    for x in it["parts"]:
        r = recs.get(f"{cond}|{it['item_id']}|part:{x['key']}")
        if r is None or r["parsed"] is None:
            return None
        out[x["x"]] = float(r["parsed"])
    return out


def piece_counts(cond, recs, items, kind, vocab, m):
    """(#items attempted, #items complete) for pos+hard."""
    att = comp = 0
    for it in items.values():
        if it["vocab"] != vocab or it["m"] != m or it["cls"] == "easy":
            continue
        if cond == "jev":
            keys = [f"jev|{it['item_id']}|full"]
        else:
            keys = [f"{cond}|{it['item_id']}|{kind}:{x['key']}" for x in it[kind + "s"]]
        present = [k in recs for k in keys]
        if any(present):
            att += 1
            f = row_answers if kind == "row" else part_answers
            comp += f(cond, recs, it) is not None
    return att, comp


# ------------------------------------------------------------------ logistic slope
def penalized_logit(X, y, lam=0.1, iters=100):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        g = X.T @ (y - p) - lam * beta
        H = X.T @ (X * W[:, None]) + lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def slope_fit(rows_):
    """rows_: list of (m, a, dec). Returns (logit slope, linear-probability slope)."""
    ms = sorted({r[0] for r in rows_})
    X = np.array([[1.0 if r[0] == mm else 0.0 for mm in ms] + [r[1]] for r in rows_])
    y = np.array([r[2] for r in rows_], float)
    bl = penalized_logit(X, y)[-1]
    blin = np.linalg.lstsq(X, y, rcond=None)[0][-1]
    return bl, blin


def slope_boot(rows_, b=1000, seed=RNG_SEED):
    est = slope_fit(rows_)
    by = defaultdict(list)
    for r in rows_:
        by[r[0]].append(r)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        s = []
        for mm, v in by.items():
            idx = rng.integers(0, len(v), len(v))
            s += [v[i] for i in idx]
        vals.append(slope_fit(s))
    vals = np.array(vals)
    return [(est[k], *np.percentile(vals[:, k], [2.5, 97.5])) for k in (0, 1)]


# ------------------------------------------------------------------ main
def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def cats_txt(c):
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items())) if c else "0"


def main():
    if not os.path.isfile(os.path.join(RAW, "jev.jsonl")):
        raise SystemExit("Cannot recompute E2 statistics without model-response logs in results/raw/. Regenerate them with run.py first.")
    os.makedirs(TAB, exist_ok=True)
    items = load_items()
    recs = {c: load_cond(c) for c in ALL}
    have = [c for c in ALL if recs[c]]
    fig = defaultdict(dict)
    md = [f"Spec {SPEC}. n per cell given in every table. CIs: 95%, item bootstrap within class (2000 resamples); "
          "Wilson score interval where a proportion is exactly 0 or 1. Thresholds: Jev P >= 0.5; LLM stated "
          "probability > 50 (a stated 50 counts as no; see sensitivity); per-piece P >= 0.5 / Yes.\n"]

    # ================= 1. holistic, primary conditions
    md.append("## 1. Holistic readout by set size: positives vs matched hard negatives (AUROC primary)\n")
    md.append("| Model | readout | vocab | m | n pos/hard/easy | non-valid | AUROC | TPR | FPR (hard neg) | acc (matched) | easy-neg acc | acc on easy test |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    hol_csv = []
    main_conds = [c for c in PRIMARY + ["sonnet5_reason", "jev_holonly"] if c in have]
    for c in main_conds:
        for task, rname in readouts_for(c):
            allrows = holistic_rows(c, recs[c], items, task)
            for vocab in ("names", "codes", "pooled"):
                for m in (4, 8, 12, 16):
                    rr = [r for r in allrows if r["it"]["m"] == m and (vocab == "pooled" or r["it"]["vocab"] == vocab)]
                    if not rr or (vocab == "pooled" and len({r["it"]["vocab"] for r in rr}) < 2):
                        continue
                    res = class_metrics(rr, pooled=(vocab == "pooled"))
                    if not res["n_pos"] or not res["n_hard"]:
                        continue
                    nv = res["n_nonvalid"]
                    extra = []
                    if res["n_exact50"]:
                        extra.append(f"{res['n_exact50']} at 50")
                    if res["n_amb"]:
                        extra.append(f"{res['n_amb']} ambiguous-scale")
                    md.append(f"| {LABEL[c]} | {rname} | {vocab} | {m} | {res['n_pos']}/{res['n_hard']}/{res['n_easy']} | "
                              f"{nv}{' (' + cats_txt(res['cats']) + ')' if nv else ''}{'; ' + ', '.join(extra) if extra else ''} | "
                              f"{fmt(res['AUROC'])} | {fmt(res['TPR'])} | {fmt(res['FPR_hard'])} | {fmt(res['acc_matched'])} | "
                              f"{fmt(res['easy_acc']) if 'easy_acc' in res else 'n/a'} | {fmt(res['acc_easy_test']) if 'acc_easy_test' in res else 'n/a'} |")
                    row = {"spec": SPEC, "cond": c, "readout": task, "vocab": vocab, "m": m,
                           **{k: res[k] for k in ("n_pos", "n_hard", "n_easy", "n_nonvalid", "n_exact50", "n_amb")}}
                    for k in ("AUROC", "TPR", "FPR_hard", "acc_matched", "easy_acc", "acc_easy_test"):
                        if k in res:
                            row[k], row[k + "_lo"], row[k + "_hi"] = res[k]
                    hol_csv.append(row)
                    fig[(c, task, vocab, m)]["holistic"] = res
    write_csv(os.path.join(TAB, "holistic_by_m.csv"), hol_csv)

    # ================= 2. direction of error
    md.append("\n## 2. Direction of error per model (names and codes pooled, stated-probability / Noul readout)\n")
    md.append("False-alarm rate = FPR on matched hard negatives; miss rate = 1 - TPR. 'false alarms' if FPR exceeds the miss rate by > 0.10, 'misses' if the reverse, else 'mixed'.\n")
    md.append("| Model | " + " | ".join(f"m={m}: FPR / miss" for m in (4, 8, 12, 16)) + " | direction (m=8-16) |")
    md.append("|---|---|---|---|---|---|")
    for c in [x for x in PRIMARY + ["sonnet5_reason"] if x in have]:
        cells, dirs = [], []
        for m in (4, 8, 12, 16):
            h = fig.get((c, "prob", "pooled", m), {}).get("holistic") or fig.get((c, "prob", "names", m), {}).get("holistic")
            if not h:
                cells.append("n/a")
                continue
            fpr, miss = h["FPR_hard"][0], 1 - h["TPR"][0]
            cells.append(f"{fpr:.2f} / {miss:.2f}")
            if m >= 8:
                dirs.append(fpr - miss)
        d = mean(dirs) if dirs else float("nan")
        lab = "n/a" if np.isnan(d) else ("false alarms" if d > 0.10 else ("misses" if d < -0.10 else "mixed"))
        md.append(f"| {LABEL[c]} | " + " | ".join(cells) + f" | {lab} |")

    # ================= 3. sensitivity: 50 counts as yes; non-valid counted as wrong
    md.append("\n## 3. Sensitivity checks (stated-probability readouts)\n")
    md.append("(a) a stated 50 counts as yes (AUROC is unaffected by this); (b) non-valid outputs excluded, imputed at 0.5 "
              "(decision no), imputed at the normative answer, or counted as wrong (v3.2 item 6).\n")
    md.append("| Model | vocab | m | #50 | #non-valid | TPR / FPR / acc (50 = no, primary) | TPR / FPR / acc (50 = yes) | AUROC non-valid excluded | AUROC non-valid imputed 0.5 | AUROC non-valid imputed normative | AUROC non-valid as wrong |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in [x for x in PRIMARY_LLM if x in have]:
        rp = holistic_rows(c, recs[c], items, "prob")
        ry = holistic_rows(c, recs[c], items, "prob", fifty_yes=True)
        for vocab in ("names", "codes"):
            for m in (4, 8, 12, 16):
                a_ = [r for r in rp if r["it"]["vocab"] == vocab and r["it"]["m"] == m]
                b_ = [r for r in ry if r["it"]["vocab"] == vocab and r["it"]["m"] == m]
                if not a_:
                    continue
                n50 = sum(r["exact50"] for r in a_)
                nvw = class_metrics(a_, nonvalid_wrong=True)
                if n50 == 0 and sum(not r["valid"] for r in a_) == 0:
                    continue
                ra, rb = class_metrics(a_), class_metrics(b_)
                ih, inn = class_metrics(a_, impute="half"), class_metrics(a_, impute="normative")
                md.append(f"| {LABEL[c]} | {vocab} | {m} | {n50} | {ra['n_nonvalid']} | {ra['TPR'][0]:.2f} / {ra['FPR_hard'][0]:.2f} / {fmt(ra['acc_matched'])} | "
                          f"{rb['TPR'][0]:.2f} / {rb['FPR_hard'][0]:.2f} / {fmt(rb['acc_matched'])} | {fmt(ra['AUROC'])} | {fmt(ih['AUROC'])} | "
                          f"{fmt(inn['AUROC'])} | {fmt(nvw['AUROC'])} |")
    md.append("\nRows are shown only where a stated 50 or a non-valid output occurs.\n")

    # ================= 4. FPR against unpaired parts, and within-size slope
    md.append("\n## 4. False alarms against the number of unpaired listed parts\n")
    md.append("(a) Rates by a (negatives: a unpaired parts; easy negatives a = 0), pooled over m. TPR by a shown for positives (a-2 unpaired parts).\n")
    bins = [(0, 0, "0 (easy)"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 9, "7-9"), (10, 12, "10-12")]
    md.append("| Model | vocab | " + " | ".join(f"FPR a={b[2]}" for b in bins) + " | " + " | ".join(f"TPR a={b[2]}" for b in bins[1:]) + " |")
    md.append("|" + "---|" * (2 + 2 * len(bins) - 1))
    fpr_csv = []
    for c in [x for x in PRIMARY if x in have]:
        allrows = [r for r in holistic_rows(c, recs[c], items, "prob") if r["valid"]]
        for vocab in ("names", "codes"):
            rr = [r for r in allrows if r["it"]["vocab"] == vocab]
            cells = []
            for kind, bb in (("neg", bins), ("pos", bins[1:])):
                for lo, hi, name in bb:
                    sel = [r for r in rr if lo <= r["it"]["a"] <= hi and
                           ((r["it"]["cls"] in ("hard", "easy")) if kind == "neg" else r["it"]["cls"] == "pos")]
                    if not sel:
                        cells.append("n/a")
                        continue
                    e = boot(prop(sel), {"x": sel}, prop_n=len(sel))
                    cells.append(f"{fmt(e)} (n={len(sel)})")
                    fpr_csv.append({"spec": SPEC, "cond": c, "vocab": vocab, "kind": kind, "a_bin": name, "n": len(sel),
                                    "rate": e[0], "lo": e[1], "hi": e[2]})
            md.append(f"| {LABEL[c]} | {vocab} | " + " | ".join(cells) + " |")
    write_csv(os.path.join(TAB, "fpr_by_unpaired.csv"), fpr_csv)
    md.append("\n(b) Within-size slope of false alarms on unpaired listed parts (hard negatives only; set-size fixed effects; "
              "logit slope with L2 penalty 0.1, and linear-probability slope; item bootstrap within m, 1000 resamples).\n")
    md.append("| Model | readout | vocab | n hard neg | logit slope per unpaired part | linear-probability slope (per part) |")
    md.append("|---|---|---|---|---|---|")
    slope_csv = []
    for c in [x for x in PRIMARY if x in have]:
        for task, rname in readouts_for(c):
            allrows = [r for r in holistic_rows(c, recs[c], items, task) if r["valid"] and r["it"]["cls"] == "hard"]
            for vocab in ("names", "codes", "pooled"):
                rr = [(r["it"]["m"], r["it"]["a"], r["dec"]) for r in allrows if vocab == "pooled" or r["it"]["vocab"] == vocab]
                if len(rr) < 20:
                    continue
                (bl, bll, blh), (bp, bpl, bph) = slope_boot(rr)
                md.append(f"| {LABEL[c]} | {rname} | {vocab} | {len(rr)} | {bl:+.2f} [{bll:+.2f}, {blh:+.2f}] | {bp:+.3f} [{bpl:+.3f}, {bph:+.3f}] |")
                slope_csv.append({"spec": SPEC, "cond": c, "readout": task, "vocab": vocab, "n": len(rr),
                                  "logit_slope": bl, "lo": bll, "hi": blh, "lp_slope": bp, "lp_lo": bpl, "lp_hi": bph})
    write_csv(os.path.join(TAB, "fpr_slope_unpaired.csv"), slope_csv)

    # ================= 5. per-piece checks and remedies
    md.append("\n## 5. Per-piece checks and remedies (positives vs matched hard negatives)\n")
    md.append("Per-row: 12 isolated 'Are both X and Y on the roster?' questions per item; per-part: 24 'Is X on the roster?' questions. "
              "Row types: both / one / neither member on the roster. Items counted only when all 12 rows (or 24 parts) are valid.\n")
    md.append("### 5a. Per-row checks and per-row OR remedy\n")
    md.append("| Model | vocab | m | items complete/attempted | per-row acc | row FP rate, one member present | row FP, neither | row miss, both present | OR remedy TPR / FPR | OR remedy acc | holistic acc (same items) | OR minus holistic acc (paired) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rem_csv = []
    flags = defaultdict(list)
    piece_conds = [c for c in PRIMARY + ["llama70b"] if c in have]
    for c in piece_conds:
        hol = {r["it"]["item_id"]: r for r in holistic_rows(c, recs[c], items, "prob")}
        for vocab in ("names", "codes"):
            for m in (4, 8, 12, 16):
                att, comp = piece_counts(c, recs[c], items, "row", vocab, m)
                if comp == 0:
                    continue
                data = []
                for it in items.values():
                    if it["vocab"] != vocab or it["m"] != m or it["cls"] == "easy":
                        continue
                    rows = row_answers(c, recs[c], it)
                    if rows is None:
                        continue
                    roster = {ln[2:] for ln in it["state"].split("\n") if ln.startswith("- ")}
                    types = [(int(r["x"] in roster) + int(r["y"] in roster), int(p >= 0.5), t)
                             for r, p, t in zip(it["rows"], rows, it["row_truth"])]
                    h = hol.get(it["item_id"])
                    data.append(dict(it=it, types=types, rd=int(any(p >= 0.5 for p in rows)), rs=max(rows),
                                     hdec=(h["dec"] if h and h["valid"] else None)))
                G = {"pos": [d for d in data if d["it"]["cls"] == "pos"], "hard": [d for d in data if d["it"]["cls"] == "hard"]}
                nrows = 12 * len(data)
                racc = boot(lambda d: mean([int(p == t) for x in d["pos"] + d["hard"] for (_, p, t) in x["types"]]), G, prop_n=nrows)
                fp1 = [p for x in data for (k, p, t) in x["types"] if k == 1]
                fp0 = [p for x in data for (k, p, t) in x["types"] if k == 0]
                ms2 = [1 - p for x in data for (k, p, t) in x["types"] if k == 2]
                tpr = mean([x["rd"] for x in G["pos"]])
                fpr = mean([x["rd"] for x in G["hard"]])
                oacc = boot(lambda d: (mean([x["rd"] for x in d["pos"]]) + 1 - mean([x["rd"] for x in d["hard"]])) / 2, G,
                            prop_n=len(data))
                hs = {k: [x for x in v if x["hdec"] is not None] for k, v in G.items()}
                hacc = boot(lambda d: (mean([x["hdec"] for x in d["pos"]]) + 1 - mean([x["hdec"] for x in d["hard"]])) / 2, hs)
                diff = boot(lambda d: ((mean([x["rd"] for x in d["pos"]]) - mean([x["rd"] for x in d["hard"]])) -
                                       (mean([x["hdec"] for x in d["pos"]]) - mean([x["hdec"] for x in d["hard"]]))) / 2, hs)
                md.append(f"| {LABEL[c]} | {vocab} | {m} | {len(G['pos'])}+{len(G['hard'])} / {att} | {fmt(racc, 3)} | "
                          f"{mean(fp1):.3f} ({sum(fp1):.0f}/{len(fp1)}) | {mean(fp0):.3f} ({sum(fp0):.0f}/{len(fp0)}) | "
                          f"{mean(ms2):.3f} ({sum(ms2):.0f}/{len(ms2)}) | {tpr:.2f} / {fpr:.2f} | {fmt(oacc)} | {fmt(hacc)} | "
                          f"{diff[0]:+.2f} [{diff[1]:+.2f}, {diff[2]:+.2f}] |")
                if c in PRIMARY and (racc[0] < 0.95 or oacc[0] < 0.90):
                    flags[c].append(f"{vocab} m={m}: per-row acc {racc[0]:.3f}, OR acc {oacc[0]:.2f}")
                rem_csv.append({"spec": SPEC, "cond": c, "vocab": vocab, "m": m, "n_items": len(data), "attempted": att,
                                "row_acc": racc[0], "row_acc_lo": racc[1], "row_acc_hi": racc[2],
                                "row_fp_one": mean(fp1), "row_fp_neither": mean(fp0), "row_miss_both": mean(ms2),
                                "or_tpr": tpr, "or_fpr": fpr, "or_acc": oacc[0], "or_acc_lo": oacc[1], "or_acc_hi": oacc[2],
                                "hol_acc_same": hacc[0], "or_minus_hol": diff[0], "or_minus_hol_lo": diff[1], "or_minus_hol_hi": diff[2]})
                if c in PRIMARY:
                    fig[(c, "prob", vocab, m)]["rows"] = {"row_acc": racc, "or_acc": oacc}
    write_csv(os.path.join(TAB, "rows_and_or_remedy.csv"), rem_csv)

    md.append("\n### 5b. Per-part checks and per-part join remedy\n")
    md.append("| Model | vocab | m | items complete/attempted | per-part acc | part FP (absent name said present) | part miss | join TPR / FPR | join acc | join AUROC (Jev graded) | holistic acc (same items) | join minus holistic acc (paired) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    join_csv = []
    for c in piece_conds:
        hol = {r["it"]["item_id"]: r for r in holistic_rows(c, recs[c], items, "prob")}
        for vocab in ("names", "codes"):
            for m in (4, 8, 12, 16):
                att, comp = piece_counts(c, recs[c], items, "part", vocab, m)
                if comp == 0:
                    continue
                data = []
                for it in items.values():
                    if it["vocab"] != vocab or it["m"] != m or it["cls"] == "easy":
                        continue
                    parts = part_answers(c, recs[c], it)
                    if parts is None:
                        continue
                    ok = [(t, int(parts[x["x"]] >= 0.5)) for x, t in zip(it["parts"], it["part_truth"])]
                    jd = int(any(parts[r["x"]] >= 0.5 and parts[r["y"]] >= 0.5 for r in it["rows"]))
                    js = max(min(parts[r["x"]], parts[r["y"]]) for r in it["rows"])
                    h = hol.get(it["item_id"])
                    data.append(dict(it=it, ok=ok, jd=jd, js=js, hdec=(h["dec"] if h and h["valid"] else None)))
                G = {"pos": [d for d in data if d["it"]["cls"] == "pos"], "hard": [d for d in data if d["it"]["cls"] == "hard"]}
                pacc = boot(lambda d: mean([int(t == p) for x in d["pos"] + d["hard"] for (t, p) in x["ok"]]), G, prop_n=24 * len(data))
                pfp = [p for x in data for (t, p) in x["ok"] if t == 0]
                pms = [1 - p for x in data for (t, p) in x["ok"] if t == 1]
                tpr = mean([x["jd"] for x in G["pos"]])
                fpr = mean([x["jd"] for x in G["hard"]])
                jacc = boot(lambda d: (mean([x["jd"] for x in d["pos"]]) + 1 - mean([x["jd"] for x in d["hard"]])) / 2, G,
                            prop_n=len(data))
                jauc = boot(lambda d: auroc([x["js"] for x in d["pos"]], [x["js"] for x in d["hard"]]), G) if c == "jev" else None
                hs = {k: [x for x in v if x["hdec"] is not None] for k, v in G.items()}
                hacc = boot(lambda d: (mean([x["hdec"] for x in d["pos"]]) + 1 - mean([x["hdec"] for x in d["hard"]])) / 2, hs)
                diff = boot(lambda d: ((mean([x["jd"] for x in d["pos"]]) - mean([x["jd"] for x in d["hard"]])) -
                                       (mean([x["hdec"] for x in d["pos"]]) - mean([x["hdec"] for x in d["hard"]]))) / 2, hs)
                md.append(f"| {LABEL[c]} | {vocab} | {m} | {len(G['pos'])}+{len(G['hard'])} / {att} | {fmt(pacc, 3)} | "
                          f"{mean(pfp):.3f} ({sum(pfp):.0f}/{len(pfp)}) | {mean(pms):.3f} ({sum(pms):.0f}/{len(pms)}) | "
                          f"{tpr:.2f} / {fpr:.2f} | {fmt(jacc)} | {fmt(jauc) if jauc else 'n/a (binary)'} | {fmt(hacc)} | "
                          f"{diff[0]:+.2f} [{diff[1]:+.2f}, {diff[2]:+.2f}] |")
                join_csv.append({"spec": SPEC, "cond": c, "vocab": vocab, "m": m, "n_items": len(data), "attempted": att,
                                 "part_acc": pacc[0], "part_acc_lo": pacc[1], "part_acc_hi": pacc[2],
                                 "join_tpr": tpr, "join_fpr": fpr, "join_acc": jacc[0], "join_acc_lo": jacc[1],
                                 "join_acc_hi": jacc[2], "hol_acc_same": hacc[0], "join_minus_hol": diff[0],
                                 "join_minus_hol_lo": diff[1], "join_minus_hol_hi": diff[2]})
                if c in PRIMARY:
                    fig[(c, "prob", vocab, m)]["parts"] = {"part_acc": pacc, "join_acc": jacc}
    write_csv(os.path.join(TAB, "parts_and_join_remedy.csv"), join_csv)

    md.append("\n### 5c. Decision rule ('each piece right' requires per-row acc >= 0.95 and OR-remedy acc >= 0.90 at every m measured)\n")
    for c in [x for x in PRIMARY if x in have]:
        tested = any(k[0] == c and "rows" in v for k, v in fig.items())
        if not tested:
            md.append(f"- {LABEL[c]}: per-row checks not measured")
        elif flags[c]:
            md.append(f"- {LABEL[c]}: **does not hold** ({'; '.join(flags[c])})")
        else:
            md.append(f"- {LABEL[c]}: holds at every measured cell")

    # Jev easy negatives with remedies
    if "jev" in have:
        md.append("\n### 5d. Jev remedies on easy negatives\n")
        md.append("| vocab | m | n | holistic easy acc | OR remedy easy acc | join remedy easy acc |")
        md.append("|---|---|---|---|---|---|")
        for vocab in ("names", "codes"):
            for m in (4, 8, 12, 16):
                d = []
                for it in items.values():
                    if it["vocab"] == vocab and it["m"] == m and it["cls"] == "easy":
                        r = recs["jev"].get(f"jev|{it['item_id']}|full")
                        rows, parts = row_answers("jev", recs["jev"], it), part_answers("jev", recs["jev"], it)
                        if r and rows and parts:
                            d.append((int(r["parsed"]["holistic"] >= 0.5), int(any(p >= 0.5 for p in rows)),
                                      int(any(parts[x["x"]] >= 0.5 and parts[x["y"]] >= 0.5 for x in it["rows"]))))
                if d:
                    md.append(f"| {vocab} | {m} | {len(d)} | {1 - mean([x[0] for x in d]):.2f} | {1 - mean([x[1] for x in d]):.2f} | {1 - mean([x[2] for x in d]):.2f} |")

    # ================= 6. names vs codes paired
    md.append("\n## 6. Names minus codes (item-paired bootstrap; AUROC and matched accuracy)\n")
    md.append("| Model | readout | m | AUROC names - codes | acc names - codes | n base items |")
    md.append("|---|---|---|---|---|---|")
    for c in [x for x in PRIMARY if x in have]:
        for task, rname in readouts_for(c):
            allrows = {r["it"]["item_id"]: r for r in holistic_rows(c, recs[c], items, task) if r["valid"]}
            for m in (4, 8, 12, 16):
                pairs = {"pos": [], "hard": []}
                for it in items.values():
                    if it["vocab"] == "names" and it["m"] == m and it["cls"] in pairs:
                        a_, b_ = allrows.get(it["item_id"]), allrows.get(it["item_id"].replace("names-", "codes-", 1))
                        if a_ and b_:
                            pairs[it["cls"]].append((a_, b_))
                if not pairs["pos"] or not pairs["hard"]:
                    continue
                da = boot(lambda d: auroc([x[0]["score"] for x in d["pos"]], [x[0]["score"] for x in d["hard"]]) -
                          auroc([x[1]["score"] for x in d["pos"]], [x[1]["score"] for x in d["hard"]]), pairs)
                dc = boot(lambda d: ((mean([x[0]["dec"] for x in d["pos"]]) - mean([x[0]["dec"] for x in d["hard"]])) -
                                     (mean([x[1]["dec"] for x in d["pos"]]) - mean([x[1]["dec"] for x in d["hard"]]))) / 2, pairs)
                md.append(f"| {LABEL[c]} | {rname} | {m} | {da[0]:+.2f} [{da[1]:+.2f}, {da[2]:+.2f}] | {dc[0]:+.2f} [{dc[1]:+.2f}, {dc[2]:+.2f}] | {len(pairs['pos'])}+{len(pairs['hard'])} |")

    # ================= 7. Jev holistic-only control
    if "jev" in have and "jev_holonly" in have:
        md.append("\n## 7. Jev: holistic-only call versus the same call with 36 per-piece branches\n")
        md.append("| m | n items | mean abs diff in holistic P | max abs diff | decision agreement | AUROC shared call (names+codes) | AUROC holistic-only | median latency s (only / +36) | mean input tokens (only / +36) |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for m in (4, 8, 12, 16, "all"):
            prs = []
            for it in items.values():
                if m != "all" and it["m"] != m:
                    continue
                a_ = recs["jev"].get(f"jev|{it['item_id']}|full")
                b_ = recs["jev_holonly"].get(f"jev_holonly|{it['item_id']}|full")
                if a_ and b_:
                    prs.append((it, a_, b_))
            if not prs:
                continue
            d = [abs(a_["parsed"]["holistic"] - b_["parsed"]["holistic"]) for _, a_, b_ in prs]
            agree = mean([int((a_["parsed"]["holistic"] >= 0.5) == (b_["parsed"]["holistic"] >= 0.5)) for _, a_, b_ in prs])
            pa = [a_["parsed"]["holistic"] for it, a_, _ in prs if it["cls"] == "pos"]
            ha = [a_["parsed"]["holistic"] for it, a_, _ in prs if it["cls"] == "hard"]
            pb = [b_["parsed"]["holistic"] for it, _, b_ in prs if it["cls"] == "pos"]
            hb = [b_["parsed"]["holistic"] for it, _, b_ in prs if it["cls"] == "hard"]
            md.append(f"| {m} | {len(prs)} | {mean(d):.3f} | {max(d):.2f} | {agree:.3f} | {auroc(pa, ha):.3f} | {auroc(pb, hb):.3f} | "
                      f"{float(np.median([b_['latency'] for _, _, b_ in prs])):.2f} / {float(np.median([a_['latency'] for _, a_, _ in prs])):.2f} | "
                      f"{mean([b_['usage']['input_tokens'] for _, _, b_ in prs]):.0f} / {mean([a_['usage']['input_tokens'] for _, a_, _ in prs]):.0f} |")

    # ================= 8. Haiku diagnostic
    diag = os.path.join(RAW, "haiku_diag.jsonl")
    if os.path.exists(diag):
        md.append("\n## 8. Haiku m = 4 diagnostic (names, 60 pos + 60 matched hard neg, Yes/No)\n")
        md.append("| arm | n | non-valid | TPR | FPR | median output tokens |")
        md.append("|---|---|---|---|---|---|")
        dr = {}
        for line in open(diag):
            r = json.loads(line)
            if not r.get("error"):
                dr[r["key"]] = r
        hk = holistic_rows("haiku", recs["haiku"], items, "yn")
        main_arm = [r for r in hk if r["it"]["vocab"] == "names" and r["it"]["m"] == 4 and r["it"]["cls"] != "easy"]
        pos = [r["dec"] for r in main_arm if r["valid"] and r["it"]["cls"] == "pos"]
        neg = [r["dec"] for r in main_arm if r["valid"] and r["it"]["cls"] == "hard"]
        md.append(f"| main (system prompt 'Answer immediately…') | {len(main_arm)} | {sum(not r['valid'] for r in main_arm)} | "
                  f"{fmt(boot(prop(None), {'x': [{'dec': v} for v in pos]}, prop_n=len(pos)))} | "
                  f"{fmt(boot(prop(None), {'x': [{'dec': v} for v in neg]}, prop_n=len(neg)))} | 1 |")
        for arm, desc in (("nosys", "no system prompt"), ("rephrase", "reworded question"), ("cot", "step-by-step allowed")):
            rs = [r for r in dr.values() if r["arm"] == arm]
            pos = [r["parsed"] for r in rs if r.get("parsed") is not None and "-pos-" in r["item_id"]]
            neg = [r["parsed"] for r in rs if r.get("parsed") is not None and "-hard-" in r["item_id"]]
            md.append(f"| {arm} ({desc}) | {len(rs)} | {sum(r.get('parsed') is None for r in rs)} | "
                      f"{fmt(boot(prop(None), {'x': [{'dec': v} for v in pos]}, prop_n=len(pos)))} | "
                      f"{fmt(boot(prop(None), {'x': [{'dec': v} for v in neg]}, prop_n=len(neg)))} | "
                      f"{float(np.median([r['usage']['output_tokens'] for r in rs])):.0f} |")

    # ================= R. v3.2 robustness
    md.append("\n## R0. Primary metric (v3.2 analysis): matched-hard-negative AUROC, change from m = 4 to m = 16\n")
    md.append("Independent items at each m; difference CI from independent bootstraps (2000 resamples each). 'falls' = CI of the difference below 0.\n")
    md.append("| Model | readout | vocab | AUROC m=4 | AUROC m=16 | m=16 minus m=4 | falls? | m=16 CI includes 0.5? |")
    md.append("|---|---|---|---|---|---|---|---|")
    cover = defaultdict(dict)
    for c in [x for x in PRIMARY + ["sonnet5_reason"] if x in have]:
        task = "prob"
        rows_all = holistic_rows(c, recs[c], items, task)
        for vocab in ("names", "codes"):
            r4 = [r for r in rows_all if r["valid"] and r["it"]["vocab"] == vocab and r["it"]["m"] == 4 and r["it"]["cls"] != "easy"]
            r16 = [r for r in rows_all if r["valid"] and r["it"]["vocab"] == vocab and r["it"]["m"] == 16 and r["it"]["cls"] != "easy"]
            if not r4 or not r16:
                continue
            G = {"p4": [r for r in r4 if r["it"]["cls"] == "pos"], "h4": [r for r in r4 if r["it"]["cls"] == "hard"],
                 "p16": [r for r in r16 if r["it"]["cls"] == "pos"], "h16": [r for r in r16 if r["it"]["cls"] == "hard"]}
            a4 = boot(lambda d: auroc([r["score"] for r in d["p4"]], [r["score"] for r in d["h4"]]), G)
            a16 = boot(lambda d: auroc([r["score"] for r in d["p16"]], [r["score"] for r in d["h16"]]), G)
            dd = boot(lambda d: auroc([r["score"] for r in d["p16"]], [r["score"] for r in d["h16"]]) -
                      auroc([r["score"] for r in d["p4"]], [r["score"] for r in d["h4"]]), G)
            falls = dd[2] < 0
            cover[c][vocab] = (falls, a16[1] <= 0.5 <= a16[2])
            md.append(f"| {LABEL[c]} | {readouts_for(c)[0][1]} | {vocab} | {fmt(a4)} | {fmt(a16)} | {dd[0]:+.2f} [{dd[1]:+.2f}, {dd[2]:+.2f}] | "
                      f"{'yes' if falls else 'no'} | {'yes' if a16[1] <= 0.5 <= a16[2] else 'no'} |")
    md.append("\nModel coverage (v3.2 item 4: claim needs Jev plus at least 3 of the 5 other primary families): AUROC falls from m=4 to m=16 in "
              + "; ".join(f"{vocab}: " + ", ".join(LABEL[c] for c in PRIMARY if cover.get(c, {}).get(vocab, (False,))[0])
                         for vocab in ("names", "codes")) + ".\n")

    para_items = load_para_items()
    if para_items:
        subset_ids = {it["base_id"] for it in para_items.values()}
        t0_items = {k: v for k, v in items.items() if v["vocab"] == "names" and v["base_id"] in subset_ids}
        md.append("\n## R1. Wording robustness (v3.2 item 1): matched-hard-negative AUROC per template\n")
        md.append("Subset: names, m = 4 and 16, 40 positives + 40 matched hard negatives per m (same base items in every template). "
                  "T0 = main wording; T1, T2 = paraphrased rules text, roster layout and question. Jev: holistic-only call in every template "
                  "(T0 also shown for the shared call). LLMs: AUROC from the stated probability; matched accuracy for both readouts in separate columns.\n")
        md.append("| Model | m | T0 AUROC | T1 AUROC | T2 AUROC | T0 / T1 / T2 acc (stated prob) | T0 / T1 / T2 acc (yes/no) | non-valid T1/T2 |")
        md.append("|---|---|---|---|---|---|---|---|")
        para_recs = {c: load_cond(c, "_para") for c in ["jev_holonly"] + PRIMARY_LLM}
        for c in ["jev"] + PRIMARY_LLM:
            pc = "jev_holonly" if c == "jev" else c
            if not para_recs.get(pc):
                continue
            for m in (4, 16):
                cells, accs, accy, nvs = [], [], [], []
                for tpl in ("T0", "T1", "T2"):
                    for task in (("prob",) if c == "jev" else ("prob", "yn")):
                        if tpl == "T0":
                            src = recs.get("jev_holonly" if c == "jev" else c, {})
                            its = {k: v for k, v in t0_items.items() if v["m"] == m}
                            rr = holistic_rows(pc, src, its, task)
                        else:
                            its = {k: v for k, v in para_items.items() if v["m"] == m and v["template"] == tpl}
                            rr = holistic_rows(pc, para_recs[pc], its, task)
                        if not rr:
                            (cells if task == "prob" else accy).append("n/a")
                            if task == "prob":
                                accs.append("n/a")
                            continue
                        res = class_metrics(rr)
                        if task == "prob":
                            cells.append(fmt(res["AUROC"]))
                            accs.append(f"{res['acc_matched'][0]:.2f}")
                            if tpl != "T0":
                                nvs.append(str(res["n_nonvalid"]))
                        else:
                            accy.append(f"{res['acc_matched'][0]:.2f}")
                        if tpl != "T0" and task == "prob":
                            fig[(c, "para", tpl, m)]["holistic"] = res
                extra = ""
                if c == "jev" and "jev" in have:
                    its = {k: v for k, v in t0_items.items() if v["m"] == m}
                    rs = class_metrics(holistic_rows("jev", recs["jev"], its))
                    extra = f" (shared call {rs['AUROC'][0]:.2f})"
                md.append(f"| {LABEL[c]} | {m} | {cells[0]}{extra} | {cells[1]} | {cells[2]} | {' / '.join(accs)} | "
                          f"{' / '.join(accy) if accy else 'n/a'} | {'/'.join(nvs)} |")

    jc = load_cond("jev_choice")
    if jc:
        md.append("\n## R2. Jev readout robustness (v3.2 item 2): Noul versus two-option Choice\n")
        md.append("Choice criteria: 'contains' / 'does_not_contain' (score = P(contains)); names, 40 pos + 40 matched hard per m. "
                  "Noul values are from the main shared call and the holistic-only call on the same items.\n")
        md.append("| m | n pos/hard | AUROC Noul (shared call) | AUROC Noul (holistic-only) | AUROC Choice | FPR Noul / Choice | TPR Noul / Choice |")
        md.append("|---|---|---|---|---|---|---|")
        for m in (4, 8, 12, 16):
            ids = {k.split("|")[1] for k in jc}
            its = {k: v for k, v in items.items() if k in ids and v["m"] == m}
            if not its:
                continue
            rc = class_metrics(holistic_rows("jev_choice", jc, its))
            rn = class_metrics(holistic_rows("jev", recs["jev"], its)) if "jev" in have else None
            rh = class_metrics(holistic_rows("jev_holonly", recs["jev_holonly"], its)) if "jev_holonly" in have else None
            md.append(f"| {m} | {rc['n_pos']}/{rc['n_hard']} | {fmt(rn['AUROC']) if rn else 'n/a'} | {fmt(rh['AUROC']) if rh else 'n/a'} | {fmt(rc['AUROC'])} | "
                      f"{rn['FPR_hard'][0]:.2f} / {rc['FPR_hard'][0]:.2f} | {rn['TPR'][0]:.2f} / {rc['TPR'][0]:.2f} |")

    md.append("\n## R3. Test-retest (v3.2 item 3): identical calls repeated once\n")
    md.append("| Model | n pairs | exact agreement of output | decision agreement | mean abs diff (probability scale) | max abs diff | per-piece decision agreement (Jev) |")
    md.append("|---|---|---|---|---|---|---|")
    for c in ["jev"] + PRIMARY_LLM + ["sonnet5_reason"]:
        rt = load_cond(c, "_rt1")
        if not rt or c not in have:
            continue
        pairs = [(recs[c][k], r) for k, r in rt.items() if k in recs[c]]
        if c == "jev":
            v = [(a_["parsed"]["holistic"], b_["parsed"]["holistic"]) for a_, b_ in pairs]
            pk = [(a_["parsed"][q] >= 0.5) == (b_["parsed"][q] >= 0.5) for a_, b_ in pairs for q in a_["parsed"] if q != "holistic"]
            dec = [(x >= 0.5) == (y >= 0.5) for x, y in v]
            piece = f"{mean(pk):.4f} ({len(pk)} pieces)"
        else:
            v = [(a_["parsed"] / 100, b_["parsed"] / 100) for a_, b_ in pairs if a_["parsed"] is not None and b_["parsed"] is not None]
            dec = [(x > 0.5) == (y > 0.5) for x, y in v]
            piece = "n/a"
        dif = [abs(x - y) for x, y in v]
        md.append(f"| {LABEL[c]} | {len(v)} | {mean([x == y for x, y in v]):.3f} | {mean(dec):.3f} | {mean(dif):.3f} | {max(dif) if dif else float('nan'):.2f} | {piece} |")
    md.append("\nSonnet 5 rejects the temperature parameter (default sampling), so its row is sampling spread, not API nondeterminism.\n")

    # ================= 9. superseded: pinned vs unpinned
    md.append("\n## 9. Superseded unpinned runs versus pinned primary runs (holistic, stated probability)\n")
    md.append("| model | vocab | m | AUROC unpinned | AUROC pinned | acc unpinned | acc pinned | providers (unpinned) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for old, new in (("llama70b", "llama70b_pin"), ("mistral24b", "mistral24b_pin")):
        if old not in have or new not in have:
            continue
        ro, rn = holistic_rows(old, recs[old], items, "prob"), holistic_rows(new, recs[new], items, "prob")
        prov = Counter(r.get("provider") for r in recs[old].values() if r["task"] in ("prob", "yn"))
        for vocab in ("names", "codes"):
            for m in (4, 8, 12, 16):
                a_ = class_metrics([r for r in ro if r["it"]["vocab"] == vocab and r["it"]["m"] == m])
                b_ = class_metrics([r for r in rn if r["it"]["vocab"] == vocab and r["it"]["m"] == m])
                if a_["n_pos"] and b_["n_pos"]:
                    md.append(f"| {LABEL[new]} | {vocab} | {m} | {fmt(a_['AUROC'])} | {fmt(b_['AUROC'])} | {fmt(a_['acc_matched'])} | "
                              f"{fmt(b_['acc_matched'])} | {', '.join(f'{k} {v}' for k, v in prov.most_common(4))}… |")

    # ================= 10. cost / tokens / latency / providers / categories
    md.append("\n## 10. Calls, readout categories, tokens, spend, latency, providers (per-call logs)\n")
    md.append("| condition | task | calls | valid / refusal / attempted-reasoning / invalid | ambiguous-scale | mean in tok | mean out tok | USD | median latency s | providers | spec of log rows |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    cost_csv = []
    for c in have:
        by = defaultdict(list)
        for r in recs[c].values():
            by[r["task"].split(":")[0]].append(r)
        for t, rs in sorted(by.items()):
            u = [r.get("usage") or {} for r in rs]
            tin = [x.get("input_tokens", x.get("prompt_tokens", 0)) for x in u]
            tout = [x.get("output_tokens", x.get("completion_tokens", 0)) for x in u]
            usd = sum(float(r.get("usd") or 0) for r in rs)
            lat = float(np.median([r["latency"] for r in rs]))
            cat = Counter(r.get("category", "valid" if r.get("valid") else "invalid") for r in rs)
            amb = sum(1 for r in rs if r.get("ambiguous_scale"))
            prov = Counter(r.get("provider") for r in rs)
            specs = Counter(r.get("spec", "pre-v3") for r in rs)
            md.append(f"| {LABEL[c]} | {t} | {len(rs)} | {cat['valid']} / {cat['refusal']} / {cat['attempted-reasoning']} / {cat['invalid']} | {amb} | "
                      f"{mean(tin):.0f} | {mean(tout):.1f} | {usd:.4f} | {lat:.2f} | {', '.join(f'{k} {v}' for k, v in prov.most_common(5))} | "
                      f"{', '.join(f'{k} {v}' for k, v in specs.items())} |")
            if c.startswith("sonnet"):
                md.append(f"|  | output tokens min / median / max | | | | | {min(tout)} / {float(np.median(tout)):.0f} / {max(tout)} | | | | |")
            cost_csv.append({"spec": SPEC, "cond": c, "task": t, "calls": len(rs), **{f"n_{k}": v for k, v in cat.items()},
                             "mean_in": mean(tin), "mean_out": mean(tout), "usd": usd, "median_latency": lat})
    write_csv(os.path.join(TAB, "cost.csv"), cost_csv)

    with open(os.path.join(HERE, "results", "summary_tables.md"), "w") as f:
        f.write(f"# E2 pairs: generated tables (analyze.py, spec {SPEC})\n\n" + "\n".join(md) + "\n")
    make_figure(fig)
    print("wrote results/summary_tables.md, results/tables/*.csv, results/fig1b.{pdf,png}")


# ------------------------------------------------------------------ figure
def make_figure(fig, m=16, vocab="names"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [c for c in PRIMARY + ["sonnet5_reason"] if (c, "prob", vocab, m) in fig]
    if not conds:
        return
    short = {"jev": "Jev", "llama70b_pin": "Llama-70B", "mistral24b_pin": "Mistral-24B", "gemma27b": "Gemma-27B",
             "deepseekv32": "DeepSeek-V3.2", "haiku": "Haiku 4.5", "sonnet5_reason": "Sonnet 5\n(reasoning)"}
    bars = [("per-part check acc", "#0072B2", "parts", "part_acc"),
            ("per-row check acc", "#56B4E9", "rows", "row_acc"),
            ("holistic AUROC", "#000000", "holistic", "AUROC"),
            ("holistic false-alarm rate", "#D55E00", "holistic", "FPR_hard"),
            ("holistic miss rate", "#E69F00", "holistic", "MISS"),
            ("remedy: per-row OR acc", "#009E73", "rows", "or_acc"),
            ("remedy: per-part join acc", "#CC79A7", "parts", "join_acc")]
    f, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.1), gridspec_kw={"width_ratios": [2.3, 1]})
    w = 0.11
    for bi, (name, col, src, key) in enumerate(bars):
        xs, ys, lo, hi = [], [], [], []
        for ci, c in enumerate(conds):
            d = fig[(c, "prob", vocab, m)].get(src)
            if not d:
                continue
            if key == "MISS":
                e, l, h = d["TPR"]
                val = (1 - e, 1 - h, 1 - l)
            else:
                val = d[key]
            xs.append(ci + (bi - 3) * w)
            ys.append(val[0])
            lo.append(max(val[0] - val[1], 0))
            hi.append(max(val[2] - val[0], 0))
        ax.bar(xs, ys, w, color=col, label=name, yerr=[lo, hi], capsize=1.0, error_kw={"lw": 0.5})
    labels = []
    for c in conds:
        h = fig[(c, "prob", vocab, m)]["holistic"]
        labels.append(f"{short[c]}\n(n={h['n_pos']}+{h['n_hard']})")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy / rate / AUROC", fontsize=8)
    ax.set_title(f"Roster of m = {m} ({vocab}), positives vs matched hard negatives", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    # right: holistic AUROC vs m
    cols = {"jev": "#000000", "llama70b_pin": "#D55E00", "mistral24b_pin": "#E69F00", "gemma27b": "#009E73",
            "deepseekv32": "#0072B2", "haiku": "#CC79A7"}
    for c in [x for x in PRIMARY if any((x, "prob", vocab, mm) in fig for mm in (4, 8, 12, 16))]:
        ms = [mm for mm in (4, 8, 12, 16) if (c, "prob", vocab, mm) in fig and "holistic" in fig[(c, "prob", vocab, mm)]]
        ys = [fig[(c, "prob", vocab, mm)]["holistic"]["AUROC"][0] for mm in ms]
        ax2.plot(ms, ys, marker="o", ms=3, lw=1.4 if c == "jev" else 0.9, color=cols[c], label=short[c])
    for c in ["jev"]:
        ms = [mm for mm in (4, 8, 12, 16) if "parts" in fig.get((c, "prob", vocab, mm), {})]
        ax2.plot(ms, [fig[(c, "prob", vocab, mm)]["parts"]["join_acc"][0] for mm in ms], ls="--", color="#000000",
                 lw=0.9, label="Jev per-part join (acc)")
        ms = [mm for mm in (4, 8, 12, 16) if "holistic" in fig.get((c, "prob", vocab, mm), {})]
        ax2.plot(ms, [fig[(c, "prob", vocab, mm)]["holistic"]["easy_acc"][0] for mm in ms], ls=":", color="#777777",
                 lw=0.9, label="Jev easy-negative acc")
    ax2.set_xticks([4, 8, 12, 16])
    ax2.set_xlabel("roster size m", fontsize=8)
    ax2.set_ylabel("holistic AUROC", fontsize=8)
    ax2.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax2.set_ylim(0.4, 1.02)
    ax2.tick_params(labelsize=7)
    ax2.set_title(f"Holistic AUROC vs roster size ({vocab})", fontsize=8)
    ax2.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    f.tight_layout()
    for ext in ("pdf", "png"):
        f.savefig(os.path.join(HERE, "results", f"fig1b.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(f)


if __name__ == "__main__":
    main()
