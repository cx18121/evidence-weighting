"""E1/E3 analysis, spec v3.1-v3.3. Re-parses every value from the raw logged outputs (never the stored `parsed`
field) and writes every number in results/summary.md, plus results/metrics.json, results/invalid_by_cell.csv,
results/fig1a.{pdf,png} and results/fig_gt_fits.{pdf,png}.

Analysis choices:
- Probabilities (model and normative) are clipped to [0.005, 0.995] before logits.
- Primary: valid replies only. ambiguous_scale replies (0 < x < 1 written as a decimal) are excluded from the
  primary analysis and reported in four sensitivity variants: read on the 0-100 scale (amb100), read as a 0-1
  probability (amb01), and every non-valid reply imputed at 0.5 (imp05) or at the normative answer (impnorm).
- CIs: two-stage cluster bootstrap (B=1000): within each design cell (n, k, r, format) resample the three
  (domain, n, k, r) clusters with replacement, then seeds within each cluster; percentile 95% intervals.
  Proportions whose point estimate is 0 or 1 use Wilson intervals on the case count.
- Signed logit error = (L(model) - L(normative)) * sign(2k - n), cells with k = n/2 excluded: positive means more
  extreme than normative in the correct direction (overconfident), negative means underconfident.
- Slopes and parametric fits use only cells whose normative answer lies inside the clip range
  (|normative log-odds| <= logit(0.995)), so a normative reporter has slope 1 exactly.
- Expected Brier and log loss use the normative posterior q as the outcome probability.
"""
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate import SPEC, posterior  # noqa: E402
from parsing import (parse_answer_line, parse_count, parse_noul, parse_prob, parse_yesno,  # noqa: E402
                     parse_choice)

RAW = HERE / "results" / "raw"
OUT = HERE / "results"
EPS = 0.005
LCLIP = math.log((1 - EPS) / EPS)
B = int(os.environ.get("E1_BOOT", 1000))
MODELS = ["jev", "llama", "mistral", "gemma", "deepseek", "haiku"]
LLMS = ["llama", "mistral", "gemma", "deepseek", "haiku"]
NAMES = {"jev": "Jev", "llama": "Llama-3.3-70B", "mistral": "Mistral-Small-3.2", "gemma": "Gemma-3-27B",
         "deepseek": "DeepSeek-V3.2", "haiku": "Haiku 4.5", "sonnet": "Sonnet 5 (reasoning)",
         "jev_remedy": "Jev per-clue remedy"}
VARIANTS = ["primary", "amb100", "amb01", "imp05", "impnorm"]
R_GRID = (0.52, 0.55, 0.70)
N_GRID = (4, 8, 16, 32, 64)
PHASES = ["main", "count", "e3", "holonly", "jevchoice", "yesno", "pad", "flip", "clue", "retest", "sonnet",
          "sonnet_retest", "unpinned"]

# The released aggregate metrics do not substitute for raw responses. A full
# reanalysis requires fresh calls for every experimental phase.
required = [HERE / "cases.jsonl", HERE.parents[1] / "api" / "ledger.jsonl"]
required += [RAW / f"{phase}.jsonl" for phase in PHASES]
missing = [str(p.relative_to(HERE.parents[1])) for p in required if not p.is_file()]
if missing:
    raise SystemExit("Cannot recompute model results without new run data: " + ", ".join(missing))

CASES = {}
for line in open(HERE / "cases.jsonl"):
    c = json.loads(line)
    CASES[c["id"]] = c


def logit(x):
    x = np.clip(np.asarray(x, float), EPS, 1 - EPS)
    return np.log(x / (1 - x))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))

# ================================================================ loading and re-parsing

def raw_text(row):
    """Text and finish reason re-extracted from the raw provider response."""
    raw = row.get("raw") or {}
    if row["provider"] == "openrouter":
        ch = (raw.get("choices") or [{}])[0]
        return (ch.get("message") or {}).get("content"), ch.get("finish_reason")
    if row["provider"] == "anthropic":
        text = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
        return text, raw.get("stop_reason")
    return None, None


def reparse(row):
    kind = row["kind"]
    if row.get("category") == "api_error":
        return dict(value=None, cat="api_error", alt=None)
    if row["model"] == "jev":
        d = row["raw"]
        if kind == "jev_count":
            probs = d["answers"]["count"]["probabilities"]
            bins = d["answers"]["sharebin"]["probabilities"]
            arg = max(probs, key=probs.get)
            return dict(value=int(arg), cat="valid", alt=None,
                        count_exp=sum(int(k) * v for k, v in probs.items()) / max(sum(probs.values()), 1e-12),
                        sharebin=max(bins, key=bins.get))
        if kind in ("choice", "choice_rev"):
            v, cat = parse_choice(d, "holistic")
        else:
            v, cat = parse_noul(d, "holistic")
        out = dict(value=v, cat=cat, alt=None)
        if kind == "holistic_clues":
            out["clues"] = [parse_noul(d, f"c{j:02d}")[0] for j in range(row["case"]["n"])]
        return out
    text, fin = raw_text(row)
    if kind == "count":
        v, cat = parse_count(text, row["case"]["n"], fin)
        return dict(value=v, cat=cat, alt=None)
    if kind in ("yesno", "clue"):
        v, cat = parse_yesno(text, fin)
        return dict(value=v, cat=cat, alt=None)
    if kind == "sonnet":
        v, cat, alt = parse_answer_line(text, fin)
        return dict(value=v, cat=cat, alt=alt)
    v, cat, alt = parse_prob(text, fin)
    return dict(value=v, cat=cat, alt=alt)


def load(phase):
    path = RAW / f"{phase}.jsonl"
    if not path.exists():
        return []
    best = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("spec") != SPEC:
            continue
        k = r["key"]
        if k not in best or best[k].get("category") == "api_error":
            best[k] = r
    recs = []
    for r in best.values():
        rec = dict(r["case"])
        rec.update(phase=phase, model=r["model"], kind=r["kind"], clue_idx=r.get("clue_idx"),
                   provider_returned=r.get("provider_returned"), model_returned=r.get("model_returned"),
                   usage=r.get("usage"), latency=r.get("latency_s"), stored_cat=r.get("category"),
                   text=raw_text(r)[0] if r["model"] != "jev" else None, clue_truth=r.get("clue_truth"),
                   reasoning_tokens=r.get("reasoning_tokens"), max_tokens=r.get("max_tokens"),
                   system=r.get("system"), finish=raw_text(r)[1] if r["model"] != "jev" else None)
        rec.update(reparse(r))
        recs.append(rec)
    return recs


def value(rec, variant):
    cat, v = rec["cat"], rec["value"]
    if cat == "valid":
        return v
    if variant == "amb100" and cat == "ambiguous_scale":
        return v
    if variant == "amb01" and cat == "ambiguous_scale":
        return rec["alt"]
    if variant == "imp05":
        return 0.5
    if variant == "impnorm":
        return rec["truth"]
    return None

# ================================================================ bootstrap

class Data:
    """Arrays for one (model, condition) set of E1 records with a usable value."""

    def __init__(self, recs, vals):
        keep = [(r, v) for r, v in zip(recs, vals) if v is not None]
        self.recs = [r for r, _ in keep]
        g = lambda k: np.array([r[k] for r in self.recs])  # noqa: E731
        self.n = g("n").astype(float) if keep else np.array([])
        self.k = g("k").astype(float) if keep else np.array([])
        self.r = g("r").astype(float) if keep else np.array([])
        self.share = g("share").astype(float) if keep else np.array([])
        self.q = g("truth").astype(float) if keep else np.array([])
        self.p = np.array([v for _, v in keep], float)
        self.template = g("template") if keep else np.array([])
        self.domain = [r["domain"] for r in self.recs]
        self.z = (2 * self.k - self.n) * np.log(self.r / (1 - self.r)) if keep else np.array([])
        strata = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(self.recs):
            strata[(r["n"], r.get("k"), r["r"], r.get("format"))][r["domain"]].append(i)
        self.strata = [[np.array(v) for v in d.values()] for d in strata.values()]

    def subset(self, idx):
        o = object.__new__(Data)
        for a in ("n", "k", "r", "share", "q", "p", "template", "z"):
            setattr(o, a, getattr(self, a)[idx])
        return o

    def draw(self, rng):
        parts = []
        for clusters in self.strata:
            for ci in rng.integers(0, len(clusters), len(clusters)):
                cl = clusters[ci]
                parts.append(cl[rng.integers(0, len(cl), len(cl))])
        return np.concatenate(parts) if parts else np.array([], int)


def wilson(x, n, zc=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = x / n
    den = 1 + zc ** 2 / n
    c = (ph + zc ** 2 / (2 * n)) / den
    h = zc * math.sqrt(ph * (1 - ph) / n + zc ** 2 / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def boot(data, fn, B=B, seed=0, props=()):
    """Point estimates and 95% CIs for every key of fn(data). props: keys that are proportions,
    mapped to their denominators via fn(data)['_den_'+key]."""
    point = fn(data)
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for _ in range(B):
        d = data.subset(data.draw(rng))
        res = fn(d)
        for k, v in res.items():
            if not k.startswith("_"):
                draws[k].append(v)
    out = {}
    for k, v in point.items():
        if k.startswith("_"):
            continue
        arr = np.array([x for x in draws[k] if x is not None and np.isfinite(x)], float)
        lo, hi = (np.percentile(arr, 2.5), np.percentile(arr, 97.5)) if len(arr) > 20 else (float("nan"),) * 2
        if k in props and v is not None and np.isfinite(v) and v in (0.0, 1.0):
            den = point.get("_den_" + k, 0)
            lo, hi = wilson(v * den, den)
        out[k] = dict(est=None if v is None or not np.isfinite(v) else float(v), lo=float(lo), hi=float(hi),
                      frac_defined=len(arr) / B)
    return out


def ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan")
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def ols_multi(X, y):
    if len(y) <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        return np.full(X.shape[1], np.nan)
    return np.linalg.lstsq(X, y, rcond=None)[0]

# ================================================================ E1 metrics

def e1_metrics(d, rs=R_GRID):
    out = {}
    if len(d.p) == 0:
        return out
    Lp, Lq = logit(d.p), logit(d.q)
    s = np.sign(2 * d.k - d.n)
    for r in rs:
        mr = np.isclose(d.r, r)
        if not mr.any():
            continue
        un = mr & (np.abs(d.z) <= LCLIP)
        out[f"slope|{r}"] = ols(d.z[un], Lp[un])
        for n in N_GRID:
            m = mr & (d.n == n)
            ms = m & (s != 0)
            if ms.any():
                out[f"err_signed|{r}|{n}"] = float(np.mean((Lp[ms] - Lq[ms]) * s[ms]))
            for f, tag in ((0.75, "75"), (0.25, "25"), (0.5, "50")):
                mf = m & np.isclose(d.share, f)
                if mf.any():
                    out[f"err{tag}|{r}|{n}"] = float(np.mean(Lp[mf] - Lq[mf]))
                    out[f"p{tag}|{r}|{n}"] = float(np.mean(d.p[mf]))
                    out[f"L{tag}|{r}|{n}"] = float(np.mean(Lp[mf]))
            if m.any():
                out[f"sshare|{r}|{n}"] = ols(d.share[m], Lp[m])
                out[f"sshare_ref|{r}|{n}"] = ols(d.share[m], Lq[m])
                q, pc = d.q[m], np.clip(d.p[m], EPS, 1 - EPS)
                out[f"brier|{r}|{n}"] = float(np.mean((d.p[m] - q) ** 2 + q * (1 - q)))
                out[f"brier_ref|{r}|{n}"] = float(np.mean(q * (1 - q)))
                out[f"ll|{r}|{n}"] = float(np.mean(-q * np.log(pc) - (1 - q) * np.log(1 - pc)))
                qq = np.clip(q, 1e-15, 1 - 1e-15)
                out[f"ll_ref|{r}|{n}"] = float(np.mean(-qq * np.log(qq) - (1 - qq) * np.log(1 - qq)))
        for f in (0.25, 0.375, 0.625, 0.75):
            m = mr & np.isclose(d.share, f)
            if m.any():
                out[f"sn|{r}|{f}"] = ols(np.log2(d.n[m]), Lp[m])
                out[f"sn_ref|{r}|{f}"] = ols(np.log2(d.n[m]), Lq[m])
        ms = mr & (s != 0)
        out[f"top|{r}"] = float(np.mean(np.sign(Lp[ms]) == s[ms])) if ms.any() else float("nan")
        out[f"_den_top|{r}"] = int(ms.sum())
        for g in (0.95, 0.9):
            hi, lo = mr & (d.q > g), mr & (d.q < g)
            mid = mr & (d.q < g) & (d.q >= 0.5)
            for key, mask, fire in (("miss", hi, False), ("ff", lo, True), ("ffmid", mid, True)):
                if mask.any():
                    v = np.mean(d.p[mask] >= g) if fire else np.mean(d.p[mask] < g)
                    out[f"{key}|{g}|{r}"] = float(v)
                    out[f"_den_{key}|{g}|{r}"] = int(mask.sum())
        # gamma on unsaturated cells
        used, D, Dref = [], [], []
        for n in N_GRID:
            a = mr & (d.n == n) & np.isclose(d.share, 0.75)
            b = mr & (d.n == n) & np.isclose(d.share, 0.25)
            if not (a.any() and b.any()):
                continue
            La, Lb = Lp[a].mean(), Lp[b].mean()
            qa = float(np.mean(d.q[a]))
            Dn = (La - Lb) / 2
            out[f"D|{r}|{n}"] = float(Dn)
            out[f"Dref|{r}|{n}"] = float((n / 2) * math.log(r / (1 - r)))
            if 0.02 <= qa <= 0.98 and all(0.02 <= float(sigmoid(x)) <= 0.98 for x in (La, Lb)):
                used.append(n); D.append(Dn)
        out[f"_gamma_cells|{r}"] = used
        out[f"gamma|{r}"] = (ols(np.log(np.array(used, float)), np.log(np.array(D))) if len(used) >= 2 and
                             min(D) > 0 else float("nan"))
    # sensitivity to r at fixed (n, k): cell means of s*L at r=.70 minus r=.52 (k != n/2)
    for ra, rb in ((0.70, 0.52), (0.55, 0.52)):
        diffs, refs = [], []
        for n in N_GRID:
            for k in range(n + 1):
                if 2 * k == n:
                    continue
                a = np.isclose(d.r, ra) & (d.n == n) & (d.k == k)
                b = np.isclose(d.r, rb) & (d.n == n) & (d.k == k)
                if a.any() and b.any():
                    sg = np.sign(2 * k - n)
                    diffs.append(sg * (Lp[a].mean() - Lp[b].mean()))
                    refs.append(sg * (Lq[a].mean() - Lq[b].mean()))
        if diffs:
            out[f"dr|{ra}-{rb}"] = float(np.mean(diffs))
            out[f"dr_ref|{ra}-{rb}"] = float(np.mean(refs))
    return out


def gt_fits(d):
    """v3.3 parametric fits on cells with |normative log-odds| <= logit(.995), all r pooled."""
    out = {}
    m = np.abs(d.z) <= LCLIP
    if m.sum() < 10:
        return out
    Lp = logit(d.p[m]); z = d.z[m]; sh = d.share[m]; n = d.n[m]; r = d.r[m]
    ls, ln, lr = np.log(sh / (1 - sh)), np.log(n), np.log(r / (1 - r))
    XA = np.column_stack([np.ones_like(ls), ls, ln, lr, ls * ln, ls * lr])
    for name, y in (("A", Lp), ("Aref", z)):
        coef = ols_multi(XA, y)
        for nm, c in zip(("b0", "b_s", "b_w", "b_r", "b_sw", "b_sr"), coef):
            out[f"{name}|{nm}"] = float(c)
    XB = np.column_stack([z, ls, np.ones_like(z)])
    coef = ols_multi(XB, Lp)
    out["B|c"], out["B|d"], out["B|e"] = map(float, coef)
    # Griffin-Tversky log-log fit on share-folded cells (k != n/2) with folded logit > 0
    s = np.sign(2 * d.k[m] - n)
    f2 = np.abs(2 * sh - 1)
    yf = s * Lp
    g = (s != 0) & (yf > 0)
    out["_C_excluded"] = int(((s != 0) & ~(yf > 0)).sum())
    out["_C_used"] = int(g.sum())
    if g.sum() > 10:
        XC = np.column_stack([np.ones(g.sum()), np.log(f2[g]), np.log(n[g]), np.log(np.log(r[g] / (1 - r[g])))])
        coef = ols_multi(XC, np.log(yf[g]))
        out["C|a0"], out["C|beta_s"], out["C|beta_w"], out["C|beta_r"] = map(float, coef)
        out["C|share_minus_weight"] = float(coef[1] - coef[2])
    return out

# ================================================================ helpers for reporting

def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def ci(m, key, nd=2):
    if key not in m:
        return "n/a"
    v = m[key]
    return f"{fmt(v['est'], nd)} [{fmt(v['lo'], nd)}, {fmt(v['hi'], nd)}]"


def table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def boot_simple(vals, fn=np.mean, B=B, seed=1):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = [fn(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(B)]
    return float(fn(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def case_boot_prop(groups, B=B, seed=2):
    """groups: list of (correct, total) per case; case-resampled bootstrap of pooled proportion."""
    g = np.array(groups, float)
    if len(g) == 0:
        return (float("nan"),) * 3
    est = g[:, 0].sum() / g[:, 1].sum()
    if est in (0.0, 1.0):
        lo, hi = wilson(g[:, 0].sum(), g[:, 1].sum())
        return est, lo, hi
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        s = g[rng.integers(0, len(g), len(g))]
        bs.append(s[:, 0].sum() / s[:, 1].sum())
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def s3(t):
    return f"{fmt(t[0])} [{fmt(t[1])}, {fmt(t[2])}]"

# ================================================================ main

def main():
    recs = {ph: load(ph) for ph in PHASES}
    OUT.mkdir(exist_ok=True)
    metrics = {"spec": SPEC, "bootstrap_B": B}
    L = []  # summary lines
    w = L.append
    w(f"# E1 (amount of evidence) and E3 (one decisive clue): results, spec {SPEC} (v3.1-v3.3)\n")
    w("Generated by `python3 analyze.py` from results/raw/*.jsonl (every value re-parsed from the raw output). "
      f"Tag `onepass-e1v31`. CIs: two-stage cluster bootstrap over (domain, n, k, r) clusters and seeds within "
      f"design cells, B = {B}, percentile 95%; Wilson intervals for proportions at 0 or 1. Probabilities clipped "
      "to [0.005, 0.995] before logits. Primary = valid replies only; sensitivity variants: amb100 "
      "(0–100 reading), amb01 (0–1 reading), imp05 (impute non-valid at 0.5), impnorm "
      "(impute at the specified posterior). Earlier pilots are excluded. Historical raw logs are not distributed.\n")

    main_recs = [r for r in recs["main"] if r["exp"] == "e1"]

    # ------------------------------------------------ invalid rates
    w("## 1. Output categories and invalid rates\n")
    rows = []
    for ph in PHASES:
        by = defaultdict(Counter)
        for r in recs[ph]:
            by[r["model"]][r["cat"]] += 1
        for mdl, cnt in sorted(by.items()):
            tot = sum(cnt.values())
            nonvalid = tot - cnt["valid"]
            rows.append([ph, NAMES.get(mdl, mdl), tot, cnt["valid"], cnt["ambiguous_scale"], cnt["truncated"],
                         cnt["attempted_reasoning"], cnt["refusal"], cnt["invalid"] + cnt["empty"], cnt["api_error"],
                         f"{nonvalid / tot:.3f}"])
    w(table(["phase", "model", "calls", "valid", "ambiguous_scale", "truncated", "attempted_reasoning", "refusal",
             "invalid/empty", "api_error", "non-valid rate"], rows))
    # per-cell invalid table
    with open(OUT / "invalid_by_cell.csv", "w") as f:
        f.write("phase,model,format,r,n,share,cases,valid,ambiguous_scale,truncated,attempted_reasoning,refusal,invalid,api_error\n")
        for ph in ("main",):
            cells = defaultdict(Counter)
            for r in recs[ph]:
                cells[(r["model"], r["format"], r["r"], r["n"], r["share"])][r["cat"]] += 1
            for key in sorted(cells):
                c = cells[key]
                f.write(",".join(map(str, (ph,) + key + (sum(c.values()), c["valid"], c["ambiguous_scale"], c["truncated"],
                                                         c["attempted_reasoning"], c["refusal"], c["invalid"] + c["empty"],
                                                         c["api_error"]))) + "\n")
    # worst cells summary
    w("\nNon-valid rate by model, r and share (main E1, both formats; full per-(model, format, r, n, share) table in "
      "`results/invalid_by_cell.csv`). Non-valid includes ambiguous_scale.\n")
    rows = []
    for mdl in MODELS:
        for r in R_GRID:
            row = [NAMES[mdl], r]
            for f in (0.25, 0.375, 0.5, 0.625, 0.75):
                sel = [x for x in main_recs if x["model"] == mdl and x["r"] == r and x["share"] == f]
                nv = sum(x["cat"] != "valid" for x in sel)
                row.append(f"{nv}/{len(sel)}")
            sel = [x for x in main_recs if x["model"] == mdl and x["r"] == r and x["n"] == 64]
            row.append(f"{sum(x['cat'] != 'valid' for x in sel)}/{len(sel)}")
            rows.append(row)
    w(table(["model", "r", "f=.25", "f=.375", "f=.5", "f=.625", "f=.75", "n=64 (all f)"], rows))
    rows = []
    for mdl in LLMS:
        sel = [r for r in main_recs if r["model"] == mdl]
        c73 = sum((r["text"] or "").strip() in ("73", "73%", "73.0") for r in sel)
        rows.append([NAMES[mdl], len(sel), c73, f"{c73 / len(sel):.3f}"])
    w("\nReplies exactly equal to the scale example in the v3.1 prompt ending ('where 73 means 73%'), main E1:\n")
    w(table(["model", "calls", "replies '73'", "rate"], rows))
    w("\nNon-valid replies concentrate at shares below 0.5 (answers such as `0.31` or `0.0000001`, i.e. the "
      "0-1 scale or tiny values despite the 0-100 instruction) and are therefore n- and share-dependent; every "
      "headline number below is also given under imp05 and impnorm.\n")

    # ------------------------------------------------ E1 main metrics per model x format x variant
    datasets = {}
    for mdl in MODELS:
        for fm in ("listed", "count"):
            sel = [r for r in main_recs if r["model"] == mdl and r["format"] == fm]
            for var in VARIANTS:
                datasets[(mdl, fm, var)] = Data(sel, [value(r, var) for r in sel])
    # pseudo-models: Jev per-clue remedy (listed), n-aware transform of every model's holistic answer
    jl = [r for r in main_recs if r["model"] == "jev" and r["format"] == "listed"]
    rem_vals = []
    for r in jl:
        cl = r.get("clues")
        khat = sum(v >= 0.5 for v in cl) if cl and all(v is not None for v in cl) else None
        rem_vals.append(None if khat is None else posterior(khat, r["n"], r["r"]))
    datasets[("jev_remedy", "listed", "primary")] = Data(jl, rem_vals)
    for mdl in MODELS:
        for fm in ("listed", "count"):
            sel = [r for r in main_recs if r["model"] == mdl and r["format"] == fm]
            vals = []
            for r in sel:
                v = value(r, "primary")
                vals.append(None if v is None else float(sigmoid((2 * v * r["n"] - r["n"]) * math.log(r["r"] / (1 - r["r"])))))
            datasets[(mdl + "_naware", fm, "primary")] = Data(sel, vals)
    results = {}
    for key, d in datasets.items():
        results[key] = boot(d, e1_metrics, props=[f"{a}|{g}|{r}" for a in ("miss", "ff", "ffmid") for g in (.95, .9)
                                                  for r in R_GRID] + [f"top|{r}" for r in R_GRID])
        pt = e1_metrics(d)
        for r in R_GRID:
            results[key][f"gamma_cells|{r}"] = pt.get(f"_gamma_cells|{r}")
        results[key]["_n_cases"] = len(d.p)
    metrics["e1"] = {"|".join(k): v for k, v in results.items()}

    # ------------------------------------------------ primary metrics
    w("## 2. E1 primary metrics (v3.2 analysis: sensitivity slope to the normative log-odds and logit "
      "error by n, at r = 0.52; listed format primary, stated-count format alongside)\n")
    w("Slope: OLS of model logit on normative log-odds (normative = 1). Signed logit error: positive = "
      "overconfident, negative = underconfident (k = n/2 excluded). Cases = valid replies used.\n")
    for fm in ("listed", "count"):
        w(f"\n### Format: {fm}, r = 0.52\n")
        rows = []
        mods = MODELS + (["jev_remedy"] if fm == "listed" else [])
        for mdl in mods:
            m = results[(mdl, fm, "primary")]
            row = [NAMES.get(mdl, mdl), m["_n_cases"] if "_n_cases" in m else "", ci(m, "slope|0.52")]
            row += [ci(m, f"err_signed|0.52|{n}") for n in N_GRID]
            rows.append(row)
        w(table(["model", "cases (all r)", "slope"] + [f"signed err n={n}" for n in N_GRID], rows))
        w("\nSensitivity of the primary numbers to invalid handling (slope at r=.52; signed error at n=4 and n=64):\n")
        rows = []
        for mdl in MODELS:
            row = [NAMES[mdl]]
            for var in VARIANTS:
                m = results[(mdl, fm, var)]
                row.append(f"{ci(m, 'slope|0.52')}; {fmt(m.get('err_signed|0.52|4', {}).get('est'))} / "
                           f"{fmt(m.get('err_signed|0.52|64', {}).get('est'))}")
            rows.append(row)
        w(table(["model"] + VARIANTS, rows))

    # ------------------------------------------------ direction summary / coverage rule
    w("\n### Direction of error and coverage (v3.2 rule 4)\n")
    w("A model 'fails to scale as the probability model requires' at r = 0.52 (listed) if the slope CI excludes 1 "
      "under the primary analysis and under both imputations (imp05, impnorm). Direction from the slope and from "
      "the signed error at n = 64.\n")
    rows = []
    holds = []
    for mdl in MODELS:
        verdicts = []
        for var in ("primary", "imp05", "impnorm"):
            m = results[(mdl, "listed", var)]["slope|0.52"]
            verdicts.append(not (m["lo"] <= 1 <= m["hi"]))
        m = results[(mdl, "listed", "primary")]
        e4, e64 = m.get("err_signed|0.52|4", {}).get("est"), m.get("err_signed|0.52|64", {}).get("est")
        sl = m["slope|0.52"]["est"]
        direction = ("slope > 1 (too steep)" if sl > 1 else "slope < 1 (too flat)")
        ok = all(verdicts)
        if ok:
            holds.append(mdl)
        rows.append([NAMES[mdl], fmt(sl), " / ".join("excl. 1" if v else "incl. 1" for v in verdicts), direction,
                     fmt(e4), fmt(e64), "holds" if ok else "does not hold"])
    w(table(["model", "slope r=.52", "CI vs 1 (primary / imp05 / impnorm)", "direction", "signed err n=4",
             "signed err n=64", "claim"], rows))
    others = [m for m in holds if m != "jev"]
    w(f"\nClaim holds for: {', '.join(NAMES[m] for m in holds) or 'none'}. Rule 4 requires Jev plus at least three "
      f"of the five other families: {'MET' if 'jev' in holds and len(others) >= 3 else 'NOT MET'} "
      f"({len(others)} of 5 others).\n")
    metrics["coverage_e1"] = holds

    # ------------------------------------------------ all r: slopes, gamma, gates, top, r-sensitivity
    w("## 3. E1 secondary metrics, every r (listed; primary variant)\n")
    for fm in ("listed", "count"):
        w(f"\n### Slope, gamma (unsaturated cells only), top-label agreement, format {fm}\n")
        rows = []
        mods = MODELS + (["jev_remedy"] if fm == "listed" else []) + [m + "_naware" for m in MODELS]
        for mdl in mods:
            m = results[(mdl, fm, "primary")]
            row = [NAMES.get(mdl, mdl.replace("_naware", " n-aware transform"))]
            for r in R_GRID:
                row += [ci(m, f"slope|{r}"), f"{ci(m, f'gamma|{r}')} (n={m.get(f'gamma_cells|{r}')})",
                        ci(m, f"top|{r}")]
            rows.append(row)
        w(table(["model"] + [f"{h} r={r}" for r in R_GRID for h in ("slope", "gamma (cells)", "top-label")], rows))
    w("\nNormative gamma = 1; gamma uses only n where the normative answer at share .75 and the model's mean answers "
      "at shares .75 and .25 lie in [0.02, 0.98]; `n=[...]` lists the n used; n/a = fewer than 2 usable n or D <= 0.\n")

    w("\n### Mean signed logit error by n (listed; positive = overconfident)\n")
    rows = []
    for mdl in MODELS + ["jev_remedy"]:
        for r in R_GRID:
            m = results[(mdl, "listed", "primary")]
            rows.append([NAMES.get(mdl, mdl), r] + [ci(m, f"err_signed|{r}|{n}") for n in N_GRID])
    w(table(["model", "r"] + [f"n={n}" for n in N_GRID], rows))

    w("\n### Confidence at share 0.75 (mean P(YES), listed) against the normative answer\n")
    rows = []
    for r in R_GRID:
        rows.append(["normative", r] + [fmt(posterior(int(.75 * n), n, r), 3) for n in N_GRID])
        for mdl in MODELS + ["jev_remedy"]:
            for fm in ("listed", "count"):
                if mdl == "jev_remedy" and fm == "count":
                    continue
                m = results[(mdl, fm, "primary")]
                rows.append([f"{NAMES.get(mdl, mdl)} ({fm})", r] + [ci(m, f"p75|{r}|{n}", 3) for n in N_GRID])
    w(table(["model", "r"] + [f"n={n}" for n in N_GRID], rows))

    w("\n### Gate errors (listed, primary): miss = P(p < g | normative > g); false fire = P(p >= g | normative < g); "
      "ffmid restricts false fires to normative in [0.5, g)\n")
    rows = []
    for mdl in MODELS + ["jev_remedy"]:
        m = results[(mdl, "listed", "primary")]
        for r in R_GRID:
            rows.append([NAMES.get(mdl, mdl), r] + [ci(m, f"{a}|{g}|{r}") for g in (0.95, 0.9) for a in ("miss", "ff", "ffmid")])
    w(table(["model", "r"] + [f"{a} @{g}" for g in (0.95, 0.9) for a in ("miss", "false fire", "ffmid")], rows))
    w("\n(r = 0.52: the normative answer never exceeds 0.95, so there is no miss denominator; n/a.)\n")

    w("\n### Sensitivity to share at fixed n and to n at fixed share (listed, primary; model slope vs normative-"
      "through-the-same-clip slope)\n")
    rows = []
    for mdl in MODELS:
        m = results[(mdl, "listed", "primary")]
        for r in R_GRID:
            rows.append([NAMES[mdl], r] +
                        [f"{fmt(m.get(f'sshare|{r}|{n}', {}).get('est'))} vs {fmt(m.get(f'sshare_ref|{r}|{n}', {}).get('est'))}" for n in N_GRID] +
                        [f"{fmt(m.get(f'sn|{r}|0.75', {}).get('est'))} vs {fmt(m.get(f'sn_ref|{r}|0.75', {}).get('est'))}",
                         f"{fmt(m.get(f'sn|{r}|0.25', {}).get('est'))} vs {fmt(m.get(f'sn_ref|{r}|0.25', {}).get('est'))}"])
    w(table(["model", "r"] + [f"d logit/d share, n={n}" for n in N_GRID] + ["d logit/d log2 n, f=.75", "f=.25"], rows))
    w("\n### Sensitivity to r at fixed (n, k) (listed; mean sign-adjusted logit difference; normative through clip)\n")
    rows = []
    for mdl in MODELS + ["jev_remedy"]:
        m = results[(mdl, "listed", "primary")]
        rows.append([NAMES.get(mdl, mdl), ci(m, "dr|0.7-0.52"), fmt(m.get("dr_ref|0.7-0.52", {}).get("est")),
                     ci(m, "dr|0.55-0.52"), fmt(m.get("dr_ref|0.55-0.52", {}).get("est"))])
    w(table(["model", "r .70 - .52", "normative", "r .55 - .52", "normative"], rows))

    w("\n### Expected Brier score and log loss by n (listed, r = 0.52; outcome ~ normative posterior)\n")
    rows = []
    for mdl in MODELS + ["jev_remedy"]:
        m = results[(mdl, "listed", "primary")]
        rows.append([NAMES.get(mdl, mdl)] + [f"{fmt(m.get(f'brier|0.52|{n}', {}).get('est'), 3)} / "
                                            f"{fmt(m.get(f'll|0.52|{n}', {}).get('est'), 3)}" for n in N_GRID])
    m = results[("jev", "listed", "primary")]
    rows.append(["normative (floor)"] + [f"{fmt(m.get(f'brier_ref|0.52|{n}', {}).get('est'), 3)} / "
                                         f"{fmt(m.get(f'll_ref|0.52|{n}', {}).get('est'), 3)}" for n in N_GRID])
    w(table(["model"] + [f"Brier / LL n={n}" for n in N_GRID], rows))

    # ------------------------------------------------ templates (v3.2 rule 1)
    w("\n## 4. Wording robustness: primary metrics per story template (listed, r = 0.52)\n")
    rows = []
    tmpl_ok = {}
    for mdl in MODELS:
        sel = [r for r in main_recs if r["model"] == mdl and r["format"] == "listed" and r["r"] == 0.52]
        row = [NAMES[mdl]]
        oks = []
        for t in (0, 1, 2):
            st = [r for r in sel if r["template"] == t]
            m = boot(Data(st, [value(r, "primary") for r in st]), lambda d: {
                k: v for k, v in e1_metrics(d, rs=(0.52,)).items() if k in ("slope|0.52", "err_signed|0.52|64")}, B=400)
            row.append(f"{ci(m, 'slope|0.52')}; err64 {fmt(m.get('err_signed|0.52|64', {}).get('est'))}")
            oks.append(not (m["slope|0.52"]["lo"] <= 1 <= m["slope|0.52"]["hi"]))
        tmpl_ok[mdl] = all(oks)
        row.append("yes" if all(oks) else "no")
        rows.append(row)
    w(table(["model", "template 0", "template 1", "template 2", "slope CI excludes 1 on every template"], rows))
    w("\nEach template is used by two of the six seeds per domain (template = (seed index + domain index) mod 3); "
      "per-template CIs use B = 400.\n")
    metrics["template_ok"] = tmpl_ok

    # ------------------------------------------------ v3.3 stages
    w("\n## 5. Which stage fails (v3.3): read, count, map count to confidence\n")
    stage = {}
    # stage 1: per-clue
    w("### Stage 1 (read each report): per-clue accuracy (P >= 0.5 counts as yes)\n")
    rows = []
    for mdl in MODELS:
        if mdl == "jev":
            groups_all, by_n = [], defaultdict(list)
            for r in jl:
                sg = CASES[r["id"]]["signs"]
                cl = r.get("clues") or []
                corr = sum((v is not None and (v >= 0.5) == bool(s)) for v, s in zip(cl, sg))
                groups_all.append((corr, len(sg))); by_n[r["n"]].append((corr, len(sg)))
            src = "main listed calls, all r (every listed case)"
        else:
            per_case = defaultdict(lambda: [0, 0])
            by_n = defaultdict(list)
            invalid = 0
            for r in recs["clue"]:
                if r["model"] != mdl:
                    continue
                v = value(r, "primary")
                pc = per_case[r["id"]]
                pc[1] += 1
                if v is None:
                    invalid += 1
                elif (v >= 0.5) == bool(r["clue_truth"]):
                    pc[0] += 1
            groups_all = [tuple(v) for v in per_case.values()]
            for cid, v in per_case.items():
                by_n[CASES[cid]["n"]].append(tuple(v))
            src = (f"{len(per_case)} cases (r=.52 seeds 0-1, r=.55 seed 0; shares .25/.75; every n and domain); one "
                   f"Yes/No call per report; non-valid counted wrong ({invalid})")
        acc = case_boot_prop(groups_all)
        stage.setdefault(mdl, {})["s1"] = acc[0]
        if mdl == "jev":
            stage[mdl]["s1_correct"] = sum(correct for correct, _ in groups_all)
            stage[mdl]["s1_total"] = sum(total for _, total in groups_all)
        rows.append([NAMES[mdl], s3(acc)] + [fmt(case_boot_prop(by_n[n])[0], 3) for n in N_GRID] +
                    [sum(g[1] for g in groups_all), src])
    w(table(["model", "accuracy [95% CI]"] + [f"n={n}" for n in N_GRID] + ["reports", "source"], rows))
    # LLM remedy on clue cases
    w("\nPer-clue remedy for LLMs (count of per-clue yes, formula in code) versus the holistic answer on the same "
      "cases (primary; cases with any non-valid clue dropped):\n")
    rows = []
    for mdl in LLMS:
        per_case = defaultdict(dict)
        for r in recs["clue"]:
            if r["model"] == mdl:
                per_case[r["id"]][r["clue_idx"]] = value(r, "primary")
        errs_rem, errs_hol, dropped = [], [], 0
        for cid, cl in per_case.items():
            c = CASES[cid]
            if len(cl) != c["n"] or any(v is None for v in cl.values()):
                dropped += 1
                continue
            khat = sum(v >= 0.5 for v in cl.values())
            s = np.sign(2 * c["k"] - c["n"])
            errs_rem.append(float((logit(posterior(khat, c["n"], c["r"])) - logit(c["truth"])) * s))
            hol = [x for x in main_recs if x["id"] == cid and x["model"] == mdl]
            hv = value(hol[0], "primary") if hol else None
            if hv is not None:
                errs_hol.append(float((logit(hv) - logit(c["truth"])) * s))
        rows.append([NAMES[mdl], len(errs_rem), dropped, s3(boot_simple(errs_rem)), s3(boot_simple(errs_hol)),
                     s3(boot_simple(np.abs(errs_rem))), s3(boot_simple(np.abs(errs_hol)))])
    w(table(["model", "cases", "dropped", "remedy signed err", "holistic signed err", "remedy |err|", "holistic |err|"], rows))

    # stage 2: count
    w("\n### Stage 2 (count): exact-count accuracy, mean absolute error, share-bin accuracy (listed, r = 0.52, every case)\n")
    w("LLMs: \"How many reports say exactly '<label>'? Reply with the number only.\" Jev: Choice over counts 0..n "
      "(argmax; expected count also reported) and a Choice over ten share bins, in one call as isolated branches. "
      "LLM share bin derived from the stated count. Share 0.5 lies on a bin boundary (bin 50-60% is strictly "
      "correct); lenient accuracy also accepts 40-50% there.\n")
    rows = []
    for mdl in MODELS:
        sel = [r for r in recs["count"] if r["model"] == mdl and r["r"] == 0.52]
        byn = defaultdict(list)
        bins_ok, bins_len, nonvalid = [], [], 0
        for r in sel:
            v = r["value"] if r["cat"] == "valid" else None
            if v is None:
                nonvalid += 1
                continue
            byn[r["n"]].append((v, r["k"], r.get("count_exp")))
            true_bin = min(int(r["k"] / r["n"] * 10 + 1e-9), 9)
            if mdl == "jev":
                b = int(r["sharebin"][1:])
            else:
                b = min(int(v / r["n"] * 10 + 1e-9), 9)
            bins_ok.append(b == true_bin)
            bins_len.append(b == true_bin or (r["k"] * 2 == r["n"] and b == 4))
        row = [NAMES[mdl], len(sel), nonvalid]
        mae_small = []
        for n in N_GRID:
            a = byn[n]
            acc = np.mean([x[0] == x[1] for x in a]) if a else float("nan")
            mae = np.mean([abs(x[0] - x[1]) for x in a]) if a else float("nan")
            if n <= 16:
                mae_small += [abs(x[0] - x[1]) for x in a]
            extra = f"; E-count MAE {fmt(np.mean([abs(x[2] - x[1]) for x in a]))}" if mdl == "jev" and a else ""
            row.append(f"{fmt(acc)} / {fmt(mae)}{extra}")
        mae16 = boot_simple(mae_small)
        binacc = case_boot_prop([(int(b), 1) for b in bins_ok])
        row += [s3(mae16), s3(binacc), fmt(np.mean(bins_len))]
        stage[mdl]["mae16"] = mae16[0]
        stage[mdl]["binacc"] = binacc[0]
        rows.append(row)
    w(table(["model", "cases", "non-valid"] + [f"acc / MAE n={n}" for n in N_GRID] +
            ["MAE n<=16 [CI]", "share-bin acc [CI]", "lenient bin acc"], rows))
    # r=.55 subset
    rows = []
    for mdl in MODELS:
        sel = [r for r in recs["count"] if r["model"] == mdl and r["r"] == 0.55 and r["cat"] == "valid"]
        rows.append([NAMES[mdl], len(sel), fmt(np.mean([r["value"] == r["k"] for r in sel]) if sel else float('nan')),
                     fmt(np.mean([abs(r["value"] - r["k"]) for r in sel]) if sel else float('nan'))])
    w("\nCount readout control at r = 0.55 (seeds 0-2, listed, all n):\n")
    w(table(["model", "valid cases", "exact acc", "MAE"], rows))

    # stage 3: stated count
    w("\n### Stage 3 (map count to confidence): stated-count format versus listed, r = 0.52\n")
    w("Paired by (domain, seed, n, k): mean of (L(stated) - L(listed)) * sign(2k - n) (positive = stated count "
      "more extreme); stated-count slope from section 2.\n")
    rows = []
    for mdl in MODELS:
        pairs = defaultdict(dict)
        for r in main_recs:
            if r["model"] == mdl and r["r"] == 0.52:
                v = value(r, "primary")
                if v is not None:
                    pairs[(r["domain"], r["seed_idx"], r["n"], r["k"])][r["format"]] = v
        byn = defaultdict(list)
        for (dm, sd, n, k), d in pairs.items():
            if "listed" in d and "count" in d and 2 * k != n:
                byn[n].append(float((logit(d["count"]) - logit(d["listed"])) * np.sign(2 * k - n)))
        mc = results[(mdl, "count", "primary")]
        rows.append([NAMES[mdl], ci(mc, "slope|0.52")] + [s3(boot_simple(byn[n])) for n in N_GRID])
        stage[mdl]["s3_slope"] = mc["slope|0.52"]
    w(table(["model", "stated-count slope r=.52"] + [f"stated - listed n={n}" for n in N_GRID], rows))

    # interpretation rule
    w("\n### Exploratory stage-localization rule (v3.3 analysis), applied per model\n")
    w("Stage 1 accurate: per-clue accuracy >= 0.95. Stage 2 accurate: count MAE <= 1 for n <= 16 AND share-bin "
      "accuracy >= 0.9 overall. 'Stated-count confidence off the normative curve' is operationalised (by this builder; "
      "the rule text does not fix a test) as: stated-count sensitivity slope at r = 0.52 has a 95% CI excluding 1. "
      "This is a behavioral diagnostic, not proof of internal stages: accurate clue and count answers "
      "with an off-curve probability are consistent with a mapping problem; an inaccurate count "
      "answer is consistent with a counting problem.\n")
    rows = []
    verdicts = {}
    for mdl in MODELS:
        s = stage[mdl]
        s1 = s["s1"] >= 0.95
        s2 = s["mae16"] <= 1 and s["binacc"] >= 0.9
        sl = s["s3_slope"]
        off = not (sl["lo"] <= 1 <= sl["hi"])
        if not s1:
            v = "reading failure (stage 1 < 0.95); rule not applicable"
        elif not s2:
            v = "counting failure (stage 2)" + ("; stated counts also off-curve" if off else "")
        elif off:
            v = "stage 3 (mapping) failure; attention-averaging disfavored"
        else:
            v = "stated counts on the normative curve: no stage-3 failure"
        verdicts[mdl] = v
        rows.append([NAMES[mdl], fmt(s["s1"], 3), "yes" if s1 else "no", fmt(s["mae16"]), fmt(s["binacc"]),
                     "yes" if s2 else "no", f"{fmt(sl['est'])} [{fmt(sl['lo'])}, {fmt(sl['hi'])}]",
                     "yes" if off else "no", v])
    w(table(["model", "per-clue acc", "S1 ok", "count MAE n<=16", "share-bin acc", "S2 ok", "stated-count slope",
             "off-curve", "verdict"], rows))
    metrics["stage_verdicts"] = verdicts
    metrics["stages"] = {m: {k: (v if not isinstance(v, dict) else v) for k, v in s.items()} for m, s in stage.items()}

    # ------------------------------------------------ parametric fits
    w("\n## 6. Parametric model of the bias (v3.3 item 4; listed, all r pooled, cells with |normative log-odds| "
      "<= logit(.995); primary)\n")
    w("Fit A: L = b0 + b_s logit(f) + b_w log n + b_r logit(r) + b_sw logit(f) log n + b_sr logit(f) logit(r); "
      "the same regression fitted to the normative log-odds gives the reference row. Fit B: L = c z + d logit(f) + e "
      "(normative c = 1, d = 0). Fit C (Griffin-Tversky, share-folded, k != n/2, folded logit > 0): log L_folded = "
      "a0 + beta_s log|2f - 1| + beta_w log n + beta_r log logit(r) (normative 1, 1, 1). **One number per model: "
      "share-minus-weight = beta_s - beta_w** (normative 0; > 0 means confidence follows the share more than the "
      "amount of evidence).\n")
    fits = {}
    rows = []
    for mdl in MODELS + ["jev_remedy"]:
        for fm in ("listed", "count"):
            if mdl == "jev_remedy" and fm == "count":
                continue
            d = datasets[(mdl, fm, "primary")]
            fb = boot(d, gt_fits, B=B)
            pt = gt_fits(d)
            fits[(mdl, fm)] = fb
            rows.append([f"{NAMES.get(mdl, mdl)} ({fm})", ci(fb, "C|share_minus_weight"), ci(fb, "C|beta_s"),
                         ci(fb, "C|beta_w"), ci(fb, "C|beta_r"), f"{pt.get('_C_used')}/{pt.get('_C_excluded')}",
                         ci(fb, "B|c"), ci(fb, "B|d")])
    w(table(["model (format)", "share-minus-weight", "beta_s", "beta_w", "beta_r", "C cases used/excluded",
             "B: c (normative weight)", "B: d (extra share weight)"], rows))
    rows = []
    for mdl in MODELS:
        row = [NAMES[mdl]]
        for var in ("primary", "amb01", "imp05", "impnorm"):
            fb = fits[(mdl, "listed")] if var == "primary" else boot(datasets[(mdl, "listed", var)], gt_fits, B=300)
            fits[(mdl, "listed", var)] = fb
            row.append(f"{ci(fb, 'C|share_minus_weight')}; c {fmt(fb.get('B|c', {}).get('est'))}")
        rows.append(row)
    w("\nShare-minus-weight and Fit-B c under invalid-output variants (listed; imputation variants B = 300):\n")
    w(table(["model", "primary", "amb01", "imp05", "impnorm"], rows))
    rows = []
    ref = gt_fits(datasets[("jev", "listed", "primary")])
    rows.append(["normative (reference)"] + [fmt(ref.get(f"Aref|{k}")) for k in ("b0", "b_s", "b_w", "b_r", "b_sw", "b_sr")])
    for mdl in MODELS:
        fb = fits[(mdl, "listed")]
        rows.append([NAMES[mdl]] + [ci(fb, f"A|{k}") for k in ("b0", "b_s", "b_w", "b_r", "b_sw", "b_sr")])
    w("\nFit A coefficients (listed):\n")
    w(table(["model", "b0", "b_s", "b_w", "b_r", "b_sw", "b_sr"], rows))
    metrics["fits"] = {"|".join(k): v for k, v in fits.items()}

    # ------------------------------------------------ controls
    w("\n## 7. Controls (r = 0.55 unless noted)\n")

    def paired(a_recs, b_recs, transform_a=lambda v: v):
        bmap = {(r["id"], r["model"]): value(r, "primary") for r in b_recs}
        out = defaultdict(list)
        for r in a_recs:
            va = value(r, "primary")
            vb = bmap.get((r["id"], r["model"]))
            if va is None or vb is None:
                continue
            out[r["model"]].append((r, float(logit(transform_a(va)) - logit(vb))))
        return out

    hol = recs["holonly"]
    w("### Padding to the n = 64 length (listed, seeds 0-2, n < 64): paired logit difference padded - unpadded "
      "(Jev baseline: holistic-only call; LLM baseline: main)\n")
    pr = paired(recs["pad"], main_recs + [dict(x, model="jev") for x in []])
    pj = paired([r for r in recs["pad"] if r["model"] == "jev"], hol)
    rows = []
    for mdl in MODELS:
        lst = pj.get("jev", []) if mdl == "jev" else pr.get(mdl, [])
        diffs = [d for _, d in lst]
        sdiffs = [d * np.sign(2 * r["k"] - r["n"]) for r, d in lst if 2 * r["k"] != r["n"]]
        rows.append([NAMES[mdl], len(diffs), s3(boot_simple(diffs)), s3(boot_simple(sdiffs)), s3(boot_simple(np.abs(diffs)))])
    w(table(["model", "pairs", "mean diff", "sign-adjusted diff (+ = more extreme)", "mean |diff|"], rows))

    w("\n### Flipped wording (probability of NO; listed, seeds 0-2, all n): P(YES) = 1 - P(NO); paired against the "
      "YES question (Jev baseline: holistic-only)\n")
    rows = []
    fl = [r for r in recs["flip"]]
    pfl = paired([r for r in fl if r["model"] != "jev"], main_recs, lambda v: 1 - v)
    pfj = paired([r for r in fl if r["model"] == "jev"], hol, lambda v: 1 - v)
    for mdl in MODELS:
        lst = pfj.get("jev", []) if mdl == "jev" else pfl.get(mdl, [])
        diffs = [d for _, d in lst]
        base = {(r["id"]): value(r, "primary") for r in (hol if mdl == "jev" else main_recs) if r["model"] == mdl}
        coh = []
        for r in fl:
            if r["model"] == mdl and value(r, "primary") is not None and base.get(r["id"]) is not None:
                coh.append(value(r, "primary") + base[r["id"]] - 1)
        sel = [r for r in fl if r["model"] == mdl]
        d = Data(sel, [None if value(r, "primary") is None else 1 - value(r, "primary") for r in sel])
        mm = boot(d, lambda x: {k: v for k, v in e1_metrics(x, rs=(0.55,)).items() if k == "slope|0.55"}, B=400)
        mb = results[(mdl, "listed", "primary")] if mdl != "jev" else None
        rows.append([NAMES[mdl], len(diffs), s3(boot_simple(diffs)), s3(boot_simple(coh)), ci(mm, "slope|0.55"),
                     ci(mb, "slope|0.55") if mb else "see holonly"])
    w(table(["model", "pairs", "logit(1-P(NO)) - logit(P(YES))", "P(YES)+P(NO)-1", "slope r=.55 (flip)",
             "slope r=.55 (main, all seeds)"], rows))

    w("\n### Jev holistic-only versus holistic with per-clue branches in the same call (all listed cases, all r)\n")
    pj2 = paired(hol, [r for r in main_recs if r["model"] == "jev"])
    diffs = [d for _, d in pj2.get("jev", [])]
    dho = Data([r for r in hol], [value(r, "primary") for r in hol])
    mho = boot(dho, e1_metrics, B=400)
    w(f"Paired logit difference (holistic-only minus combined): {s3(boot_simple(diffs))}, mean |diff| "
      f"{s3(boot_simple(np.abs(diffs)))}, n = {len(diffs)} pairs; identical Noul in "
      f"{np.mean(np.abs(diffs) < 1e-9):.3f} of pairs. Holistic-only slope r=.52 {ci(mho, 'slope|0.52')} vs combined "
      f"{ci(results[('jev', 'listed', 'primary')], 'slope|0.52')}; signed err n=64 r=.52 "
      f"{ci(mho, 'err_signed|0.52|64')} vs {ci(results[('jev', 'listed', 'primary')], 'err_signed|0.52|64')}.\n")

    w("\n### Readout robustness (v3.2 item 2)\n")
    w("**Jev two-option Choice** (same YES/NO question as a Choice; all r = 0.52 cases, both formats; reversed option "
      "order on listed seeds 0-1):\n")
    ch = [r for r in recs["jevchoice"] if r["kind"] == "choice"]
    rows = []
    for fm in ("listed", "count"):
        sel = [r for r in ch if r["format"] == fm]
        mc = boot(Data(sel, [value(r, "primary") for r in sel]), lambda x: e1_metrics(x, rs=(0.52,)), B=400)
        if fm == "listed":
            metrics["_choice_listed"] = mc
        base = results[("jev", fm, "primary")]
        rows.append([fm, len(sel), ci(mc, "slope|0.52"), ci(base, "slope|0.52")] +
                    [f"{fmt(mc.get(f'err_signed|0.52|{n}', {}).get('est'))} vs {fmt(base.get(f'err_signed|0.52|{n}', {}).get('est'))}" for n in N_GRID] +
                    [f"{fmt(mc.get('p75|0.52|64', {}).get('est'), 3)} vs {fmt(base.get('p75|0.52|64', {}).get('est'), 3)}"])
    w(table(["format", "cases", "Choice slope", "Noul slope"] + [f"signed err n={n} (Choice vs Noul)" for n in N_GRID] +
            ["P at 48/64 (Choice vs Noul)"], rows))
    rev = {r["id"]: value(r, "primary") for r in recs["jevchoice"] if r["kind"] == "choice_rev"}
    od = [float(logit(value(r, "primary")) - logit(rev[r["id"]])) for r in ch if r["id"] in rev]
    w(f"\nOption-order effect (YES-first minus NO-first, logit): {s3(boot_simple(od))}, n = {len(od)}.\n")
    metrics["jev_choice_order_effect"] = boot_simple(od)

    w("\n**LLM yes/no decision** (listed, r = 0.52, seeds 0-1, 138 cases): agreement with the normative top label "
      "(k != n/2), compared with the stated-probability top label on the same cases, and P(yes) at k = n/2.\n")
    rows = []
    for mdl in LLMS:
        sel = [r for r in recs["yesno"] if r["model"] == mdl]
        prob = {r["id"]: value(r, "primary") for r in main_recs if r["model"] == mdl}
        agree, agree_p, half, byn = [], [], [], defaultdict(list)
        for r in sel:
            if r["cat"] != "valid":
                continue
            if 2 * r["k"] == r["n"]:
                half.append(r["value"]); continue
            truth = 1 if 2 * r["k"] > r["n"] else 0
            agree.append(r["value"] == truth); byn[r["n"]].append(r["value"] == truth)
            pv = prob.get(r["id"])
            if pv is not None:
                agree_p.append((pv > 0.5) == bool(truth))
        nv = sum(r["cat"] != "valid" for r in sel)
        rows.append([NAMES[mdl], len(sel), nv, s3(case_boot_prop([(int(a), 1) for a in agree])),
                     fmt(np.mean(agree_p)) if agree_p else "n/a", fmt(np.mean(half)) if half else "n/a"] +
                    [fmt(np.mean(byn[n])) if byn[n] else "n/a" for n in N_GRID])
    w(table(["model", "cases", "non-valid", "yes/no agreement", "stated-prob agreement (same cases)", "P(yes) at k=n/2"] +
            [f"agree n={n}" for n in N_GRID], rows))

    w("\n### Test-retest (v3.2 item 3): 100 identical repeats of main-condition calls (r = 0.52 subset)\n")
    rows = []
    for mdl in MODELS:
        base = {r["id"]: r for r in main_recs if r["model"] == mdl}
        same_text, dl, both = 0, [], 0
        sel = [r for r in recs["retest"] if r["model"] == mdl]
        catsame = 0
        for r in sel:
            b = base.get(r["id"])
            if b is None:
                continue
            catsame += r["cat"] == b["cat"]
            if mdl == "jev":
                eq = abs(r["value"] - b["value"]) < 1e-12
            else:
                eq = (r["text"] or "").strip() == (b["text"] or "").strip()
            same_text += eq
            va, vb = value(r, "primary"), value(b, "primary")
            if va is not None and vb is not None:
                both += 1
                dl.append(abs(float(logit(va) - logit(vb))))
        rows.append([NAMES[mdl], len(sel), fmt(same_text / max(len(sel), 1), 3), fmt(catsame / max(len(sel), 1), 3),
                     both, s3(boot_simple(dl)), fmt(np.max(dl) if dl else float("nan"))])
    sn = {r["id"]: r for r in recs["sonnet"]}
    sd = []
    for r in recs["sonnet_retest"]:
        b = sn.get(r["id"])
        va, vb = value(r, "amb100"), value(b, "amb100") if b else None
        if va is not None and vb is not None:
            sd.append(abs(float(logit(va) - logit(vb))))
    rows.append(["Sonnet 5 reasoning (default sampling, no temperature control)", len(recs["sonnet_retest"]), "", "",
                 len(sd), s3(boot_simple(sd)), fmt(np.max(sd) if sd else float("nan"))])
    w(table(["model", "repeats", "identical output", "same category", "both valid", "mean |logit diff| [CI]",
             "max |logit diff|"], rows))

    w("\n### Provider pinning: unpinned versus pinned (listed, r = 0.55, seed 0, 69 cases)\n")
    rows = []
    for mdl in ("llama", "mistral"):
        base = {r["id"]: r for r in main_recs if r["model"] == mdl}
        sel = [r for r in recs["unpinned"] if r["model"] == mdl]
        provs = Counter(r["provider_returned"] for r in sel)
        dl, same = [], 0
        for r in sel:
            b = base[r["id"]]
            same += (r["text"] or "").strip() == (b["text"] or "").strip()
            va, vb = value(r, "primary"), value(b, "primary")
            if va is not None and vb is not None:
                dl.append(float(logit(va) - logit(vb)))
        nvu = sum(r["cat"] != "valid" for r in sel)
        nvp = sum(base[r["id"]]["cat"] != "valid" for r in sel)
        rows.append([NAMES[mdl], len(sel), dict(provs), f"{nvu} vs {nvp}", fmt(same / len(sel), 3),
                     s3(boot_simple(dl)), s3(boot_simple(np.abs(dl)))])
    w(table(["model", "cases", "providers served (unpinned)", "non-valid unpinned vs pinned", "identical text",
             "mean logit diff", "mean |logit diff|"], rows))
    provs = Counter((r["model"], r["provider_returned"]) for ph in PHASES for r in recs[ph] if r["model"] in LLMS
                    and ph != "unpinned")
    w(f"\nProviders that served the pinned runs (all phases): {dict(provs)}. Reasoning tokens reported by OpenRouter "
      f"for pinned open models: max = {max((r['reasoning_tokens'] or 0) for ph in PHASES for r in recs[ph] if r['model'] in LLMS and r['model'] != 'haiku')}.\n")

    # ------------------------------------------------ Sonnet reference
    w("\n## 8. Reasoning reference: Sonnet 5 (listed, seed 0, shares .25/.75, every n, 3 domains, every r; max_tokens "
      "6000; final line 'Answer: N')\n")
    ss = recs["sonnet"]
    rows = []
    for var in ("primary", "amb100"):
        d = Data(ss, [value(r, var) for r in ss])
        m = boot(d, e1_metrics, B=400)
        if var == "amb100":
            metrics["_sonnet"] = m
        rows.append([var, len(d.p)] + [ci(m, f"slope|{r}") for r in R_GRID] +
                    [ci(m, f"err_signed|0.52|{n}") for n in N_GRID])
    w(table(["variant", "cases", "slope r=.52", "slope r=.55", "slope r=.70"] + [f"signed err r=.52 n={n}" for n in N_GRID], rows))
    toks = [r["usage"]["output_tokens"] for r in ss if r.get("usage")]
    lat = [r["latency"] for r in ss if r.get("latency")]
    amb = [r for r in ss if r["cat"] == "ambiguous_scale"]
    closer100 = sum(abs(float(logit(r["value"]) - logit(r["truth"]))) < abs(float(logit(r["alt"]) - logit(r["truth"])))
                    for r in amb)
    w(f"\nOutput tokens median {np.median(toks):.0f} (IQR {np.percentile(toks, 25):.0f}-{np.percentile(toks, 75):.0f}), "
      f"latency median {np.median(lat):.1f} s. {len(amb)} ambiguous_scale replies (e.g. 'Answer: 0.16' where the "
      f"normative answer is 0.16%); the 0-100 reading is closer to the normative answer in {closer100} of {len(amb)}; "
      f"the amb100 row reads them on the 0-100 scale as instructed.\n")

    # ------------------------------------------------ E3
    w("\n## 9. E3: one decisive report (99%) among n - 1 weak reports (55%, split evenly)\n")
    e3 = recs["e3"]
    w("Primary (v3.2): confidence at n = 65 for decisive positive and negative (mean P(YES), unpadded, early and "
      "late pooled; normative 0.99 / 0.01).\n")
    rows = []
    e3m = {}
    for mdl in MODELS:
        row = [NAMES[mdl]]
        for dec in (1, 0):
            for var in ("primary", "imp05", "impnorm"):
                vals = [value(r, var) for r in e3 if r["model"] == mdl and r["control"] == "decisive" and r["n"] == 65
                        and r["decisive"] == dec]
                vals = [v for v in vals if v is not None]
                t = boot_simple(vals)
                e3m[(mdl, dec, var)] = t
                row.append(f"{s3(t)} (n={len(vals)})" if var == "primary" else fmt(t[0], 3))
        rows.append(row)
    w(table(["model", "dec + primary", "imp05", "impnorm", "dec - primary", "imp05", "impnorm"], rows))
    metrics["e3_primary"] = {f"{m}|{d}|{v}": t for (m, d, v), t in e3m.items()}
    w("\nConfidence toward the decisive report, P_dec = P(YES) if decisive positive else 1 - P(YES), by n (unpadded, "
      "early+late); normative 0.99 at every n. Ceiling: share of decisive cases with P_dec >= 0.95.\n")
    rows = []
    for mdl in MODELS:
        row = [NAMES[mdl]]
        ceil = []
        for n in (5, 9, 17, 33, 65):
            vals = []
            for r in e3:
                if r["model"] == mdl and r["control"] == "decisive" and r["n"] == n:
                    v = value(r, "primary")
                    if v is not None:
                        vals.append(v if r["decisive"] else 1 - v)
            ceil += [v >= 0.95 for v in vals]
            row.append(fmt(np.mean(vals), 3) if vals else "n/a")
        row.append(fmt(np.mean(ceil), 3))
        # dilution slope logit(P_dec) ~ log2 n
        xs, ys = [], []
        for r in e3:
            if r["model"] == mdl and r["control"] == "decisive":
                v = value(r, "primary")
                if v is not None:
                    xs.append(math.log2(r["n"])); ys.append(float(logit(v if r["decisive"] else 1 - v)))
        row.append(fmt(ols(np.array(xs), np.array(ys))))
        rows.append(row)
    w(table(["model", "n=5", "n=9", "n=17", "n=33", "n=65", "ceiling rate (P_dec>=.95)", "dilution slope (logit per doubling)"], rows))
    w("\nEarly versus late decisive report, weak-only control (normative 0.5) and padding control (padded to the "
      "n = 65 length), P_dec means pooled over n (weak-only: mean P(YES) and mean |P - 0.5|):\n")
    rows = []
    for mdl in MODELS:
        def pdec(filt):
            vals = []
            for r in e3:
                if r["model"] == mdl and filt(r):
                    v = value(r, "primary")
                    if v is not None:
                        vals.append(v if r["decisive"] else 1 - v)
            return vals
        early = pdec(lambda r: r["control"] == "decisive" and r["location"] == "early")
        late = pdec(lambda r: r["control"] == "decisive" and r["location"] == "late")
        padv = pdec(lambda r: r["control"] == "padding")
        unp = pdec(lambda r: r["control"] == "decisive" and r["n"] != 65)
        wk = [value(r, "primary") for r in e3 if r["model"] == mdl and r["control"] == "weak_only"]
        wk = [v for v in wk if v is not None]
        rows.append([NAMES[mdl], fmt(np.mean(early), 3), fmt(np.mean(late), 3), fmt(np.mean(unp), 3), fmt(np.mean(padv), 3),
                     fmt(np.mean(wk), 3), fmt(np.mean(np.abs(np.array(wk) - 0.5)), 3), len(wk)])
    w(table(["model", "early", "late", "unpadded n<65", "padded n<65", "weak-only mean P", "weak-only mean |P-.5|", "weak-only n"], rows))

    # ------------------------------------------------ costs
    w("\n## 10. Cost (ledger, tag onepass-e1v31)\n")
    led = defaultdict(lambda: [0, 0.0])
    for line in open(HERE.parents[1] / "api" / "ledger.jsonl"):
        r = json.loads(line)
        if r["tag"] == "onepass-e1v31":
            led[(r.get("provider", "anthropic"), r["model"])][0] += 1
            led[(r.get("provider", "anthropic"), r["model"])][1] += r["usd"]
    rows = [[p, m, n, f"${u:.3f}"] for (p, m), (n, u) in sorted(led.items())]
    tot = defaultdict(float)
    for (p, _), (_, u) in led.items():
        tot[p] += u
    rows.append(["total", "", "", ", ".join(f"{p} ${u:.2f}" for p, u in sorted(tot.items()))])
    w(table(["provider", "model", "calls", "USD"], rows))
    metrics["cost"] = dict(tot)

    w("\n## Exclusions\n")
    w("Earlier pilot and smoke runs are excluded. They used a wrong-scale parser, unpinned providers, "
      "identical prompts across seeds, unmatched padding, or a per-clue wording that elicited report "
      "reliability instead of reading the report. The final per-clue check uses a yes/no reading "
      "question. Historical raw responses are not included in this release.\n")

    H = ["## 0. Headline numbers (all generated below; primary analysis unless stated)\n"]
    mj = results[("jev", "listed", "primary")]
    H.append(f"- **E1 primary, r = 0.52, listed.** Sensitivity slope to the normative log-odds (normative 1): " +
             "; ".join(f"{NAMES[m]} {ci(results[(m, 'listed', 'primary')], 'slope|0.52')}" for m in MODELS) +
             f"; Jev per-clue remedy {ci(results[('jev_remedy', 'listed', 'primary')], 'slope|0.52')}.")
    H.append("- **Signed logit error at n = 4 -> n = 64 (r = 0.52, listed; + = overconfident):** " +
             "; ".join(f"{NAMES[m]} {fmt(results[(m, 'listed', 'primary')].get('err_signed|0.52|4', {}).get('est'))} -> "
                       f"{fmt(results[(m, 'listed', 'primary')].get('err_signed|0.52|64', {}).get('est'))}" for m in MODELS) + ".")
    H.append(f"- **Jev at 48 of 64 reports (share .75):** r = .52 {ci(mj, 'p75|0.52|64', 3)} vs normative 0.928; "
             f"r = .55 {ci(mj, 'p75|0.55|64', 3)} vs 0.998; r = .70 {ci(mj, 'p75|0.70|64', 3) if 'p75|0.70|64' in mj else ci(mj, 'p75|0.7|64', 3)} vs 1.000. "
             f"Jev per-clue accuracy {stage['jev']['s1_correct']:,}/{stage['jev']['s1_total']:,} over listed reports; 0.95-gate miss rate at r = .55 "
             f"{ci(mj, 'miss|0.95|0.55')}.")
    H.append(f"- **Coverage (v3.2 rule 4):** slope CI excludes 1 under primary, imp05 and impnorm for "
             f"{', '.join(NAMES[m] + (' (too flat)' if results[(m, 'listed', 'primary')]['slope|0.52']['est'] < 1 else ' (too steep)') for m in holds)}; "
             f"CI includes 1 for {', '.join(NAMES[m] for m in MODELS if m not in holds) or 'none'}. Slope CI excludes 1 on every "
             f"template for {', '.join(NAMES[m] for m in MODELS if tmpl_ok.get(m)) or 'none'} (section 4).")
    H.append("- **Stages (v3.3 rule):** " + "; ".join(f"{NAMES[m]}: {verdicts[m]}" for m in MODELS) + ".")
    H.append("- **Share-minus-weight (Griffin-Tversky elasticity difference, listed; normative 0):** " +
             "; ".join(f"{NAMES[m]} {ci(fits[(m, 'listed')], 'C|share_minus_weight')}" for m in MODELS) + ".")
    H.append("- **E3 primary, n = 65, mean P(YES) decisive + / decisive - (normative 0.99 / 0.01):** " +
             "; ".join(f"{NAMES[m]} {fmt(e3m[(m, 1, 'primary')][0])} / {fmt(e3m[(m, 0, 'primary')][0])}" for m in MODELS) +
             ". The decisive report is diluted by weak reports for every model (section 9).")
    H.append(f"- **Readout:** Jev asked the same YES/NO question as a two-option Choice has slope "
             f"{ci(metrics['_choice_listed'], 'slope|0.52')} (listed, r = .52), i.e. overconfident, versus Noul "
             f"{ci(mj, 'slope|0.52')}. Sonnet 5 with reasoning: slope r=.52 "
             f"{ci(metrics['_sonnet'], 'slope|0.52')}.\n")
    metrics.pop("_choice_listed", None); metrics.pop("_sonnet", None)
    L[2:2] = H
    (OUT / "summary.md").write_text("\n".join(L))
    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer)) else o.item())
    make_figures(results, fits)
    print("wrote", OUT / "summary.md")

# ================================================================ figures

def make_figures(results, fits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"jev": "#d62728", "llama": "#1f77b4", "mistral": "#ff7f0e", "gemma": "#2ca02c", "deepseek": "#9467bd",
              "haiku": "#8c564b", "jev_remedy": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax = axes[0]
    ns = np.array(N_GRID)
    nf = np.geomspace(4, 64, 60)
    ax.plot(nf, logit(sigmoid((2 * 0.75 - 1) * nf * math.log(0.52 / 0.48))), "k--", lw=1.5, label="normative")
    for mdl in MODELS + ["jev_remedy"]:
        m = results[(mdl, "listed", "primary")]
        y = [m.get(f"L75|0.52|{n}", {}).get("est") for n in N_GRID]
        lo = [logit(sigmoid(m.get(f"L75|0.52|{n}", {}).get("lo", np.nan))) for n in N_GRID]
        ls = ":" if mdl == "jev_remedy" else "-"
        lw = 2.2 if mdl == "jev" else 1.1
        ax.plot(ns, y, ls, marker="o" if mdl != "jev_remedy" else "s", ms=3, lw=lw, color=colors[mdl],
                mfc="white" if mdl == "jev_remedy" else colors[mdl], label=NAMES[mdl])
    ticks = [0.5, 0.75, 0.9, 0.95, 0.99, 0.995]
    ax.set_yticks(logit(ticks)); ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xscale("log", base=2); ax.set_xticks(N_GRID); ax.set_xticklabels(N_GRID)
    ax.set_xlabel("reports n (share 0.75, r = 0.52)"); ax.set_ylabel("P(YES), mean on logit scale")
    ax.set_title("(a) confidence vs amount", fontsize=9)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for i, mdl in enumerate(MODELS + ["jev_remedy"]):
        m = results[(mdl, "listed", "primary")]
        y = np.array([m.get(f"err_signed|0.52|{n}", {}).get("est", np.nan) for n in N_GRID], float)
        lo = np.array([m.get(f"err_signed|0.52|{n}", {}).get("lo", np.nan) for n in N_GRID], float)
        hi = np.array([m.get(f"err_signed|0.52|{n}", {}).get("hi", np.nan) for n in N_GRID], float)
        x = ns * (1 + 0.04 * (i - 3))
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=":s" if mdl == "jev_remedy" else "-o", ms=3,
                    lw=2.2 if mdl == "jev" else 1.1, color=colors[mdl], capsize=1.5,
                    mfc="white" if mdl == "jev_remedy" else colors[mdl])
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks(N_GRID); ax.set_xticklabels(N_GRID)
    ax.set_xlabel("reports n (r = 0.52, share $\\neq$ 0.5)")
    ax.set_ylabel("signed logit error\n(+ over-, $-$ underconfident)")
    ax.set_title("(b) error direction by n", fontsize=9)
    ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / "fig1a.pdf"); fig.savefig(OUT / "fig1a.png", dpi=200)
    plt.close(fig)
    # supplement: G&T fitted elasticities
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for mdl in MODELS + ["jev_remedy"]:
        for fm, mk in (("listed", "o"), ("count", "^")):
            if (mdl, fm) not in fits:
                continue
            f = fits[(mdl, fm)]
            if "C|beta_w" not in f or f["C|beta_w"]["est"] is None:
                continue
            ax.errorbar(f["C|beta_w"]["est"], f["C|beta_s"]["est"],
                        xerr=[[f["C|beta_w"]["est"] - f["C|beta_w"]["lo"]], [f["C|beta_w"]["hi"] - f["C|beta_w"]["est"]]],
                        yerr=[[f["C|beta_s"]["est"] - f["C|beta_s"]["lo"]], [f["C|beta_s"]["hi"] - f["C|beta_s"]["est"]]],
                        fmt=mk, color=colors[mdl], ms=5, capsize=1.5,
                        mfc="white" if fm == "count" else colors[mdl],
                        label=f"{NAMES[mdl]} ({fm})")
    ax.plot([1], [1], "k*", ms=10, label="normative")
    lim = ax.get_xlim()
    ax.plot([-1, 3], [-1, 3], color="grey", lw=0.6, ls=":")
    ax.set_xlim(lim)
    ax.set_xlabel("weight elasticity beta_w (log n)"); ax.set_ylabel("strength elasticity beta_s (log|2f-1|)")
    ax.legend(fontsize=5, ncol=2, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gt_fits.pdf"); fig.savefig(OUT / "fig_gt_fits.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
