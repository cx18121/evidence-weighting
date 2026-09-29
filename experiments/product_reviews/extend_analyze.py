"""Disjoint empirical target, controlled shares, and human-label-trained Jev remedy."""

import json, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from target import make_draws, load, fit, feature, rates, KS

ROOT = Path(__file__).parent
R = ROOT / "results"
DATA = ROOT / "data"
rng = np.random.default_rng(9907)
if not (DATA / "products.jsonl").exists():
    raise SystemExit("Missing product sample. Run sample.py and extend_sample.py first.")
EVAL = {o["asin"]: o for l in (DATA / "products.jsonl").open() if (o := json.loads(l))}


def readlog(path):
    r = {}
    attempts = []
    for l in path.open():
        x = json.loads(l)
        attempts.append(x)
        if "error" not in x and (
            (
                x["model"] == "jev"
                and isinstance(x.get("parsed"), dict)
                and all(isinstance(v, (float, int)) for v in x["parsed"].values())
            )
            or (x["model"] != "jev" and isinstance(x.get("parsed"), (int, float)))
        ):
            r[x["key"]] = x
    return r, attempts


# Use the corrected calls when present; earlier logs used different wording.
if (R / "v31_calls.jsonl").exists():
    natural = {}
    controlled = {}
    attempts = [json.loads(l) for l in (R / "v31_calls.jsonl").open()]
    for x in attempts:
        if x.get("readout_category") != "valid" or "error" in x:
            continue
        if x.get("condition") == "natural":
            natural[f"{x['model']}:{x['asin']}:{x['k']}"] = x
        elif x.get("condition") == "controlled":
            controlled[f"{x['model']}:{x['asin']}:share{x['share']}:{x['k']}"] = x
else:
    natural, _ = readlog(R / "calls.jsonl")
    controlled, attempts = readlog(R / "controlled_calls.jsonl")


def value(r):
    return r["parsed"]["holistic"] if r["model"] == "jev" else r["parsed"]


def fmt(x):
    return f"{x:.3f}"


def interval(vals):
    v = np.array(vals, dtype=float)
    v = v[np.isfinite(v)]
    return (
        f"{np.quantile(v, 0.025):.3f}, {np.quantile(v, 0.975):.3f}" if len(v) else "NA"
    )


def score(a, kind):
    if not len(a):
        return float("nan")
    y, p = a[:, 0], a[:, 1]
    if kind == "Brier":
        return np.mean((y - p) ** 2)
    if kind == "log loss":
        return np.mean(
            -y * np.log(np.clip(p, 0.005, 0.995))
            - (1 - y) * np.log(np.clip(1 - p, 0.005, 0.995))
        )
    if kind == "AUROC":
        return roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")
    if kind == "gate":
        return (
            np.mean(p[a[:, 2] > 0.9] > 0.9) if np.any(a[:, 2] > 0.9) else float("nan")
        )


def boot_metric(records, kind, B=250):
    # records (asin,label,p,target); resample PRODUCT rather than prompt.
    groups = defaultdict(list)
    for a, y, p, t in records:
        groups[a].append([y, p, t])
    ids = list(groups)
    v = score(np.array([z for a in ids for z in groups[a]]), kind)
    draws = []
    for _ in range(B):
        sampled = rng.choice(ids, len(ids), replace=True)
        array = np.array([z for a in sampled for z in groups[a]])
        draws.append(score(array, kind))
    if kind == "gate" and v in (0, 1):
        data = np.array([z for a in ids for z in groups[a]])
        n = sum(data[:, 2] > 0.9)
        if n:
            z = 1.96
            center = (v + z * z / (2 * n)) / (1 + z * z / n)
            half = (
                z * math.sqrt(v * (1 - v) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            )
            return v, f"{max(0, center - half):.3f}, {min(1, center + half):.3f}"
    return v, interval(draws)


def main():
    make_draws()
    objects = load()
    clf = fit(objects)
    pred = lambda k, f: float(clf.predict_proba([feature(k, f)])[0, 1])
    # product-cluster bootstraps of both empirical bins and smooth curve.
    B = 180
    N = len(objects)
    bootidx = [rng.integers(N, size=N) for _ in range(B)]
    bootmodels = [fit(objects, inds) for inds in bootidx]
    curves = {}
    for k in (4, 8, 16, 32):
        for share in (25, 75):
            f = share / 100
            v = pred(k, f)
            bv = []
            for model in bootmodels:
                bv.append(float(model.predict_proba([feature(k, f)])[0, 1]))
            curves[k, share] = (v, interval(bv))
    # Determine empirical bin rates and bootstrap via product sums, not treating 16 correlated draws as independent.
    cells = {}
    for k in KS:
        for bi, (lo, hi) in enumerate(
            zip((0, 0.25, 0.5, 0.75), (0.25, 0.5, 0.75, 1.00001))
        ):
            counts = np.array(
                [
                    sum(lo <= j / k < hi for j in o["draws"].get(str(k), []))
                    for o in objects
                ]
            )
            labs = np.array([o["label"] for o in objects])
            den = counts.sum()
            if not den:
                continue
            v = np.sum(counts * labs) / den
            bs = []
            for ix in bootidx:
                d = sum(counts[ix])
                bs.append(np.dot(counts[ix], labs[ix]) / d if d else float("nan"))
            cells[k, bi] = (v, interval(bs), int(den), int(sum(counts > 0)))
    lines = [
        "\n## Extension: independent empirical target and fixed shares",
        "",
        f"Empirical target (spec v3.1): **{N} eligible calibration products, disjoint from all 240 evaluation products**, including every other product meeting ≥60 text reviews; k-specific calculations require ≥k shown-pool reviews. Their held-out label prevalence is {sum(o['label'] for o in objects)}/{N}={np.mean([o['label'] for o in objects]):.3f}; evaluation was deliberately balanced 120/120. For each calibration product and k, 16 independent uniform shown-pool draws without replacement *within each set*; held-out reviews never enter shown sets. Target bins pool the resulting draws; intervals resample products (180 draws), not reviews. Smooth reference for exactly 25%/75% uses human-label-only weighted logistic regression of held-out label on fraction, fraction², log₂(k) and their interactions; each product has total fitting weight one (over its feasible k). Bootstrap the entire fit over products (180 draws). It is observational, regularized and potentially extrapolates at rare k/share cells—not an exact normative posterior.",
        "",
        "| k | shown positive fraction bin | target P(good) [95% CI] | sampled sets | contributing products |",
        "|---:|---|---|---:|---:|",
    ]
    for k in KS:
        for bi, (lo, hi) in enumerate(
            zip((0, 0.25, 0.5, 0.75), (0.25, 0.5, 0.75, 1.00001))
        ):
            if (k, bi) not in cells:
                continue
            t, ci, n, products = cells[k, bi]
            lines.append(
                f"| {k} | [{lo:.2f},{min(hi, 1):.2f}{']' if hi > 1 else ')'} | {fmt(t)} [{ci}] | {n} | {products} |"
            )
    lines += [
        "",
        "### Controlled shares, texts without stars",
        "",
        "One independently randomized review set per feasible product and (k, share), sampled from its original shown pool. Feasibility requires exactly j positive (human rating 4–5) and k−j other reviews; skipped products are **not** replaced. The same text set goes to all three models; Jev per-review Nouls are in the same call as holistic. The disjoint-calibration smooth target estimates a random-shown-set conditional probability, whereas controlled sets include each feasible product once: target/model gaps are descriptive and selection/composition can differ. Jev Noul and LLM stated numbers are different readout interfaces, not interchangeable calibrated probabilities. CIs for model means resample products.",
        "",
        "| k | positive share | feasible/240 (skipped) | exact-j calibration draws | smooth target [95% CI] | model | valid | mean confidence [95% CI] | model − target |",
        "|---:|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    sets = [json.loads(l) for l in (DATA / "controlled.jsonl").open()]
    by = {(x["asin"], x["k"], x["share"]): x for x in sets}
    plotted = {}
    for k in (4, 8, 16, 32):
        for share in (25, 75):
            selected = [x for x in sets if x["k"] == k and x["share"] == share]
            t, ci = curves[k, share]
            support = sum(
                sum(j == k * share // 100 for j in o["draws"].get(str(k), []))
                for o in objects
            )
            for model in ("jev", "llama", "haiku"):
                rec = []
                for x in selected:
                    r = controlled.get(f"{model}:{x['asin']}:share{share}:{k}")
                    if r:
                        rec.append((x["asin"], x["label"], value(r), t))
                if not rec:
                    continue
                v = np.mean([z[2] for z in rec])
                ids = np.array([z[2] for z in rec])
                boots = [
                    np.mean(ids[rng.integers(len(ids), size=len(ids))])
                    for _ in range(250)
                ]
                plotted[k, share, model] = (v, interval(boots), len(rec))
                lines.append(
                    f"| {k} | {share}% | {len(selected)}/240 ({240 - len(selected)}) | {support} | {fmt(t)} [{ci}] | {model} | {len(rec)} | {fmt(v)} [{interval(boots)}] | {v - t:+.3f} |"
                )
    lines.insert(
        2,
        f"**Primary v3.2, exact 75% positive shown stars:** target {curves[4, 75][0]:.3f} (k=4) → {curves[32, 75][0]:.3f} (k=32); Jev {plotted[4, 75, 'jev'][0]:.3f} → {plotted[32, 75, 'jev'][0]:.3f}; Together-pinned Llama {plotted[4, 75, 'llama'][0]:.3f} → {plotted[32, 75, 'llama'][0]:.3f}; Haiku {plotted[4, 75, 'haiku'][0]:.3f} → {plotted[32, 75, 'haiku'][0]:.3f}. Feasible product n changes with k; common-product sensitivity appears below.**",
    )
    lines.insert(3, "")
    lines += [
        "",
        "**Primary reported E4 contrast (v3.2 analysis):** confidence against k for the **exactly 75%** controlled sets, compared with the human-label empirical target in the table and figure above. Other confidence contrasts and the 25% arm are secondary. As k increases, different products remain feasible; the table is not a paired fixed-product comparison across all k.",
        "",
        "**Common-product robustness (75% share):** restrict all k to products feasible at k=32; these are the same products at k=4,8,16,32. Their review sets are independent per k (not nested).",
        "",
        "| k | same products | smooth target | Jev mean [95% product CI] | Llama mean [95% product CI] | Haiku mean [95% product CI] |",
        "|---:|---:|---:|---|---|---|",
    ]
    common = sorted(x["asin"] for x in sets if x["k"] == 32 and x["share"] == 75)
    for k in (4, 8, 16, 32):
        texts = [f"| {k} | {len(common)} | {curves[k, 75][0]:.3f}"]
        for model in ("jev", "llama", "haiku"):
            v = np.array(
                [
                    value(controlled[f"{model}:{a}:share75:{k}"])
                    for a in common
                    if f"{model}:{a}:share75:{k}" in controlled
                ]
            )
            b = [np.mean(v[rng.integers(len(v), size=len(v))]) for _ in range(250)]
            texts.append(f"{np.mean(v):.3f} [{interval(b)}] (n={len(v)})")
        lines.append(" | ".join(texts) + " |")
    lines += [
        "",
        "### Jev per-review count remedy on disjoint evaluation products",
        "",
        f"The same human-only calibration logistic rule above is trained **only** on the {N} non-evaluation products; at evaluation its input is k and the fraction of Jev per-review Nouls ≥0.5, never human stars. No Jev output enters training, tuning or model selection. Both outcomes below are the held-out human label. Log loss clips probability to [0.005,0.995]; AUROC is threshold-free. Product-cluster bootstrap 95% CIs, 250 draws. Controlled table pools both shares at each k, clustering duplicate products.",
        "",
        "| set | k | valid products | condition rows | method | Brier [95% CI] | log loss [95% CI] | AUROC [95% CI] | target>0.9 products/rows | gate P(pred>0.9 | target>0.9) [95% CI] |",
        "|---|---:|---:|---:|---|---|---|---|---:|---|",
    ]
    for setting in ("natural", "controlled"):
        for k in KS if setting == "natural" else (4, 8, 16, 32):
            rec = {method: [] for method in ("holistic", "per-review rule")}
            examples = (
                [
                    {
                        "asin": a,
                        "label": o["label"],
                        "k": k,
                        "share": None,
                        "reviews": o["reviews"][:k],
                    }
                    for a, o in EVAL.items()
                ]
                if setting == "natural"
                else [x for x in sets if x["k"] == k]
            )
            for x in examples:
                a = x["asin"]
                suffix = (
                    f"{a}:{k}" if setting == "natural" else f"{a}:share{x['share']}:{k}"
                )
                row = (natural if setting == "natural" else controlled).get(
                    "jev:" + suffix
                )
                if not row:
                    continue
                t = pred(k, sum(r["rating"] >= 4 for r in x["reviews"]) / k)
                fjev = sum(row["parsed"][f"r{i}"] >= 0.5 for i in range(k)) / k
                rec["holistic"].append((a, x["label"], row["parsed"]["holistic"], t))
                rec["per-review rule"].append((a, x["label"], pred(k, fjev), t))
            for method in rec:
                a = rec[method]
                n_gate = sum(r[3] > 0.9 for r in a)
                met = []
                for name in ("Brier", "log loss", "AUROC", "gate"):
                    v, ci = boot_metric(a, name)
                    met.append(
                        "NA (zero target-positive rows)"
                        if math.isnan(v)
                        else f"{fmt(v)} [{ci}]"
                    )
                lines.append(
                    f"| {setting} | {k} | {len(set(r[0] for r in a))} | {len(a)} | {method} | {met[0]} | {met[1]} | {met[2]} | {n_gate} | {met[3]} |"
                )
    lines += [
        "",
        "**Paired Brier improvement at k=32 (remedy minus holistic; negative is better):**",
    ]
    for condition in ("natural", "controlled"):
        records = []
        for x in (
            [
                {"asin": a, "label": o["label"], "reviews": o["reviews"][:32]}
                for a, o in EVAL.items()
            ]
            if condition == "natural"
            else [o for o in sets if o["k"] == 32]
        ):
            k = 32
            a = x["asin"]
            suffix = (
                f"{a}:{k}" if condition == "natural" else f"{a}:share{x['share']}:{k}"
            )
            row = (natural if condition == "natural" else controlled).get(
                "jev:" + suffix
            )
            if not row:
                continue
            h = row["parsed"]["holistic"]
            det = sum(row["parsed"][f"r{i}"] >= 0.5 for i in range(k)) / k
            rr = pred(k, det)
            records.append((a, (rr - x["label"]) ** 2 - (h - x["label"]) ** 2))
        groups = defaultdict(list)
        for a, d in records:
            groups[a].append(d)
        ids = list(groups)
        bs = []
        for _ in range(250):
            sample = rng.choice(ids, size=len(ids), replace=True)
            bs.append(np.mean([v for a in sample for v in groups[a]]))
        lines.append(
            f"{condition}: {np.mean([d for _, d in records]):+.3f} [{interval(bs)}] (n={len(groups)} products, {len(records)} rows; 95% product-cluster bootstrap)."
        )
    lines.append(
        "The Jev per-review rule improves Brier on natural nested sets at k=32, but **worsens it on fixed-share controlled sets**; no general remedy success is claimed."
    )
    lines += [
        "",
        "**Gate caveat:** A zero target>0.9 denominator makes recall at that gate undefined, not 0%. The 25%/75% controlled shares may never cross the 0.9 empirical-target threshold; natural shown sets are also reported to make this explicit. Target and remedy are human-label-trained estimates, not independent normative truth. The evaluation set was balanced by held-out label; its calibration/Brier/log-loss do not estimate the original population prevalence.",
        "",
    ]
    # Sensitivity to excluded/invalid outcomes: 0.5 versus the empirical-target score.
    lines += [
        "",
        "### Invalid/excluded readouts and sensitivity (v3.2)",
        "",
        "For each exact-share cell, missing/invalid model probabilities are imputed first at 0.5 and then at the smooth empirical target. The latter is a human-data reference, **not a normative answer** (none exists for real text). The same feasible products are used in both scenarios; no silent dropping.",
        "",
        "| model | arm | k | valid / feasible | categories for recorded calls | complete-case mean | impute 0.5 | impute empirical target |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for model in ("jev", "llama", "haiku"):
        for share in (25, 75):
            for k in (4, 8, 16, 32):
                relevant = [x for x in sets if x["share"] == share and x["k"] == k]
                v = [
                    value(controlled[f"{model}:{x['asin']}:share{share}:{k}"])
                    for x in relevant
                    if f"{model}:{x['asin']}:share{share}:{k}" in controlled
                ]
                t = curves[k, share][0]
                missing = len(relevant) - len(v)
                cats = Counter(
                    r.get("readout_category", "invalid")
                    for r in attempts
                    if r.get("condition") == "controlled"
                    and r["model"] == model
                    and r.get("k") == k
                    and r.get("share") == share
                )
                lines.append(
                    f"| {model} | {share}% | {k} | {len(v)}/{len(relevant)} | {dict(cats)} | {np.mean(v):.3f} | {(sum(v) + missing * 0.5) / len(relevant):.3f} | {(sum(v) + missing * t) / len(relevant):.3f} |"
                )
    cost = defaultdict(float)
    for x in attempts:
        cost[x["model"]] += x.get("usd", 0)
    lines.append(
        f"Clean v3.1 calls: {len(attempts)} attempts, {len(controlled)} valid distinct controlled calls for {len(sets)} feasible product-condition sets; cost "
        + ", ".join(f"{m} ${cost[m]:.4f}" for m in ("jev", "llama", "haiku"))
        + f", total ${sum(cost.values()):.4f} (natural + controlled). Failed/invalid records remain in `v31_calls.jsonl`; `controlled_calls.jsonl` is superseded."
    )
    if (R / "retest_calls.jsonl").exists():
        rep = [json.loads(l) for l in (R / "retest_calls.jsonl").open()]
        lines += [
            "",
            "### Identical-input test–retest (v3.2)",
            "",
            "50 fixed products in the controlled 75%-positive k=8 arm were queried once more per model with byte-identical prompt/question text and request configuration. Agreement means identical numeric readout (not just class); differences may reflect provider/model nondeterminism. Retest CIs are product-bootstrap unless agreement equals 0 or 1, where Wilson bounds apply.",
            "",
            "| model | valid paired / 50 | exactly identical [95% CI] | mean absolute shift [95% CI] | max shift |",
            "|---|---:|---:|---:|---:|",
        ]
        for m in ("jev", "llama", "haiku"):
            pairs = []
            for x in rep:
                if x["model"] != m or x.get("readout_category") != "valid":
                    continue
                original = controlled.get(f"{m}:{x['asin']}:share75:8")
                if original:
                    pairs.append((value(original), value(x)))
            if pairs:
                d = np.abs(np.diff(np.array(pairs), axis=1).reshape(-1))
                ident = np.mean(d == 0)
                if ident in (0, 1):
                    n = len(d)
                    z = 1.96
                    center = (ident + z * z / (2 * n)) / (1 + z * z / n)
                    half = (
                        z
                        * math.sqrt(ident * (1 - ident) / n + z * z / (4 * n * n))
                        / (1 + z * z / n)
                    )
                    agreeci = (
                        f"{max(0, center - half):.3f}, {min(1, center + half):.3f}"
                    )
                else:
                    agreeci = interval(
                        [
                            np.mean(d[rng.integers(len(d), size=len(d))] == 0)
                            for _ in range(250)
                        ]
                    )
                spread = interval(
                    [np.mean(d[rng.integers(len(d), size=len(d))]) for _ in range(250)]
                )
                lines.append(
                    f"| {m} | {len(pairs)}/50 | {ident:.3f} [{agreeci}] | {np.mean(d):.3f} [{spread}] | {max(d):.3f} |"
                )
    path = R / "summary.md"
    base = path.read_text().split("\n## Extension:")[0].rstrip()
    path.write_text(base + "\n" + "\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    for ax, share in zip(axes, (25, 75)):
        for model, color in (
            ("target", "#222222"),
            ("jev", "#2c7783"),
            ("llama", "#b96236"),
            ("haiku", "#8176a6"),
        ):
            xx = []
            yy = []
            low = []
            high = []
            for k in (4, 8, 16, 32):
                entry = (
                    curves[k, share]
                    if model == "target"
                    else plotted.get((k, share, model))
                )
                if not entry:
                    continue
                v, ci = entry[:2]
                lo, hi = map(float, ci.split(", "))
                xx.append(k)
                yy.append(v)
                low.append(max(0, v - lo))
                high.append(max(0, hi - v))
            ax.errorbar(
                xx, yy, yerr=[low, high], fmt="o-", label=model, color=color, capsize=2
            )
        sizes = [
            sum(x["k"] == k and x["share"] == share for x in sets)
            for k in (4, 8, 16, 32)
        ]
        ax.set(
            title=f"{share}% positive shown stars",
            xlabel=f"k=4,8,16,32; n={','.join(map(str, sizes))}",
            xticks=(4, 8, 16, 32),
            ylim=(0, 1),
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean confidence / P(held-out good)")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        f"E4 v3.2 | {N} calibration products | 95% product-bootstrap intervals",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(R / "controlled_confidence.pdf")
    fig.savefig(R / "controlled_confidence.png", dpi=170)
    plt.close(fig)
    print(
        "extension written",
        len(sets),
        "controlled sets",
        len(controlled),
        "valid calls",
        flush=True,
    )


if __name__ == "__main__":
    main()
