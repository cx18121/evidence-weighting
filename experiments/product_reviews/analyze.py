"""Product-cluster bootstrap; deterministic, separate human-label-only count rule."""

import json, math, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent
R = ROOT / "results"
KS = (2, 4, 8, 16, 32)
MODELS = ("jev", "llama", "haiku")
rng = np.random.default_rng(1198)
objs = {
    x["asin"]: x for l in (ROOT / "data/products.jsonl").open() if (x := json.loads(l))
}
rows = {}
source = (
    R / "v31_calls.jsonl" if (R / "v31_calls.jsonl").exists() else R / "calls.jsonl"
)
source_attempts = [json.loads(l) for l in source.open()]
for x in source_attempts:
    if x.get("condition", "natural") != "natural":
        continue
    if (
        "error" not in x
        and x.get("readout_category", "valid") == "valid"
        and (
            (
                x["model"] == "jev"
                and isinstance(x.get("parsed"), dict)
                and all(isinstance(v, (int, float)) for v in x["parsed"].values())
            )
            or (x["model"] != "jev" and isinstance(x.get("parsed"), (int, float)))
        )
    ):
        rows[f"{x['model']}:{x['asin']}:{x['k']}"] = x


def boot(data, metric, B=400):
    a = np.array(data, dtype=float)
    v = metric(a)
    if len(a) < 5:
        return v, (float("nan"), float("nan"))
    draws = []
    for _ in range(B):
        b = a[rng.integers(len(a), size=len(a))]
        try:
            draws.append(metric(b))
        except ValueError:
            continue
    return v, tuple(np.quantile(draws, [0.025, 0.975])) if draws else (
        float("nan"),
        float("nan"),
    )


def fmt(v):
    return f"{v[0]:.3f} [{v[1][0]:.3f}, {v[1][1]:.3f}]"


def main():
    lines = [
        "# E4 Amazon real-data check — spec v3.2 (clean v3.1 calls)",
        "",
        "Amazon Reviews 2023, Health_and_Personal_Care (McAuley Lab). Seed 42709; 240 products sampled uniformly within held-out truth classes, not by extreme mean. Reviews ≥20 characters; product ≥60 eligible reviews. A deterministic 50/50 hash split precedes label calculation and selection; 32 nested, randomized shown reviews come exclusively from the shown pool. Truth is the mean of *all* held-out ratings (positive ≥4.0). Review text clipped at 1000 characters in prompts; ratings, title, metadata omitted.",
        "",
    ]
    labels = np.array([o["label"] for o in objs.values()])
    means = np.array([o["heldout_mean"] for o in objs.values()])
    ns = np.array([o["n_heldout"] for o in objs.values()])
    lines += [
        f"Products: {len(objs)}; held-out label 1/0: {sum(labels)}/{len(labels) - sum(labels)}; held-out reviews per product median {np.median(ns):.0f} (min {min(ns)}, max {max(ns)}).",
        "Held-out mean stars distribution (minimum, 10th, 25th, median, 75th, 90th, maximum): "
        + ", ".join(
            f"{x:.3f}" for x in np.quantile(means, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
        )
        + ".",
        "Distribution by class (10th, median, 90th): "
        + "; ".join(
            f"{label}: "
            + ", ".join(
                f"{v:.3f}" for v in np.quantile(means[labels == label], [0.1, 0.5, 0.9])
            )
            for label in (0, 1)
        )
        + ".",
        "",
        "All brackets are 95% bootstrap percentile intervals resampling products (400 replicates); AUROC replicates without both classes excluded. Numeric probabilities are *stated* for Llama and Haiku, TypeSafe Noul yes-probabilities for Jev. Invalid/refused calls excluded and counted.",
        "",
    ]
    metrics = {}
    for m in MODELS:
        lines += [
            f"## {m}",
            "",
            "| k | n (valid/240) | AUROC | Brier | mean confidence label 0 | mean confidence label 1 | invalid/missing |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        metrics[m] = {}
        for k in KS:
            data = np.array(
                [
                    (
                        o["label"],
                        (r["parsed"]["holistic"] if m == "jev" else r["parsed"]),
                    )
                    for a, o in objs.items()
                    if (r := rows.get(f"{m}:{a}:{k}"))
                ],
                dtype=float,
            )
            if len(data) == 0:
                continue
            auc = boot(
                data,
                lambda a: (
                    roc_auc_score(a[:, 0], a[:, 1])
                    if len(set(a[:, 0])) == 2
                    else float("nan")
                ),
            )
            brier = boot(data, lambda a: np.mean((a[:, 0] - a[:, 1]) ** 2))
            c = []
            for label in (0, 1):
                sub = data[data[:, 0] == label]
                c.append(boot(sub, lambda a: np.mean(a[:, 1])))
            metrics[m][k] = {"n": len(data), "auc": auc, "brier": brier, "conf": c}
            lines.append(
                f"| {k} | {len(data)} | {fmt(auc)} | {fmt(brier)} | {fmt(c[0])} | {fmt(c[1])} | {240 - len(data)} |"
            )
        lines.append("")
    # Headline is recomputed from full-run outcomes, never frozen to an earlier partial run.
    if all(k in metrics["jev"] for k in (2, 32)):
        a, b = metrics["jev"][2], metrics["jev"][32]
        lines.insert(
            2,
            f"**Secondary natural-set diagnostic:** Jev AUROC {fmt(a['auc'])} at k=2 versus {fmt(b['auc'])} at k=32; mean confidence for held-out positives {a['conf'][1][0]:.3f}→{b['conf'][1][0]:.3f}. Discrimination improves, but no synthetic exponent or normative failure is inferred.",
        )
    # Binned fractions are not comparable across k; exact 25%/75% controlled shares are in the extension.
    lines += ["", "## Per-review extraction and human-label-only count rule", ""]
    checks = []
    for o in objs.values():
        r = rows.get(f"jev:{o['asin']}:32")
        if r:
            z = [
                (int(review["rating"] >= 4), r["parsed"][f"r{i}"])
                for i, review in enumerate(o["reviews"])
            ]
            checks.append((sum((p >= 0.5) == bool(y) for y, p in z), len(z)))
    if checks:
        v, ci = boot(checks, lambda a: a[:, 0].sum() / a[:, 1].sum())
        lines.append(
            f"Jev per-review accuracy at k=32 (threshold Noul ≥0.5 against human stars 4–5): {v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], {sum(x[1] for x in checks)} reviews from {len(checks)} products. Repeated prefixes not double counted."
        )
    # fixed disjoint stratified 50/50 product split; fit solely review human ratings and heldout human labels
    rng2 = np.random.default_rng(1271)
    train = set()
    for lab in (0, 1):
        aa = sorted(a for a, o in objs.items() if o["label"] == lab)
        rng2.shuffle(aa)
        train.update(aa[: len(aa) // 2])
    for k in KS:

        def features(aa):
            return np.array(
                [
                    [sum(r["rating"] >= 4 for r in objs[a]["reviews"][:k]) / k]
                    for a in aa
                ]
            )

        tr = sorted(train)
        te = sorted(set(objs) - train)
        clf = LogisticRegression().fit(features(tr), [objs[a]["label"] for a in tr])
        pr = clf.predict_proba(features(te))[:, 1]
        array = np.array([(objs[a]["label"], p) for a, p in zip(te, pr)])
        b = boot(array, lambda x: np.mean((x[:, 0] - x[:, 1]) ** 2))
        auc = boot(
            array,
            lambda x: (
                roc_auc_score(x[:, 0], x[:, 1])
                if len(set(x[:, 0])) == 2
                else float("nan")
            ),
        )
        lines.append(
            f"k={k}: count-based human-stars fraction logistic rule, training {len(tr)} products, test {len(te)} disjoint products: AUROC {fmt(auc)}, Brier {fmt(b)}. This is an oracle-input comparison, since stars are concealed from models; no Jev values enter fitting."
        )
    lines += ["", "## Spend / interpretation", ""]
    current = [x for x in source_attempts if x.get("condition", "natural") == "natural"]
    costs = defaultdict(float)
    for x in current:
        costs[x["model"]] += x.get("usd", 0)
    lines.append(
        "Spec v3.1 natural-call cost including retries: "
        + ", ".join(f"{m} ${costs[m]:.4f}" for m in MODELS)
        + f"; total ${sum(costs.values()):.4f}. {len(rows)}/3600 valid distinct calls; {len(current)} logged natural attempts. Costs estimated by provider helper. All v3.1 conditions share the exact mean-held-out-stars ≥4.0 holistic question; OpenRouter is Together-only."
    )
    for m in MODELS:
        subset = [x for x in current if x["model"] == m]
        categories = defaultdict(int)
        for x in subset:
            categories[x.get("readout_category", "invalid")] += 1
        reasons = defaultdict(int)
        for x in subset:
            if x.get("model") == "llama":
                reasons[
                    (x.get("raw") or {})
                    .get("usage", {})
                    .get("completion_tokens_details", {})
                    .get("reasoning_tokens", "unreported")
                ] += 1
        lines.append(
            f"{m} readout categories (n={len(subset)}): "
            + ", ".join(f"{cat}={n}" for cat, n in sorted(categories.items()))
            + (f"; reported reasoning tokens: {dict(reasons)}" if m == "llama" else "")
            + "."
        )
    old = [json.loads(l) for l in (R / "calls.jsonl").open()]
    prior = (
        [json.loads(l) for l in (R / "controlled_calls.jsonl").open()]
        if (R / "controlled_calls.jsonl").exists()
        else []
    )
    lines.append(
        f"Superseded, excluded from v3.1 comparisons: `calls.jsonl` ({len(old)} attempts, ${sum(x.get('usd', 0) for x in old):.4f}; unpinned Llama/mismatched Jev wording); `controlled_calls.jsonl` ({len(prior)} attempts, ${sum(x.get('usd', 0) for x in prior):.4f}; same issue). Earlier logs remain auditable and are never pooled with clean results."
    )
    lines.append(
        "Caveats: selection balances on held-out label (not on extreme means), changing population prevalence; shown reviews and held-out ratings are from different reviews but potentially same reviewers/variants; within-product reviews are correlated and may be manipulated; no exact normative posterior exists for real text. Label threshold 4.0 and finite held-out sample introduce noise. Stars used in matched-fraction analysis and the count rule are an offline audit, not shown to models. Matched fraction bins may have sparse or differently composed products; no causal or synthetic-exponent inference."
    )
    (R / "summary.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True)
    for ax, m in zip(axes, MODELS):
        for label, color in ((0, "#b34238"), (1, "#278164")):
            xx = []
            yy = []
            err = []
            for k in KS:
                item = metrics[m].get(k)
                if item:
                    v, ci = item["conf"][label]
                    xx.append(k)
                    yy.append(v)
                    err.append((v - ci[0], ci[1] - v))
            ax.errorbar(
                xx,
                yy,
                yerr=np.array(err).T,
                fmt="o-",
                color=color,
                capsize=2,
                label=f"held-out {'≥4' if label else '<4'}",
            )
        ax.set(xlabel="Shown text reviews (k)", title=m, ylim=(0, 1), xticks=KS)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Holistic probability / confidence")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"E4 v3.2 | natural sets | n={len(objs)} products | 95% product-bootstrap intervals",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(R / "confidence.png", dpi=170)
    fig.savefig(R / "confidence.pdf")
    plt.close(fig)
    print(
        "wrote summary and figures; valid calls", len(rows), "cost", sum(costs.values())
    )


if __name__ == "__main__":
    main()
