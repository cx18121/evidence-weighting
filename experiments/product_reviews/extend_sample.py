"""Stream full raw corpus; disjoint eligible calibration products and controlled text sets."""

import argparse, json, random
from collections import defaultdict
from pathlib import Path
from sample import rows, valid, shown, rank, SEED

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-target", action="store_true")
    args = ap.parse_args()
    source = DATA / "products.jsonl"
    if not source.is_file():
        raise SystemExit(f"Run sample.py first: {source} is missing")
    EVAL = {o["asin"]: o for l in source.open() if (o := json.loads(l))}
    target = DATA / "target_products.jsonl"
    controlled = DATA / "controlled.jsonl"
    if (target.exists() or controlled.exists()) and not args.refresh_target:
        raise SystemExit("existing processed sample; not overwriting")
    if args.refresh_target and not controlled.exists():
        raise SystemExit("controlled sample missing")
    stats = defaultdict(lambda: [0, 0, 0, 0.0])
    for i, d in rows():
        if not valid(d):
            continue
        s = stats[d["parent_asin"]]
        s[0] += 1
        if shown(i):
            s[1] += 1
        else:
            s[2] += 1
            s[3] += float(d["rating"])
    eligible = {
        a: s for a, s in stats.items() if s[0] >= 60 and s[1] >= 2 and s[2] >= 1
    }
    assert all(
        a in eligible
        and abs(eligible[a][3] / eligible[a][2] - o["heldout_mean"]) < 1e-9
        for a, o in EVAL.items()
    )
    train = {a: s for a, s in eligible.items() if a not in EVAL}
    print(
        "eligible",
        len(eligible),
        "eval",
        len(EVAL),
        "disjoint target",
        len(train),
        flush=True,
    )
    # All shown-pool human ratings for calibration (no text); full shown-pool texts for eval only.
    ratings = {a: [] for a in train}
    texts = {a: [] for a in EVAL}
    for i, d in rows():
        a = d.get("parent_asin")
        if a not in ratings and a not in texts:
            continue
        if not valid(d) or not shown(i):
            continue
        if a in ratings:
            ratings[a].append((rank(i), int(float(d["rating"]) >= 4)))
        else:
            texts[a].append((rank(i), i, d["text"].strip(), float(d["rating"])))
    with target.open("w") as f:
        for a in sorted(train):
            s = train[a]
            v = ratings[a]
            assert len(v) == s[1]
            f.write(
                json.dumps(
                    {
                        "asin": a,
                        "label": int(s[3] / s[2] >= 4),
                        "heldout_mean": s[3] / s[2],
                        "n_valid": s[0],
                        "n_shown_pool": s[1],
                        "n_heldout": s[2],
                        "shown_positive": [p for _, p in sorted(v)],
                    }
                )
                + "\n"
            )
    conditions = [(k, share) for k in (4, 8, 16, 32) for share in (25, 75)]
    counts = defaultdict(int)
    if args.refresh_target:
        print("existing controlled text sets preserved; target refreshed", flush=True)
        return
    with controlled.open("w") as f:
        for a in sorted(EVAL):
            v = texts[a]
            assert len(v) == EVAL[a]["n_shown_pool"]
            for k, share in conditions:
                pos = sorted((x for x in v if x[3] >= 4), key=lambda x: x[0])
                neg = sorted((x for x in v if x[3] < 4), key=lambda x: x[0])
                j = k * share // 100
                if len(pos) < j or len(neg) < k - j:
                    counts[(k, share, "skip")] += 1
                    continue
                # Deterministic independent random subsets/order for each k/share, not nested.
                rng = random.Random(f"{SEED}:control:{a}:{k}:{share}")
                picks = rng.sample(pos, j) + rng.sample(neg, k - j)
                rng.shuffle(picks)
                obj = {
                    "asin": a,
                    "k": k,
                    "share": share,
                    "label": EVAL[a]["label"],
                    "reviews": [
                        {"text": t, "rating": r, "source_line": line}
                        for _, line, t, r in picks
                    ],
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                counts[(k, share, "valid")] += 1
    print(
        "condition valid/skips", dict(counts), "target products", len(train), flush=True
    )


if __name__ == "__main__":
    main()
