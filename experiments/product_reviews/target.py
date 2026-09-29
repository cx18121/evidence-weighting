"""Human-stars-only target: 16 random shown sets/product/k; disjoint eval products."""

import json, random, math
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent
DATA = ROOT / "data"
KS = (2, 4, 8, 16, 32)


def make_draws():
    path = DATA / "target_draws.jsonl"
    if path.exists():
        return
    source = DATA / "target_products.jsonl"
    if not source.exists():
        raise SystemExit("Missing calibration sample. Run sample.py and extend_sample.py first.")
    with path.open("w") as out:
        for line in source.open():
            obj = json.loads(line)
            pool = obj["shown_positive"]
            rng = random.Random(f"42709:target:{obj['asin']}")
            # Some ≥60-review products have <32 shown-pool reviews after hash split;
            # retain them for feasible k, exclude only for k above their pool size.
            draws = {
                str(k): [sum(rng.sample(pool, k)) for _ in range(16)]
                for k in KS
                if len(pool) >= k
            }
            out.write(
                json.dumps({"asin": obj["asin"], "label": obj["label"], "draws": draws})
                + "\n"
            )


def load():
    return [json.loads(l) for l in (DATA / "target_draws.jsonl").open()]


def feature(k, f):
    t = math.log2(k)
    return [f, f * f, t, t * f, t * f * f]


def fit(objs, inds=None):
    if inds is None:
        inds = range(len(objs))
    X = []
    y = []
    weights = []
    for ix in inds:
        o = objs[ix]
        weight = 1 / (16 * len(o["draws"]))
        for k in KS:
            for j in o["draws"].get(str(k), []):
                X.append(feature(k, j / k))
                y.append(o["label"])
                weights.append(weight)
    clf = LogisticRegression(C=10, max_iter=500).fit(X, y, sample_weight=weights)
    return clf


def rates(objs, k, lo, hi, inds=None):
    if inds is None:
        inds = range(len(objs))
    num = den = 0
    for ix in inds:
        o = objs[ix]
        c = sum(lo <= j / k < hi for j in o["draws"].get(str(k), []))
        den += c
        num += c * o["label"]
    return num / den if den else float("nan"), den


if __name__ == "__main__":
    make_draws()
    objects = load()
    print("products", len(objects), "labels", sum(o["label"] for o in objects))
    clf = fit(objects)
    for k in KS:
        for f in (0.25, 0.75, 1):
            print(k, f, round(clf.predict_proba([feature(k, f)])[0, 1], 3))
