"""Independent raw-log checker for the E4 primary 75%-share mean and target."""

import json, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent
R = ROOT / "results"
D = ROOT / "data"


def parse(row):
    if row.get("error") or row.get("finish_reason") in ("length", "max_tokens"):
        return None
    if row["model"] == "jev":
        try:
            return float(row["raw"]["answers"]["holistic"]["noul"])
        except (KeyError, TypeError, ValueError):
            return None
    text = row["raw"].get("text")
    if not isinstance(text, str):
        return None
    m = re.fullmatch(
        r"\s*(?:\*\*)?\s*(\d+(?:\.\d*)?|\.\d+)(?:[eE]([+-]?\d+))?\s*%?\s*(?:\*\*)?\s*",
        text,
    )
    if not m:
        return None
    x = float(m[1]) * (10 ** int(m[2] or 0))
    if not 0 <= x <= 100 or 0 < x < 1:
        return None  # ambiguous 0–1 number (scale)
    return x / 100


def main():
    rows = {}
    for l in (R / "v31_calls.jsonl").open():
        x = json.loads(l)
        if x.get("condition") != "controlled" or x.get("share") != 75:
            continue
        p = parse(x)
        if p is not None:
            rows[x["model"], x["k"], x["asin"]] = p
    # Separate reimplementation of human-stars-only logistic fitting; product equal weight.
    X = []
    y = []
    w = []
    objs = [json.loads(l) for l in (D / "target_draws.jsonl").open()]
    for o in objs:
        ks = o["draws"]
        for key, js in ks.items():
            k = int(key)
            t = math.log2(k)
            for j in js:
                f = j / k
                X.append([f, f * f, t, t * f, t * f * f])
                y.append(o["label"])
                w.append(1 / (len(ks) * len(js)))
    model = LogisticRegression(C=10, max_iter=500).fit(X, y, sample_weight=w)
    summary = (R / "summary.md").read_text()
    start = summary.index("### Controlled shares")
    end = summary.index("### Jev per-review", start)
    table = summary[start:end]
    lines = [
        "# Independent E4 primary check — spec v3.2",
        "",
        f"Read raw API responses and human-label calibration draws independently from analysis code. The target is refitted on {len(objs)} disjoint products using human labels alone; numbers agree with summary to displayed precision. This checks point estimates; 95% confidence intervals in summary.md use product-cluster percentile bootstraps (target 180, model 250 draws).",
        "",
        "| k | model | independently reparsed n | mean | human-star target | summary absolute discrepancy |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for k in (4, 8, 16, 32):
        t = float(
            model.predict_proba(
                [
                    [
                        0.75,
                        0.75**2,
                        math.log2(k),
                        math.log2(k) * 0.75,
                        math.log2(k) * 0.75**2,
                    ]
                ]
            )[0, 1]
        )
        target_match = re.search(
            rf"^\| {k} \| 75% \|[^\n]*?\| \d+ \| ([0-9.]+) \[", table, re.M
        )
        if not target_match or abs(float(target_match[1]) - t) >= 0.00051:
            raise RuntimeError(f"target mismatch at k={k}")
        for m in ("jev", "llama", "haiku"):
            a = [p for (mm, kk, _), p in rows.items() if mm == m and kk == k]
            v = np.mean(a)
            match = re.search(
                rf"^\| {k} \| 75% \|[^\n]*\| {m} \| (\d+) \| ([0-9.]+) \[", table, re.M
            )
            if not match:
                raise RuntimeError(f"missing summary cell {k},{m}")
            assert int(match[1]) == len(a)
            diff = abs(float(match[2]) - v)
            assert diff < 0.00051, (k, m, diff)
            lines.append(f"| {k} | {m} | {len(a)} | {v:.6f} | {t:.6f} | {diff:.6f} |")
    (R / "independent_check.md").write_text("\n".join(lines) + "\n")
    print("independent primary cells verified", len(lines) - 6)


if __name__ == "__main__":
    main()
