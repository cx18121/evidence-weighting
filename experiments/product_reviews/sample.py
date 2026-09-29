"""Two-pass streaming, deterministic product-balanced sampling; raw corpus never saved."""

import argparse
import hashlib, json, random, requests
from collections import defaultdict
from pathlib import Path

URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Health_and_Personal_Care.jsonl"
OUT = Path(__file__).parent / "data/products.jsonl"
SEED = 42709


def rows():
    with requests.get(URL, stream=True, timeout=(30, 180)) as r:
        r.raise_for_status()
        for i, line in enumerate(r.iter_lines()):
            if line:
                try:
                    yield i, json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue


def valid(d):
    text = d.get("text")
    try:
        rating = float(d.get("rating"))
    except (ValueError, TypeError):
        return False
    return (
        isinstance(text, str)
        and len(text.strip()) >= 20
        and rating in (1, 2, 3, 4, 5)
        and bool(d.get("parent_asin"))
    )


def shown(i):
    return (
        int.from_bytes(
            hashlib.blake2b(f"{SEED}:split:{i}".encode(), digest_size=8).digest(), "big"
        )
        % 2
        == 0
    )


def rank(i):
    return int.from_bytes(
        hashlib.blake2b(f"{SEED}:order:{i}".encode(), digest_size=8).digest(), "big"
    )


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(
            "sample exists; remove explicitly only if intending to resample"
        )
    # each entry: valid count, shown count, held-out count, held-out star sum
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
        a: s for a, s in stats.items() if s[0] >= 60 and s[1] >= 32 and s[2] >= 20
    }
    groups = {False: [], True: []}
    for a, s in eligible.items():
        groups[s[3] / s[2] >= 4].append(a)
    rng = random.Random(SEED)
    for g in groups.values():
        rng.shuffle(g)
    n = min(120, *(len(g) for g in groups.values()))
    selected = {a: eligible[a] for label in groups for a in groups[label][:n]}
    print(
        "valid products",
        len(stats),
        "eligible",
        len(eligible),
        "eligible labels",
        {str(k): len(v) for k, v in groups.items()},
        "sample per label",
        n,
        flush=True,
    )
    if n < 50:
        raise RuntimeError(
            "insufficient products for balanced sample; do not silently select extreme means"
        )
    samples = {a: [] for a in selected}
    for i, d in rows():
        a = d.get("parent_asin")
        if a in selected and valid(d) and shown(i):
            samples[a].append((rank(i), i, d["text"].strip(), float(d["rating"])))
    with OUT.open("w") as f:
        for a in sorted(selected):
            s = selected[a]
            rev = sorted(samples[a])[:32]
            assert len(rev) == 32
            obj = {
                "asin": a,
                "n_valid": s[0],
                "n_shown_pool": s[1],
                "n_heldout": s[2],
                "heldout_mean": s[3] / s[2],
                "label": int(s[3] / s[2] >= 4),
                "reviews": [
                    {"text": t, "rating": rating, "source_line": i}
                    for _, i, t, rating in rev
                ],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(selected), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
