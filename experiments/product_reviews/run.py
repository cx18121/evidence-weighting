"""Resumable calls; --smoke runs the first product only; --limit limits products."""

import argparse, concurrent.futures, fcntl, json, re, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api"))
import providers as api

# The shared ledger may be written concurrently; synchronize reads with
# its writer lock to prevent transient partial JSON from aborting a long run.
_original_refresh = api._refresh
_refresh_thread_lock = threading.Lock()


def _locked_refresh():
    # The helper also keeps a mutable read cursor; serialize threads *within* process.
    with _refresh_thread_lock, open(api.LEDGER, "a+") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_SH)
        try:
            _original_refresh()
        finally:
            fcntl.flock(lockfile, fcntl.LOCK_UN)


api._refresh = _locked_refresh
from providers import ask, jev, openrouter, spent

ROOT = Path(__file__).parent
DATA = ROOT / "data/products.jsonl"
LOG = ROOT / "results/calls.jsonl"
TAG = "onepass-e4"
MODELS = ["jev", "llama", "haiku"]
KS = (2, 4, 8, 16, 32)
HOLISTIC = "Is the mean star rating of other, unseen reviews for this product at least 4.0 out of 5?"
NUMERIC = "Give the probability as a number from 0 to 100, where 73 means 73%. Reply with the number only."
SYSTEM = (
    "Answer the question using only the review texts. "
    + NUMERIC
    + " Do not explain or reason aloud."
)


def prompt(reviews):
    return (
        "Below are customer reviews of one product. Their star ratings have been removed. "
        "Based only on the review texts, answer the question about the held-out reviews.\n\n"
        + "\n\n".join(
            f"Review {i + 1}: {r['text'][:1000]}" for i, r in enumerate(reviews)
        )
        + "\n\n"
        + HOLISTIC
        + "\n"
        + NUMERIC
    )


def question(k):
    return {
        "holistic": {"type": "noul", "instructions": HOLISTIC},
        **{
            f"r{i}": {
                "type": "noul",
                "instructions": f"Is reviewer {i + 1} satisfied, 4 to 5 stars?",
            }
            for i in range(k)
        },
    }


def parse(text):
    if not isinstance(text, str):
        return None
    match = re.fullmatch(
        r"\s*(?:\*\*)?\s*(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*%?\s*(?:\*\*)?\s*",
        text,
    )
    if not match:
        return None
    x = float(text.strip().strip("*").strip().rstrip("%").strip())
    return x / 100 if 0 <= x <= 100 else None


def category(text, parsed, stop_reason=None):
    if parsed is not None:
        return "valid"
    if not isinstance(text, str):
        return "invalid"
    if re.search(r"(?i)cannot|unable to|refuse|won.t provide|not able to", text):
        return "refusal"
    if stop_reason == "max_tokens" or re.match(
        r"(?i)\s*(let.me|first|because|the probability|based on|i think|to answer|we need)",
        text,
    ):
        return "attempted-reasoning"
    return "invalid"


def call(job):
    model, obj, k = job
    rev = obj["reviews"][:k]
    p = prompt(rev)
    start = time.monotonic()
    row = {
        "key": f"{model}:{obj['asin']}:{k}",
        "model": model,
        "model_requested": {
            "jev": "jev-latest",
            "llama": "meta-llama/llama-3.3-70b-instruct",
            "haiku": "claude-haiku-4-5-20251001",
        }[model],
        "asin": obj["asin"],
        "k": k,
        "prompt": p,
        "temperature": None if model == "jev" else 0,
        "tag": TAG,
        "spec": "v3.1",
        "provider_requested": None if model != "llama" else "Together",
        "max_tokens": None if model == "jev" else 12,
        "system": None if model == "jev" else SYSTEM,
    }
    try:
        if model == "jev":
            q = question(k)
            row["questions"] = q
            raw = jev(state=p, questions=q, tag=TAG)
            row["raw"] = raw
            row["model_returned"] = raw.get("model")
            row["usage"] = raw.get("usage")
            row["usd"] = (
                (
                    raw.get("usage", {}).get("input_tokens", 0)
                    + raw.get("usage", {}).get("output_tokens", 0)
                )
                * 0.042
                / 1e6
            )
            # Parse only explicit Noul yes probabilities; retain raw for schema audit.
            answers = raw.get("answers", raw.get("results", {}))

            def probability(x):
                if isinstance(x, dict):
                    for name in (
                        "noul",
                        "probability",
                        "p_yes",
                        "yes_probability",
                        "prob",
                        "yes",
                    ):
                        if name in x:
                            return probability(x[name])
                if (
                    isinstance(x, (int, float))
                    and not isinstance(x, bool)
                    and 0 <= x <= 1
                ):
                    return float(x)
                return None

            row["parsed"] = (
                {name: probability(answers.get(name)) for name in q}
                if isinstance(answers, dict)
                else {}
            )
            row["readout_category"] = (
                "valid"
                if len(row["parsed"]) == len(q)
                and all(v is not None for v in row["parsed"].values())
                else "invalid"
            )
        elif model == "llama":
            raw = openrouter(
                p,
                model=row["model_requested"],
                system=SYSTEM,
                max_tokens=12,
                temperature=0,
                tag=TAG,
            )
            row.update(
                raw=raw,
                parsed=parse(raw.get("text")),
                usage=raw.get("usage"),
                usd=raw.get("usd"),
                provider_returned=raw.get("provider"),
            )
            row["finish_reason"] = raw.get("usage", {}).get("_e4_finish_reason")
            row["readout_category"] = category(
                raw.get("text"), row["parsed"], row["finish_reason"]
            )
            if row["finish_reason"] == "length":
                row["readout_category"] = "invalid"
                row["parsed"] = None
            if row["parsed"] is not None and 0 < row["parsed"] < 0.01:
                row["readout_category"] = "ambiguous_scale"
                row["parsed"] = None
        else:
            raw = ask(
                p,
                model=row["model_requested"],
                system=SYSTEM,
                max_tokens=12,
                temperature=0,
                tag=TAG,
            )
            row.update(
                raw=raw,
                parsed=parse(raw.get("text")),
                usage=raw.get("usage"),
                usd=raw.get("usd"),
            )
            row["finish_reason"] = raw.get("stop_reason")
            row["readout_category"] = category(
                raw.get("text"), row["parsed"], row["finish_reason"]
            )
            if row["finish_reason"] in ("max_tokens", "length"):
                row["readout_category"] = "invalid"
                row["parsed"] = None
            if row["parsed"] is not None and 0 < row["parsed"] < 0.01:
                row["readout_category"] = "ambiguous_scale"
                row["parsed"] = None
    except Exception as exc:
        row["error"] = str(exc)
        row["readout_category"] = "invalid"
    row["latency_s"] = round(time.monotonic() - start, 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=240)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    objs = [json.loads(l) for l in DATA.open()][: 1 if args.smoke else args.limit]
    previous = (
        {
            r["key"]
            for l in LOG.open()
            if (r := json.loads(l))
            and "error" not in r
            and (
                all(v is not None for v in r.get("parsed", {}).values())
                if r.get("model") == "jev"
                else r.get("parsed") is not None
            )
        }
        if LOG.exists()
        else set()
    )
    jobs = [
        (m, o, k)
        for o in objs
        for k in KS
        for m in MODELS
        if f"{m}:{o['asin']}:{k}" not in previous
    ]
    if args.smoke:
        jobs = [j for j in jobs if j[2] == 2]
    print("jobs", len(jobs), "prior spend", spent(tag=TAG), flush=True)
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool,
        LOG.open("a") as f,
    ):
        # Batches cap overshoot if price unpredictable; errors are not skipped on retry.
        for b in range(0, len(jobs), args.workers):
            if spent(tag=TAG) >= 5.75:
                print("stopping at spend cap", flush=True)
                break
            for row in pool.map(call, jobs[b : b + args.workers]):
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                f.flush()
                if "error" in row or b == 0:
                    print(
                        row["key"],
                        "error",
                        row.get("error"),
                        "parsed",
                        str(row.get("parsed"))[:250],
                        "usd",
                        row.get("usd"),
                        flush=True,
                    )
            if b % 150 == 0:
                print(
                    "done",
                    b + args.workers,
                    "/",
                    len(jobs),
                    "spent",
                    spent(tag=TAG),
                    flush=True,
                )
    print("final spend", spent(tag=TAG), flush=True)


if __name__ == "__main__":
    main()
