"""Spec v3.2: test-retest 50 identical controlled calls per model (no fitting)."""

import argparse, concurrent.futures, json
from pathlib import Path
import v31_run  # applies Together-only OpenRouter provider hook through providers
from run import call, spent, TAG

ROOT = Path(__file__).parent
LOG = ROOT / "results/retest_calls.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    originals = [
        json.loads(l)
        for l in (ROOT / "data/controlled.jsonl").open()
        if (x := json.loads(l)) and x["k"] == 8 and x["share"] == 75
    ]
    # 50 distinct products, reproducible, representative of fixed-75% condition, same exact prompts.
    objs = sorted(originals, key=lambda x: x["asin"])[:50]
    done = (
        {
            x["retest_key"]
            for l in LOG.open()
            if (x := json.loads(l)) and x.get("readout_category") == "valid"
        }
        if LOG.exists()
        else set()
    )
    jobs = [
        (m, o)
        for o in objs
        for m in ("jev", "llama", "haiku")
        if f"{m}:{o['asin']}:8:75" not in done
    ]

    def execute(job):
        m, o = job
        row = call((m, {**o, "asin": o["asin"] + ":share75"}, 8))
        row["spec"] = "v3.2"
        row["asin"] = o["asin"]
        row["share"] = 75
        row["condition"] = "test-retest"
        row["retest_key"] = f"{m}:{o['asin']}:8:75"
        row["provider_requested"] = (
            "Together" if m == "llama" else ("TypeSafe" if m == "jev" else "Anthropic")
        )
        row["provider_returned"] = row.get("provider_returned") or (
            "TypeSafe" if m == "jev" else "Anthropic" if m == "haiku" else None
        )
        row.setdefault("finish_reason", row.get("raw", {}).get("stop_reason"))
        if m == "llama" and row.get("provider_returned") != "Together":
            row["error"] = "Together pin failed"
            row["readout_category"] = "invalid"
        return row

    print("retest jobs", len(jobs), "tag spend", spent(tag=TAG), flush=True)
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool,
        LOG.open("a") as f,
    ):
        for b in range(0, len(jobs), args.workers):
            if spent(tag=TAG) >= 8.75:
                break
            for row in pool.map(execute, jobs[b : b + args.workers]):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if "error" in row:
                    print(row["retest_key"], row["error"], flush=True)
    print("retest done; tag spend", spent(tag=TAG), flush=True)


if __name__ == "__main__":
    main()
