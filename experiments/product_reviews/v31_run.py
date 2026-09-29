"""Clean E4 v3.1 holistic wording rerun, including Together-pinned Llama."""

import argparse, concurrent.futures, json
from pathlib import Path
from run import api, spent, TAG, call

ROOT = Path(__file__).parent
LOG = ROOT / "results/v31_calls.jsonl"
_original_post = api._post


def together_post(url, body, headers, **kwargs):
    if "openrouter.ai/api/v1/chat/completions" in url:
        body = {**body, "provider": {"order": ["Together"], "allow_fallbacks": False}}
    response = _original_post(url, body, headers, **kwargs)
    if "openrouter.ai/api/v1/chat/completions" in url:
        response.setdefault("usage", {})["_e4_finish_reason"] = response["choices"][
            0
        ].get("finish_reason")
    return response


api._post = together_post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    natural = [json.loads(l) for l in (ROOT / "data/products.jsonl").open()]
    controlled = [json.loads(l) for l in (ROOT / "data/controlled.jsonl").open()]
    jobs = [
        (kind, o, k, m)
        for kind, sets in [
            ("natural", [(o, k) for o in natural for k in (2, 4, 8, 16, 32)]),
            ("controlled", [(o, o["k"]) for o in controlled]),
        ]
        for o, k in sets
        for m in ("jev", "llama", "haiku")
    ]
    if args.smoke:
        jobs = [jobs[0], jobs[1], jobs[2], jobs[-3], jobs[-2], jobs[-1]]
    previous = (
        {
            x["v31_key"]
            for l in LOG.open()
            if (x := json.loads(l))
            and x.get("readout_category") == "valid"
            and (x["model"] != "llama" or x.get("provider_returned") == "Together")
        }
        if LOG.exists()
        else set()
    )
    jobs = [
        (kind, o, k, m)
        for kind, o, k, m in jobs
        if f"{kind}:{m}:{o['asin']}:{o.get('share', 0)}:{k}" not in previous
    ]
    print("v3.1 jobs", len(jobs), "tag spend", spent(tag=TAG), flush=True)

    def execute(job):
        kind, o, k, m = job
        item = (
            o if kind == "natural" else {**o, "asin": o["asin"] + f":share{o['share']}"}
        )
        row = call((m, item, k))
        row["spec"] = "v3.1"
        row["condition"] = kind
        row["v31_key"] = f"{kind}:{m}:{o['asin']}:{o.get('share', 0)}:{k}"
        row["asin"] = o["asin"]
        row["share"] = o.get("share")
        row["provider_requested"] = (
            "Together" if m == "llama" else ("TypeSafe" if m == "jev" else "Anthropic")
        )
        row["provider_returned"] = row.get("provider_returned") or (
            "TypeSafe" if m == "jev" else "Anthropic" if m == "haiku" else None
        )
        row["finish_reason"] = row.get(
            "finish_reason", row.get("raw", {}).get("stop_reason")
        )
        if m == "llama" and row.get("provider_returned") not in (None, "Together"):
            row["error"] = "Provider unexpectedly not Together: " + str(
                row["provider_returned"]
            )
            row["readout_category"] = "invalid"
        return row

    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool,
        LOG.open("a") as f,
    ):
        for b in range(0, len(jobs), args.workers):
            if spent(tag=TAG) >= 8.75:
                print("stop before ~ $6 extra spend", flush=True)
                break
            for row in pool.map(execute, jobs[b : b + args.workers]):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if b == 0 or "error" in row:
                    print(
                        row["v31_key"],
                        "provider",
                        row.get("provider_returned"),
                        "category",
                        row.get("readout_category"),
                        "error",
                        row.get("error"),
                        flush=True,
                    )
            if b % 600 == 0:
                print(
                    "done",
                    b + args.workers,
                    "/",
                    len(jobs),
                    "tag spend",
                    spent(tag=TAG),
                    flush=True,
                )
    print("tag spend", spent(tag=TAG), flush=True)


if __name__ == "__main__":
    main()
