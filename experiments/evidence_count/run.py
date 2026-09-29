"""Run the independent-report experiment and its controls.

Each call writes its request, response, model settings, and usage to
results/raw/<phase>.jsonl. Calls without API errors are skipped on a rerun.

Phases
  smoke      small slice for every model (parser, finish reasons, provider pinning, reasoning tokens)
  main       Overall probability from each model; Jev also answers per-report questions on listed cases
  holonly    Jev overall probability without per-report questions
  pad        Listed reports padded to the longest comparable prompt
  flip       Probability of NO instead of YES
  count      Count the positive reports
  clue       Classify one report at a time
  sonnet     Reasoning reference using Sonnet 5
  e3         One decisive report among weak reports
  unpinned   Llama and Mistral without fixed OpenRouter providers

Examples
  python3 run.py smoke
  python3 run.py main --r 0.52 --models llama mistral
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path

from generate import DOMAINS, SPEC
from parsing import parse_answer_line, parse_choice, parse_count, parse_noul, parse_prob, parse_yesno

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "api"))
import providers as cc  # noqa: E402

TAG = "onepass-e1v31"
RAW = HERE / "results" / "raw"
# Experiment spending thresholds in USD.
LIMITS = {"anthropic": 23.0, "openrouter": 11.0, "typesafe": 4.5}
WORKERS = {"typesafe": 12, "openrouter": 8, "anthropic": 8}

# Serialize ledger reads across worker threads.
_refresh_lock = threading.Lock()
_orig_refresh = cc._refresh


def _locked_refresh():
    with _refresh_lock:
        _orig_refresh()


cc._refresh = _locked_refresh

# Capture the raw response to record the OpenRouter finish reason.
_local = threading.local()
_orig_post = cc._post


def _capturing_post(url, body, headers, **kw):
    d = _orig_post(url, body, headers, **kw)
    _local.last = {"url": url, "body": body, "response": d}
    return d


cc._post = _capturing_post

# The smoke slice compared two system prompts. Asking for a single number, without
# calculations, reduced Haiku truncation from 11/28 calls to 2/28 calls. Use the
# same prompt for every non-reasoning LLM.
SYSTEM = "Respond with one number only. Do not explain or show calculations; answer in a single step."
SYSTEM_YESNO = "Respond with one word only. Do not explain or show calculations; answer in a single step."
MAX_TOKENS = 20          # every non-reasoning LLM readout (not raised for the main condition)
SONNET_MAX_TOKENS = 6000
MODELS = {
    "jev": dict(provider="typesafe", model="jev-latest"),
    "llama": dict(provider="openrouter", model="meta-llama/llama-3.3-70b-instruct", providers=["Together"]),
    "mistral": dict(provider="openrouter", model="mistralai/mistral-small-3.2-24b-instruct", providers=["DeepInfra"]),
    "gemma": dict(provider="openrouter", model="google/gemma-3-27b-it", providers=["DeepInfra"]),
    "deepseek": dict(provider="openrouter", model="deepseek/deepseek-v3.2", providers=["DeepInfra"]),
    "haiku": dict(provider="anthropic", model="claude-haiku-4-5-20251001"),
    "sonnet": dict(provider="anthropic", model="claude-sonnet-5"),
}
PRIMARY = ["jev", "llama", "mistral", "gemma", "deepseek", "haiku"]
LLMS = ["llama", "mistral", "gemma", "deepseek", "haiku"]
ENDING = "Give the probability as a number from 0 to 100, where 73 means 73%. Reply with the number only."


def code_sha():
    h = hashlib.sha256()
    for f in ("run.py", "generate.py", "parsing.py"):
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:12]


CODE_SHA = code_sha()

# ---------------------------------------------------------------- prompts

def yes_def(c):
    D = DOMAINS[c["domain"]]
    return f'the {D["item"]} is {D["pos_state"]}'


def no_def(c):
    D = DOMAINS[c["domain"]]
    return f'the {D["item"]} is {D["neg_state"]}'


def llm_prompt(c, kind, idx=None):
    D = DOMAINS[c["domain"]]
    state = c["state_padded"] if kind == "pad" else c["state"]
    if kind in ("holistic", "pad"):
        q = f"What is the probability that the hidden state is YES ({yes_def(c)}), given all the reports? {ENDING}"
    elif kind == "flip":
        q = f"What is the probability that the hidden state is NO ({no_def(c)}), given all the reports? {ENDING}"
    elif kind == "yesno":
        q = f"Is the hidden state YES ({yes_def(c)}), given all the reports? Reply with Yes or No only."
    elif kind == "count":
        q = f"How many reports say exactly '{D['pos']}'? Reply with the number only."  # v3.3 wording
    elif kind == "clue":
        q = f'Did the report from {D["reporter"]} {c["ids"][idx]} say "{D["pos"]}"? Reply with Yes or No only.'
    elif kind == "sonnet":
        q = (f"What is the probability that the hidden state is YES ({yes_def(c)}), given all the reports? "
             "Work it out as carefully as you need, then end your reply with a final line of the form "
             '"Answer: N", where N is the probability as a number from 0 to 100 (73 means 73%).')
    else:
        raise ValueError(kind)
    return state + "\n" + q


SHARE_BINS = {f"b{i}": (f"from {10 * i}% up to but not including {10 * i + 10}% of the reports" if i < 9
                         else "from 90% up to and including 100% of the reports") for i in range(10)}


def jev_request(c, kind):
    D = DOMAINS[c["domain"]]
    state = c["state_padded"] if kind == "pad" else c["state"]
    if kind == "flip":
        qs = {"holistic": {"type": "noul", "instructions": f"Is the hidden state NO ({no_def(c)})?"}}
    elif kind == "jev_count":  # v3.3 stage 2: Choice over counts 0..n plus a share-bin Choice (isolated branches)
        qs = {"count": {"type": "choice",
                        "instructions": f'How many reports say exactly "{D["pos"]}"?',
                        "criteria": {str(j): f'exactly {j} of the reports say "{D["pos"]}"' for j in range(c["n"] + 1)}},
              "sharebin": {"type": "choice",
                           "instructions": f'What share of the reports say exactly "{D["pos"]}"?',
                           "criteria": SHARE_BINS}}
    elif kind in ("choice", "choice_rev"):
        crit = {"YES": yes_def(c), "NO": no_def(c)}
        if kind == "choice_rev":
            crit = {"NO": no_def(c), "YES": yes_def(c)}
        qs = {"holistic": {"type": "choice", "instructions": "Which is the hidden state?", "criteria": crit}}
    else:
        qs = {"holistic": {"type": "noul", "instructions": f"Is the hidden state YES ({yes_def(c)})?"}}
    if kind == "holistic_clues":
        for j, i in enumerate(c["ids"]):
            qs[f"c{j:02d}"] = {"type": "noul", "instructions": f'Did the report from {D["reporter"]} {i} say "{D["pos"]}"?'}
    return state, qs

# ---------------------------------------------------------------- task selection

def tasks_for(phase, c, model, args):
    e1, e3 = c["exp"] == "e1", c["exp"] == "e3"
    listed = c.get("format") == "listed"
    sub55 = e1 and listed and c["r"] == 0.55 and c["seed_idx"] <= 2
    if phase == "smoke":
        ok = (e1 and c["seed_idx"] == 0 and c["domain"] == "food" and c["r"] == 0.52 and c["share"] in (0.25, 0.75)
              and c["n"] in (4, 64)) or (e3 and c["seed_idx"] == 0 and c["domain"] == "food" and c["n"] in (5, 65)
                                          and c["control"] == "decisive" and c["location"] == "late")
        if not ok:
            return []
        if model == "jev":
            return [("holistic_clues", None), ("choice", None)] if e1 and listed else [("holistic", None)]
        if model == "sonnet":
            return [("sonnet", None)] if e1 and listed and c["n"] == 4 else []
        out = [("holistic", None)]
        if e1 and listed and c["n"] == 4:
            out += [("count", None), ("flip", None), ("clue", 0), ("pad", None), ("yesno", None)]
        return out
    if phase == "main":
        if not e1 or model == "sonnet" or (args.r is not None and abs(c["r"] - args.r) > 1e-9):
            return []
        if model == "jev":
            return [("holistic_clues", None)] if listed else [("holistic", None)]
        return [("holistic", None)]
    if phase == "holonly":
        return [("holistic", None)] if model == "jev" and e1 and listed and (args.r is None or abs(c["r"] - args.r) < 1e-9) else []
    if phase == "pad":
        return [("pad", None)] if sub55 and c["n"] < 64 and model != "sonnet" else []
    if phase == "flip":
        return [("flip", None)] if sub55 and model != "sonnet" else []
    if phase == "count":  # v3.3 stage 2: every listed r=.52 case; v3 control: r=.55 seeds 0-2
        if not (e1 and listed and (c["r"] == 0.52 or sub55)):
            return []
        if model in LLMS:
            return [("count", None)]
        return [("jev_count", None)] if model == "jev" else []
    if phase == "clue":
        if model in LLMS and e1 and listed and c["share"] in (0.25, 0.75) and (
                (c["r"] == 0.55 and c["seed_idx"] == 0) or (c["r"] == 0.52 and c["seed_idx"] <= 1)):
            return [("clue", j) for j in range(c["n"])]
        return []
    if phase == "sonnet":
        if model == "sonnet" and e1 and listed and c["seed_idx"] == 0 and c["share"] in (0.25, 0.75):
            return [("sonnet", None)]
        return []
    if phase == "e3":
        return [("holistic", None)] if e3 and model != "sonnet" else []
    if phase == "yesno":
        return [("yesno", None)] if model in LLMS and e1 and listed and c["r"] == 0.52 and c["seed_idx"] <= 1 else []
    if phase == "jevchoice":
        if model != "jev" or not e1 or c["r"] != 0.52:
            return []
        return [("choice", None)] + ([("choice_rev", None)] if listed and c["seed_idx"] <= 1 else [])
    if phase == "retest":  # identical repeat of main-condition requests on a fixed 100-case r=.52 subset
        if not e1 or c["r"] != 0.52 or c["id"] not in _retest_ids():
            return []
        if model == "sonnet":
            return []
        if model == "jev":
            return [("holistic_clues", None)] if listed else [("holistic", None)]
        return [("holistic", None)]
    if phase == "sonnet_retest":
        if model == "sonnet" and e1 and listed and c["r"] == 0.52 and c["seed_idx"] == 0 and c["share"] in (0.25, 0.75):
            return [("sonnet", None)]
        return []
    if phase == "unpinned":
        if model in ("llama", "mistral") and e1 and listed and c["r"] == 0.55 and c["seed_idx"] == 0:
            return [("holistic", None)]
        return []
    raise ValueError(phase)

@cache
def _retest_ids():
    import random
    ids = sorted(json.loads(l)["id"] for l in open(HERE / "cases.jsonl")
                 if '"exp": "e1"' in l and '"r": 0.52' in l)
    return set(random.Random(20260927).sample(ids, 100))

# ---------------------------------------------------------------- execution

def case_meta(c):
    keep = ("id", "exp", "domain", "seed", "seed_idx", "template", "n", "k", "share", "r", "format", "truth",
            "decisive", "location", "control", "n_reports", "pad_lines")
    return {k: c[k] for k in keep if k in c}


def execute(phase, c, model, kind, idx, unpinned=False):
    cfg = MODELS[model]
    key = f"{phase}|{model}|{kind}|{c['id']}" + (f"|{idx}" if idx is not None else "")
    row = dict(key=key, spec=SPEC, tag=TAG, phase=phase, model=model, kind=kind, clue_idx=idx,
               case=case_meta(c), provider=cfg["provider"], model_requested=cfg["model"],
               code_sha=CODE_SHA, t_start=time.time())
    if idx is not None:
        row["clue_truth"] = c["signs"][idx]
    _local.last = None
    t0 = time.monotonic()
    try:
        if model == "jev":
            state, qs = jev_request(c, kind)
            row.update(state=state, questions=qs, max_tokens=None, temperature=None, system=None,
                       providers_requested=None)
            d = cc.jev(state, qs, model=cfg["model"], tag=TAG)
            row.update(raw=d, model_returned=d.get("model"), provider_returned="typesafe", usage=d.get("usage"),
                       finish_reason=None)
            if kind == "jev_count":
                probs = ((d.get("answers") or {}).get("count") or {}).get("probabilities") or {}
                bins = ((d.get("answers") or {}).get("sharebin") or {}).get("probabilities") or {}
                p = max(probs, key=probs.get) if probs else None
                cat = "valid" if p is not None and p.isdigit() else "invalid"
                row.update(count_probs=probs, sharebin_probs=bins,
                           sharebin_argmax=max(bins, key=bins.get) if bins else None)
                p = int(p) if cat == "valid" else None
            elif kind in ("choice", "choice_rev"):
                p, cat = parse_choice(d, "holistic")
            else:
                p, cat = parse_noul(d, "holistic")
            row.update(parsed=p, category=cat)
            if kind == "holistic_clues":
                cl = [parse_noul(d, f"c{j:02d}") for j in range(len(c["ids"]))]
                row["clue_parsed"] = [v for v, _ in cl]
                row["clue_categories"] = [k for _, k in cl]
        else:
            prompt = llm_prompt(c, kind, idx)
            providers = None if unpinned else cfg.get("providers")
            if model == "sonnet":
                mt, temp, system = SONNET_MAX_TOKENS, None, None
            else:
                mt, temp, system = MAX_TOKENS, 0.0, (SYSTEM_YESNO if kind in ("yesno", "clue") else SYSTEM)
            row.update(prompt=prompt, max_tokens=mt, temperature=temp, system=system, providers_requested=providers)
            if cfg["provider"] == "anthropic":
                out = cc.ask(prompt, model=cfg["model"], system=system, max_tokens=mt,
                             temperature=0.0 if temp is None else temp, tag=TAG)
                last = _local.last or {}
                resp = last.get("response") or {}
                row.update(text=out.get("text"), finish_reason=out.get("stop_reason"), usage=out.get("usage"),
                           usd=out.get("usd"), model_returned=resp.get("model"), provider_returned="anthropic",
                           raw=resp, request_body_temperature=(last.get("body") or {}).get("temperature"))
            else:
                out = cc.openrouter(prompt, model=cfg["model"], system=system, max_tokens=mt, temperature=temp,
                                    tag=TAG, providers=providers)
                last = _local.last or {}
                resp = last.get("response") or {}
                ch = (resp.get("choices") or [{}])[0]
                usage = out.get("usage") or {}
                row.update(text=out.get("text"), finish_reason=ch.get("finish_reason"),
                           native_finish_reason=ch.get("native_finish_reason"), usage=usage, usd=out.get("usd"),
                           provider_returned=out.get("provider"), model_returned=resp.get("model"),
                           reasoning_tokens=((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")),
                           reasoning_text=(ch.get("message") or {}).get("reasoning"), raw=resp,
                           request_provider=(last.get("body") or {}).get("provider"))
            if kind == "count":
                v, cat = parse_count(row["text"], c["n"], row["finish_reason"])
                row.update(parsed=v, category=cat)
            elif kind in ("yesno", "clue"):
                v, cat = parse_yesno(row["text"], row["finish_reason"])
                row.update(parsed=v, category=cat)
            elif kind == "sonnet":
                v, cat, alt = parse_answer_line(row["text"], row["finish_reason"])
                row.update(parsed=v, category=cat, alt_parsed=alt)
            else:
                v, cat, alt = parse_prob(row["text"], row["finish_reason"])
                row.update(parsed=v, category=cat, alt_parsed=alt)
    except Exception as e:  # logged, retried on the next run
        row.update(parsed=None, category="api_error", error=str(e)[:500])
    row["latency_s"] = round(time.monotonic() - t0, 3)
    return row


def done_keys(path):
    keys = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("category") != "api_error":
                keys.add(r["key"])
    return keys


def spend():
    return {p: round(cc.spent(p, TAG), 4) for p in ("anthropic", "openrouter", "typesafe")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["smoke", "main", "holonly", "pad", "flip", "count", "clue", "sonnet", "e3",
                                      "unpinned", "yesno", "jevchoice", "retest", "sonnet_retest"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--r", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None, help="override concurrency (e.g. 2 for DeepInfra 429s)")
    args = ap.parse_args()
    default = {"sonnet": ["sonnet"], "sonnet_retest": ["sonnet"], "yesno": LLMS, "jevchoice": ["jev"], "smoke": PRIMARY + ["sonnet"], "unpinned": ["llama", "mistral"],
               "count": LLMS + ["jev"], "clue": LLMS, "holonly": ["jev"]}
    models = args.models or default.get(args.phase, PRIMARY)
    cases = [json.loads(line) for line in open(HERE / "cases.jsonl")]
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{args.phase}.jsonl"
    done = done_keys(path)
    lock = threading.Lock()
    for model in models:
        prov = MODELS[model]["provider"]
        todo = []
        for c in cases:
            for kind, idx in tasks_for(args.phase, c, model, args):
                key = f"{args.phase}|{model}|{kind}|{c['id']}" + (f"|{idx}" if idx is not None else "")
                if key not in done:
                    todo.append((c, kind, idx))
        if args.limit:
            todo = todo[:args.limit]
        if not todo:
            print(f"{args.phase} {model}: nothing to do", flush=True)
            continue
        if cc.spent(prov, TAG) >= LIMITS[prov]:
            print(f"{model}: local limit reached for {prov}; skipping", flush=True)
            continue
        print(f"{args.phase} {model}: {len(todo)} calls", flush=True)
        n_done, n_err, cats = 0, 0, {}
        stop = threading.Event()

        def work(item):
            if stop.is_set():
                return None
            c, kind, idx = item
            return execute(args.phase, c, model, kind, idx, unpinned=args.phase == "unpinned")

        with open(path, "a", buffering=1) as out, ThreadPoolExecutor(args.workers or WORKERS[prov]) as pool:
            futs = [pool.submit(work, it) for it in todo]
            for f in as_completed(futs):
                row = f.result()
                if row is None:
                    continue
                with lock:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
                cats[row["category"]] = cats.get(row["category"], 0) + 1
                if row["category"] == "api_error":
                    n_err += 1
                    if "budget" in row.get("error", "") or n_err > 50:
                        stop.set()
                if n_done % 200 == 0:
                    print(f"  {model} {n_done}/{len(todo)} {cats} spent {spend()}", flush=True)
                if cc.spent(prov, TAG) >= LIMITS[prov]:
                    stop.set()
        print(f"{args.phase} {model}: {n_done} rows {cats} spent {spend()}", flush=True)
        if stop.is_set():
            print(f"  stopped early for {model} (errors {n_err})", flush=True)


if __name__ == "__main__":
    main()
