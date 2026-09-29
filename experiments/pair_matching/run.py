"""Run the pair-matching task. Responses go to results/raw/<cond>.jsonl.

Conditions
  jev_holonly  control: holistic Noul alone in the call (latency/token cost; branch independence)
  jev        TypeSafe jev-latest. ONE call per item: holistic Noul + 12 per-row Nouls + 24 per-part
             Nouls, all isolated branches over the same state.
  llama70b   meta-llama/llama-3.3-70b-instruct via OpenRouter, temperature 0 (non-reasoning model)
  mistral24b mistralai/mistral-small-3.2-24b-instruct via OpenRouter, temperature 0
  haiku      claude-haiku-4-5-20251001, temperature 0, no reasoning
  sonnet5    claude-sonnet-5 (always reasons; default temperature); reasoning reference only
LLM tasks (one isolated call each)
  prob       holistic question, stated probability 0..100 (integer only)
  yn         holistic question, Yes/No
  pieces     24 per-part and 12 per-row Yes/No questions, each a separate call with the same state
  rows/parts only the 12 per-row or only the 24 per-part questions (same keys as pieces)

Examples
  python run.py --cond jev --subset smoke
  python run.py --cond haiku --tasks prob,yn --vocab names
  python run.py --cond haiku --tasks pieces --vocab names --m 8,16 --classes pos,hard
  python run.py --cond sonnet5 --tasks prob --vocab names --subset third
Re-running the same command skips calls already logged without an API error.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "api"))
import providers as cc  # noqa: E402

from generate import HOLISTIC_Q, part_question, row_question  # noqa: E402

# Serialize ledger reads across worker threads.
_refresh_lock = threading.Lock()
_orig_refresh = cc._refresh


def _locked_refresh():
    with _refresh_lock:
        _orig_refresh()


cc._refresh = _locked_refresh

TAG = "onepass-e2"
SPEC = "v3.1"  # written into every new log line
# Experiment spending thresholds in USD.
LIMITS = {"anthropic": 20.0, "openrouter": 10.0, "typesafe": 3.0}
CONDS = {
    "jev": {"provider": "typesafe", "model": "jev-latest"},
    # control: the holistic Noul alone (no per-piece branches), to measure the token/latency cost of the
    # extra branches and to check that adding branches does not change the holistic answer
    "jev_holonly": {"provider": "typesafe", "model": "jev-latest"},
    "jev_choice": {"provider": "typesafe", "model": "jev-latest"},
    "llama70b": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct", "max_tokens": 10},
    "mistral24b": {"provider": "openrouter", "model": "mistralai/mistral-small-3.2-24b-instruct", "max_tokens": 10},
    # Provider-pinned runs (no fallbacks) for the open-weight models.
    "llama70b_pin": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct", "max_tokens": 10,
                     "providers": ["Together"]},
    "mistral24b_pin": {"provider": "openrouter", "model": "mistralai/mistral-small-3.2-24b-instruct",
                       "max_tokens": 10, "providers": ["DeepInfra"]},  # v3: DeepInfra (fallback Venice)
    "gemma27b": {"provider": "openrouter", "model": "google/gemma-3-27b-it", "max_tokens": 10,
                 "providers": ["DeepInfra"]},
    "deepseekv32": {"provider": "openrouter", "model": "deepseek/deepseek-v3.2", "max_tokens": 10,
                    "providers": ["DeepInfra"]},
    "haiku": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "max_tokens": 10, "temperature": 0.0},
    # Sonnet 5 rejects temperature (default sampling). "direct" uses the same system prompt as the other
    # LLMs; "reason" is the reasoning reference: no system prompt, explicit step-by-step working allowed,
    # final line "Final answer: <integer>".
    "sonnet5_direct": {"provider": "anthropic", "model": "claude-sonnet-5", "max_tokens": 4000},
    "sonnet5_reason": {"provider": "anthropic", "model": "claude-sonnet-5", "max_tokens": 6000, "reason": True},  # >= 4000 per v3.1
}
WORKERS = {"typesafe": 12, "openrouter": 8, "anthropic": 8}

PROB_SUFFIX = ("Give the probability, from 0 to 100, that the answer is Yes. "
               "Reply with only an integer from 0 to 100 and nothing else.")
YN_SUFFIX = "Reply with only Yes or No."
# v2 (after smoke v1, where most LLM probability answers began with step-by-step text and were
# truncated): a system prompt that asks for an immediate answer. Same text for every LLM condition.
SYSTEM = ("Answer immediately with only the requested answer. Do not explain, do not show any "
          "working, and do not write anything else.")
# v3.1: final line "Answer: N"
REASON_SUFFIX = ("Give the probability, from 0 to 100, that the answer is Yes. Work through the problem "
                 "step by step, then end your reply with a final line of the form 'Answer: N', where N "
                 "is a number from 0 to 100.")
FINAL_RE = re.compile(r"(?:Final )?Answer:\s*\**\s*(\d+(?:\.\d+)?)\s*%?\s*\**\s*\.?\s*$", re.IGNORECASE)
# v3 parser: a single number (integer, decimal or scientific notation), optional percent sign
PROB_RE = re.compile(r"^\s*\**\s*(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|\.\d+)\s*%?\s*\**\s*\.?\s*$")  # "**76.9**" ok
REFUSAL_RE = re.compile(r"\b(I can(?:no|')t|I am unable|I'm unable|I'm not able|I won't|sorry)\b", re.IGNORECASE)
YN_RE = re.compile(r"^\s*(yes|no)\s*\.?\s*$", re.IGNORECASE)


def parse_prob(text):
    if text is None:
        return None
    mt = PROB_RE.match(text)
    if not mt:
        return None
    v = float(mt.group(1))
    return v if 0 <= v <= 100 else None


def categorize(text, parsed, stop_reason=None):
    """v3 readout categories: valid, refusal, attempted-reasoning, invalid."""
    if parsed is not None:
        return "valid"
    t = (text or "").strip()
    if REFUSAL_RE.search(t):
        return "refusal"
    if stop_reason in ("max_tokens", "length", "length(inferred)") or len(t.split()) >= 4:
        return "attempted-reasoning"
    return "invalid"


def parse_yn(text):
    if text is None:
        return None
    mt = YN_RE.match(text)
    return None if not mt else int(mt.group(1).lower() == "yes")


def parse_final(text):
    if text is None:
        return None
    mt = FINAL_RE.search(text.strip())
    if not mt:
        return None
    v = int(mt.group(1))
    return v if 0 <= v <= 100 else None


def llm_prompt(state, question, kind):
    if kind == "reason":
        return f"{state}\n\nQuestion: {question}\n{REASON_SUFFIX}"
    return f"{state}\n\nQuestion: {question}\n{PROB_SUFFIX if kind == 'prob' else YN_SUFFIX}"


CHOICE_CRITERIA = {  # v3.2 readout robustness: Jev two-option Choice (API field "criteria": {label: description})
    "contains": "The proposed team roster contains at least one forbidden pair (both members of one listed pair).",
    "does_not_contain": "The proposed team roster does not contain any forbidden pair.",
}


def jev_questions(it):
    q = {"holistic": {"type": "noul", "instructions": it.get("question", HOLISTIC_Q)}}
    for r in it.get("rows", []):
        q[r["key"]] = {"type": "noul", "instructions": row_question(r["x"], r["y"])}
    for p in it.get("parts", []):
        q[p["key"]] = {"type": "noul", "instructions": part_question(p["x"])}
    return q


def select(items, a):
    vocabs, ms, classes = a.vocab.split(","), [int(x) for x in a.m.split(",")], a.classes.split(",")
    out = []
    for it in items:
        if it["vocab"] not in vocabs or it["m"] not in ms or it["cls"] not in classes:
            continue
        j = int(it["base_id"].rsplit("-", 1)[1])
        if a.subset == "smoke" and not (it["seed"] == 0 and (j < 2 if it["cls"] != "easy" else j < 1)):
            continue
        if a.subset == "third" and j % 3 != 0:  # same j across pos/hard keeps a matched
            continue
        # 20 pos + 20 matched hard per m (j = 0, 3, ..., 18 in each seed, minus seed 2 j = 18); no easy items
        if a.subset == "sonnet40" and (it["cls"] == "easy" or j % 3 != 0 or (it["seed"] == 2 and j == 18)):
            continue
        # 40 pos + 40 matched hard per m (same selection as the paraphrase subset)
        if a.subset == "para40" and (it["cls"] == "easy" or j % 3 == 2 or (it["seed"] == 2 and j >= 18)):
            continue
        # test-retest: 25 pos + 25 matched hard per m (global index g = 20*seed + j, g % 12 < 5)
        if a.subset == "rt100" and (it["cls"] == "easy" or (20 * it["seed"] + j) % 12 >= 5):
            continue
        out.append(it)
    return out[: a.limit] if a.limit else out


def jobs_for(it, cond, tasks):
    """Yield (key, task, payload)."""
    if cond == "jev":
        yield f"jev|{it['item_id']}|full", "full", {"state": it["state"], "questions": jev_questions(it)}
        return
    if cond == "jev_holonly":
        yield (f"jev_holonly|{it['item_id']}|full", "full",
               {"state": it["state"], "questions": {"holistic": jev_questions(it)["holistic"]}})
        return
    if cond == "jev_choice":
        yield (f"jev_choice|{it['item_id']}|full", "full",
               {"state": it["state"], "questions": {"choice": {"type": "choice", "instructions": it.get("question", HOLISTIC_Q),
                                                               "criteria": CHOICE_CRITERIA}}})
        return
    for t in tasks:
        if t in ("prob", "yn"):
            kind = "reason" if (t == "prob" and CONDS[cond].get("reason")) else t
            yield (f"{cond}|{it['item_id']}|{t}", t,
                   {"prompt": llm_prompt(it["state"], it.get("question", HOLISTIC_Q), kind), "kind": kind})
        if t in ("pieces", "rows"):
            for r in it["rows"]:
                yield (f"{cond}|{it['item_id']}|row:{r['key']}", f"row:{r['key']}",
                       {"prompt": llm_prompt(it["state"], row_question(r["x"], r["y"]), "yn"), "kind": "yn"})
        if t in ("pieces", "parts"):
            for p in it["parts"]:
                yield (f"{cond}|{it['item_id']}|part:{p['key']}", f"part:{p['key']}",
                       {"prompt": llm_prompt(it["state"], part_question(p["x"]), "yn"), "kind": "yn"})


def call(cond, payload):
    c = CONDS[cond]
    t0 = time.time()
    if cond in ("jev", "jev_holonly", "jev_choice"):
        d = cc.jev(state=payload["state"], questions=payload["questions"], model=c["model"], tag=TAG)
        lat = time.time() - t0
        ans = d.get("answers", {})
        if cond == "jev_choice":  # score = P("contains")
            ans = {k: {"noul": (v.get("probabilities") or {}).get("contains")} for k, v in ans.items()}
        parsed = {k: (v.get("noul") if isinstance(v, dict) else None) for k, v in ans.items()}
        valid = all(isinstance(parsed.get(k), (int, float)) and 0 <= parsed[k] <= 1 for k in payload["questions"])
        return {"raw": d, "parsed": parsed, "valid": valid, "usage": d.get("usage"),
                "model_version": d.get("model"), "provider": "typesafe", "latency": lat,
                "usd": sum(d.get("usage", {}).get(k, 0) for k in ("input_tokens", "output_tokens")) * cc.JEV_PRICE[0] / 1e6}
    if c["provider"] == "openrouter":
        d = cc.openrouter(prompt=payload["prompt"], model=c["model"], max_tokens=c["max_tokens"],
                          temperature=0.0, system=SYSTEM, tag=TAG, providers=c.get("providers"))
        lat = time.time() - t0
        text = d.get("text")
        # providers.openrouter does not return finish_reason; infer truncation from the token count
        ctoks = (d.get("usage") or {}).get("completion_tokens") or 0
        meta = {"provider": d.get("provider"), "model_version": c["model"], "usage": d.get("usage"),
                "usd": d.get("usd"),
                "stop_reason": "length(inferred)" if ctoks >= c["max_tokens"] else "stop(inferred)"}
    else:
        kw = {"prompt": payload["prompt"], "model": c["model"], "max_tokens": c["max_tokens"], "tag": TAG}
        if not c.get("reason"):
            kw["system"] = SYSTEM
        if "temperature" in c:
            kw["temperature"] = c["temperature"]
        d = cc.ask(**kw)
        lat = time.time() - t0
        text = d.get("text")
        meta = {"provider": "anthropic", "model_version": c["model"], "usage": d.get("usage"),
                "usd": d.get("usd"), "stop_reason": d.get("stop_reason")}
    parsed = {"prob": parse_prob, "yn": parse_yn, "reason": parse_final}[payload["kind"]](text)
    if meta.get("stop_reason") in ("max_tokens", "length", "length(inferred)"):
        parsed = None  # v3.1: truncated replies are invalid
    meta["ambiguous_scale"] = bool(payload["kind"] in ("prob", "reason") and parsed is not None and 0 < parsed < 1)
    return {"raw": text, "parsed": parsed, "valid": parsed is not None,
            "category": categorize(text, parsed, meta.get("stop_reason")), "latency": lat, **meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=sorted(CONDS))
    ap.add_argument("--tasks", default="prob,yn")
    ap.add_argument("--vocab", default="names,codes")
    ap.add_argument("--m", default="4,8,12,16")
    ap.add_argument("--classes", default="pos,hard,easy")
    ap.add_argument("--subset", default="all", choices=["all", "third", "smoke", "sonnet40", "para40", "rt100"])
    ap.add_argument("--items", default="items.jsonl", help="items file (items_para.jsonl for the paraphrase subset)")
    ap.add_argument("--log-suffix", default="", help="e.g. _para or _rt1: separate log results/raw/<cond><suffix>.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--repair-invalid-via", default=None,
                    help="re-call keys of --cond whose logged output is invalid, using this condition's config "
                         "(e.g. a pinned provider); rows are appended to --cond's log with repair_via set")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(os.path.join(HERE, a.items))]
    sel = select(items, a)
    out_dir = os.path.join(HERE, "results", "raw")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{a.cond}{a.log_suffix}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("error"):
                done.add(r["key"])
    call_cond = a.repair_invalid_via or a.cond
    if a.repair_invalid_via:
        latest = {}
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("error"):
                latest[r["key"]] = r
        bad = {k for k, r in latest.items() if not r.get("valid")}
        jobs = [(it, k, t, p) for it in sel for k, t, p in jobs_for(it, a.cond, a.tasks.split(",")) if k in bad]
    else:
        jobs = [(it, k, t, p) for it in sel for k, t, p in jobs_for(it, a.cond, a.tasks.split(",")) if k not in done]
    prov = CONDS[call_cond]["provider"]
    print(f"{a.cond}: {len(sel)} items, {len(jobs)} calls to do ({len(done)} already logged); "
          f"spent so far on {prov}/{TAG}: ${cc.spent(prov, TAG):.4f}", flush=True)
    if a.dry or not jobs:
        return
    lock = threading.Lock()
    stop = threading.Event()
    n_done = [0]
    t_start = time.time()

    def work(job):
        it, key, task, payload = job
        if stop.is_set():
            return
        if cc.spent(prov, TAG) >= LIMITS[prov]:
            stop.set()
            print(f"STOP: own limit ${LIMITS[prov]} reached on {prov}", flush=True)
            return
        rec = {"spec": SPEC, "key": key, "cond": a.cond, "item_id": it["item_id"], "task": task, "t": time.time(),
               "prompt": payload.get("prompt") or {"state": payload["state"], "questions": payload["questions"]},
               "items_file": a.items, "log_suffix": a.log_suffix,
               "system": None if (a.cond.startswith("jev") or CONDS[a.cond].get("reason")) else SYSTEM}
        cfg = CONDS[call_cond]
        rec["request"] = {"model": cfg["model"], "provider_family": cfg["provider"],
                          "providers_requested": cfg.get("providers"), "max_tokens": cfg.get("max_tokens"),
                          "temperature": (None if cfg["model"] == "claude-sonnet-5" else 0.0)
                          if cfg["provider"] != "typesafe" else None,
                          "system": rec["system"], "spec": SPEC}
        if call_cond != a.cond:
            rec["repair_via"] = call_cond
        try:
            rec.update(call(call_cond, payload))
            rec["error"] = None
        except Exception as e:  # logged; retried on the next run
            rec["error"] = repr(e)[:500]
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_done[0] += 1
            if n_done[0] % 200 == 0:
                el = time.time() - t_start
                print(f"  {n_done[0]}/{len(jobs)} in {el:.0f}s; ${cc.spent(prov, TAG):.4f}", flush=True)

    with ThreadPoolExecutor(WORKERS[prov]) as pool:
        for f in as_completed([pool.submit(work, j) for j in jobs]):
            f.result()
    recs = [json.loads(l) for l in open(path)]
    errs = sum(1 for r in recs if r.get("error"))
    inval = sum(1 for r in recs if not r.get("error") and not r.get("valid"))
    print(f"done {a.cond}: log has {len(recs)} lines, {errs} API errors, {inval} invalid outputs; "
          f"spent ${cc.spent(prov, TAG):.4f} on {prov}/{TAG}", flush=True)


if __name__ == "__main__":
    main()
