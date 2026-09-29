"""HTTP calls to Anthropic, OpenRouter, and TypeSafe with a local usage ledger."""
import fcntl
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.jsonl")
CAPS = {  # USD estimates (provider total, per tag), not hard billing limits
    "anthropic": (150.0, 25.0),
    "openrouter": (40.0, 12.0),
    "typesafe": (25.0, 5.0),
}
CLAUDE_PRICES = {  # USD per million tokens (input, output); prefer claude-sonnet-5 (default) or claude-opus-5-5
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-opus-4-6": (5.0, 25.0),
}
JEV_PRICE = (0.042, 0.042)  # input price is published; output assumed equal (tiny either way)
WORKERS = {"anthropic": 8, "openrouter": 8, "typesafe": 16}
_keys = {}


def _key(provider):
    if provider not in _keys:
        name = {"anthropic": "ANTHROPIC_API_KEY", "typesafe": "TYPESAFE_API_KEY",
                "openrouter": "OPENROUTER_API_KEY"}[provider]
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Set {name} before calling {provider}")
        _keys[provider] = value
    return _keys[provider]


_ledger_pos = 0
_totals = {}  # (provider, tag) -> usd, built incrementally from new ledger lines


def _refresh():
    global _ledger_pos
    if not os.path.exists(LEDGER):
        return
    with open(LEDGER) as f:
        f.seek(_ledger_pos)
        for line in f:
            if not line.endswith("\n"):
                break  # partial line being written; read it next time
            row = json.loads(line)
            key = (row.get("provider", "anthropic"), row["tag"])
            _totals[key] = _totals.get(key, 0.0) + row["usd"]
            _ledger_pos += len(line.encode())


def spent(provider=None, tag=None):
    _refresh()
    return sum(v for (p, t), v in _totals.items()
               if (provider is None or p == provider) and (tag is None or t == tag))


def _check_budget(provider, tag):
    total_cap, tag_cap = CAPS[provider]
    if spent(provider) >= total_cap:
        raise RuntimeError(f"{provider} pilot budget ${total_cap} reached")
    if spent(provider, tag) >= tag_cap:
        raise RuntimeError(f"{provider} budget ${tag_cap} for tag {tag} reached")


def _record(row):
    with open(LEDGER, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def _post(url, body, headers, retries=4, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**headers, "content-type": "application/json"})
    for attempt in range(retries + 1):
        try:
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as e:
            if e.code in (408, 429, 500, 502, 503, 529) and attempt < retries:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries:
                time.sleep(2 ** attempt * 2)
                continue
            raise


NO_TEMPERATURE = {"claude-sonnet-5", "claude-opus-5-5"}  # these models reject the temperature parameter


def ask(prompt=None, model="claude-sonnet-5", system=None, max_tokens=256,
        temperature=0.0, tag="untagged", messages=None):
    if model not in CLAUDE_PRICES:
        raise ValueError(f"model {model} not allowed; choose from {sorted(CLAUDE_PRICES)}")
    _check_budget("anthropic", tag)
    body = {"model": model, "max_tokens": max_tokens,
            "messages": messages or [{"role": "user", "content": prompt}]}
    if model not in NO_TEMPERATURE:
        body["temperature"] = temperature
    if system:
        body["system"] = system
    d = _post("https://api.anthropic.com/v1/messages", body,
              {"x-api-key": _key("anthropic"), "anthropic-version": "2023-06-01"})
    usage = d["usage"]
    pin, pout = CLAUDE_PRICES[model]
    usd = (usage["input_tokens"] * pin + usage["output_tokens"] * pout) / 1e6
    _record({"t": time.time(), "provider": "anthropic", "tag": tag, "model": model,
             "in": usage["input_tokens"], "out": usage["output_tokens"], "usd": usd})
    text = "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")
    return {"text": text, "usage": usage, "usd": usd, "stop_reason": d.get("stop_reason")}


def jev(state, questions, model="jev-latest", tag="untagged"):
    _check_budget("typesafe", tag)
    d = _post("https://api.typesafe.ai/v1/systemone", {"state": state, "model": model, "questions": questions},
              {"Authorization": "Bearer " + _key("typesafe")}, timeout=60)
    usage = d.get("usage", {})
    usd = (usage.get("input_tokens", 0) * JEV_PRICE[0] + usage.get("output_tokens", 0) * JEV_PRICE[1]) / 1e6
    _record({"t": time.time(), "provider": "typesafe", "tag": tag, "model": d.get("model", model),
             "in": usage.get("input_tokens", 0), "out": usage.get("output_tokens", 0), "usd": usd})
    return d


def openrouter(prompt=None, model="meta-llama/llama-3.1-8b-instruct", system=None, max_tokens=256,
               temperature=0.0, logprobs=False, top_logprobs=5, tag="untagged", messages=None,
               providers=None):
    """providers: optional list of OpenRouter provider names to pin (no fallbacks), e.g. ["DeepInfra"]."""
    _check_budget("openrouter", tag)
    msgs = messages or ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature,
            "usage": {"include": True}}
    if any(m in model for m in ("qwen3", "deepseek-r1", "thinking", "gpt-oss")):
        body["reasoning"] = {"enabled": False}
    if logprobs:
        # route only to providers that honor logprobs
        body.update({"logprobs": True, "top_logprobs": top_logprobs, "provider": {"require_parameters": True}})
    if providers:
        body.setdefault("provider", {}).update({"order": list(providers), "allow_fallbacks": False})
    d = _post("https://openrouter.ai/api/v1/chat/completions", body,
              {"Authorization": "Bearer " + _key("openrouter")})
    usage = d.get("usage", {})
    usd = float(usage.get("cost") or 0.0)
    _record({"t": time.time(), "provider": "openrouter", "tag": tag, "model": model,
             "in": usage.get("prompt_tokens", 0), "out": usage.get("completion_tokens", 0), "usd": usd})
    c = d["choices"][0]
    return {"text": c["message"].get("content"), "logprobs": (c.get("logprobs") or {}).get("content"),
            "usage": usage, "usd": usd, "provider": d.get("provider")}


def _many(fn, provider, requests, workers, **fixed):
    def run(r):
        try:
            return fn(**fixed, **r)
        except Exception as e:  # keep partial results; the caller sees the error per item
            return {"error": str(e)}
    with ThreadPoolExecutor(min(workers or WORKERS[provider], WORKERS[provider])) as pool:
        return list(pool.map(run, requests))


def ask_many(requests, model="claude-sonnet-5", tag="untagged", workers=None):
    return _many(ask, "anthropic", requests, workers, model=model, tag=tag)


def jev_many(requests, model="jev-latest", tag="untagged", workers=None):
    return _many(jev, "typesafe", requests, workers, model=model, tag=tag)


def openrouter_many(requests, model="meta-llama/llama-3.1-8b-instruct", tag="untagged", workers=None):
    return _many(openrouter, "openrouter", requests, workers, model=model, tag=tag)
