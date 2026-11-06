"""Cached OpenAI-compatible client for the EAB pilot.

Differences from the original: (1) the cache key includes a repeat index `rep`, so repeated runs of the same
request are real, separate API calls; (2) vendor refusals (HTTP 400 content filter) are cached and returned
as results instead of raised; (3) every call logs latency, tokens and list-price dollars; (4) a per-model daily
spend guard stops the run before the shared cap is reached. Keys are read from LLM_KEYS_JSON.
"""
import hashlib, json, os, time, threading, datetime
import requests

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Keys live in a JSON file outside the repository (see README); point LLM_KEYS_JSON at it. Never commit keys.
_KEYS_PATH = os.path.expanduser(os.environ.get("LLM_KEYS_JSON", "~/.config/llm_keys.json"))
KEYS = json.load(open(_KEYS_PATH)) if os.path.exists(_KEYS_PATH) else {}  # analyses run without keys
CACHE_DIR = os.path.join(HERE, "cache")
LOG = os.path.join(CACHE_DIR, "usage.log")
os.makedirs(CACHE_DIR, exist_ok=True)

# name -> (key entry, vendor model id); same physical endpoints as code/label.py
MODELS = {"ds-flash": ("deepseek", "deepseek-flash"), "ds-pro": ("deepseek", "deepseek-v4-pro"),
          "glm-flash": ("glm", "glm-4.5-flash"), "glm-air": ("glm2", "glm-4.5-air"),
          "kimi-k2.6": ("kimi", "kimi-k2.6"), "kimi-k3": ("kimi", "kimi-k3"),
          # same request as glm-air through the extra key; Zhipu serves it as glm-5.3-flash (see "served")
          "glm-air-k3": ("glm3", "glm-4.5-air"),
          "glm-5.3": ("glmplan", "glm-5.3"), "glm-5.3-flash": ("glmplan", "glm-5.3-flash"),
          # international labelers through the company gateway (a router, unlike the first-party endpoints above)
          "gpt-6-sol": ("gateway", "gpt-6-sol"), "claude-opus-5.5": ("gateway", "claude-opus-5-5"),
          # open-weight, deterministic: Qwen2.5-3B-Instruct (Q4_K_M GGUF) with greedy decoding, no request batching
          "qwen2.5-3b": ("local", "qwen2.5-3b-instruct-q4_k_m")}
OFF = {"thinking": {"type": "disabled"}}
FIXED = {"ds-flash": OFF, "ds-pro": OFF, "glm-flash": OFF, "glm-air": OFF, "glm-air-k3": OFF,
         "kimi-k2.6": {**OFF, "temperature": 0.6}, "kimi-k3": {**OFF, "temperature": 0.6}}
# USD per 1M tokens (input, output), identical to code/label.py (list prices, 2026-11-03)
PRICES = {"glm-flash": (0.0, 0.0), "glm-air": (0.20, 1.10), "ds-flash": (0.30, 1.20), "ds-pro": (1.32, 3.96),
          "kimi-k2.6": (0.95, 4.00), "kimi-k3": (3.00, 15.00)}
PRICES["glm-air-k3"] = PRICES["glm-air"]  # placeholder: the served model's own price is not verified
# Coding Plan calls have no per-token bill; usd stays 0 and tokens are logged so that list prices can be applied later
PRICES.update({"glm-5.3": (0.0, 0.0), "glm-5.3-flash": (0.0, 0.0)})
FIXED.update({"glm-5.3": {"reasoning": {"effort": "minimal"}}, "glm-5.3-flash": {"reasoning": {"effort": "minimal"}}})
# gateway calls are billed to the company, not per token to us: usd stays 0 and tokens are logged
PRICES.update({"gpt-6-sol": (0.0, 0.0), "claude-opus-5.5": (0.0, 0.0), "qwen2.5-3b": (0.0, 0.0)})
FIXED["claude-opus-5.5"] = {"temperature": None}  # "temperature is deprecated for this model" on part of the gateway
# cache entries older than CACHE_AFTER (unix time) are ignored and re-fetched, e.g. to keep pilot-era answers out of
# the main runs (the main runs use CACHE_AFTER = the time the protocol was fixed)
CACHE_AFTER = float(os.environ.get("CACHE_AFTER", 0))
DAILY_CAP = float(os.environ.get("DAILY_CAP", 15.0))  # stay below the $20/model/day share
_lock = threading.Lock()
_spent = {}


def cost(model, usage):
    pi, po = PRICES[model]
    return (usage.get("prompt_tokens", 0) * pi + usage.get("completion_tokens", 0) * po) / 1e6


def _load_spent():
    today = datetime.date.today().isoformat()
    if os.path.exists(LOG):
        for l in open(LOG):
            d = json.loads(l)
            if d.get("day") == today:
                _spent[d["m"]] = _spent.get(d["m"], 0) + d.get("usd", 0)


_load_spent()


def request(model, messages, temperature=0.0, max_tokens=512, rep=0):
    """Request body and cache key (SHA-256 of provider, body and repetition index) of a call."""
    prov, name = MODELS[model]
    kw = {k: v for k, v in FIXED.get(model, {}).items() if k != "temperature"}
    temperature = FIXED.get(model, {}).get("temperature", temperature)
    responses = KEYS.get(prov, {}).get("api") == "responses"
    if responses:  # OpenAI Responses API: messages go in `input`, the token limit is max_output_tokens
        body = {"model": name, "input": messages, "temperature": temperature, "max_output_tokens": max_tokens, **kw}
    else:
        body = {"model": name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kw}
    if temperature is None:  # the model rejects the parameter; its default sampling is used
        body.pop("temperature")
    return prov, body, responses, hashlib.sha256(json.dumps([prov, body, rep], sort_keys=True).encode()).hexdigest()


def chat(model, messages, temperature=0.0, max_tokens=512, rep=0, retries=8):
    """Returns dict(text, usage, usd, latency, status) where status is 'ok' or 'refused'."""
    prov, body, responses, key = request(model, messages, temperature, max_tokens, rep)
    path = os.path.join(CACHE_DIR, key[:2], key + ".json")
    if os.path.exists(path) and os.path.getmtime(path) >= CACHE_AFTER:
        return json.load(open(path))
    if _spent.get(model, 0) > DAILY_CAP:
        raise RuntimeError(f"daily cap reached for {model}")
    cfg, err = KEYS[prov], None
    for i in range(retries):
        try:
            t0 = time.time()
            r = requests.post(cfg["base_url"].rstrip("/") + ("/responses" if responses else "/chat/completions"),
                              headers={"Authorization": "Bearer " + cfg["api_key"]}, json=body, timeout=180)
            lat = time.time() - t0
            if r.status_code == 200 and responses:
                d = r.json()
                u = d.get("usage") or {}
                u = {"prompt_tokens": u.get("input_tokens", 0), "completion_tokens": u.get("output_tokens", 0),
                     "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
                     "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0)}
                text = "".join(c.get("text", "") for o in d.get("output", []) if o.get("type") == "message"
                               for c in o.get("content", []))
                out = {"text": text, "usage": u, "usd": cost(model, u), "latency": lat, "status": "ok",
                       "finish": (d.get("incomplete_details") or {}).get("reason") or d.get("status"), "served": d.get("model")}
            elif r.status_code == 200 and "choices" not in r.text[:2000] and '"error"' in r.text[:2000]:
                # DeepSeek under overload keeps the connection open up to 900 s and then answers 200 with an error body
                err = f"200-with-error {r.text.strip()[:200]}"
                time.sleep(min(60, 2 ** i))
                continue
            elif r.status_code == 200:
                d = r.json()
                u = d.get("usage", {})
                out = {"text": d["choices"][0]["message"]["content"] or "", "usage": u, "usd": cost(model, u),
                       "latency": lat, "status": "ok", "finish": d["choices"][0].get("finish_reason"),
                       "served": d.get("model")}
            elif r.status_code == 400 and ("contentFilter" in r.text or "1301" in r.text or "cyber_policy" in r.text or "risk" in r.text.lower()):
                out = {"text": "", "usage": {}, "usd": 0.0, "latency": lat, "status": "refused", "err": r.text[:300]}
            else:
                err = f"{r.status_code} {r.text[:200]}"
                time.sleep(min(60, 2 ** i) + (5 if r.status_code == 429 else 0))
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with _lock:
                json.dump(out, open(path, "w"))
                _spent[model] = _spent.get(model, 0) + out["usd"]
                with open(LOG, "a") as f:
                    f.write(json.dumps({"day": datetime.date.today().isoformat(), "t": time.time(), "m": model,
                                        "usd": out["usd"], "lat": lat, "status": out["status"],
                                        "served": out.get("served"), **out["usage"]}) + "\n")
            return out
        except requests.RequestException as e:
            err = repr(e)
            time.sleep(min(60, 2 ** i))
    raise RuntimeError(err)
