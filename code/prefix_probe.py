"""The one rerun flip of a greedy open-weight labeler (Qwen2.5-3B, Hate, 2026-11-07), reproduced.

The canonical copy of the first Hate call ran right after the availability probe of main_runs.py ("Say OK"), so the
llama.cpp server reused only 5 cached prompt tokens and evaluated the remaining 491; its rerun ran after a Hate call
and reused 66. The same prompt, evaluated in a different split, gives different greedy answers to three of its ten
tuples. This script sends the call after the probe and after a Hate call, twice each, without the response cache.
Usage: LOCAL_LLM_URL=http://<host>:8080/v1 python prefix_probe.py"""
import os, sys
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot

URL = os.environ["LOCAL_LLM_URL"].rstrip("/") + "/chat/completions"
NAME = "qwen2.5-3b-instruct-q4_k_m"


def ask(order, k):
    spec, df = pilot.TASKS["hate"], pilot.load("hate")
    tids = pilot.order_of(df, order, 500)[10 * k:10 * k + 10]
    body = {"model": NAME, "messages": pilot.prompt_batch(spec, [df.text[i] for i in tids]), "temperature": 0.0,
            "max_tokens": 800}
    return requests.post(URL, json=body, timeout=600).json()["choices"][0]["message"]["content"]


def probe():
    requests.post(URL, json={"model": NAME, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5}, timeout=120)


if __name__ == "__main__":
    for prev in ["probe", "hate call", "probe", "hate call"]:
        probe() if prev == "probe" else ask("perm1", 49)
        lines = ask("shuffled", 0).split("\n")
        print(f"after {prev:9s}: {lines[0]} | {lines[1]} | {lines[7]}", flush=True)
