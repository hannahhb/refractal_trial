"""Thin OpenRouter client with a disk cache, cost tracking, a hard budget cap and a dry-run mode.

Every call is cached on disk by (model, messages, params) hash, so reruns are free and reproducible,
and every call is appended to outputs/llm_calls.jsonl for auditing.

Env vars:
  OPENROUTER_API_KEY   key (sk-or-v1-...)
  LLM_BUDGET_USD       hard spend cap for this process (default 2.00)
  LLM_DRY_RUN=1        never call the API; return None and log the token estimate
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import threading

_LOCK = threading.Lock()
CACHE_DIR = ROOT / "cache" / "llm"
CALL_LOG = ROOT / "outputs" / "llm_calls.jsonl"

# USD per 1M tokens (input, output), from openrouter.ai/api/v1/models on 2026-11-18.
PRICES = {
    "openai/gpt-oss-20b": (0.02, 0.09),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
    "google/gemini-3.5-flash-lite": (0.30, 2.50),
    "google/gemini-3.8-flash": (0.83, 4.12),
    "openai/gpt-6-luna": (0.10, 0.50),
    "openai/gpt-6.1-terra": (1.00, 5.00),
}
# Cheap models for iterating; strong one for final runs / judging.
CHEAP_MODEL = "anthropic/claude-haiku-5.5"
CHEAP_MODEL_2 = "google/gemini-3.5-flash-lite"   # second family, for cross-model agreement
STRONG_MODEL = "anthropic/claude-sonnet-5.5"


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token). Good enough for budgeting."""
    return max(1, len(text) // 4)


def estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    pin, pout = PRICES.get(model, (5.0, 25.0))  # unknown model: assume expensive
    return (in_tokens * pin + out_tokens * pout) / 1e6


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMClient:
    budget_usd: float = field(default_factory=lambda: float(os.getenv("LLM_BUDGET_USD", "2.0")))
    dry_run: bool = field(default_factory=lambda: os.getenv("LLM_DRY_RUN", "0") == "1")
    spent_usd: float = 0.0
    n_calls: int = 0
    n_cache_hits: int = 0

    def __post_init__(self):
        self._client = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CALL_LOG.parent.mkdir(parents=True, exist_ok=True)

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            key = os.getenv("OPENROUTER_API_KEY", "")
            if not key.startswith("sk-or-"):
                raise RuntimeError("OPENROUTER_API_KEY missing or not an OpenRouter key")
            self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        return self._client

    def _key(self, model, messages, params) -> str:
        blob = json.dumps({"m": model, "msgs": messages, "p": params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(
        self,
        messages: list[dict],
        model: str = CHEAP_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        json_mode: bool = True,
        tag: str = "",
        seed_variant: int = 0,  # bump to get a distinct cached sample (self-consistency runs)
    ) -> str | None:
        params = {"temperature": temperature, "max_tokens": max_tokens, "json": json_mode, "v": seed_variant}
        key = self._key(model, messages, params)
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            self.n_cache_hits += 1
            return json.loads(path.read_text())["content"]

        in_est = sum(estimate_tokens(m["content"]) for m in messages)
        worst_case = estimate_cost(model, in_est, max_tokens)
        if self.dry_run:
            self._log({"tag": tag, "model": model, "dry_run": True, "in_tokens_est": in_est,
                       "worst_case_usd": round(worst_case, 5)})
            return None
        with _LOCK:
            if self.spent_usd + worst_case > self.budget_usd:
                raise BudgetExceeded(
                    f"call would exceed budget: spent ${self.spent_usd:.3f} + worst case ${worst_case:.3f} "
                    f"> ${self.budget_usd:.2f} (raise LLM_BUDGET_USD to allow)"
                )
            self.spent_usd += worst_case  # reserve; corrected after the call

        # data_collection=deny: only route to providers that don't store/train on prompts (dataset is private)
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                      extra_body={"provider": {"data_collection": "deny"}})
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.time()
        resp = None
        try:
            for attempt in range(4):
                try:
                    resp = self.client.chat.completions.create(**kwargs)
                    break
                except Exception as e:  # rate limits / transient errors / unsupported json mode
                    if "response_format" in kwargs and "400" in str(e):
                        kwargs.pop("response_format")
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt * 2)
        finally:
            with _LOCK:
                self.spent_usd -= worst_case
        content = (resp.choices[0].message.content or "") if resp.choices else ""
        u = resp.usage
        p_tok = u.prompt_tokens if u else in_est
        c_tok = u.completion_tokens if u else estimate_tokens(content)
        cost = estimate_cost(model, p_tok, c_tok)
        with _LOCK:
            self.spent_usd += cost
            self.n_calls += 1
        path.write_text(json.dumps({"model": model, "content": content}))
        self._log({"tag": tag, "model": model, "in_tokens": p_tok, "out_tokens": c_tok,
                   "usd": round(cost, 6), "latency_s": round(time.time() - t0, 2), "cache_key": key})
        return content

    def _log(self, rec: dict):
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with _LOCK, CALL_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def summary(self) -> str:
        return (f"LLM calls: {self.n_calls} live, {self.n_cache_hits} cached, "
                f"spent ≈ ${self.spent_usd:.4f} of ${self.budget_usd:.2f} budget"
                + (" [DRY RUN]" if self.dry_run else ""))


def parse_json(text: str | None) -> dict | None:
    """Parse model JSON output, tolerating ```json fences and leading prose."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        s, e = t.find("{"), t.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(t[s : e + 1])
            except json.JSONDecodeError:
                return None
    return None
