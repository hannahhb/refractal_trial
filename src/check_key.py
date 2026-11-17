"""Check OpenRouter keys in .env using the FREE /key endpoint (no model call, no spend).

  python -m src.check_key
"""
import json
import os
import urllib.request

from .llm import ROOT  # noqa: F401  (loads .env)


def check(var: str) -> str:
    key = os.getenv(var, "")
    if not key:
        return "not set"
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=15))["data"]
        rem = d.get("limit_remaining")
        return f"VALID  (key cap remaining ${rem:.2f}; spent ${d['usage']:.2f})"
    except urllib.error.HTTPError as e:
        return f"REJECTED ({e.code}: {json.load(e).get('error', {}).get('message', '')})"


def credits() -> str:
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                 headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=15))["data"]
        return f"account balance ${d['total_credits'] - d['total_usage']:.2f}"
    except Exception:
        return "account balance: unavailable (OPENROUTER_API_KEY not valid)"


if __name__ == "__main__":
    for v in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        print(f"{v}: {check(v)}")
    print(credits())
