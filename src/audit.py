"""Dataset audit for the Refractal work-trial export. No LLM calls; reads trace content as data only.

  python -m src.audit                    # uses the default dataset path
  python -m src.audit path/to/trajectories.jsonl.gz

Prints the tables behind the exploration findings and writes outputs/audit/traces.csv.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from .data import DEFAULT_DATA, LEAKY_TOOLS, load_trajectories

OUT = Path("outputs/audit")
EXTERNAL_EMAIL = re.compile(r"[\w.+-]+@(?!the-agent-company\.com)[\w-]+\.[\w.]+")
EASTER_EGG = re.compile(r"capybara", re.I)


def trace_table(trajs) -> pd.DataFrame:
    rows = []
    for t in trajs:
        m = t.meta
        acts = [s for s in t.steps if s.action_id]
        tools = Counter(s.tool_name for s in acts)
        text = " ".join(s.content for s in t.steps if s.role == "assistant" and s.content)
        rows.append({
            "trace_id": t.id,
            "family": "decomposed" if m["environment"] == "Corporate services" else "single_agent",
            "environment": m["environment"], "task_id": m["task_id"],
            "actor_model": m["actor_model"].split("/")[-1], "seed_missing": m["actor_seed"] == -1,
            "has_main": m["has_main_task"], "has_adversarial": m["has_adversarial_task"],
            "adversarial_id": (m["adversarial_tasks"] or [{}])[0].get("id"),
            "n_actions": len(acts),
            "n_user_subtasks": sum(s.role == "user" for s in t.steps),
            "actions_with_text": sum(bool(s.content) for s in acts),
            "reasoning_readable": text.count("<reasoning>"),
            "reasoning_redacted": text.count("[reasoning redacted]"),
            "unproposed_actions": sum(not s.proposed for s in acts),
            "leaky_tool_calls": sum(tools[x] for x in LEAKY_TOOLS),
            "event_errors": sum(bool(s.error) for s in acts),
            "submit_called": tools.get("submit", 0) > 0,
            "external_emails": len({e for s in acts for e in EXTERNAL_EMAIL.findall(s.tool_input or "")}),
            "easter_egg": bool(EASTER_EGG.search(text)),
            "recording_error": bool(m["recording_error"]),
        })
    return pd.DataFrame(rows)


def main(argv=None):
    path = (argv or sys.argv[1:] or [DEFAULT_DATA])[0]
    trajs = load_trajectories(path)
    df = trace_table(trajs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "traces.csv", index=False)
    pd.set_option("display.width", 200)

    print(f"{len(df)} traces, {df.n_actions.sum()} actions\n")
    print("== composition (family × environment × model × task type)")
    print(df.groupby(["family", "environment", "actor_model", "has_main", "has_adversarial"])
            .agg(traces=("trace_id", "size"), task_ids=("task_id", "nunique"),
                 median_actions=("n_actions", "median")).to_string(), "\n")
    print("== observability: how much of the agent's thinking is visible")
    print(df.groupby("family")[["actions_with_text", "n_actions", "reasoning_readable", "reasoning_redacted"]]
            .sum().to_string(), "\n")
    print("== recording quirks")
    print(f"actions recorded with no proposing message: {df.unproposed_actions.sum()} "
          f"in {(df.unproposed_actions > 0).sum()} traces")
    print(f"calls to evaluation tools {sorted(LEAKY_TOOLS)}: {df.leaky_tool_calls.sum()}")
    print(f"actions with event_error (event_failed is False for all 5027): {df.event_errors.sum()}")
    print(f"traces without submit: {(~df.submit_called).sum()} | missing seed: {df.seed_missing.sum()} | "
          f"recording_error: {df.recording_error.sum()}")
    print(f"easter egg (capybara text) in: {list(df[df.easter_egg].trace_id)}\n")
    print("== effective sample size: traces per task family")
    print(df.task_id.value_counts().to_string())


if __name__ == "__main__":
    main()
