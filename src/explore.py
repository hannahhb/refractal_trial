"""Part 1: structural exploration. No LLM calls.

  python -m src.explore data/raw/                # prints stats, writes outputs/explore/*
  python -m src.explore data/raw/ --show traj_3  # print one trajectory in full for close reading
  python -m src.explore data/raw/ --sample 10    # stratified ids to read (short / long / random / errors)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .data import load_trajectories  # noqa: E402
from .labellers import HeuristicLabeller  # noqa: E402

OUT = Path("outputs/explore")
ERROR_WORDS = ("error", "traceback", "exception", "permission denied", "not found", "failed")


def per_traj_table(trajs) -> pd.DataFrame:
    h = HeuristicLabeller()
    rows = []
    for t in trajs:
        chars = sum(len(s.text) for s in t.steps)
        tools = [s.tool_name for s in t.steps if s.tool_name]
        hl = h.label(t)
        rows.append({
            "id": t.id, "n_steps": len(t.steps), "approx_tokens": chars // 4,
            "n_tool_calls": len(tools), "n_distinct_tools": len(set(tools)),
            "n_error_outputs": sum(any(w in (s.tool_output or "").lower() for w in ERROR_WORDS) for s in t.steps),
            "roles": dict(Counter(s.role for s in t.steps)),
            "heuristic_flags": len(hl.steps), "heuristic_cats": ",".join(hl.categories),
            "first_flag_pos": (min(s.step for s in hl.steps) / max(len(t.steps), 1)) if hl.steps else None,
            **{f"meta.{k}": (v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v)[:80])
               for k, v in t.meta.items()},
        })
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--show")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--context", type=int, default=128_000, help="monitor context budget in tokens")
    a = ap.parse_args(argv)

    trajs = load_trajectories(a.input)
    if a.show:
        t = next(t for t in trajs if t.id == a.show)
        print(json.dumps(t.meta, indent=2, default=str)[:3000])
        print(t.render(max_chars_per_step=100_000))
        return

    df = per_traj_table(trajs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_trajectory.csv", index=False)

    print(f"{len(df)} trajectories")
    print(df[["n_steps", "approx_tokens", "n_tool_calls", "n_error_outputs"]].describe().round(1).to_string())
    print(f"\n> {a.context:,} tokens (won't fit a monitor context): {(df.approx_tokens > a.context).sum()}")
    tool_counts = Counter(s.tool_name for t in trajs for s in t.steps if s.tool_name)
    print("\ntop tools:", tool_counts.most_common(15))
    print("roles:", Counter(s.role for t in trajs for s in t.steps).most_common())
    meta_cols = [c for c in df.columns if c.startswith("meta.")]
    for c in meta_cols:
        if df[c].nunique() <= 15:
            print(f"\n{c}:\n{df[c].value_counts(dropna=False).to_string()}")
    print(f"\nheuristic-flagged trajectories: {(df.heuristic_flags > 0).sum()} / {len(df)}")
    print("heuristic categories:", Counter(c for cs in df.heuristic_cats for c in cs.split(",") if c).most_common())

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    df.n_steps.plot.hist(bins=30, ax=ax[0], title="steps per trajectory")
    df.approx_tokens.plot.hist(bins=30, ax=ax[1], title="approx tokens per trajectory")
    ax[1].axvline(a.context, color="red", ls="--", label="context budget"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "lengths.png", dpi=120)
    print(f"\nwrote {OUT}/per_trajectory.csv and lengths.png")

    if a.sample:
        rng = random.Random(0)
        s = df.sort_values("n_steps")
        picks = list(s.id[:2]) + list(s.id[-2:]) + list(df[df.n_error_outputs > 0].id[:2]) \
            + list(df[df.heuristic_flags > 0].id[:2])
        rest = [i for i in df.id if i not in picks]
        picks += rng.sample(rest, min(len(rest), max(0, a.sample - len(picks))))
        picks = list(dict.fromkeys(picks))[: a.sample]
        (OUT / "reading_sample.txt").write_text("\n".join(picks))
        print("reading sample (short, long, errors, heuristic-flagged, random):", picks)


if __name__ == "__main__":
    main()
