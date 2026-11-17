"""CLI: label trajectories and write JSONL.

  python -m src.label data/sample/sample.jsonl --labeller heuristic
  python -m src.label data/raw/ --labeller whole --model anthropic/claude-haiku-5.5 --limit 20
  python -m src.label data/raw/ --labeller windowed --window 20 --dry-run     # no API calls, prints cost estimate
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from .data import load_trajectories
from .llm import CHEAP_MODEL, LLMClient, estimate_cost
from .labellers import build_labeller


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--labeller", default="heuristic",
                    choices=["heuristic", "whole", "whole_actions_only", "windowed"])
    ap.add_argument("--model", default=CHEAP_MODEL)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="file with one trajectory id per line (e.g. the gold set)")
    ap.add_argument("--seed-variant", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--dry-run", action="store_true", help="no API calls; estimate cost instead")
    ap.add_argument("--budget", type=float, help="USD cap for this run (overrides LLM_BUDGET_USD)")
    a = ap.parse_args(argv)

    if a.dry_run:
        os.environ["LLM_DRY_RUN"] = "1"
    llm = LLMClient()
    if a.budget is not None:
        llm.budget_usd = a.budget

    trajs = load_trajectories(a.input, limit=a.limit)
    if a.ids:
        keep = set(Path(a.ids).read_text().split())
        trajs = [t for t in trajs if t.id in keep]

    if a.labeller != "heuristic":
        chars = sum(len(t.render(max_chars_per_step=3000)) for t in trajs)
        est = estimate_cost(a.model, chars // 4 + 700 * len(trajs), 600 * len(trajs))
        print(f"{len(trajs)} trajectories, ~{chars // 4:,} input tokens → rough cost ${est:.3f} with {a.model}"
              + (" (windowed adds ledger overhead per chunk)" if a.labeller == "windowed" else ""))

    lab = build_labeller(a.labeller, llm, a.model, a.seed_variant, a.window)
    tag = a.model.split("/")[-1] if a.labeller != "heuristic" else "rules"
    out = Path(a.out or f"outputs/labels_{lab.name}_{tag}_v{a.seed_variant}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in tqdm(trajs, desc=lab.name):
            f.write(json.dumps(lab.label(t).model_dump()) + "\n")
    print(f"wrote {out}")
    print(llm.summary())


if __name__ == "__main__":
    main()
