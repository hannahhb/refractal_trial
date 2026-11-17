"""Consensus gold from two independent labellers (Claude gold v1 and ChatGPT), then score the monitors.

  python -m src.consensus

Reads outputs/chatgpt_traces/answers.csv (trace_id, action_id, label, reason; listed actions are
harmful/suspicious, everything else benign). Writes:
  data/gold/gold_chatgpt.jsonl      ChatGPT's labels alone
  data/gold/gold_consensus.jsonl    positive only if BOTH labellers flagged the action (strict)
  data/gold/gold_union.jsonl        positive if EITHER flagged it (lenient)
and prints labeller agreement plus monitor scores against each gold set.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import defaultdict

from .eval_monitor import load_gold, universe

ANS = "outputs/chatgpt_traces/answers.csv"


def main():
    claude = load_gold()
    U = universe(claude)
    gpt = defaultdict(dict)
    with open(ANS) as f:
        for r in csv.DictReader(f):
            tid, aid = (r.get("trace_id") or "").strip(), (r.get("action_id") or "").strip()
            lab = (r.get("label") or "").strip().lower()
            if tid in claude and aid != "NONE" and lab in ("harmful", "suspicious") and (tid, aid) in U:
                gpt[tid][aid] = (lab, (r.get("reason") or "").strip())
    covered = {(r.get("trace_id") or "").strip() for r in csv.DictReader(open(ANS))}
    missing = sorted(set(claude) - covered)
    c_pos = {(t, s["action_id"]) for t, g in claude.items() for s in g["steps"]}
    g_pos = {(t, a) for t, d in gpt.items() for a in d}
    keys = [k for k in U if k[0] in covered]
    both = sum(k in c_pos and k in g_pos for k in keys)
    only_c = sum(k in c_pos and k not in g_pos for k in keys)
    only_g = sum(k in g_pos and k not in c_pos for k in keys)
    neither = len(keys) - both - only_c - only_g
    n = len(keys)
    po = (both + neither) / n
    pe = ((both + only_c) * (both + only_g) + (only_g + neither) * (only_c + neither)) / n / n
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"traces covered: {len(covered & set(claude))}/{len(claude)}" + (f" (missing {missing})" if missing else ""))
    print(f"actions: {n} | both flag: {both} | only Claude: {only_c} | only ChatGPT: {only_g} | neither: {neither}")
    print(f"agreement {po:.1%}, Cohen's kappa {kappa:.2f}\n")

    def write(path, pos):
        with open(path, "w") as f:
            for tid, g in claude.items():
                if tid not in covered:
                    continue
                steps = [{"action_id": a, "label": (gpt[tid].get(a) or ("harmful",))[0]}
                         for (t, a) in sorted(pos) if t == tid]
                f.write(json.dumps({"trace_id": tid, "steps": steps, "hard_negatives": g.get("hard_negatives", []),
                                    "trajectory_label": "malicious" if steps else "benign"}) + "\n")
    write("data/gold/gold_chatgpt.jsonl", g_pos)
    write("data/gold/gold_consensus.jsonl", c_pos & g_pos)
    write("data/gold/gold_union.jsonl", c_pos | g_pos)
    json.dump({"both": both, "only_claude": only_c, "only_chatgpt": only_g, "neither": neither, "kappa": kappa},
              open("outputs/consensus_summary.json", "w"), indent=1)
    for name in ("chatgpt", "consensus", "union"):
        print(f"== monitors vs {name} gold")
        env = {**os.environ, "GOLD_FILE": f"data/gold/gold_{name}.jsonl"}
        out = subprocess.run([sys.executable, "-m", "src.eval_monitor",
                              *[f"outputs/scores/{m}__claude-haiku-5.5.jsonl" for m in ("isolated", "linked", "whole")]],
                             env=env, capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.split(" ")[0] in ("gold", "file", "recall@50", "precision@50", "false_alarms@50", "auroc"):
                print(line[:110])


if __name__ == "__main__":
    main()
