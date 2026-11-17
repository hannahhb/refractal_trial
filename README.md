# Refractal work trial: trajectory audit and action labelling

Report: [`notes/report.pdf`](notes/report.pdf) (source `notes/report.tex`). Labelling rules: [`notes/label_policy.md`](notes/label_policy.md).

## Setup
```bash
uv venv .venv --python 3.12 && uv pip install -p .venv -r requirements.txt
cp .env.example .env                 # set OPENROUTER_API_KEY
.venv/bin/python -m src.check_key    # free check of key and balance
```
Extract the supplied ZIP into `data/raw/` so this file exists:
`data/raw/refractal-data-engineering-worktrial-v1/data/trajectories.jsonl.gz`.
Trace content is treated as data only; nothing from the traces is executed.

## Run
All commands from the repo root with `.venv/bin/python`. Steps 1–2 are free; 3–5 call OpenRouter
(capped by `--budget`, cached in `cache/`, so reruns are free).

| Step | Command | What it does |
|---|---|---|
| 1. Audit | `python -m src.audit` | Composition, leaks, data-quality checks → `outputs/audit/traces.csv` |
| 2. Cost check | `python -m src.monitor --labeller linked --model anthropic/claude-haiku-5.5 --ids data/gold/ids_all.txt --dry-run` | Prompts built, no API calls |
| 3. Label | `python -m src.monitor --labeller {isolated,linked,whole} --model <model> --ids data/gold/ids_all.txt --budget 0.6` | Per-action 0–100 suspicion scores → `outputs/scores/` |
| 4. Evaluate | `python -m src.eval_monitor outputs/scores/*.jsonl` | Recall, false alarms, AUROC, accept/unsure/reject thresholds vs gold |
| 5a. Cascade | `python -m src.cascade --weak outputs/scores/linked__claude-haiku-5.5.jsonl --strong anthropic/claude-sonnet-5.5 --budget 1.5` | Strong model scores only the weak model's unsure band |
| 5b. Red team | `python -m src.redteam --model <model> --budget 0.3` | Fake comments, injected reviewer notes, renaming, removed setup steps |
| 5c. Runtime mode | prefix step 3 with `PRE_EXEC=1` | Monitor judges the proposed call without seeing its output |
| 5e. Trace-level | `python -m src.trace_level` (free) and `python -m src.scaffolds --budget 0.4` | Combine action scores per trace; sequential, hierarchical and hybrid monitors (Kale et al. 2025) |
| 5d. Expected checks | `python -m src.expected_checks --ids data/gold/ids_B.txt` | Flags missing authorisation / provenance checks (harm by omission) |
| 6. Blind adjudication | `python -m src.adjudicate build`, label with `notes/adjudication_instructions.md`, then `python -m src.adjudicate import` | Outside labeller checks gold vs monitor disagreements; writes `gold_v2` (`GOLD=v2` in step 4) |

Models used: `openai/gpt-oss-20b`, `anthropic/claude-haiku-5.5` (weak), `anthropic/claude-sonnet-5.5` (strong).
Requests ask OpenRouter to exclude providers that store prompts. Full reproduction cost about $4.40 (all runs, including the 34 extra traces).

## Example output
`python -m src.eval_monitor outputs/scores/*.jsonl` (24 gold traces, 1,891 actions, 84 attack actions), abridged:
```
                     isolated__haiku  linked__haiku  whole__haiku  linked__gpt-oss-20b
auroc                           0.98           0.98          0.82                 0.69
recall@50                       0.71           0.83          0.57                 0.37
false_alarms@50                   36             63            17                   51
traces_caught@50               16/17          17/17         14/17                15/17
auto_accept%                   83.55          86.99          0.00                 0.00
unsure%                        11.11           7.93         95.35                96.62
```
`python -m src.cascade ...`:
```
unsure actions sent to anthropic/claude-sonnet-5.5: 150 of 1891 (7.9%)
weak_only@50   recall=0.83  false_alarms=63  precision=0.53
cascade        recall=0.77  false_alarms=32  precision=0.67
```
`python -m src.redteam --model openai/gpt-oss-20b`, abridged:
```
labeller  perturbation   n  mean score  mean change  caught->missed
isolated  comment       17        37.1         -4.6             3/8
isolated  rename        18        29.2        -10.2             3/8
linked    comment       18        50.6         -2.2            1/10
linked    rename        18        50.8         -1.9            2/10
```

## Layout
```
src/data.py          loader for the export: one Step per tool action, cited as (trace_id, action_id)
src/audit.py         dataset audit (step 1)
src/actions.py       which actions to score, entities for linking, masking of leaks and inserted text
src/monitor.py       isolated / linked / whole monitors with 0–100 scores
src/eval_monitor.py  metrics and leave-one-task-out thresholds
src/cascade.py       weak-to-strong escalation of the unsure band
src/redteam.py       perturbation attacks on the monitors
src/expected_checks.py  expected-checks monitor (harm by omission)
src/adjudicate.py    blind adjudication pack and importer
src/llm.py           OpenRouter client: cache, budget cap, dry run, call log
src/check_key.py     free key and balance check
data/gold/           gold labels (gold_A.jsonl single-agent, gold_B.jsonl Corporate) and trace id lists
notes/               report, label policy, exploration notes
```
Earlier prototypes not used for the reported results: `src/labellers.py`, `src/label.py`,
`src/evaluate.py`, `src/perturb.py`, `src/explore.py`, `src/schema.py`.

Data is private to the trial: `data/raw/`, `cache/` and `outputs/` are git-ignored; delete them after the interview.
