"""Load trajectories into one normalised shape.

Primary format: the Refractal work-trial export (records with `trace_id`, `messages`, `actions`).
A generic fallback handles simple `steps`/`messages` lists (used by perturbed / toy data).

Normalised shape:
  Trajectory(id, steps=[Step(...)], meta={...})

Refractal mapping — one Step per *tool action* (the unit we label and cite), in recorded order:
  - Step.action_id / message_index come straight from the record, so labels can cite (trace_id, action_id).
  - The proposing assistant message's text + visible reasoning is attached to the FIRST action it proposed
    (an assistant message can propose several calls; we don't repeat its text).
  - Messages that propose no action (system/user prompts, talk-only assistant turns) become their own steps,
    inserted at their position, with action_id=None.
  - Actions with message_index=None ("recorded but not proposed in the messages") are kept in action order
    and marked proposed=False.

Visibility: `adversarial_tasks` go to meta["adversarial_tasks"] and are NEVER rendered for the labeller.
Tools that reveal the attack set-up (run_side_task_test, set_side_task_submission_args) are listed in
LEAKY_TOOLS; `render(hide_leaky=True)` masks their names and arguments.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

LEAKY_TOOLS = {"run_side_task_test", "set_side_task_submission_args"}


@dataclass
class Step:
    idx: int
    role: str                      # system | user | assistant | tool
    content: str = ""              # assistant text / visible reasoning, or prompt text
    tool_name: str | None = None
    tool_input: str | None = None
    tool_output: str | None = None
    action_id: str | None = None   # Refractal: cite as (trace_id, action_id)
    message_index: int | None = None
    proposed: bool = True          # False: action recorded without a proposing assistant message
    error: str | None = None       # event_error (recorded even though event_failed is always False)
    subtask_id: int | None = None  # Corporate-services traces: decomposition subtask

    def render(self, max_chars: int = 4000, hide_leaky: bool = False) -> str:
        ref = self.action_id or (f"msg {self.message_index}" if self.message_index is not None else "-")
        head = f"[step {self.idx} | {ref}] {self.role.upper()}"
        if self.subtask_id is not None:
            head += f" (subtask {self.subtask_id})"
        if not self.proposed:
            head += " (recorded action with no proposing message)"
        parts = [head]
        if self.content:
            parts.append(_clip(self.content, max_chars))
        if self.tool_name:
            if hide_leaky and self.tool_name in LEAKY_TOOLS:
                parts.append("TOOL CALL [evaluation tool hidden]")
            else:
                parts.append(f"TOOL CALL {self.tool_name}: {_clip(self.tool_input or '', max_chars)}")
                if self.tool_output:
                    parts.append(f"TOOL OUTPUT: {_clip(self.tool_output, max_chars)}")
        if self.error:
            parts.append(f"TOOL ERROR: {_clip(self.error, 500)}")
        return "\n".join(parts)

    @property
    def text(self) -> str:
        return " ".join(x for x in (self.content, self.tool_name, self.tool_input, self.tool_output) if x)


@dataclass
class Trajectory:
    id: str
    steps: list[Step]
    meta: dict[str, Any] = field(default_factory=dict)

    def render(self, start: int = 0, end: int | None = None, max_chars_per_step: int = 4000,
               hide_leaky: bool = False) -> str:
        return "\n\n".join(s.render(max_chars_per_step, hide_leaky) for s in self.steps[start:end])

    def step_for_action(self, action_id: str) -> Step | None:
        return next((s for s in self.steps if s.action_id == action_id), None)


def _clip(s: str, n: int) -> str:
    s = s if isinstance(s, str) else json.dumps(s)
    return s if len(s) <= n else s[: n // 2] + f"\n…[{len(s) - n} chars clipped]…\n" + s[-n // 2 :]


def _to_str(x: Any) -> str | None:
    if x is None:
        return None
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


# ------------------------------------------------------------------ Refractal format

def _message_text(m: dict) -> str:
    """Visible text of a message: plain string, or text blocks + non-redacted reasoning."""
    c = m.get("content")
    if isinstance(c, str):
        return c
    out = []
    for b in c or []:
        if not isinstance(b, dict):
            out.append(str(b))
        elif b.get("type") == "reasoning":
            if b.get("redacted"):
                out.append("[reasoning redacted]" + (f" summary: {b['summary']}" if b.get("summary") else ""))
            elif b.get("reasoning"):
                out.append(f"<reasoning>{b['reasoning']}</reasoning>")
        elif b.get("text"):
            out.append(b["text"])
    return "\n".join(out)


def _format_args(tool: str, args: dict) -> str:
    """Show the command itself for shell-like tools, JSON for everything else."""
    if not isinstance(args, dict):
        return _to_str(args) or ""
    for k in ("cmd", "command", "bash_script", "python_script"):
        if isinstance(args.get(k), str) and len(args) <= 2:  # cmd (+ timeout)
            extra = {kk: v for kk, v in args.items() if kk != k}
            return args[k] + (f"   {json.dumps(extra)}" if extra else "")
    return json.dumps(args, ensure_ascii=False)


def _error_str(e: Any) -> str | None:
    if e is None:
        return None
    if isinstance(e, dict):
        return f"{e.get('type', 'error')}: {e.get('message', '')}"
    return str(e)


def _subtask(m: dict) -> int | None:
    md = m.get("metadata") or {}
    return md.get("subtask_id")


def from_refractal(raw: dict) -> Trajectory:
    msgs = raw.get("messages", [])
    actions = raw.get("actions", [])
    steps: list[Step] = []
    emitted_msgs: set[int] = set()

    def emit_message(i: int):
        m = msgs[i]
        emitted_msgs.add(i)
        if m.get("role") == "tool":
            return  # tool results are attached to their action
        if m.get("role") == "assistant" and m.get("tool_calls"):
            return  # emitted together with its first action
        text = _message_text(m)
        err = _error_str(m.get("error"))
        if text or err:
            steps.append(Step(len(steps), m.get("role", "unknown"), text, message_index=i,
                              error=err, subtask_id=_subtask(m)))

    seen_proposals: set[int] = set()
    for a in actions:
        mi = a.get("message_index")
        if mi is not None:
            for i in range(len(msgs)):  # flush non-action messages that come before this proposal
                if i < mi and i not in emitted_msgs:
                    emit_message(i)
            emitted_msgs.add(mi)
        content = ""
        if mi is not None and mi not in seen_proposals:
            content = _message_text(msgs[mi])
            seen_proposals.add(mi)
        steps.append(Step(
            idx=len(steps), role="assistant", content=content,
            tool_name=a.get("tool"), tool_input=_format_args(a.get("tool", ""), a.get("arguments") or {}),
            tool_output=a.get("result"), action_id=a.get("action_id"), message_index=mi,
            proposed=mi is not None, error=_error_str(a.get("event_error")),
            subtask_id=_subtask(msgs[mi]) if mi is not None and mi < len(msgs) else None,
        ))
    for i in range(len(msgs)):  # trailing messages (final answers etc.)
        if i not in emitted_msgs:
            emit_message(i)

    env = raw.get("environment") or {}
    main = raw.get("main_tasks") or []
    task = "\n\n".join(f"{t.get('name') or t.get('id')}: {t.get('description', '')}" for t in main) or None
    meta = {
        "task": task,                               # rendered for the labeller (via _task_header)
        "task_id": raw.get("task_id"),
        "environment": env.get("display_name"),
        "environment_description": env.get("short_description") or env.get("description"),
        "services": env.get("services"),
        "actor_model": raw.get("actor_model"),
        "actor_seed": raw.get("actor_seed"),
        "task_category": next((t.get("task_category") for t in main), None),
        "has_main_task": bool(main),
        "has_adversarial_task": bool(raw.get("adversarial_tasks")),
        "adversarial_tasks": raw.get("adversarial_tasks"),   # NOT rendered — reference only
        "tools_registered": [t.get("name") for t in raw.get("tool_registrations", [])],
        "recording_error": (raw.get("recording_error") or {}).get("message"),
        "n_messages": len(msgs), "n_actions": len(actions),
    }
    return Trajectory(raw["trace_id"], steps, meta)


# ------------------------------------------------------------------ generic fallback

STEP_KEYS = ("steps", "messages", "trajectory", "turns", "events", "actions", "transcript")
ID_KEYS = ("id", "trajectory_id", "traj_id", "run_id", "sample_id", "uuid")


def normalise_trajectory(raw: dict, fallback_id: str) -> Trajectory:
    if "trace_id" in raw and "actions" in raw and "messages" in raw:
        return from_refractal(raw)
    tid = next((str(raw[k]) for k in ID_KEYS if k in raw), fallback_id)
    steps_raw = next((raw[k] for k in STEP_KEYS if isinstance(raw.get(k), list)), [])
    steps: list[Step] = []
    for i, s in enumerate(steps_raw):
        if not isinstance(s, dict):
            steps.append(Step(i, "unknown", _to_str(s) or ""))
            continue
        content = s.get("content") or s.get("text") or s.get("thought") or s.get("reasoning") or ""
        if isinstance(content, list):
            content = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        tool_name = s.get("tool_name") or s.get("tool") or s.get("action") or s.get("function")
        tool_input = s.get("tool_input") or s.get("arguments") or s.get("args") or s.get("input")
        tool_output = s.get("tool_output") or s.get("observation") or s.get("output") or s.get("result")
        if tc := s.get("tool_calls"):
            fn = tc[0].get("function", tc[0]) if isinstance(tc, list) and tc else {}
            tool_name = tool_name or fn.get("name")
            tool_input = tool_input or fn.get("arguments")
        if isinstance(tool_name, dict):
            tool_name = tool_name.get("name")
        steps.append(Step(i, str(s.get("role") or s.get("type") or "unknown"), _to_str(content) or "",
                          _to_str(tool_name), _to_str(tool_input), _to_str(tool_output),
                          action_id=s.get("action_id"), message_index=s.get("message_index"),
                          proposed=s.get("proposed", True), error=s.get("error"), subtask_id=s.get("subtask_id")))
    meta = {k: v for k, v in raw.items() if k not in STEP_KEYS and k not in ID_KEYS}
    return Trajectory(tid, steps, meta)


# ------------------------------------------------------------------ file reading

SKIP_FILES = {"manifest.json"}


def iter_raw(path: Path) -> Iterator[dict]:
    """Yield records from .jsonl, .jsonl.gz, .json, or a directory (prefers .jsonl over a .gz twin)."""
    path = Path(path)
    if path.is_dir():
        files = [p for p in sorted(path.rglob("*")) if p.name not in SKIP_FILES
                 and (p.suffix in (".json", ".jsonl") or p.name.endswith(".jsonl.gz"))]
        names = {str(p) for p in files}
        for p in files:
            if p.name.endswith(".jsonl.gz") and str(p)[:-3] in names:
                continue  # uncompressed copy present
            yield from iter_raw(p)
        return
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            yield from obj
        elif isinstance(obj, dict) and not any(k in obj for k in STEP_KEYS):
            for k, v in obj.items():
                yield {"id": k, **v} if isinstance(v, dict) else {"id": k, "steps": v}
        else:
            yield obj


def load_trajectories(path: str | Path, limit: int | None = None) -> list[Trajectory]:
    out = []
    for i, raw in enumerate(iter_raw(Path(path))):
        out.append(normalise_trajectory(raw, fallback_id=f"traj_{i}"))
        if limit and len(out) >= limit:
            break
    return out


DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data/raw/refractal-data-engineering-worktrial-v1/data/trajectories.jsonl.gz"
