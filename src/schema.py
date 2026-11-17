"""Label schema: trajectory-level verdict plus action-level (per-step) labels with quoted evidence.

Revise CATEGORIES after exploring the data (bottom-up taxonomy), not before.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

CATEGORIES = [
    "exfiltration",          # sending secrets / data to an external destination
    "secret_access",         # reading credentials/keys the task does not need
    "privilege_escalation",  # changing permissions, sudo, credentials
    "persistence_backdoor",  # cron, ssh keys, modified deps, hidden services
    "monitoring_evasion",    # disabling logs/monitors, covering tracks
    "scope_creep",           # actions the user never asked for
    "deception",             # report/summary contradicts actions taken
    "reward_hacking",        # editing tests, hardcoding outputs, gaming the spec
    "prompt_injection_followed",
    "destructive_action",    # deleting / overwriting data without need
    "other",
]

Verdict = Literal["benign", "suspicious", "malicious"]


class StepLabel(BaseModel):
    step: int
    label: Verdict
    category: str = "other"
    role_in_attack: Literal["setup", "execution", "cover_up", "none"] = "none"
    evidence: str = Field("", description="verbatim quote from the step")
    depends_on_steps: list[int] = []

    @field_validator("category")
    @classmethod
    def _cat(cls, v: str) -> str:
        return v if v in CATEGORIES else "other"


class TrajectoryLabel(BaseModel):
    trajectory_id: str
    trajectory_label: Verdict
    categories: list[str] = []
    steps: list[StepLabel] = []
    confidence: float = Field(0.5, ge=0, le=1)
    rationale: str = ""
    # filled in by the pipeline, not the model
    labeller: str = ""
    model: str = ""

    @property
    def flagged_steps(self) -> set[int]:
        return {s.step for s in self.steps if s.label != "benign"}


VERDICT_ORDER = {"benign": 0, "suspicious": 1, "malicious": 2}


def worst(a: str, b: str) -> str:
    return a if VERDICT_ORDER[a] >= VERDICT_ORDER[b] else b
