"""Load the work-trial dataset and check its supplied file hashes."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_traces():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        if name.startswith("data/"):
            data = (ROOT / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"Checksum mismatch: {name}")
    with gzip.open(ROOT / "data/trajectories.jsonl.gz", "rt", encoding="utf-8") as f:
        traces = [json.loads(line) for line in f]
    if len(traces) != manifest["trajectory_count"]:
        raise ValueError("Trajectory count differs from manifest")
    return traces


if __name__ == "__main__":
    traces = load_traces()
    print(f"Loaded {len(traces)} trajectories")
    print(f"Recorded tool events: {sum(len(trace['actions']) for trace in traces)}")
    print(f"First trace: {traces[0]['trace_id']}")
