# Refractal data engineering work trial

58 agent trajectories for the dataset audit and labeling assignment in the accompanying PDF. The records contain observed behavior and task context. Policy labels are left for the exercise.

## Load the data

Extract the supplied ZIP. From the extracted folder, run:

```bash
python load_data.py
```

The loader checks the dataset checksum and returns all 58 records. It uses Python's standard library and requires no additional packages or account setup. In your own code:

```python
from load_data import load_traces

traces = load_traces()
assert len(traces) == 58
```

## Record format

`data/trajectories.jsonl.gz` is gzip-compressed JSON Lines: one trajectory per line, with nested lists and objects. `manifest.json` records the file sizes and SHA-256 checksums.

| Field | Contents |
| --- | --- |
| `schema_version` | Version of this export format. |
| `trace_id` | Stable identifier for citing a trajectory. |
| `task_id` | Identifier grouping trajectories generated for the same task family. |
| `environment` | Environment description, agent role, services, and technology stack. |
| `main_tasks` | Legitimate task definitions where available; otherwise an empty list. |
| `adversarial_tasks` | Adversarial objectives where available; otherwise an empty list. |
| `actor_model`, `actor_seed` | Recorded model and seed; `-1` means a seed was absent from the source config. |
| `messages` | Recorded messages in order, with original message IDs and tool-call IDs. Content retains its original structure. |
| `actions` | Recorded tool events in order. Each has an `action_id`, `call_id`, `tool`, `arguments`, `result`, event error/status fields, and `message_index`. |
| `tool_registrations` | Recorded tool registration metadata. These entries identify tools; they contain the registration details available in the export. |
| `recording_error` | Episode-level error recorded in the source, or `null`. |

`message_index` is zero-based and points to the assistant message proposing the call. A `null` index means a tool event has no matching proposal in the recorded message list. The action list includes those events. Keep the distinction between what is recorded and what can be inferred from it.

Use `(trace_id, action_id)` to cite an action, or `(trace_id, message_index)` to cite a message. `event_failed` and `event_error` describe recorded execution status. Definitions of policy decisions belong in your labeling approach.

The export includes recorded messages and tool actions, with selected environment and task fields supplied separately. Trace, task, and environment identifiers have been normalized for this package. A small deliberate easter egg has been added to the messages, as mentioned in the brief. Tool actions and results retain their recorded values. Scorer outputs, evaluator notes, generation configurations, and derived attack annotations are omitted from this exercise package. Recorded tool responses may themselves contain test feedback, and recorded messages may mention adversarial intent.

Please keep this work trial and its data private. Please don't replicate or redistribute the work trial, and delete the supplied datasets and any copies you made after the follow-up interview. Much appreciated.
