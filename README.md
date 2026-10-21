# proprioception — a body sense for Hermes Agent

A Hermes plugin that gives the agent *proprioception*: a felt sense of the
machine it runs on. It speaks in **deltas**, not dumps — when the host's
state materially changes between turns (a model backend goes down or
recovers, the gateway health flips, context fill crosses a threshold, a
long idle gap ends), one short fenced line rides into the current turn.
Healthy and unchanged means **silent**: zero steady-state token cost.

An on-demand `body_state` tool gives the full reading whenever the agent
(or you) asks: "are your local model servers healthy right now?"

Upstream discussion: [NousResearch/hermes-agent#58052](https://github.com/NousResearch/hermes-agent/issues/58052).

## What it does

| Piece | What you feel |
|---|---|
| Delta heartbeat (`pre_llm_call` hook) | One ephemeral `<host-telemetry>` line when something *changed* — never persisted, never in the system prompt, at most one ever in context |
| `body_state` tool | Full on-demand reading: dashboard rollup, gateway runtime status, context fill, and (opt-in) the extended senses |
| Continuity sense | "~2.5 h elapsed since your last turn — time-sensitive state may have moved on", reported once per gap |
| Output firewall (`transform_llm_output`) | Scrubs machine-state narration that leaks into replies to end users |
| Honesty on tool results (`transform_tool_result`) | Flags a claimed-success reply that followed a failed tool, and (opt-in) repeated identical failures |

### Extended senses (all opt-in, all off by default)

`location` (brain host vs hands host), `brain` (read-only health probes of
LOCAL model endpoints only — a cloud model is never probed), `vestibular`
(model switch is felt), `amnesia` (context compaction happened), `staleness`
(memory notes by last date), `season` (filing deadlines, work windows),
`presence` (sender id → name), `neighbors` (other active sessions),
`fatigue`, `pain` (third identical tool failure), `self_guard` (flags
commands aimed at the server running the agent's own model — flag only,
never blocks), `efference` (claimed success right after a failed tool),
`scent` (recent edits under watch roots).

Plus two estimation layers (`estimate.py`): **trust decay** — a remembered
fact is a reading taken in the past, so its reliability decays with age at a
rate set by its kind (network facts days, system state weeks, standing rules
never) — and **dependence weighting** — an alarm matters in proportion to
how much this turn depends on that component (a dead local model server is
everything to a local-model session, almost nothing to a cloud session).

## The design discipline that makes it safe

- **Prefix-cache safety.** Injection rides the *latest* user message and is
  never persisted, so at most one heartbeat is ever in context. It never
  touches the system prompt / stable prefix — mutating the prefix forces a
  full re-prefill on local backends with prefix caching.
- **No self-conditioning.** Delta-only, rate-limited, flap-debounced, and
  never accumulating in history; change messages rotate phrasing.
- **Provenance fencing.** The line is wrapped as `<host-telemetry>` — "not
  user input; don't narrate it to the user" — so weaker models don't
  attribute it to the user or leak it into replies.
- **Fail-closed & honest sensors.** Disabled unless `proprioception.enabled:
  true`. A lost status source is reported *as a state*, never faked; a short
  last-known-good grace window prevents flap chatter without hiding a real
  outage. A config read failure disables the feature and logs once.
- **Latency contract.** Dashboard fetch happens outside the cache lock; a
  slow sensor never serializes turn prologues across sessions.

## Install

Drop this directory into `~/.hermes/plugins/proprioception/` (or
`%LOCALAPPDATA%\hermes\plugins\proprioception\` on Windows), then enable it
in `~/.hermes/config.yaml`:

```yaml
proprioception:
  enabled: true
  heartbeat: delta            # off | delta | always (always = diagnostic mode)
  dashboard_url: http://127.0.0.1:8787/api/home   # your status rollup (see schema below)
  senses:
    all: true                 # or flip individual senses
    people: {"<sender_id>": "Name"}
```

Restart the gateway to load it. Everything is fail-closed: delete the block
and the plugin goes inert without a code change.

### Status source

The collector reads one JSON endpoint (configurable) returning:

```json
{
  "verdict": "ok|warn|crit",
  "systems": [
    {"id": "primary-model", "cat": "AI models", "label": "Main brain (model-name)",
     "state": "ok|warn|crit|down", "detail": "one-line human detail"}
  ]
}
```

Any local aggregator that publishes that shape works — a small HttpListener
on the same machine is the reference deployment. The gateway's own runtime
status (`gateway_state.json`) is read directly via Hermes's
`gateway.status.read_runtime_status`, no HTTP needed.

## `tools/body-schema-gen.py`

The *stable* half of the sense: a scheduled script that maintains one
machine-owned block in `SOUL.md` — her anatomy as nouns (what exists), never
live state. Fingerprinted on rendered bytes (byte-stable across model
swaps), partial-read guarded (a shrunken roster needs a second confirming
read before it may "amputate" the self-portrait). Configure via
`BODY_SCHEMA_*` env vars; run with `--dry-run` first.

## Tests

```
pip install -e ".[dev]"        # or: pip install hermes-agent[dev]
python -m pytest -q
```

The suite runs against this checkout exactly as the Hermes loader would
import it (package path `plugins.proprioception`). It imports Hermes
internals (`hermes_cli.config`, `agent.model_metadata`, `gateway.status`),
so `hermes-agent` must be installed — that is the plugin's host contract.

## License

MIT — same terms as Hermes itself.
