Shipped as a standalone plugin: **https://github.com/lEWFkRAD/hermes-proprioception**

This is the implementation of what this issue describes — an ephemeral, delta-only body sense:

- **`pre_llm_call` heartbeat** — when the host's state *materially changes* between turns (model backends up/down, gateway health flips, context fill crosses a threshold, a long idle gap ends), one short fenced `<host-telemetry>` line rides into the current turn. Unchanged and healthy = silent. Never persisted, never in the system prompt, at most one ever in context — the prefix cache is never touched.
- **`body_state` tool** — full on-demand reading: status rollup, gateway runtime status, context fill, plus opt-in extended senses (location, brain-endpoint health, model-switch vestibular, compaction amnesia, memory staleness, presence, neighbors, fatigue/pain).
- **`transform_llm_output` / `transform_tool_result`** — an output firewall that scrubs machine-state narration leaking into user replies, and honesty flags on claimed-success-after-failed-tool.
- **Estimation layers** — trust decay (a remembered fact is a reading taken in the past; reliability decays at a rate set by its kind) and dependence weighting (an alarm matters in proportion to how much the current turn depends on that component).

Built to the catalog admission rules: public surfaces only (hooks + `ctx.register_tool`, no core patching), no self-updating code, fail-closed (`proprioception.enabled: true` or it's inert), no outbound telemetry, disclosures in the README. `hermes plugins validate` passes clean on a fresh clone at the pinned commit (security scan: safe); 96 tests run against the checkout exactly as the loader imports it.

`requires_hermes: ">=0.21.7"` — the three hooks this uses all exist in v0.21.7.

Ready to file the one-file `plugin-catalog/proprioception.yaml` PR (entry drafted in the repo, pinned to `e370bfb838699501810511e2e03cfa953c71e01e`) — given the open-PR cap, happy for a maintainer to pull the entry from a sweep instead. Running in production on my own box since June; it's the sense that lets me say "my model backend flapped twice this morning" without being told.
