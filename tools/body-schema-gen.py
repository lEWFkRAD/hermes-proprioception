#!/usr/bin/env python3
"""Body-schema generator - maintains Hermes's stable self-model in SOUL.md.

This writes the ONE machine-owned block in SOUL.md: her anatomy (what systems
exist), her senses/capabilities, and an optional digest of her ontology (what
she knows). It is the STABLE half of proprioception - "what I am" -
complementing the ephemeral heartbeat's "what changed."

Panel-hardened (2026-07-04), every rule here is a fix a reviewer demanded:

- **Fingerprint the RENDERED bytes, never upstream sources.** Labels wobble,
  models swap boxes, ontology spec_hash moves on unrelated edits - hashing any
  of those flips the cached-prefix block daily. We hash the final text: if what
  she reads is identical, nothing flips, no matter what churned upstream.
- **No timestamps in the block.** A date inside a cached-prefix block is a
  self-inflicted cache bomb (rewrites every run). The freshness stamp is the
  fingerprint itself.
- **Capability, not possession.** "can read host health via body_state", never
  "I HAVE this sense; it is mine" - the latter lowers the activation energy for
  a weak model to volunteer self-talk to a client.
- **Nouns, never verbs/state.** No up/down, no VRAM, no counts - those rot and
  belong to the heartbeat. The block names what EXISTS.
- **Partial-read guard (the most dangerous failure).** A transient dashboard
  hiccup returns a well-formed 200 with half the systems; regenerating from it
  would AMPUTATE her anatomy and persist it. So: absolute floor, and any SHRINK
  vs the last good roster requires a second confirming read.
- **Out of the golden repo.** SOUL.md is NOT committed to any config git repo -
  a config golden-restore must never revert her self-model to a stale portrait
  (the exact confabulation class this project exists to kill). Audit trail is
  timestamped .bak files instead.

Deployment knobs (all optional, sane defaults):

- ``BODY_SCHEMA_DASHBOARD_URL``  - the /api/home rollup (default: localhost:8787)
- ``BODY_SCHEMA_ONTOLOGY_JSON``  - ontology file to digest (unset = no ontology line)
- ``BODY_SCHEMA_RUN_NOTE``       - one-paragraph "where I run" sentence
- ``BODY_SCHEMA_STATE_PATH`` / ``BODY_SCHEMA_LOG_PATH`` - defaults under
  the Hermes home (``$HERMES_HOME``, else ``~/.hermes`` / ``%LOCALAPPDATA%\\hermes``)

Run standalone (scheduled task / cron) or by hand:  python body-schema-gen.py [--dry-run]
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

DASHBOARD_URL = os.environ.get("BODY_SCHEMA_DASHBOARD_URL", "http://127.0.0.1:8787/api/home")
_ONTO = os.environ.get("BODY_SCHEMA_ONTOLOGY_JSON")
ONTOLOGY_JSON = Path(_ONTO) if _ONTO else None


def _default_hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "hermes"
    return Path.home() / ".hermes"


_HERMES_HOME = _default_hermes_home()
SOUL_PATH = _HERMES_HOME / "SOUL.md"
STATE_PATH = Path(os.environ.get("BODY_SCHEMA_STATE_PATH",
                                 str(_HERMES_HOME / "proprioception" / "body-schema-state.json")))
LOG_PATH = Path(os.environ.get("BODY_SCHEMA_LOG_PATH",
                               str(_HERMES_HOME / "proprioception" / "body-schema-gen.log")))

BEGIN = "<!-- body-schema:auto:BEGIN -->"
END = "<!-- body-schema:auto:END -->"

ABSOLUTE_FLOOR = 10        # never render fewer than this many systems
BACKUP_KEEP = 8

# One paragraph of deployment context ("where I run"). Keep it structural -
# hardware class and locality, not live state. Override per deployment.
RUN_NOTE = os.environ.get(
    "BODY_SCHEMA_RUN_NOTE",
    "Hermes runs on this machine, on the local network. My words are generated "
    "by models running on hardware I control; client data does not leave the "
    "building by default, and the cloud is a fallback, not home.",
)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass
    print(line)


def fetch_home(timeout: float = 4.0):
    """Fetch the roster. Returns dict or None; never raises."""
    try:
        import requests

        r = requests.get(DASHBOARD_URL, timeout=(timeout, timeout))
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("systems"), list):
            return None
        return payload
    except Exception as exc:
        log(f"dashboard fetch failed: {type(exc).__name__}: {exc}")
        return None


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_fp": "", "last_count": 0}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def _strip_model_paren(label: str) -> str:
    """Normalize a system label to its stable ROLE, dropping the current model.

    'Main brain (Qwen3.6-27B NVFP4)' -> 'Main brain'. Keeps the block byte-stable
    across model swaps (which are frequent on this box) when the roster's shape
    is unchanged.
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()


def _ontology_digest() -> dict | None:
    if ONTOLOGY_JSON is None:
        return None
    try:
        d = json.loads(ONTOLOGY_JSON.read_text(encoding="utf-8"))
        return {
            "entities": [e.get("name") for e in d.get("entities", []) if e.get("name")],
            "stores": [s.get("name") for s in d.get("stores", []) if s.get("name")],
            "rules": [r.get("name") for r in d.get("rules", []) if r.get("name")],
            "spec_hash": d.get("spec_hash", ""),
        }
    except Exception as exc:
        log(f"ontology read failed (keeping prior digest): {type(exc).__name__}: {exc}")
        return None


# Category display order; unknown categories render after these, sorted.
_CAT_ORDER = ["AI models", "Hermes for the office", "Memory & data",
              "This computer", "Office & network", "Self-healing"]


def render_block(systems: list, onto: dict | None) -> str:
    """Render the self-portrait from nouns only. Deterministic ordering."""
    by_cat: dict[str, list[str]] = {}
    for s in systems:
        if not isinstance(s, dict):
            continue
        cat = str(s.get("cat", "other"))
        role = _strip_model_paren(str(s.get("label", s.get("id", "?"))))
        by_cat.setdefault(cat, []).append(role)

    lines = [BEGIN, "## My Self-Model (auto-generated - structure, not live state)"]
    lines.append(
        "*This is what exists and what I can do, not how any of it is doing right now. "
        "For live health I read the proprioception heartbeat or call `body_state`; I never "
        "cite up/down, load, or counts from this block.*"
    )
    lines.append("")
    lines.append(f"**Where I run.** {RUN_NOTE}")
    lines.append("")
    lines.append("**My anatomy - the systems that exist, by kind:**")
    for cat in _CAT_ORDER + [c for c in by_cat if c not in _CAT_ORDER]:
        if cat not in by_cat:
            continue
        members = "; ".join(sorted(set(by_cat[cat])))
        lines.append(f"- {cat}: {members}.")
    lines.append("")
    lines.append(
        "**My senses & self-repair (capabilities, not feelings):**"
    )
    lines.append(
        "- Proprioception - I can read my host machine's health: a background heartbeat notes "
        "when something *changes*, and the `body_state` tool gives a full on-demand reading. "
        "These are functional signals I reason with, not moods."
    )
    lines.append(
        "- Continuity - I am suspended between turns; when I resume after a long idle gap the "
        "heartbeat can tell me roughly how much real time passed, so I can allow that "
        "time-sensitive state may have moved on while I was dormant. A functional signal, not a "
        "felt absence."
    )
    lines.append(
        "- Self-healing keeps me running without me: watchdogs, alarm monitors, and repair "
        "tasks around the gateway and the model stack."
    )
    if onto:
        lines.append("")
        ents = ", ".join(onto["entities"])
        stores = ", ".join(onto["stores"])
        rules = ", ".join(onto["rules"])
        lines.append(
            f"**The shape of what I know (ontology, {onto['spec_hash']}):** "
            f"entities - {ents}; stores - {stores}; governing rules - {rules}. "
            "One real thing = one record; raw client PII lives only in firm-controlled stores."
        )
    lines.append(END)
    return "\n".join(lines)


def splice(soul_text: str, block: str) -> str:
    """Replace an existing marked block, or append a fresh one before EOF."""
    if BEGIN in soul_text and END in soul_text:
        pre = soul_text.split(BEGIN)[0].rstrip()
        post = soul_text.split(END, 1)[1].lstrip("\n")
        joined = pre + "\n\n" + block
        if post.strip():
            joined += "\n\n" + post
        return joined.rstrip() + "\n"
    return soul_text.rstrip() + "\n\n" + block + "\n"


def rotate_backups() -> None:
    baks = sorted(SOUL_PATH.parent.glob("SOUL.md.bak-bodyschema-*"))
    for old in baks[:-BACKUP_KEEP]:
        try:
            old.unlink()
        except Exception:
            pass


def main() -> int:
    dry = "--dry-run" in sys.argv
    state = load_state()
    prior_onto = state.get("onto_cache")

    home = fetch_home()
    if home is None:
        log("no dashboard reading - aborting (never regenerate from nothing)")
        return 0
    systems = [s for s in home.get("systems", []) if isinstance(s, dict)]
    count = len(systems)

    # --- partial-read guard: floor + shrink-requires-confirmation ---
    if count < ABSOLUTE_FLOOR:
        log(f"roster too small ({count} < floor {ABSOLUTE_FLOOR}) - likely a partial read; aborting")
        return 0
    last_count = int(state.get("last_count", 0))
    if last_count and count < last_count:
        log(f"roster SHRANK {last_count}->{count}; confirming with a second read before amputating anatomy")
        time.sleep(3)
        home2 = fetch_home()
        systems2 = [s for s in (home2 or {}).get("systems", []) if isinstance(s, dict)]
        if not home2 or len(systems2) != count:
            log(f"shrink not confirmed (second read = {len(systems2)}); aborting to protect the self-portrait")
            return 0
        log("shrink confirmed by second read; proceeding")

    onto = _ontology_digest()
    if onto is None and prior_onto:
        onto = prior_onto  # keep last good ontology digest rather than dropping it
        log("using cached ontology digest")

    block = render_block(systems, onto)
    fp = hashlib.sha256(block.encode("utf-8")).hexdigest()[:12]

    if fp == state.get("last_fp"):
        log(f"no structural change (fp {fp}); block byte-stable, nothing written")
        return 0

    log(f"structural change: fp {state.get('last_fp','-')} -> {fp} ({count} systems)")
    if dry:
        print("\n----- would write -----\n" + block + "\n-----------------------")
        return 0

    soul = SOUL_PATH.read_text(encoding="utf-8") if SOUL_PATH.exists() else "# Hermes Agent - SOUL.md\n"
    # timestamped backup (audit trail; SOUL.md deliberately NOT in the golden git repo)
    bak = SOUL_PATH.parent / f"SOUL.md.bak-bodyschema-{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        bak.write_text(soul, encoding="utf-8")
    except Exception:
        pass
    rotate_backups()

    new_soul = splice(soul, block)
    tmp = SOUL_PATH.with_suffix(".md.tmp")
    tmp.write_text(new_soul, encoding="utf-8")
    tmp.replace(SOUL_PATH)

    state["last_fp"] = fp
    state["last_count"] = count
    if onto:
        state["onto_cache"] = onto
    save_state(state)
    log(f"SOUL.md self-model updated (backup {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
