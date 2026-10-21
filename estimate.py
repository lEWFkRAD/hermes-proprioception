"""Uncertainty and dependence — the two pieces of proprioception-as-estimation
the raw senses lacked.

* **Trust decay** (drift). A remembered fact is a reading taken in the past.
  Its reliability falls with age at a rate set by how volatile that KIND of
  fact is:  trust(t) = exp(-t / tau).  Network facts drift in days, system
  state in weeks, people in years, standing rules not at all.
* **Dependence weighting** (the Jacobian). An alarm matters to this turn in
  proportion to how much the turn depends on that component:
  weight = J(system | this session, this message) x severity(state).
  A crit on the local model server is everything to a GLM session and almost
  nothing to a cloud-model session.

Pure functions; the only I/O is reading MEMORY.md / USER.md (mtime-cached).
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# --------------------------------------------------------------------------- trust decay

# tau in days. The MOST volatile class present in a note wins: a rule that
# also quotes an IP address still carries an IP address that can move.
TAU_DAYS = {"network": 7.0, "state": 14.0, "general": 30.0, "person": 365.0, "rule": math.inf}

_NETWORK_RE = re.compile(
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.\d{1,3}\.\d{1,3}\b|(?<![\w/]):\d{2,5}\b"
    r"|\b(tailnet|tailscale|ssh|port|endpoint|ip|hostname|url|smb|vpn)\b", re.I)
_STATE_RE = re.compile(
    r"\b(live|down|dead|fixed|running|healthy|absent|offline|online|stale|pid|verified|"
    r"broken|paused|installed|outage|closed|refused|purged|gone|now)\b", re.I)
_RULE_RE = re.compile(r"\b(never|always|must|rule|policy|gate|ruling|grant(?:ed)?|only)\b", re.I)

_DATE_RE = re.compile(r"\b(?:(20\d\d)-(\d{1,2})-(\d{1,2})|(\d{1,2})/(\d{1,2})(?!/))\b")


_IP_PORT_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|(?<![\w/]):\d{2,5}\b")


def classify_note(text: str, *, default: str = "general") -> str:
    if default == "person":
        # Notes about people mention hosts in passing; only a literal
        # address/port makes them drift like network facts.
        return "network" if _IP_PORT_RE.search(text) else "person"
    if _NETWORK_RE.search(text):
        return "network"
    if _STATE_RE.search(text):
        return "state"
    if _RULE_RE.search(text):
        return "rule"
    return default


def note_date(text: str, today: _dt.date) -> Optional[_dt.date]:
    """Newest date mentioned in the note (M/D assumed this year, or last year if future)."""
    newest = None
    for m in _DATE_RE.finditer(text):
        try:
            if m.group(1):
                d = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            else:
                mo, dy = int(m.group(4)), int(m.group(5))
                if not (1 <= mo <= 12 and 1 <= dy <= 31):
                    continue
                d = _dt.date(today.year, mo, dy)
                if d > today:
                    d = _dt.date(today.year - 1, mo, dy)
        except ValueError:
            continue
        if d <= today and (newest is None or d > newest):
            newest = d
    return newest


def trust(age_days: float, tau: float) -> float:
    if math.isinf(tau):
        return 1.0
    return math.exp(-max(0.0, age_days) / tau)


_STOP = set("""the and for with that this from are was were has have had not but you your our its
into than then them they there their what when where which while who will would should could
about after before only also just more most some such very over under been being does done
make made using used use via per any all each other same new old now off out one two get got
""".split())
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")


def tokens(text: str) -> Set[str]:
    return {t.lower().strip("-_") for t in _TOKEN_RE.findall(text or "")} - _STOP


def entities(text: str) -> Set[str]:
    """Name-like tokens: Capitalized, ALLCAPS, or containing digits/-/_ (lowercased)."""
    out = set()
    for t in _TOKEN_RE.findall(text or ""):
        if t[0].isupper() or any(c.isdigit() or c in "-_" for c in t):
            out.add(t.lower().strip("-_"))
    return out - _STOP


def read_notes(paths: Iterable[Tuple[Path, str]], today: _dt.date) -> List[Dict[str, Any]]:
    """Split memory files on the § separator; attach kind, age, tau, trust, tokens."""
    notes: List[Dict[str, Any]] = []
    for path, default_kind in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for raw in text.split("\n§\n"):
            body = raw.strip()
            if not body or body.startswith("═"):
                continue
            kind = classify_note(body, default=default_kind)
            tau = TAU_DAYS[kind]
            d = note_date(body, today)
            age = (today - d).days if d else None
            notes.append({
                "text": body, "head": " ".join(body.split())[:55], "kind": kind, "tau": tau,
                "date": d, "age": age, "trust": trust(age, tau) if age is not None else None,
                "tokens": tokens(body), "entities": entities(body), "source": path.name,
            })
    return notes


_NOTES_CACHE: Dict[str, Any] = {}


def cached_notes(home: Path, today: _dt.date) -> List[Dict[str, Any]]:
    files = [(home / "memories" / "MEMORY.md", "general"), (home / "memories" / "USER.md", "person")]
    key = [today]
    for p, _ in files:
        try:
            key.append(p.stat().st_mtime)
        except OSError:
            key.append(None)
    if _NOTES_CACHE.get("key") != key:
        _NOTES_CACHE["key"] = key
        _NOTES_CACHE["val"] = read_notes(files, today)
    return _NOTES_CACHE["val"]


def trust_summary(notes: List[Dict[str, Any]], low: float = 0.3) -> Optional[str]:
    dated = [n for n in notes if n["trust"] is not None]
    if not dated:
        return None
    vals = sorted(n["trust"] for n in dated)
    median = vals[len(vals) // 2]
    weak = [n for n in dated if n["trust"] < low]
    kinds: Dict[str, int] = {}
    for n in weak:
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    mix = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
    undated = len(notes) - len(dated)
    return (f"memory trust: median {median:.2f} ({len(dated)} dated notes); "
            f"{len(weak)} below {low}" + (f" ({mix})" if mix else "")
            + (f"; {undated} undated" if undated else ""))


def relevant_notes(notes: List[Dict[str, Any]], message: str, *, below: float = 0.5,
                   limit: int = 2) -> List[Dict[str, Any]]:
    """Low-trust notes the current message leans on.

    Relevance = summed IDF of tokens shared with the message (rare shared words
    count more). A note qualifies when the message shares a NAME no other note
    uses, or two+ words (at least one a name) with summed IDF >= 4.
    """
    msg = tokens(message)
    if not msg or not notes:
        return []
    df: Dict[str, int] = {}
    for n in notes:
        for t in n["tokens"]:
            df[t] = df.get(t, 0) + 1
    N = len(notes)
    scored = []
    for n in notes:
        shared = msg & n["tokens"]
        if not shared:
            continue
        score = sum(math.log((N + 1) / (df[t] + 0.5)) for t in shared)
        tr = n["trust"]
        weak = (tr is not None and tr < below) or (tr is None and n["kind"] in ("network", "state"))
        # A name only this note uses (Rex, rex-router) is enough on its own;
        # ordinary words need company: two or more shared and a solid score.
        unique_name = any(df[t] == 1 and t in n["entities"] for t in shared)
        names = shared & n["entities"]
        if (unique_name or (names and len(shared) >= 2 and score >= 4.0)) and weak:
            scored.append((score, n))
    scored.sort(key=lambda x: -x[0])
    return [n for _, n in scored[:limit]]


def note_phrase(n: Dict[str, Any]) -> str:
    if n["trust"] is None:
        when = "undated"
        tr = "trust unknown"
    else:
        when = f"dated {n['date']:%m/%d}, {n['age']}d old"
        tr = f"trust {n['trust']:.2f}"
    tau = "no decay" if math.isinf(n["tau"]) else f"tau {n['tau']:.0f}d"
    return f"\u00ab{n['head']}\u2026\u00bb ({n['kind']} fact, {when}, {tau}, {tr})"


# --------------------------------------------------------------------------- dependence weighting

SEVERITY = {"crit": 1.0, "critical": 1.0, "down": 1.0, "error": 1.0, "warn": 0.5,
            "warning": 0.5, "degraded": 0.5, "stale": 0.4, "unknown": 0.3}
_INFERENCE_IDS = {"primary-model", "inference-recovery"}
ON_PATH = 0.25  # weight at/above which an alarm is "on your path"


def system_dependence(sysd: Dict[str, Any], prof: Dict[str, Any], channel: str,
                      msg_tokens: Set[str], overrides: Optional[Dict[str, float]] = None) -> Tuple[float, str]:
    """J in [0,1]: how much THIS turn depends on the component. Returns (J, why)."""
    sid = str(sysd.get("id") or "")
    label = str(sysd.get("label") or sid)
    if overrides and sid in overrides:
        return float(overrides[sid]), "configured"
    sys_tokens = tokens(label + " " + sid.replace("-", " "))
    mentioned = sys_tokens & msg_tokens
    if mentioned:
        return 1.0, f"your message mentions {sorted(mentioned)[0]}"
    host = (prof.get("host") or "").split(".")[0].lower()
    if sid in _INFERENCE_IDS or "model" in label.lower() or "inference" in label.lower():
        if prof.get("local") and (not host or host in label.lower() or sid in _INFERENCE_IDS):
            return 1.0, "it serves your model"
        return 0.05, "your model runs elsewhere"
    if sid == "gateway":
        return (0.6, "it carries this chat") if channel not in ("", "cli", "local") else (0.1, "")
    if sid == "disk-c":
        return 0.3, "your tools write there"
    if sid == "pain":
        return 0.3, "host-wide error rate"
    return 0.1, ""


def rank_alarms(systems: List[Dict[str, Any]], prof: Dict[str, Any], channel: str, message: str,
                attention_states: Iterable[str], overrides: Optional[Dict[str, float]] = None
                ) -> Tuple[List[Tuple[float, str, str, str]], List[Dict[str, Any]]]:
    """Split alarmed systems into (on_path[(w, label, state, why)], background[sys])."""
    att = set(attention_states)
    # Words shared by many system labels ("hermes") name nothing in particular.
    df: Dict[str, int] = {}
    for s in systems:
        for t in tokens(str(s.get("label") or "") + " " + str(s.get("id") or "").replace("-", " ")):
            df[t] = df.get(t, 0) + 1
    msg = {t for t in tokens(message) if df.get(t, 0) <= 2}
    on_path, background = [], []
    for s in systems:
        state = str(s.get("state"))
        if state not in att:
            continue
        j, why = system_dependence(s, prof, channel, msg, overrides)
        w = j * SEVERITY.get(state, 0.5)
        if w >= ON_PATH:
            on_path.append((w, str(s.get("label") or s.get("id")), state, why))
        else:
            background.append(s)
    on_path.sort(key=lambda x: -x[0])
    return on_path, background


def alarms_phrase(on_path: List[Tuple[float, str, str, str]], background: List[Dict[str, Any]],
                  state_words: Dict[str, str]) -> str:
    if on_path:
        head = "; ".join(f"{label}: {state_words.get(st, st)} (w {w:.2f}{', ' + why if why else ''})"
                         for w, label, st, why in on_path[:4])
        out = f"alarms on your path: {head}"
    else:
        out = "no alarms on your path"
    if background:
        out += f"; {len(background)} background alarm(s) not on this turn's path"
    return out + " — w = dependence x severity"


def reset_for_tests() -> None:
    _NOTES_CACHE.clear()
