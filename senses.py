"""Extended senses: where the agent is, who is near, what time it is in the
world it serves, and which of its own readings have gone dark.

Every sense is opt-in under ``proprioception.senses`` (all off by default).
Two halves, kept apart on purpose:

* ``observe(...)`` — does the I/O (config, state.db, brain endpoint probes,
  memory file). Cached; never raises; runs OUTSIDE the per-session lock.
* ``diff(...)`` — pure: compares an observation with the per-session sense
  state and returns event lines. Only CHANGES speak (delta mode), except the
  first reading of a session, which carries the location line.

Perception, not policy: lines report state. Nothing here tells the model to
act.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import platform as _platform
import re
import sqlite3
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from plugins.proprioception import estimate

logger = logging.getLogger(__name__)

SENSE_KEYS = (
    "location",     # brain/hands split, model, provider, cwd, profile, channel
    "brain",        # live probe of a LOCAL model endpoint: reach, rtt, ceiling, slowdown
    "vestibular",   # model/provider switch mid-conversation
    "amnesia",      # context compaction happened; fallback summary = lossy
    "staleness",    # memory notes by age; dark (stale) readings
    "season",       # firm deadline season + operator work windows
    "presence",     # who is talking, on which channel
    "neighbors",    # other active sessions (crowd / contention on the same brain)
    "fatigue",      # tool-call / api-call count thresholds this session
    "pain",         # same tool error repeating (in-loop, via tool-result hook)
    "self_guard",   # flag commands that hit the server running this agent's own model
    "efference",    # last reply claimed success right after a failed tool call
    "scent",        # recent file edits in watched roots (body_state only)
    "trust",        # memory notes decay: trust = exp(-age/tau) by kind; flags low-trust notes this turn leans on
    "relevance",    # alarms weighted by this turn's dependence on them (w = J x severity)
    "tensor",       # passive recorder: each reading -> vector s_t in tensor.db (changes nothing the agent sees)
)

DEFAULT_WINDOWS = [
    {"name": "quiet hours", "start": "01:00", "end": "05:00"},
    {"name": "morning peak", "start": "09:00", "end": "11:00"},
    {"name": "admin dip", "start": "15:00", "end": "17:00"},
    {"name": "evening peak", "start": "21:00", "end": "23:00"},
]

# US federal filing dates (MM-DD). Weekend dates roll to Monday; federal
# holidays are NOT modeled — "approx" in the line says so.
DEFAULT_DEADLINES = [
    {"date": "01-15", "label": "Q4 estimates"},
    {"date": "01-31", "label": "W-2/1099 filing"},
    {"date": "03-15", "label": "partnership/S-corp returns"},
    {"date": "04-15", "label": "individual/C-corp returns"},
    {"date": "06-15", "label": "Q2 estimates"},
    {"date": "09-15", "label": "extended partnership/S-corp"},
    {"date": "10-15", "label": "extended individual returns"},
]

_COMPACTION_MARKER = "[CONTEXT COMPACTION"
# Most recent turn context seen by the heartbeat — lets the body_state tool
# (whose handler gets no model/platform kwargs) read the live brain.
LAST_SEEN: Dict[str, str] = {}
_FATIGUE_STEPS = (100, 250, 500, 1000)
_PAIN_REPEAT = 3

_SUCCESS_WORDS = re.compile(
    r"\b(done|fixed|verified|works|working now|succeeded|passing|all green|shipped|complete[d]?)\b",
    re.I,
)
_SELF_HARM_RE = re.compile(
    r"(systemctl\s+(?:--user\s+)?(?:restart|stop|kill|reload)\b"
    r"|docker\s+(?:restart|stop|kill|rm)\b"
    r"|\bpkill\b|\bkillall\b|\bkill\s+-9\b|\breboot\b|shutdown\s+-r"
    r"|/la\?on=)",
    re.I,
)


# --------------------------------------------------------------------------- settings

def sense_settings(raw: Any) -> Dict[str, Any]:
    """Sanitize ``proprioception.senses``. Unknown/malformed -> all off."""
    out: Dict[str, Any] = {k: False for k in SENSE_KEYS}
    out.update({
        "people": {}, "windows": list(DEFAULT_WINDOWS), "deadlines": list(DEFAULT_DEADLINES),
        "deadline_horizon_days": 21, "stale_days": 14, "watch_roots": [],
        "scent_minutes": 15, "format": "auto", "brain_timeout_seconds": 1.0,
        "dependence": {}, "tensor_max_rows": 500_000, "tensor_path": "",
    })
    if not isinstance(raw, dict):
        return out
    if raw.get("all") is True:
        for k in SENSE_KEYS:
            out[k] = True
    for k in SENSE_KEYS:
        if k in raw and raw[k] is not None:
            out[k] = bool(raw[k])
    if isinstance(raw.get("dependence"), dict):
        dep = {}
        for k, v in raw["dependence"].items():
            try:
                dep[str(k)] = min(1.0, max(0.0, float(v)))
            except (TypeError, ValueError):
                pass
        out["dependence"] = dep
    if isinstance(raw.get("people"), dict):
        out["people"] = {str(k): str(v) for k, v in raw["people"].items()}
    for key in ("windows", "deadlines", "watch_roots"):
        if isinstance(raw.get(key), list):
            out[key] = list(raw[key])
    for key, floor in (("deadline_horizon_days", 0), ("stale_days", 1), ("scent_minutes", 1),
                       ("brain_timeout_seconds", 0.2)):
        try:
            out[key] = max(floor, float(raw.get(key, out[key])))
        except (TypeError, ValueError):
            pass
    try:
        out["tensor_max_rows"] = max(1000, int(raw.get("tensor_max_rows", out["tensor_max_rows"])))
    except (TypeError, ValueError):
        pass
    if isinstance(raw.get("tensor_path"), str):
        out["tensor_path"] = raw["tensor_path"]
    fmt = str(raw.get("format", "auto")).lower()
    out["format"] = fmt if fmt in ("auto", "prose", "terse") else "auto"
    return out


def any_enabled(ss: Dict[str, Any]) -> bool:
    return any(ss.get(k) for k in SENSE_KEYS)


# --------------------------------------------------------------------------- helpers

def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _is_local_host(host: str) -> bool:
    host = (host or "").lower()
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".ts.net") or host.endswith(".local"):
        return True
    if re.match(r"^(10|127)\.", host) or re.match(r"^192\.168\.", host):
        return True
    m = re.match(r"^(100|172)\.(\d+)\.", host)
    if m:
        a, b = m.group(1), int(m.group(2))
        return (a == "100" and 64 <= b <= 127) or (a == "172" and 16 <= b <= 31)
    return False


def _short_host(host: str) -> str:
    return (host or "?").split(".")[0]


def _fmt_k(n: Optional[int]) -> str:
    if not n:
        return "?"
    return f"{n // 1024}k" if n % 1024 == 0 else f"{n / 1000:.0f}k"


# --------------------------------------------------------------------------- model profile

def resolve_model_profile(model: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map the turn's model name to provider/base_url/host and local-vs-cloud."""
    cfg = cfg if cfg is not None else _load_config()
    model = str(model or "")
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    mcfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    prof: Dict[str, Any] = {"model": model or "?", "provider": "", "base_url": "",
                            "context_length": None}
    hit = None
    for name, block in providers.items():
        if isinstance(block, dict) and model and model in (block.get("models") or []):
            hit = (name, block)
            break
    if hit:
        prof["provider"] = hit[0]
        prof["base_url"] = str(hit[1].get("base_url") or "")
        prof["context_length"] = hit[1].get("context_length")
    elif model and model == mcfg.get("default"):
        prof["provider"] = str(mcfg.get("provider") or "")
        prof["base_url"] = str(mcfg.get("base_url") or "")
        prof["context_length"] = mcfg.get("context_length")
    elif "/" in model:
        prof["provider"] = model.split("/", 1)[0]
    u = urlparse(prof["base_url"]) if prof["base_url"] else None
    prof["host"] = (u.hostname or "") if u else ""
    prof["port"] = (u.port or (443 if u.scheme == "https" else 80)) if u else None
    prof["local"] = bool(prof["host"]) and _is_local_host(prof["host"])
    prof["root"] = f"{u.scheme}://{u.netloc}" if (u and prof["local"]) else ""
    return prof


def model_key(prof: Dict[str, Any]) -> str:
    where = f"{_short_host(prof['host'])}:{prof['port']}" if prof.get("local") else (prof.get("provider") or "cloud")
    return f"{prof.get('model')}@{where}"


# --------------------------------------------------------------------------- brain probe

_BRAIN_LOCK = threading.Lock()
_BRAIN_CACHE: Dict[str, Dict[str, Any]] = {}
_BRAIN_TTL = 45.0
_INFO_TTL = 600.0


def _get_json(url: str, timeout: float) -> Tuple[Optional[Any], float, str]:
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
            body = r.read(1 << 20)
        dt = (time.monotonic() - t0) * 1000.0
        try:
            return json.loads(body.decode("utf-8", "replace")), dt, ""
        except ValueError:
            return None, dt, "non-json"
    except Exception as exc:  # urllib raises many shapes
        return None, (time.monotonic() - t0) * 1000.0, type(exc).__name__


def probe_brain(prof: Dict[str, Any], timeout: float = 1.0, *, force: bool = False) -> Optional[Dict[str, Any]]:
    """Cheap read-only GETs against a LOCAL model server. Cloud -> None."""
    root = prof.get("root")
    if not root:
        return None
    now = time.monotonic()
    with _BRAIN_LOCK:
        cached = _BRAIN_CACHE.get(root)
        if cached and not force and now - cached["_at"] < _BRAIN_TTL:
            return cached
    health, rtt, err = _get_json(root + "/health", timeout)
    if health is None and err not in ("non-json", "HTTPError"):
        # /health missing is fine (HTTPError); a connect failure is not.
        res = {"_at": now, "reachable": False, "error": err, "rtt_ms": None}
        with _BRAIN_LOCK:
            prev = _BRAIN_CACHE.get(root) or {}
            res["info"] = prev.get("info")
            _BRAIN_CACHE[root] = res
        return res
    res: Dict[str, Any] = {"_at": now, "reachable": True, "error": "", "rtt_ms": round(rtt)}
    with _BRAIN_LOCK:
        prev = dict(_BRAIN_CACHE.get(root) or {})
    info = prev.get("info")
    if not info or now - prev.get("_info_at", 0) > _INFO_TTL:
        si, _, _ = _get_json(root + "/server_info", timeout)
        if isinstance(si, dict):
            # exllamav3 GLM server nests the knobs under server_args.
            info = {**(si.get("server_args") if isinstance(si.get("server_args"), dict) else {}),
                    **{k: v for k, v in si.items() if k != "server_args"}}
        else:
            info = info or {}
        res["_info_at"] = now
    else:
        res["_info_at"] = prev.get("_info_at", now)
    res["info"] = info
    st, _, _ = _get_json(root + "/stats", timeout)
    if isinstance(st, dict):
        la = st.get("k_lookahead") or {}
        res["stats"] = {"requests": st.get("requests"), "steps": la.get("steps"),
                        "wait_ms": la.get("wait_ms"), "fallback": la.get("fallback"),
                        "wait_ms_per_step": la.get("wait_ms_per_step")}
        old = prev.get("stats") or {}
        try:
            dsteps = res["stats"]["steps"] - old["steps"]
            if dsteps >= 200:
                res["interval_ms_per_step"] = (res["stats"]["wait_ms"] - old["wait_ms"]) / dsteps
                res["interval_requests"] = res["stats"]["requests"] - old["requests"]
        except (KeyError, TypeError):
            pass
    with _BRAIN_LOCK:
        _BRAIN_CACHE[root] = res
    return res


def brain_phrase(prof: Dict[str, Any], b: Optional[Dict[str, Any]]) -> str:
    where = f"{_short_host(prof['host'])}:{prof['port']}"
    if b is None:
        return f"{where} {prof['model']}"
    if not b.get("reachable"):
        return f"{where} {prof['model']} UNREACHABLE ({b.get('error') or 'no answer'})"
    info = b.get("info") or {}
    bits = []
    if info.get("engine"):
        bits.append(str(info["engine"]))
    if info.get("max_batch_size") is not None:
        bits.append(f"batch {info['max_batch_size']}")
    if info.get("max_q_size") is not None:
        bits.append(f"queue<= {info['max_q_size']}")
    ctx = info.get("context_length") or prof.get("context_length")
    if ctx:
        bits.append(f"ctx {_fmt_k(int(ctx))}")
    if b.get("rtt_ms") is not None:
        bits.append(f"rtt {b['rtt_ms']}ms")
    return f"{where} {prof['model']}" + (f" ({', '.join(bits)})" if bits else "")


# --------------------------------------------------------------------------- hands / location

def hands_phrase(platform_name: str) -> str:
    node = _platform.node().split(".")[0].lower() or "?"
    if os.name == "nt":
        try:  # Windows 11 still reports release "10"; the build number tells them apart
            build = int(_platform.version().split(".")[-1])
        except (ValueError, IndexError):
            build = 0
        osname = "win11" if build >= 22000 else "win" + _platform.release()
    else:
        osname = _platform.system().lower()
    shell = "msys bash" if os.name == "nt" else os.path.basename(os.environ.get("SHELL", "sh"))
    cwd = os.environ.get("TERMINAL_CWD") or os.getcwd()
    home = _hermes_home()
    prof = home.name if home.parent.name == "profiles" else "default"
    chan = platform_name or "?"
    return f"{node} ({osname}, {shell}), cwd {cwd}, profile {prof}, channel {chan}"


# --------------------------------------------------------------------------- season / windows

def _hm(s: str) -> int:
    h, m = str(s).split(":")
    return int(h) * 60 + int(m)


def current_window(now: _dt.datetime, windows: List[Dict[str, Any]]) -> str:
    t = now.hour * 60 + now.minute
    for w in windows:
        try:
            a, b = _hm(w["start"]), _hm(w["end"])
        except (KeyError, ValueError, TypeError):
            continue
        if (a <= t < b) if a <= b else (t >= a or t < b):
            return str(w.get("name") or "window")
    return "weekend" if now.weekday() >= 5 else "working hours" if 8 * 60 <= t < 18 * 60 else "off-hours"


def next_deadline(today: _dt.date, deadlines: List[Dict[str, Any]], horizon: float) -> Optional[Tuple[str, int, _dt.date]]:
    best = None
    for d in deadlines:
        try:
            mm, dd = (int(x) for x in str(d["date"]).split("-"))
        except (KeyError, ValueError, TypeError):
            continue
        for year in (today.year, today.year + 1):
            try:
                due = _dt.date(year, mm, dd)
            except ValueError:
                continue
            while due.weekday() >= 5:
                due += _dt.timedelta(days=1)
            days = (due - today).days
            if 0 <= days <= horizon and (best is None or days < best[1]):
                best = (str(d.get("label") or d["date"]), days, due)
    return best


# --------------------------------------------------------------------------- memory staleness

_DATE_RE = re.compile(r"\b(?:(20\d\d)-(\d{1,2})-(\d{1,2})|(\d{1,2})/(\d{1,2})(?!/))\b")
_MEM_CACHE: Dict[str, Any] = {}


def memory_staleness(path: Path, today: _dt.date, stale_days: float) -> Optional[Dict[str, Any]]:
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime, today, stale_days)
    if _MEM_CACHE.get("key") == key:
        return _MEM_CACHE["val"]
    text = path.read_text(encoding="utf-8", errors="replace")
    entries = [e for e in text.split("\n§\n") if e.strip()]
    undated = stale = 0
    oldest: Optional[_dt.date] = None
    for e in entries:
        newest = None
        for m in _DATE_RE.finditer(e):
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
            newest = d if newest is None or d > newest else newest
        if newest is None:
            undated += 1
            continue
        if (today - newest).days > stale_days:
            stale += 1
            oldest = newest if oldest is None or newest < oldest else oldest
    val = {"entries": len(entries), "stale": stale, "undated": undated, "oldest": oldest}
    _MEM_CACHE.update(key=key, val=val)
    return val


# --------------------------------------------------------------------------- state.db: neighbors, fatigue, compaction quality

def session_db_facts(session_id: str, window_s: float = 600.0) -> Dict[str, Any]:
    db = _hermes_home() / "state.db"
    out: Dict[str, Any] = {}
    if not db.exists():
        return out
    try:
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=0.5)
        try:
            cutoff = time.time() - window_s
            rows = con.execute(
                "select model, count(*) from sessions where last_activity_at > ? and id != ? "
                "and ended_at is null group by model", (cutoff, session_id),
            ).fetchall()
            out["neighbors"] = {str(m or "?"): int(n) for m, n in rows}
            row = con.execute(
                "select tool_call_count, api_call_count, compression_fallback_streak from sessions where id = ?",
                (session_id,),
            ).fetchone()
            if row:
                out["tool_calls"], out["api_calls"], out["fallback_streak"] = (int(x or 0) for x in row)
        finally:
            con.close()
    except Exception:
        logger.debug("proprioception senses: state.db read failed", exc_info=True)
    return out


# --------------------------------------------------------------------------- history scans

def _content_text(msg: Dict[str, Any]) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return ""


def compaction_signature(history: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    for msg in history or []:
        txt = _content_text(msg)
        if _COMPACTION_MARKER in txt[:400]:
            return hashlib.sha1(txt[:4000].encode("utf-8", "replace")).hexdigest()[:12]
    return None


def _tool_errored(txt: str) -> bool:
    t = txt[:3000]
    if re.search(r'"exit_code":\s*[1-9]', t) or re.search(r'"status":\s*"error"', t):
        return True
    m = re.search(r'"error":\s*("([^"]*)"|null)', t)
    return bool(m and m.group(1) != "null" and m.group(2))


def efference_mismatch(history: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Previous final reply claimed success while the last tool call before it failed."""
    msgs = list(history or [])
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") == "assistant" and not m.get("tool_calls") and _content_text(m).strip():
            if not _SUCCESS_WORDS.search(_content_text(m)[:1500]):
                return None
            for j in range(i - 1, -1, -1):
                r = msgs[j].get("role")
                if r == "tool":
                    if _tool_errored(_content_text(msgs[j])):
                        return str(msgs[j].get("name") or "a tool")
                    return None
                if r == "user":
                    return None
            return None
        if m.get("role") == "user":
            continue
    return None


# --------------------------------------------------------------------------- observe

def observe(*, session_id: str, model: str, platform_name: str, sender_id: str,
            history: Optional[List[Dict[str, Any]]], ss: Dict[str, Any],
            user_message: str = "") -> Dict[str, Any]:
    obs: Dict[str, Any] = {"message": user_message or ""}
    now = _dt.datetime.now()
    if model:
        LAST_SEEN.update(session_id=session_id, model=model, platform=platform_name, sender_id=sender_id)
    try:
        if ss["location"] or ss["brain"] or ss["vestibular"] or ss["self_guard"] or ss["relevance"]:
            prof = resolve_model_profile(model)
            obs["profile"] = prof
            obs["model_key"] = model_key(prof)
            if ss["brain"] and prof["local"]:
                obs["brain"] = probe_brain(prof, ss["brain_timeout_seconds"])
            if ss["location"]:
                obs["hands"] = hands_phrase(platform_name)
            if prof.get("local"):
                _SELF_ENDPOINTS[session_id] = (prof["host"], prof["port"])
            else:
                _SELF_ENDPOINTS.pop(session_id, None)
        if ss["season"]:
            obs["window"] = current_window(now, ss["windows"])
            obs["deadline"] = next_deadline(now.date(), ss["deadlines"], ss["deadline_horizon_days"])
            obs["weekday"] = now.strftime("%A")
        if ss["presence"]:
            sid = str(sender_id or "")
            obs["speaker"] = ss["people"].get(sid) or (f"unknown sender {sid}" if sid else "local operator")
            obs["channel"] = platform_name or "?"
        if ss["amnesia"]:
            obs["compaction"] = compaction_signature(history)
        if ss["trust"]:
            obs["notes"] = estimate.cached_notes(_hermes_home(), now.date())
        if ss["staleness"] and not ss["trust"]:
            obs["memory"] = memory_staleness(_hermes_home() / "memories" / "MEMORY.md", now.date(), ss["stale_days"])
        if ss["neighbors"] or ss["fatigue"] or ss["amnesia"]:
            obs["db"] = session_db_facts(session_id)
        if ss["efference"]:
            obs["efference"] = efference_mismatch(history)
    except Exception:
        logger.debug("proprioception senses: observe failed", exc_info=True)
    return obs


# --------------------------------------------------------------------------- diff

def location_lines(obs: Dict[str, Any]) -> List[str]:
    prof = obs.get("profile")
    if not prof:
        return []
    out = []
    if prof["local"]:
        out.append(f"brain: {brain_phrase(prof, obs.get('brain'))} — your model runs on that host, "
                   "your tools do not")
    else:
        out.append(f"brain: {prof['model']} (remote API, off this network)")
    if obs.get("hands"):
        out.append(f"hands: {obs['hands']}")
    return out


def diff(st: Dict[str, Any], obs: Dict[str, Any], ss: Dict[str, Any], first: bool) -> Tuple[List[str], bool]:
    """Return (lines, urgent). Mutates ``st`` (the session's sense memory)."""
    lines: List[str] = []
    urgent = False

    if ss["location"] and first:
        lines += location_lines(obs)

    mk = obs.get("model_key")
    if mk:
        prev = st.get("model_key")
        if prev and prev != mk:
            # A different brain: its reachability/slowness history is not ours.
            st.pop("brain_reachable", None)
            st.pop("brain_slow", None)
        if ss["vestibular"] and prev and prev != mk:
            urgent = True
            lines.append(f"moved: model changed {prev} -> {mk} since your last turn; earlier "
                         "assistant turns in this conversation came from a different model")
            if ss["location"]:
                lines += location_lines(obs)
        st["model_key"] = mk

    b = obs.get("brain")
    if ss["brain"] and b is not None:
        prof = obs["profile"]
        was = st.get("brain_reachable")
        if was is not None and was != b.get("reachable"):
            urgent = True
            lines.append(f"brain {'back online' if b.get('reachable') else 'lost'}: {brain_phrase(prof, b)}")
        elif first and not b.get("reachable") and not ss["location"]:
            lines.append(f"brain: {brain_phrase(prof, b)}")
        st["brain_reachable"] = b.get("reachable")
        base = (b.get("stats") or {}).get("wait_ms_per_step")
        iv = b.get("interval_ms_per_step")
        slow = bool(iv and base and iv > 1.5 * base)
        if slow and not st.get("brain_slow"):
            lines.append(f"brain slowing: {iv:.1f} ms/step lookahead wait vs {base:.1f} lifetime "
                         f"({b.get('interval_requests', '?')} requests served since last reading)")
        st["brain_slow"] = slow

    if ss["season"]:
        win = obs.get("window")
        if win and (first or st.get("window") != win):
            if not first:
                lines.append(f"window: now {win} ({obs.get('weekday')})")
            else:
                lines.append(f"window: {win} ({obs.get('weekday')})")
        st["window"] = win
        dl = obs.get("deadline")
        dkey = f"{dl[0]}:{dl[1]}" if dl else ""
        if dl and (first or st.get("deadline") != dkey):
            when = "TODAY" if dl[1] == 0 else f"in {dl[1]} day{'s' if dl[1] != 1 else ''}"
            lines.append(f"season: {dl[0]} due {dl[2]:%a %m/%d} ({when}, approx — holidays not modeled)")
        st["deadline"] = dkey

    if ss["presence"]:
        who = (obs.get("speaker"), obs.get("channel"))
        if who[0] and (first or st.get("who") != who):
            lines.append(f"presence: {who[0]} on {who[1]}" + ("" if first else " (changed)"))
        st["who"] = who

    if ss["amnesia"]:
        sig = obs.get("compaction")
        if sig and sig != st.get("compaction"):
            st["compactions"] = st.get("compactions", 0) + 1
            urgent = True
            streak = (obs.get("db") or {}).get("fallback_streak") or 0
            lossy = "; the last summary was the deterministic FALLBACK (lossier than usual)" if streak else ""
            lines.append(f"memory: earlier turns were compacted into a summary"
                         f" ({st['compactions']} seen this session){lossy}; exact earlier details "
                         "are recoverable only via session_search")
        st["compaction"] = sig or st.get("compaction")

    if ss["staleness"] and first:
        mem = obs.get("memory")
        if mem and mem["stale"]:
            lines.append(f"dark readings: {mem['stale']} of {mem['entries']} memory notes were last dated "
                         f">{ss['stale_days']:.0f} days ago (oldest {mem['oldest']:%m/%d}); "
                         f"{mem['undated']} undated — treat as possibly stale until re-verified")

    if ss["trust"]:
        notes = obs.get("notes") or []
        if first:
            summ = estimate.trust_summary(notes)
            if summ:
                lines.append(summ)
        rel = estimate.relevant_notes(notes, obs.get("message", ""))
        key = tuple(n["head"] for n in rel)
        if rel and key != st.get("trust_rel"):
            lines.append("this turn leans on low-trust memory: "
                         + "; ".join(estimate.note_phrase(n) for n in rel)
                         + " — re-verify before relying on it")
        st["trust_rel"] = key

    if ss["relevance"] and obs.get("systems") is not None:
        on_path, _bg = estimate.rank_alarms(obs["systems"], obs.get("profile") or {},
                                            obs.get("channel_name", ""), obs.get("message", ""),
                                            obs.get("attention_states", ()), ss.get("dependence"))
        cur = {label: (st_, why) for _w, label, st_, why in on_path}
        prev = st.get("on_path")
        if prev is not None:
            new = [k for k in cur if k not in prev]
            if new:
                lines.append("now on your path: " + "; ".join(
                    f"{k}: {cur[k][0]}" + (f" ({cur[k][1]})" if cur[k][1] else "") for k in new[:3]))
        st["on_path"] = {k: v[0] for k, v in cur.items()}

    if ss["neighbors"]:
        nb = (obs.get("db") or {}).get("neighbors")
        if nb is not None:
            total = sum(nb.values())
            bucket = 0 if total == 0 else 1 if total <= 2 else 2 if total <= 5 else 3
            same = nb.get((obs.get("profile") or {}).get("model", ""), 0)
            if bucket != st.get("nb_bucket") and (not first or total):
                desc = ", ".join(f"{n} on {m}" for m, n in sorted(nb.items(), key=lambda x: -x[1])[:3])
                share = (f"; {same} share your model's single-request server"
                         if same and (obs.get("profile") or {}).get("local") else "")
                lines.append(f"neighbors: {total} other active session(s)" + (f" ({desc})" if desc else "") + share)
            st["nb_bucket"] = bucket

    if ss["fatigue"]:
        tc = (obs.get("db") or {}).get("tool_calls")
        if tc is not None:
            step = max([s for s in _FATIGUE_STEPS if tc >= s], default=0)
            if step > st.get("fatigue_step", 0 if first else step):
                lines.append(f"fatigue: {tc} tool calls this session (crossed {step})")
            st["fatigue_step"] = step

    if ss["efference"]:
        eff = obs.get("efference")
        if eff and st.get("efference") != eff and not first:
            urgent = True
            lines.append(f"mismatch: your previous reply reported success, but the last tool result "
                         f"before it ({eff}) was an error")
        st["efference"] = eff

    return lines, urgent


# --------------------------------------------------------------------------- in-loop: pain + self-guard

_PAIN_LOCK = threading.Lock()
_PAIN: Dict[str, Dict[str, int]] = {}
_SELF_ENDPOINTS: Dict[str, Tuple[str, Optional[int]]] = {}


def _command_text(args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    return " ".join(str(args.get(k) or "") for k in ("command", "code", "cmd"))


def tool_result_note(*, session_id: str, tool_name: str, args: Any, status: Any,
                     error_message: Any, ss: Dict[str, Any]) -> Optional[str]:
    """Return a one-line note to append to this tool result, or None."""
    notes: List[str] = []
    if ss["pain"] and str(status or "").lower() in ("error", "failed", "failure"):
        sig = f"{tool_name}:{' '.join(str(error_message or '').split())[:80]}"
        with _PAIN_LOCK:
            bucket = _PAIN.setdefault(session_id, {})
            bucket[sig] = bucket.get(sig, 0) + 1
            n = bucket[sig]
            if len(_PAIN) > 512:
                _PAIN.pop(next(iter(_PAIN)))
        if n >= _PAIN_REPEAT:
            notes.append(f"pain: this is the {n}th identical {tool_name} failure this session "
                         f"({sig.split(':', 1)[1] or 'same error'})")
    if ss["self_guard"]:
        ep = _SELF_ENDPOINTS.get(session_id)
        cmd = _command_text(args)
        if ep and cmd and _SELF_HARM_RE.search(cmd):
            host = _short_host(ep[0]).lower()
            hit = host in cmd.lower() or (ep[1] and str(ep[1]) in cmd) or re.search(
                r"inference|exllama|glm|strata|qwen|vllm|sglang|llama", cmd, re.I)
            if hit:
                notes.append(f"self: that command targets {host}:{ep[1]}, the server running YOUR "
                             "model — a restart there interrupts this conversation's brain")
    if not notes:
        return None
    return "[proprioception] " + " | ".join(notes)


# --------------------------------------------------------------------------- scent (tool-only)

def recent_edits(roots: List[str], minutes: float, limit: int = 12) -> List[Tuple[float, str]]:
    cutoff = time.time() - minutes * 60
    hits: List[Tuple[float, str]] = []
    home = _hermes_home()
    for root in roots:
        p = Path(os.path.expandvars(str(root))).expanduser()
        if not p.is_absolute():
            p = home / p
        if "profiles" in p.parts:  # never sniff another profile's rooms
            continue
        if not p.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    mt = os.stat(fp).st_mtime
                except OSError:
                    continue
                if mt >= cutoff:
                    hits.append((mt, fp))
    hits.sort(reverse=True)
    return hits[:limit]


def reset_for_tests() -> None:
    with _BRAIN_LOCK:
        _BRAIN_CACHE.clear()
    with _PAIN_LOCK:
        _PAIN.clear()
    _SELF_ENDPOINTS.clear()
    _MEM_CACHE.clear()
    estimate.reset_for_tests()
