"""Passive state recorder: each heartbeat reading becomes a vector s_t.

History accumulates as rows (time x slot); grouped by session it is the
session x time x slot tensor. Nothing here changes what the agent is told —
the recorder only writes, so tau (trust decay), J (dependence) and sigma
(noise per slot) can later be FITTED from data instead of guessed.

Slots are append-only: a slot name gets a fixed column index the first time it
is seen (systems and memory notes come and go). A row stores float32 values by
column; columns a row did not observe are NaN.

Storage: <hermes_home>/proprioception/tensor.db (sqlite, local only). Bounded
by ``tensor_max_rows`` (oldest rows pruned). Recording never raises.

Slot families
  t:*        hour_sin/hour_cos (time of day on a circle), dow, gap_min
  ctx:*      tokens, frac
  brain:*    local, reachable, rtt_ms, requests, wait_ms_per_step, interval_ms_per_step
  dash:*     verdict_sev, sensors_down, gateway_ok
  sys:<id>:sev  0 ok / 0.5 warn / 1 crit       sys:<id>:J  dependence this turn
  note:<h>:trust / note:<h>:age               (h = hash of the note's first 55 chars)
  db:*       neighbors, tool_calls, api_calls, fallback_streak
  mem:compacted, out:emitted, out:delta_w
out:delta_w = ||s_t - s_prev||_W over the bounded slots (sys sev weighted by
J, brain reachable, ctx frac, note trust) — the weighted-norm form of
"speak on change", logged so its threshold can be calibrated later.
"""

from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
import threading
import time
from array import array
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SLOTS: Dict[str, int] = {}          # name -> column (cache of the slots table)
_SLOTS_DB: Optional[str] = None      # db path the cache belongs to
_LAST: Dict[str, Tuple[float, Dict[str, float]]] = {}   # session -> (wall ts, features)
_INSERTS = 0
_PRUNE_EVERY = 500

_SEV = {"ok": 0.0, "info": 0.0, "warn": 0.5, "warning": 0.5, "degraded": 0.5, "stale": 0.4,
        "unknown": 0.3, "crit": 1.0, "critical": 1.0, "down": 1.0, "error": 1.0}


def note_key(head: str) -> str:
    return hashlib.sha1(head.encode("utf-8", "replace")).hexdigest()[:10]


# --------------------------------------------------------------------------- features (pure)

def features(obs: Dict[str, Any], *, ctx_tokens: int, window: int, snap: Any,
             wall: float, prev_wall: Optional[float], emitted: bool,
             ss: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Tuple[str, str]]]:
    """obs/snapshot -> ({slot: value}, {slot: (kind, label)})."""
    from . import estimate

    f: Dict[str, float] = {}
    meta: Dict[str, Tuple[str, str]] = {}
    lt = time.localtime(wall)
    hour = lt.tm_hour + lt.tm_min / 60.0
    f["t:hour_sin"] = math.sin(2 * math.pi * hour / 24)
    f["t:hour_cos"] = math.cos(2 * math.pi * hour / 24)
    f["t:dow"] = float(lt.tm_wday)
    if prev_wall is not None and wall > prev_wall:
        f["t:gap_min"] = (wall - prev_wall) / 60.0
    f["ctx:tokens"] = float(ctx_tokens)
    if window:
        f["ctx:frac"] = ctx_tokens / float(window)

    prof = obs.get("profile") or {}
    if prof:
        f["brain:local"] = 1.0 if prof.get("local") else 0.0
    b = obs.get("brain")
    if b:
        f["brain:reachable"] = 1.0 if b.get("reachable") else 0.0
        if b.get("rtt_ms") is not None:
            f["brain:rtt_ms"] = float(b["rtt_ms"])
        st = b.get("stats") or {}
        for k in ("requests", "wait_ms_per_step"):
            if isinstance(st.get(k), (int, float)):
                f[f"brain:{k}"] = float(st[k])
        if isinstance(b.get("interval_ms_per_step"), (int, float)):
            f["brain:interval_ms_per_step"] = float(b["interval_ms_per_step"])

    dash = getattr(snap, "dashboard", None)
    systems = obs.get("systems")
    if systems is None and isinstance(dash, dict):
        systems = [s for s in (dash.get("systems") or []) if isinstance(s, dict)]
    if isinstance(dash, dict):
        f["dash:verdict_sev"] = _SEV.get(str(dash.get("verdict", "")).lower(), 0.3)
    f["dash:sensors_down"] = 1.0 if getattr(snap, "sensors_down", False) else 0.0
    gws = getattr(snap, "gateway_state", None)
    if gws is not None:
        f["dash:gateway_ok"] = 1.0 if str(gws) in ("ok", "running", "healthy") else 0.0
    msg_tokens = estimate.tokens(obs.get("message", ""))
    for s in systems or []:
        sid = str(s.get("id") or s.get("label") or "?")
        f[f"sys:{sid}:sev"] = _SEV.get(str(s.get("state", "")).lower(), 0.3)
        j, _why = estimate.system_dependence(s, prof, obs.get("channel_name", ""), msg_tokens,
                                              ss.get("dependence"))
        f[f"sys:{sid}:J"] = float(j)
        meta[f"sys:{sid}:sev"] = ("sys", str(s.get("label") or sid))

    for n in obs.get("notes") or []:
        if n.get("trust") is None:
            continue
        h = note_key(n["head"])
        f[f"note:{h}:trust"] = float(n["trust"])
        f[f"note:{h}:age"] = float(n["age"])
        meta[f"note:{h}:trust"] = (n["kind"], n["head"])

    db = obs.get("db") or {}
    if "neighbors" in db:
        f["db:neighbors"] = float(sum(db["neighbors"].values()))
    for k in ("tool_calls", "api_calls", "fallback_streak"):
        if k in db:
            f[f"db:{k}"] = float(db[k])
    if "compaction" in obs:
        f["mem:compacted"] = 1.0 if obs.get("compaction") else 0.0
    f["out:emitted"] = 1.0 if emitted else 0.0
    return f, meta


def weighted_delta(cur: Dict[str, float], prev: Dict[str, float]) -> Optional[float]:
    """||s_t - s_prev||_W over bounded slots shared by both readings."""
    total, n = 0.0, 0
    for k, v in cur.items():
        if k not in prev:
            continue
        if k.endswith(":sev") and k.startswith("sys:"):
            w = cur.get(k[:-4] + ":J", 0.1)
        elif k in ("brain:reachable", "ctx:frac") or k.endswith(":trust"):
            w = 1.0
        else:
            continue
        total += (w * (v - prev[k])) ** 2
        n += 1
    return math.sqrt(total) if n else None


# --------------------------------------------------------------------------- storage

def db_path(ss: Optional[Dict[str, Any]] = None) -> Path:
    custom = (ss or {}).get("tensor_path")
    if custom:
        return Path(custom)
    from .senses import _hermes_home

    return _hermes_home() / "proprioception" / "tensor.db"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=1.0)
    con.execute("pragma journal_mode=wal")
    con.execute("create table if not exists slots (idx integer primary key, name text unique not null, "
                "kind text, label text, first_ts real)")
    con.execute("create table if not exists readings (id integer primary key autoincrement, ts real not null, "
                "session text, model text, platform text, emitted integer, vec blob not null)")
    con.execute("create index if not exists readings_session_ts on readings(session, ts)")
    return con


def _slot_index(con: sqlite3.Connection, name: str, kind: str, label: str, ts: float) -> int:
    idx = _SLOTS.get(name)
    if idx is not None:
        return idx
    con.execute("insert or ignore into slots(idx, name, kind, label, first_ts) values "
                "((select coalesce(max(idx), -1) + 1 from slots), ?, ?, ?, ?)", (name, kind, label, ts))
    idx = int(con.execute("select idx from slots where name = ?", (name,)).fetchone()[0])
    _SLOTS[name] = idx
    return idx


def record(*, session_id: str, model: str, platform: str, obs: Dict[str, Any], snap: Any,
           ctx_tokens: int, window: int, emitted: bool, ss: Dict[str, Any],
           wall: Optional[float] = None) -> Optional[int]:
    """Append one reading. Returns the row id, or None on any failure."""
    global _SLOTS_DB, _INSERTS
    try:
        wall = time.time() if wall is None else wall
        prev = _LAST.get(session_id)
        f, meta = features(obs, ctx_tokens=ctx_tokens, window=window, snap=snap, wall=wall,
                           prev_wall=prev[0] if prev else None, emitted=emitted, ss=ss)
        if prev:
            d = weighted_delta(f, prev[1])
            if d is not None:
                f["out:delta_w"] = d
        path = db_path(ss)
        with _LOCK:
            con = _connect(path)
            try:
                if _SLOTS_DB != str(path):
                    _SLOTS.clear()
                    _SLOTS.update({n: i for n, i in con.execute("select name, idx from slots")})
                    _SLOTS_DB = str(path)
                with con:
                    cols = {}
                    for name, v in f.items():
                        kind, label = meta.get(name, (name.split(":", 1)[0], ""))
                        cols[_slot_index(con, name, kind, label, wall)] = v
                    vec = array("f", [math.nan]) * (max(cols) + 1)
                    for i, v in cols.items():
                        vec[i] = v
                    cur = con.execute(
                        "insert into readings(ts, session, model, platform, emitted, vec) values (?,?,?,?,?,?)",
                        (wall, session_id, model, platform, int(emitted), vec.tobytes()))
                    rid = int(cur.lastrowid)
                    _INSERTS += 1
                    if _INSERTS % _PRUNE_EVERY == 0:
                        cap = int(ss.get("tensor_max_rows") or 500_000)
                        con.execute("delete from readings where id <= ?", (rid - cap,))
            finally:
                con.close()
        _LAST[session_id] = (wall, f)
        return rid
    except Exception:
        logger.debug("proprioception tensor: record failed", exc_info=True)
        return None


# --------------------------------------------------------------------------- read side

def load(path: Optional[Path] = None, *, since: Optional[float] = None,
         session: Optional[str] = None) -> Dict[str, Any]:
    """Dense matrix of readings: X[row, slot] (NaN = not observed).

    Returns {names, kinds, labels, ts, session, model, platform, emitted, X}.
    X is a numpy array when numpy is importable, else a list of lists.
    """
    path = path or db_path()
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, timeout=1.0)
    try:
        slots = con.execute("select idx, name, kind, label from slots order by idx").fetchall()
        q, args = "select ts, session, model, platform, emitted, vec from readings", []
        cond = []
        if since is not None:
            cond.append("ts >= ?"); args.append(since)
        if session is not None:
            cond.append("session = ?"); args.append(session)
        if cond:
            q += " where " + " and ".join(cond)
        rows = con.execute(q + " order by ts", args).fetchall()
    finally:
        con.close()
    D = (slots[-1][0] + 1) if slots else 0
    names, kinds, labels = [""] * D, [""] * D, [""] * D
    for i, n, k, lab in slots:
        names[i], kinds[i], labels[i] = n, k or "", lab or ""
    mat: List[List[float]] = []
    for r in rows:
        a = array("f"); a.frombytes(r[5])
        v = list(a) + [math.nan] * (D - len(a))
        mat.append(v)
    try:
        import numpy as np

        X: Any = np.array(mat, dtype=np.float32).reshape(len(mat), D)
    except ImportError:
        X = mat
    return {"names": names, "kinds": kinds, "labels": labels, "ts": [r[0] for r in rows],
            "session": [r[1] for r in rows], "model": [r[2] for r in rows],
            "platform": [r[3] for r in rows], "emitted": [r[4] for r in rows], "X": X}


def as_tensor(data: Dict[str, Any]):
    """Group rows by session -> (sessions, T[S, Tmax, D]) padded with NaN. Needs numpy."""
    import numpy as np

    order: Dict[str, List[int]] = {}
    for i, s in enumerate(data["session"]):
        order.setdefault(s, []).append(i)
    X = np.asarray(data["X"])
    S, D = len(order), X.shape[1] if X.ndim == 2 else 0
    Tmax = max((len(v) for v in order.values()), default=0)
    T = np.full((S, Tmax, D), np.nan, dtype=np.float32)
    for si, idxs in enumerate(order.values()):
        T[si, :len(idxs)] = X[idxs]
    return list(order), T


def stats(path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(path or db_path())
    if not path.exists():
        return {"path": str(path), "exists": False}
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=1.0)
    try:
        n, t0, t1, ns = con.execute(
            "select count(*), min(ts), max(ts), count(distinct session) from readings").fetchone()
        d = con.execute("select count(*) from slots").fetchone()[0]
    finally:
        con.close()
    size = sum(p.stat().st_size for p in path.parent.glob(path.name + "*") if p.is_file())
    return {"path": str(path), "exists": True, "rows": n, "sessions": ns, "slots": d,
            "first": t0, "last": t1, "bytes": size}


def reset_for_tests() -> None:
    global _SLOTS_DB, _INSERTS
    with _LOCK:
        _SLOTS.clear()
        _SLOTS_DB = None
        _LAST.clear()
        _INSERTS = 0
