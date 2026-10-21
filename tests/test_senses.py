"""Behavior contracts for the extended senses (senses.py + heartbeat wiring)."""

from __future__ import annotations

import datetime as dt

import pytest

from plugins.proprioception import heartbeat, senses
from plugins.proprioception.settings import DEFAULTS

CFG = {
    "providers": {
        "local-glm": {"base_url": "http://box-a.tailnet-example.ts.net:30000/v1",
                     "models": ["glm-5.3-flash"], "context_length": 131072},
        "local-qwen": {"base_url": "http://box-a.tailnet-example.ts.net:8002/v1",
                       "models": ["qwen3.8-flash-next"]},
    },
    "model": {"default": "qwen3.8-flash-next", "provider": "local-qwen"},
}


@pytest.fixture(autouse=True)
def _reset():
    senses.reset_for_tests()
    heartbeat.reset_for_tests()
    yield
    senses.reset_for_tests()
    heartbeat.reset_for_tests()


def test_all_senses_off_by_default():
    ss = senses.sense_settings(None)
    assert not senses.any_enabled(ss)
    assert senses.any_enabled(senses.sense_settings({"all": True}))
    assert senses.sense_settings({"all": True, "brain": False})["brain"] is False


def test_local_and_cloud_models_resolve_differently():
    glm = senses.resolve_model_profile("glm-5.3-flash", CFG)
    qwen = senses.resolve_model_profile("qwen3.8-flash-next", CFG)
    cloud = senses.resolve_model_profile("anthropic/claude-opus-5.5", CFG)
    assert glm["local"] and glm["port"] == 30000 and glm["root"].endswith(":30000")
    assert qwen["local"] and qwen["port"] == 8002
    assert not cloud["local"] and cloud["root"] == "" and cloud["provider"] == "anthropic"
    assert senses.model_key(glm) != senses.model_key(qwen)


def test_cloud_model_is_never_probed():
    assert senses.probe_brain(senses.resolve_model_profile("anthropic/x", CFG)) is None


def test_deadline_rolls_off_weekend_and_respects_horizon():
    # 2027-01-31 is a Sunday -> due Monday 02-01
    hit = senses.next_deadline(dt.date(2027, 1, 25), [{"date": "01-31", "label": "W-2"}], 21)
    assert hit and hit[2] == dt.date(2027, 2, 1) and hit[2].weekday() == 0
    assert senses.next_deadline(dt.date(2027, 1, 1), [{"date": "06-15", "label": "x"}], 21) is None


def test_window_wraps_midnight():
    w = [{"name": "night", "start": "23:00", "end": "02:00"}]
    assert senses.current_window(dt.datetime(2026, 10, 7, 0, 30), w) == "night"
    assert senses.current_window(dt.datetime(2026, 10, 7, 12, 0), w) != "night"


def test_memory_staleness_counts_old_and_undated(tmp_path):
    p = tmp_path / "MEMORY.md"
    p.write_text("fresh fact 10/9\n§\nold fact 8/17\n§\nno date here\n§\niso 2026-09-01", encoding="utf-8")
    res = senses.memory_staleness(p, dt.date(2026, 10, 10), 14)
    assert res["entries"] == 4 and res["undated"] == 1 and res["stale"] == 2


def test_compaction_seen_once_then_quiet():
    ss = senses.sense_settings({"amnesia": True})
    hist = [{"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary..."}]
    st: dict = {}
    obs = {"compaction": senses.compaction_signature(hist), "db": {"fallback_streak": 1}}
    lines, urgent = senses.diff(st, obs, ss, first=False)
    assert urgent and any("compacted" in line and "FALLBACK" in line for line in lines)
    lines2, _ = senses.diff(st, obs, ss, first=False)
    assert lines2 == []


def test_model_switch_is_felt():
    ss = senses.sense_settings({"vestibular": True})
    st: dict = {}
    senses.diff(st, {"model_key": "qwen3.8-flash-next@local-qwen:8002"}, ss, first=True)
    lines, urgent = senses.diff(st, {"model_key": "glm-5.3-flash@local-glm:30000"}, ss, first=False)
    assert urgent and "qwen3.8-flash-next" in lines[0] and "glm-5.3-flash" in lines[0]


def test_efference_flags_claimed_success_after_failed_tool():
    bad = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "name": "terminal", "content": '{"output": "boom", "exit_code": 1}'},
        {"role": "assistant", "content": "Done — fixed and verified."},
    ]
    good = bad[:2] + [{"role": "tool", "name": "terminal", "content": '{"output": "ok", "exit_code": 0, "error": null}'}] + bad[3:]
    assert senses.efference_mismatch(bad) == "terminal"
    assert senses.efference_mismatch(good) is None


def test_pain_appears_on_third_identical_failure():
    ss = senses.sense_settings({"pain": True})
    kw = dict(session_id="s", tool_name="terminal", args={}, status="error", error_message="cannot change to", ss=ss)
    assert senses.tool_result_note(**kw) is None
    assert senses.tool_result_note(**kw) is None
    note = senses.tool_result_note(**kw)
    assert note and "pain" in note and "cannot change to" in note


def test_self_guard_only_for_own_brain():
    ss = senses.sense_settings({"self_guard": True})
    senses._SELF_ENDPOINTS["s"] = ("box-a.tailnet-example.ts.net", 30000)
    hit = senses.tool_result_note(session_id="s", tool_name="terminal",
                                  args={"command": "ssh box-a sudo systemctl restart glm53"},
                                  status="ok", error_message="", ss=ss)
    miss = senses.tool_result_note(session_id="s", tool_name="terminal",
                                   args={"command": "systemctl restart nginx"},
                                   status="ok", error_message="", ss=ss)
    cloud = senses.tool_result_note(session_id="other", tool_name="terminal",
                                    args={"command": "ssh box-a sudo systemctl restart glm53"},
                                    status="ok", error_message="", ss=ss)
    assert hit and "YOUR" in hit and miss is None and cloud is None


def test_heartbeat_first_turn_carries_location_and_terse_for_local(monkeypatch):
    from plugins.proprioception import collector

    monkeypatch.setattr(collector, "get_snapshot", lambda *_a, **_k: collector.Snapshot(
        dashboard={"systems": [], "verdict": "ok"}, gateway=None, fetched_at=0.0))
    monkeypatch.setattr(senses, "_load_config", lambda: CFG)
    monkeypatch.setattr(senses, "probe_brain", lambda prof, *_a, **_k: {
        "reachable": False, "error": "URLError", "rtt_ms": None})
    settings = dict(DEFAULTS, enabled=True,
                    senses=senses.sense_settings({"location": True, "brain": True}))
    text = heartbeat.build_heartbeat(session_id="t1", conversation_history=[], settings=settings,
                                     model="qwen3.8-flash-next", platform_name="telegram")
    assert text and "8002" in text and "UNREACHABLE" in text and "hands:" in text
    assert heartbeat._SYSTEM_NOTE_TERSE in text  # local brain -> terse format


def test_heartbeat_senses_off_changes_nothing(monkeypatch):
    from plugins.proprioception import collector

    monkeypatch.setattr(collector, "get_snapshot", lambda *_a, **_k: collector.Snapshot(
        dashboard={"systems": [], "verdict": "ok"}, gateway=None, fetched_at=0.0))
    settings = dict(DEFAULTS, enabled=True, senses=senses.sense_settings({}))
    assert heartbeat.build_heartbeat(session_id="t2", conversation_history=[], settings=settings,
                                     model="glm-5.3-flash") is None


# ---------------------------------------------------------------- estimate.py

from plugins.proprioception import estimate  # noqa: E402


def test_trust_decays_by_kind():
    assert estimate.classify_note("GPU box 192.0.2.13 :8002 direct") == "network"
    assert estimate.classify_note("PROFILES = default+work; never add") == "rule"
    assert estimate.classify_note("Docling watchdog FIXED") == "state"
    assert estimate.trust(7, estimate.TAU_DAYS["network"]) == pytest.approx(0.3679, abs=1e-3)
    assert estimate.trust(365, estimate.TAU_DAYS["rule"]) == 1.0


def test_relevant_low_trust_note_surfaces(tmp_path):
    p = tmp_path / "MEMORY.md"
    p.write_text("Sidecar VM tailscale 198.51.100.23 relay 9/1\n\u00a7\n"
                 "PROFILES default work only, never add 9/2\n\u00a7\nDocling watchdog fixed 10/9",
                 encoding="utf-8")
    notes = estimate.read_notes([(p, "general")], dt.date(2026, 10, 10))
    rel = estimate.relevant_notes(notes, "can you ssh into the Sidecar VM for me?")
    assert len(rel) == 1 and rel[0]["kind"] == "network" and rel[0]["trust"] < 0.01
    assert estimate.relevant_notes(notes, "add a profile for work") == []  # rule: trust 1.0


def test_alarm_weight_depends_on_where_the_brain_is():
    systems = [{"id": "primary-model", "label": "Primary model (box-a)", "state": "crit"},
               {"id": "hermes-backup", "label": "Hermes local recovery", "state": "crit"},
               {"id": "disk-c", "label": "Disk C:", "state": "warn"}]
    glm = senses.resolve_model_profile("glm-5.3-flash", CFG)
    cloud = senses.resolve_model_profile("anthropic/x", CFG)
    on_glm, _ = estimate.rank_alarms(systems, glm, "telegram", "hi", {"crit", "warn"})
    on_cloud, bg = estimate.rank_alarms(systems, cloud, "telegram", "hi", {"crit", "warn"})
    assert on_glm[0][1].startswith("Primary model") and on_glm[0][0] == 1.0
    assert all(not x[1].startswith("Primary") for x in on_cloud) and len(bg) == 3  # disk w=0.3x0.5=0.15 < 0.25
    on_msg, _ = estimate.rank_alarms(systems, cloud, "telegram", "is the recovery backup ok?", {"crit", "warn"})
    assert on_msg[0][1] == "Hermes local recovery" and "mentions" in on_msg[0][3]


def test_relevance_reports_new_on_path_alarm_once():
    ss = senses.sense_settings({"relevance": True})
    systems = [{"id": "hermes-backup", "label": "Hermes local recovery", "state": "crit"}]
    base = {"systems": systems, "attention_states": ("crit",), "profile": {}, "channel_name": "telegram"}
    st: dict = {}
    senses.diff(st, {**base, "message": "hello"}, ss, first=True)
    lines, _ = senses.diff(st, {**base, "message": "check the backup recovery"}, ss, first=False)
    assert lines and "Hermes local recovery" in lines[0]
    again, _ = senses.diff(st, {**base, "message": "and the backup again"}, ss, first=False)
    assert again == []


# ---------------------------------------------------------------- tensor.py

from plugins.proprioception import tensor  # noqa: E402


class _Snap:
    dashboard = {"verdict": "warn", "systems": [
        {"id": "primary-model", "label": "Primary model (box-a)", "state": "crit"},
        {"id": "disk-c", "label": "Disk C:", "state": "ok"}]}
    sensors_down = False
    gateway_state = "ok"


def test_weighted_delta_counts_only_what_you_depend_on():
    a = {"sys:x:sev": 0.0, "sys:x:J": 1.0, "sys:y:sev": 0.0, "sys:y:J": 0.1}
    b = {"sys:x:sev": 1.0, "sys:x:J": 1.0, "sys:y:sev": 0.0, "sys:y:J": 0.1}
    c = {"sys:x:sev": 0.0, "sys:x:J": 1.0, "sys:y:sev": 1.0, "sys:y:J": 0.1}
    assert tensor.weighted_delta(b, a) == pytest.approx(1.0)
    assert tensor.weighted_delta(c, a) == pytest.approx(0.1)
    assert tensor.weighted_delta({"ctx:tokens": 5.0}, {"ctx:tokens": 1.0}) is None


def test_record_and_load_roundtrip(tmp_path):
    tensor.reset_for_tests()
    ss = senses.sense_settings({"tensor": True, "relevance": True, "tensor_path": str(tmp_path / "t.db")})
    glm = senses.resolve_model_profile("glm-5.3-flash", CFG)
    obs = {"profile": glm, "brain": {"reachable": True, "rtt_ms": 120}, "message": "", "channel_name": "telegram"}
    r1 = tensor.record(session_id="s1", model="glm-5.3-flash", platform="telegram", obs=obs, snap=_Snap(),
                       ctx_tokens=1000, window=131072, emitted=True, ss=ss, wall=1_000_000.0)
    obs2 = dict(obs, brain={"reachable": False, "rtt_ms": None})
    r2 = tensor.record(session_id="s1", model="glm-5.3-flash", platform="telegram", obs=obs2, snap=_Snap(),
                       ctx_tokens=2000, window=131072, emitted=False, ss=ss, wall=1_000_060.0)
    assert r1 and r2
    data = tensor.load(tmp_path / "t.db")
    X, names = data["X"], data["names"]
    assert X.shape[0] == 2 and len(names) == X.shape[1]
    col = names.index
    assert X[0, col("sys:primary-model:J")] == 1.0 and X[0, col("sys:primary-model:sev")] == 1.0
    assert X[1, col("t:gap_min")] == pytest.approx(1.0)
    assert X[1, col("out:delta_w")] == pytest.approx(1.0, abs=0.01)   # brain went unreachable
    import math as _m
    assert _m.isnan(X[0, col("out:delta_w")])
    sessions, T = tensor.as_tensor(data)
    assert sessions == ["s1"] and T.shape == (1, 2, X.shape[1])
    assert tensor.stats(tmp_path / "t.db")["rows"] == 2


def test_record_never_raises():
    assert tensor.record(session_id="x", model="", platform="", obs=None, snap=None,  # type: ignore[arg-type]
                         ctx_tokens=0, window=0, emitted=False, ss={}) is None
