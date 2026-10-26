"""Phase 3: incidents and storage."""
import pandas as pd
import pytest

from brick_model import BrickModel
from fm_incidents import STATUSES, all_events, allowed_next, build_incidents, set_status
from fm_store import SqliteStore


@pytest.fixture(scope="module")
def events(flagged_floors):
    return all_events(flagged_floors, BrickModel.load().units())


@pytest.fixture()
def store(tmp_path):
    return SqliteStore(tmp_path / "fm.db")


def test_one_incident_per_asset_and_pattern(events):
    inc = build_incidents(events)
    assert inc["incident_key"].is_unique
    assert len(inc) < len(events) / 3  # hundreds of events collapse into a handful of incidents
    assert (inc["occurrences"].sum() == len(events))
    assert inc["confidence"].between(0, 100).all() and inc["triage_score"].notna().all()


def test_saturday_pattern_on_f7_z1_ac4_is_one_incident(events):
    inc = build_incidents(events)
    row = inc[inc["incident_key"] == "F7_Z1_AC4|closed_running"]
    assert len(row) == 1 and row.iloc[0]["occurrences"] >= 10
    sat = events[(events["incident_key"] == "F7_Z1_AC4|closed_running") & (events["start"].dt.dayofweek == 5)]
    assert len(sat) >= 10  # the Saturday running that used to be many separate events


def test_period_only_changes_active(events):
    full = build_incidents(events)
    day = build_incidents(events, period=(pd.Timestamp("2019-07-27"), pd.Timestamp("2019-07-28")))
    assert list(full["incident_key"]) == list(day["incident_key"])  # identity does not depend on the timeframe
    assert 0 < day["active"].sum() < len(day)


def test_status_survives_restart(tmp_path, events):
    path = tmp_path / "fm.db"
    s1 = SqliteStore(path)
    set_status(s1, "F7_Z1_AC4|closed_running", "Acknowledged")
    s2 = SqliteStore(path)  # a new process would open the same file
    inc = build_incidents(events, s2.table("incident_state"))
    assert inc.set_index("incident_key").loc["F7_Z1_AC4|closed_running", "status"] == "Acknowledged"
    assert (s2.table("audit_log")["change"].str.contains("Open -> Acknowledged")).any()


def test_starts_empty_without_a_store_file(tmp_path, events):
    path = tmp_path / "missing.db"
    assert not path.exists()
    store = SqliteStore(path)
    assert store.is_empty()
    inc = build_incidents(events, store.table("incident_state"))
    assert (inc["status"] == "Open").all()
    path.unlink()
    assert SqliteStore(path).is_empty()  # deleting the file between runs is fine


def test_status_rules(store):
    with pytest.raises(ValueError):
        set_status(store, "k", "Verified")  # Open -> Verified is not allowed
    with pytest.raises(ValueError):
        set_status(store, "k", "Dismissed", "")  # needs a reason
    set_status(store, "k", "Dismissed", "booked event")
    assert store.get("incident_state", "k")["status_reason"] == "booked event"
    set_status(store, "k", "Open")
    assert set(allowed_next("Resolved")) == {"Resolved", "Verified", "Reopened", "Dismissed"}
    assert all(s in STATUSES for s in allowed_next("Open"))


def test_ids_increment(store):
    for _ in range(3):
        store.upsert("work_orders", {"wo_id": store.next_id("work_orders"), "status": "Draft"})
    assert list(store.table("work_orders")["wo_id"]) == ["WO-0001", "WO-0002", "WO-0003"]
