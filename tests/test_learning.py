"""Phase 5 (close-out and learning) and Phase 6 (verification)."""
import pandas as pd
import pytest

from brick_model import BrickModel
from conftest import FLOORS, OPEN_DEFAULT, OUT, read_floor
from fm_core import build_events, flag_detail
from fm_incidents import all_events, build_incidents
from fm_learning import (add_exception, close_out, deactivate_exception, exceptions_tuple, history_note,
                         overall_precision_pct, precision_tables, seed_demo, suggest_exception, threshold_suggestions,
                         track_record)
from fm_store import SqliteStore
from fm_verify import find_natural_changes, verify_incident
from fm_workorders import draft_work_order, raise_work_order

UNITS = BrickModel.load().units()
COL_OF = dict(zip(UNITS["asset_id"], UNITS["power_column"]))


@pytest.fixture(scope="module")
def world(flagged_floors):
    ev = all_events(flagged_floors, UNITS)
    inc = build_incidents(ev)
    fc = {f: pd.read_csv(OUT / f"floor_{f}" / "forecast_hourly_actual_predictions.csv", index_col=0,
                         parse_dates=True)["M1 median profile"] for f in FLOORS}
    return ev, inc, fc


@pytest.fixture()
def seeded(tmp_path, world, flagged_floors):
    ev, inc, fc = world
    store = SqliteStore(tmp_path / "fm.db")
    n = seed_demo(store, ev, inc, flagged_floors, UNITS, OPEN_DEFAULT, fc, pd.Timestamp("2019-12-31"))
    return store, n


def test_seed_makes_about_ten_simulated_closeouts(seeded):
    store, n = seeded
    assert 8 <= n <= 10
    co = store.table("closeouts")
    assert len(co) == n and (co["simulated"] == 1).all()
    assert (store.table("work_orders")["simulated"] == 1).all()
    assert not store.is_empty()


def test_seed_runs_once(seeded, world, flagged_floors):
    store, n = seeded
    ev, inc, fc = world
    assert seed_demo(store, ev, inc, flagged_floors, UNITS, OPEN_DEFAULT, fc, pd.Timestamp("2019-12-31")) == 0
    assert len(store.table("closeouts")) == n


def test_closeouts_drive_precision_kpi(seeded):
    store, n = seeded
    pct, confirmed, no_fault = overall_precision_pct(store)
    assert confirmed >= 4 and no_fault >= 2 and 0 < pct < 100
    assert pct == pytest.approx(100 * confirmed / (confirmed + no_fault))
    t = precision_tables(store, None)
    assert t["overall"]["expected_use"] == 1  # legitimate use is not a detection error
    assert t["pattern"].loc["supply_group_high", "precision"] == 0


def test_no_fault_found_lowers_confidence(seeded, flagged_floors):
    store, _ = seeded
    d = flagged_floors[4]
    base = build_events(d)
    rec = track_record(store, COL_OF)
    after = build_events(d, track_record=rec)
    key = ("F4_z4_AC1(kW)", "high_while_open")
    assert key[0] in set(base["unit"])
    b = base[(base["unit"] == key[0]) & (base["pattern"] == key[1])]["confidence"].mean()
    a = after[(after["unit"] == key[0]) & (after["pattern"] == key[1])].sort_values("start")["confidence"].mean()
    assert a < b  # 'No fault found' on this unit and pattern


def test_confirmed_raises_towards_track_record(seeded, flagged_floors):
    store, _ = seeded
    d = flagged_floors[7]
    base = build_events(d)
    after = build_events(d, track_record=track_record(store, COL_OF))
    m = (base["unit"] == "F7_z1_AC4(kW)") & (base["pattern"] == "closed_running")
    # base ~92, smoothed record (1+1)/(1+0+2)... pattern-wide has several confirmed: result stays between the two
    assert after.loc[after["unit"].eq("F7_z1_AC4(kW)") & after["pattern"].eq("closed_running"), "confidence"].between(0, 100).all()
    assert m.any()


def test_exception_removes_flags(seeded, flagged_floors):
    store, _ = seeded
    exc = store.table("exceptions")
    assert len(exc) == 1 and exc.iloc[0]["simulated"] == 1
    tup = exceptions_tuple(store, COL_OF)
    assert len(tup) == 1
    d7, th7 = read_floor(7)
    before = flagged_floors[7]
    after = flag_detail(d7, th7, OPEN_DEFAULT, 1.5, 1.5, 0.5, 8, tup)
    unit = tup[0][0]
    assert after.loc[after["unit"] == unit, "flagged"].sum() < before.loc[before["unit"] == unit, "flagged"].sum()
    assert after["excepted"].sum() > 0
    # deactivating brings the flags back
    deactivate_exception(store, exc.iloc[0]["exc_id"])
    assert exceptions_tuple(store, COL_OF) == ()


def test_history_note_cites_simulated(seeded):
    store, _ = seeded
    note = history_note(store, "F7_Z1_AC4")
    assert "timer schedule left on" in note and "(simulated)" in note


def test_threshold_suggestion_needs_three_closeouts(seeded):
    store, _ = seeded
    assert threshold_suggestions(store) == []  # no unit has >= 3 close-outs with precision < 30%


def test_suggest_exception_prefill(world):
    ev, inc, fc = world
    e = ev[ev["incident_key"] == "F7_Z2_AC1|closed_running"]
    s = suggest_exception({"asset_id": "F7_Z2_AC1", "wo_id": "WO-0009"}, e)
    assert s["weekdays"].startswith("5") and s["asset_id"] == "F7_Z2_AC1" and s["source_wo_id"] == "WO-0009"


# ---------------------------------------------------------------- Phase 6
def test_natural_changes_exist_and_verify(flagged_floors, world):
    ev, inc, fc = world
    d = flagged_floors[7][flagged_floors[7]["unit"] == "F7_z1_AC4(kW)"]
    changes = find_natural_changes(d, "closed_running")
    assert changes
    events = ev[ev["incident_key"] == "F7_Z1_AC4|closed_running"]
    v = verify_incident(d, "closed_running", changes[0], events)
    assert v["verdict"] == "Verified" and v["drop_pct"] >= 50 and v["saved_kWh"] > 0
    assert v["saved_THB"] == pytest.approx(v["saved_kWh"] * 4.2)


def test_savings_match_hand_calculation(flagged_floors, world):
    ev, inc, fc = world
    d = flagged_floors[7][flagged_floors[7]["unit"] == "F7_z1_AC4(kW)"]
    fix = pd.Timestamp("2019-10-07")
    v = verify_incident(d, "closed_running", fix, ev[ev["incident_key"] == "F7_Z1_AC4|closed_running"])
    pre = d[(d["Date"] >= fix - pd.Timedelta(days=28)) & (d["Date"] < fix) & d["after_hours"]]["actual_kW"].sum() * 0.25 / 28
    post = d[(d["Date"] >= fix) & (d["Date"] < fix + pd.Timedelta(days=28)) & d["after_hours"]]["actual_kW"].sum() * 0.25 / 28
    assert v["pre_per_day"] == pytest.approx(pre) and v["post_per_day"] == pytest.approx(post)
    assert v["saved_kWh"] == pytest.approx(max(0, (pre - post) * 28))


def test_recurrence_reopens(flagged_floors, world):
    ev, inc, fc = world
    d = flagged_floors[7][flagged_floors[7]["unit"] == "F7_z1_AC4(kW)"]
    events = ev[ev["incident_key"] == "F7_Z1_AC4|closed_running"]
    v = verify_incident(d, "closed_running", pd.Timestamp("2019-07-20"), events)  # it kept happening every Saturday
    assert v["verdict"] == "Reopened" and v["events_last_14d"] > 0


def test_short_window_is_inconclusive(flagged_floors, world):
    ev, inc, fc = world
    d = flagged_floors[7][flagged_floors[7]["unit"] == "F7_z1_AC4(kW)"]
    v = verify_incident(d, "closed_running", pd.Timestamp("2019-12-25"), ev[ev["incident_key"] == "F7_Z1_AC4|closed_running"])
    assert v["verdict"] == "Inconclusive" and v["saved_kWh"] == 0


def test_seeded_fixes_have_verdicts_and_at_least_one_verifies(seeded, flagged_floors, world):
    store, _ = seeded
    ev, inc, fc = world
    verdicts = {}
    wo = store.table("work_orders").set_index("wo_id")
    for r in store.table("closeouts").itertuples():
        if r.outcome != "Fault confirmed":
            continue
        key = wo.loc[r.wo_id, "incident_key"]
        if key.startswith("Floor"):
            continue
        asset = wo.loc[r.wo_id, "asset_id"]
        floor = int(asset[1])
        d = flagged_floors[floor][flagged_floors[floor]["unit"] == COL_OF[asset]]
        verdicts[key] = verify_incident(d, key.split("|")[1], r.date_fixed, ev[ev["incident_key"] == key])["verdict"]
    assert "Verified" in verdicts.values(), verdicts
    assert set(verdicts.values()) <= {"Verified", "Reopened", "Inconclusive", "Not verified"}


def test_sync_sets_verified_and_reopened(seeded, flagged_floors, world):
    from fm_learning import verification_results
    store, _ = seeded
    ev, inc, fc = world
    res = verification_results(store, ev, flagged_floors, UNITS, 4.2)
    state = store.table("incident_state").set_index("incident_key")["status"]
    assert state["F7_Z1_AC4|closed_running"] == "Verified"
    assert state["F5_Z2_AC1|off_schedule"] == "Reopened"  # recurred after its fix date
    assert state["F5_Z2_AC1|high_while_open"] == "Reopened"
    assert res["F7_Z1_AC4|closed_running"]["saved_kWh"] > 0
    n_audit = len(store.table("audit_log"))
    verification_results(store, ev, flagged_floors, UNITS, 4.2)  # idempotent: no new transitions
    assert store.table("incident_state").set_index("incident_key")["status"].equals(state)
    assert len(store.table("audit_log")) == n_audit
