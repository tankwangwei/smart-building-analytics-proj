"""Phase 7 (asset card data) and Phase 9 (maintenance log, health, risk, RUL)."""
import numpy as np
import pandas as pd
import pytest

from brick_model import BrickModel
from fm_asset import monthly_comfort, monthly_kw_per_degc, monthly_peer_gap, timeline
from fm_health import peer_weekly as _peer_weekly
from fm_health import (DEMO_LABEL, fault_risk_model, fleet_health, likely_causes, rul_degradation, rul_usage,
                       seed_maintenance, weekly_indicators)
from fm_store import SqliteStore

UNITS = BrickModel.load().units()
AS_OF = pd.Timestamp("2019-12-31")


def peer_weekly(flagged_floors):
    return _peer_weekly(flagged_floors, BrickModel.load().peer_groups())


@pytest.fixture()
def store(tmp_path):
    return SqliteStore(tmp_path / "fm.db")


def test_seed_is_reproducible_and_simulated(tmp_path):
    a, b = SqliteStore(tmp_path / "a.db"), SqliteStore(tmp_path / "b.db")
    for s in (a, b):
        seed_maintenance(s, UNITS, warm_assets={"F7_Z5_AC1"}, rising_assets={"F3_Z1_AC4"}, seed=42)
    ta, tb = a.table("maintenance_log").drop(columns="log_id"), b.table("maintenance_log").drop(columns="log_id")
    pd.testing.assert_frame_equal(ta, tb)
    assert (ta["simulated"] == 1).all() and len(ta) > 16 * 4
    assert pd.to_datetime(ta["date"]).between("2018-07-01", "2019-12-31").all()
    assert seed_maintenance(a, UNITS) == 0  # once
    f7 = ta[ta["asset_id"] == "F7_Z5_AC1"]
    assert {"dirty filter", "refrigerant leak"} <= set(f7["fault_type"])


def test_indicators_and_health(flagged_floors):
    d = flagged_floors[7]
    d = d[d["unit"] == "F7_z1_AC4(kW)"]
    ind = weekly_indicators(d)
    assert {"run_hours", "starts", "efficiency", "drift", "warm_hours"} <= set(ind.columns)
    assert (ind["run_hours"] >= 0).all()


def test_fleet_health_covers_all_units_and_rul_never_negative(flagged_floors, store):
    seed_maintenance(store, UNITS, warm_assets={"F7_Z5_AC1"})
    inds = {}
    pw = peer_weekly(flagged_floors)
    table, ind_all = fleet_health(UNITS, flagged_floors, store, AS_OF, None, pw)
    assert len(table) == 16
    assert table["health"].between(0, 100).all()
    assert (table["days_to_service"] >= 0).all()
    assert (table["rul_deg_days"].dropna() >= 0).all()
    assert table["degradation"].isin(["degrading", "no degradation detected", "not enough data"]).all()


def test_rul_degradation_on_synthetic_trend():
    idx = pd.date_range("2019-07-01", periods=20, freq="W-MON")
    rng = np.random.default_rng(0)
    ind = pd.DataFrame({"efficiency": 1.0 + 0.02 * np.arange(20) + rng.normal(0, 0.01, 20), "run_hours": 30.0}, index=idx)
    r = rul_degradation(ind, idx[0], idx[-1])
    assert r["status"] == "degrading" and r["rul_days"] >= 0 and r["low"] <= r["rul_days"] <= r["high"]
    flat = ind.assign(efficiency=1.0 + rng.normal(0, 0.01, 20))
    assert rul_degradation(flat, idx[0], idx[-1])["status"] == "no degradation detected"
    assert rul_usage(ind, idx[0], idx[-1] + pd.Timedelta(days=200))[0] == 0  # overdue is 0, never negative


def test_fault_risk_reports_time_split_auc(flagged_floors, store):
    seed_maintenance(store, UNITS, warm_assets={"F7_Z5_AC1", "F7_Z1_AC4"})
    pw = peer_weekly(flagged_floors)
    _, ind_all = fleet_health(UNITS, flagged_floors, store, AS_OF, None, pw)
    model, _, metrics, table = fault_risk_model(ind_all, store)
    assert model is not None and metrics["label"] == DEMO_LABEL
    assert 0 <= metrics["auc"] <= 1 and metrics["test_weeks"] > 3 and metrics["train_weeks"] > 5
    assert len(table) == 16 and table["risk"].between(0, 1).all()


def test_likely_causes_distribution(store):
    seed_maintenance(store, UNITS, warm_assets={"F7_Z5_AC1"})
    c = likely_causes(store, "warm_running", "F7_Z5_AC1")
    assert abs(sum(likely_causes(store, "warm_running", top=10).values()) - 1) < 0.01
    assert set(c) <= {"dirty filter", "refrigerant leak", "compressor fault", "fan motor"}


def test_asset_card_series(flagged_floors):
    d = flagged_floors[7][flagged_floors[7]["unit"] == "F7_z1_AC4(kW)"]
    m = monthly_kw_per_degc(d)
    assert len(m) >= 4 and m["kW_per_degC"].notna().all()
    assert len(monthly_comfort(d)) == 6
    peers = BrickModel.load().peer_groups()["z1_AC4"]
    g = monthly_peer_gap(flagged_floors, "F7_z1_AC4(kW)", peers)
    assert len(g) >= 4


def test_timeline(flagged_floors, store):
    seed_maintenance(store, UNITS)
    from fm_incidents import all_events
    ev = all_events(flagged_floors, UNITS)
    t = timeline("F7_Z1_AC4", ev, store)
    assert {"Flagged episode", "Maintenance"} <= set(t["kind"])
