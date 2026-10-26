"""Phase 4: work orders."""
import io

import pandas as pd
import pytest

from brick_model import BrickModel
from conftest import FLOORS, OPEN_DEFAULT, OUT
from fm_incidents import all_events, build_incidents
from fm_store import SqliteStore
from fm_workorders import (add_working_days, draft_work_order, is_working_day, next_working_day, printable_html,
                           raise_work_order, suggest_slot, to_csv, to_json)


@pytest.fixture(scope="module")
def incidents(flagged_floors):
    ev = all_events(flagged_floors, BrickModel.load().units())
    return ev, build_incidents(ev)


@pytest.fixture(scope="module")
def forecast():
    p = pd.read_csv(OUT / "floor_7" / "forecast_hourly_actual_predictions.csv", index_col=0, parse_dates=True)
    return p["M1 median profile"]


def test_working_days_skip_weekends_and_holidays():
    # Fri 11 Oct 2019 -> next working day is Mon 14 Oct, which is a substitution holiday -> Tue 15 Oct
    assert next_working_day("2019-10-11", OPEN_DEFAULT) == pd.Timestamp("2019-10-15")
    assert not is_working_day("2019-10-23", OPEN_DEFAULT)
    assert add_working_days("2019-10-21", 2, OPEN_DEFAULT) == pd.Timestamp("2019-10-24")  # Tue 22 (1), Wed 23 holiday, Thu 24 (2)
    no_hol = OPEN_DEFAULT[:3] + (False,)
    assert is_working_day("2019-10-23", no_hol)


def test_slot_is_in_opening_hours_on_a_working_day(forecast):
    for after in pd.date_range("2019-07-01", "2020-01-10", freq="3D"):
        start, end, kw = suggest_slot(forecast, after, OPEN_DEFAULT)
        assert is_working_day(start, OPEN_DEFAULT) and start > after
        assert start.hour * 60 >= OPEN_DEFAULT[1] and end.hour * 60 + end.minute <= OPEN_DEFAULT[2]
        assert kw >= 0


def test_slot_picks_the_lowest_forecast_hour(forecast):
    start, end, kw = suggest_slot(forecast, "2019-08-05", OPEN_DEFAULT)
    day = forecast[forecast.index.normalize() == start.normalize()]
    inside = day[(day.index.hour >= 8) & (day.index.hour <= 16)]
    assert kw == pytest.approx(inside.min())


def test_draft_has_every_field(incidents, forecast):
    ev, inc = incidents
    row = inc[inc["incident_key"] == "F7_Z1_AC4|closed_running"].iloc[0]
    unit = BrickModel.load().asset_for_column("F7_z1_AC4(kW)")
    wo = draft_work_order(row, pd.Timestamp("2020-01-01"), OPEN_DEFAULT, forecast, active_days=20, unit_info=unit)
    for k in ["incident_key", "asset_id", "priority", "due_date", "slot_start", "slot_end", "symptom", "checklist_json",
              "evidence_json"]:
        assert wo[k]
    assert wo["status"] == "Draft" and len(wo["checklist_json"]) == 3
    assert wo["evidence_json"]["cost_of_delay_THB_per_day"] > 0
    assert "assumed" in wo["evidence_json"]["location"]["supply_group"]
    assert pd.Timestamp(wo["due_date"]) >= pd.Timestamp(wo["slot_start"]).normalize()


def test_raise_saves_and_moves_incident(tmp_path, incidents, forecast):
    ev, inc = incidents
    store = SqliteStore(tmp_path / "fm.db")
    row = inc.iloc[0]
    draft = draft_work_order(row, pd.Timestamp("2019-12-31"), OPEN_DEFAULT, forecast, 10)
    wo = raise_work_order(store, draft)
    assert wo["wo_id"] == "WO-0001"
    assert store.get("incident_state", row["incident_key"])["status"] == "Work order raised"
    again = raise_work_order(store, draft)  # no duplicate for the same incident
    assert again["wo_id"] == "WO-0001" and len(store.table("work_orders")) == 1


def test_exports(tmp_path, incidents, forecast):
    ev, inc = incidents
    store = SqliteStore(tmp_path / "fm.db")
    for i in range(2):
        raise_work_order(store, draft_work_order(inc.iloc[i], pd.Timestamp("2019-12-31"), OPEN_DEFAULT, forecast, 10))
    wos = store.table("work_orders")
    raw = to_csv(wos)
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM so Excel reads UTF-8
    back = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    assert list(back["WO ID"]) == ["WO-0001", "WO-0002"] and "Checklist" in back.columns
    assert len(__import__("json").loads(to_json(wos))) == 2
    assert "WO-0001" in printable_html(store.get("work_orders", "WO-0001"))
