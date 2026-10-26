"""Phase 8: per-unit forecast, what-if arrays and agreement with the simulator maths."""
import numpy as np
import pandas as pd
import pytest

from conftest import FLOORS, OUT, read_floor
from fm_core import INTERVAL_HOURS
from fm_sim import Settings, prepare, simulate_series, summarize
from fm_twin import unit_forecasts


def sens(f):
    return pd.read_csv(OUT / f"floor_{f}" / "setpoint_sensitivity.csv", index_col=0)["kW_per_degC_outdoor"].clip(lower=0)


def day_frame(flagged_floors, f, day):
    d = flagged_floors[f]
    return d[d["Date"].dt.normalize() == pd.Timestamp(day)]


def test_unit_forecasts_sum_to_floor_forecast(flagged_floors):
    for f in FLOORS:
        d, _ = read_floor(f)
        fc = pd.read_csv(OUT / f"floor_{f}" / "forecast_hourly_actual_predictions.csv", index_col=0, parse_dates=True)["M1 median profile"]
        slots = pd.date_range("2019-10-07", periods=96 * 7, freq="15min")
        per_unit = unit_forecasts(d, fc, slots)
        total = sum(per_unit.values())
        expected = fc.reindex(slots.floor("h")).to_numpy()
        ok = ~np.isnan(expected)
        assert np.allclose(total.to_numpy()[ok], expected[ok], atol=1e-6), f
        assert all((v.dropna() >= 0).all() for v in per_unit.values())


def test_unit_forecast_shares_use_expected_load(flagged_floors):
    f = 7
    d, _ = read_floor(f)
    fc = pd.read_csv(OUT / "floor_7" / "forecast_hourly_actual_predictions.csv", index_col=0, parse_dates=True)["M1 median profile"]
    slots = pd.date_range("2019-10-08", periods=96, freq="15min")  # a Tuesday
    per_unit = unit_forecasts(d, fc, slots)
    noon = slots.get_loc(pd.Timestamp("2019-10-08 10:00"))
    big = max(per_unit, key=lambda u: per_unit[u].iloc[noon])
    assert big == "F7_z1_AC4(kW)"  # the unit with the largest baseline load gets the largest share


def test_twin_whatif_equals_simulator_for_the_day(flagged_floors):
    """Saved kWh for a day from the twin arrays equals the simulator tab's maths on that same day."""
    for f, day in [(7, "2019-10-08"), (4, "2019-08-14")]:
        frame = day_frame(flagged_floors, f, day)
        s = Settings(setpoint=25.0, start_h=7.0, stop_h=18.0)
        sim = prepare(frame, sens(f), 0.5)
        series = simulate_series(sim, s)
        sp, sch, comfort = summarize(series, s)
        saved_from_arrays = (series["actual_kW"] - series["sim_kW"]).sum() * INTERVAL_HOURS
        assert saved_from_arrays == pytest.approx(sp + sch)
        assert sch > 0 or sp != 0


def test_lower_setpoint_costs_energy_and_window_saves(flagged_floors):
    frame = day_frame(flagged_floors, 7, "2019-10-08")
    sim = prepare(frame, sens(7), 0.5)
    warm = summarize(simulate_series(sim, Settings(setpoint=28)), Settings(setpoint=28))[0]
    cold = summarize(simulate_series(sim, Settings(setpoint=18)), Settings(setpoint=18))[0]
    assert warm > 0 > cold
    nowin = simulate_series(sim, Settings(setpoint=sim["current_sp"].median(), start_h=0, stop_h=24))
    assert summarize(nowin, Settings())[1] == 0
