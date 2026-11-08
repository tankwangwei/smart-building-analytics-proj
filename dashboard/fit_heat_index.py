"""Fit each floor's AC sensitivity to heat index ('feels like') for the dashboard's heat-index forecast method, and
score that method on the Apr–Jun 2019 validation period so it can be chosen like the notebook's methods.

Run once from the project folder after run_all_floors.py: `python dashboard/fit_heat_index.py` (about 3 minutes; it reads
combined_by_floor.csv). Writes, in ac_analysis_outputs/:
- heat_index_sensitivity.csv: the slope used for the Jul–Dec 2019 forecast;
- heat_index_validation_metrics.csv: Apr–Jun 2019 accuracy, which the dashboard adds to each floor's validation table;
- heat_index_validation_predictions.csv: its Apr–Jun 2019 predictions, from which the dashboard learns its error bands;
- heat_index_next_24_hours.csv: the method's next-day demonstration, used when validation chooses it.

Method, baseline year only (Jul 2018 – Jun 2019, so nothing from the Jul–Dec 2019 test period is used):
- hourly floor AC load = sum of the floor's AC units;
- expected load = median for that hour of the week (public holidays and closures left out);
- anomaly = heat index minus the average weekday heat index for that hour;
- slope (kW per °C of heat index) = least squares through the origin of (load − expected) on the anomaly, over hours when
  the AC normally runs (expected above 5 % of its maximum).
The temperature slope is fitted the same way, for comparison.

Validation repeats the notebook's split: the slope, the typical weekly pattern and the weather reference levels are all
fitted on Jul 2018 – Mar 2019 only, then scored on Apr–Jun 2019 on exactly the hours the notebook scores every method on
(demand, both lags and weather all present). As a check, the script recomputes the notebook's own typical-weekly-pattern
score on those hours; it stops if that does not match forecast_validation_metrics.csv.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fm_core import CLOSED_DATES, heat_index_c, shift_profile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ac_analysis_outputs"
START = pd.Timestamp("2018-07-01")
VALIDATION_START = pd.Timestamp("2019-04-01")
TEST_START = pd.Timestamp("2019-07-01")
FLOORS = [3, 4, 5, 7]
M1 = "M1 median profile"
HEAT_MODEL = "M1 + heat index (weather)"  # same name as in app.py
MIN_MINUTES_PER_HOUR = 54  # as in the notebook
WEATHER_FEATURES = ["outdoor_temp_C", "outdoor_rh_pct", "rain_mm"]  # the notebook's weather methods need all three


def active_units(floor):
    quality = pd.read_csv(OUT / f"floor_{floor}" / "unit_quality.csv", index_col=0)
    return quality.index[quality["active_selected"].astype(str).str.lower().eq("true")].tolist()


def hourly_loads():
    """(load, demand): `load` is each floor's mean per-minute AC total (used for the slopes); `demand` is the notebook's
    hourly floor demand (sum of active units' hourly means, each needing 54 valid minutes), used for validation."""
    head = pd.read_csv(ROOT / "combined_by_floor.csv", nrows=1)
    cols = {f: [c for c in head.columns if re.match(rf"F{f}_.*AC\d+\(kW\)", c)] for f in FLOORS}
    active = {f: active_units(f) for f in FLOORS}
    load_parts, unit_parts = [], []
    for chunk in pd.read_csv(ROOT / "combined_by_floor.csv", usecols=["Date"] + sum(cols.values(), []),
                             parse_dates=["Date"], chunksize=300_000):
        chunk = chunk.set_index("Date")
        totals = pd.DataFrame({f: chunk[c].sum(axis=1, min_count=len(c)) for f, c in cols.items()})
        load_parts.append(totals.resample("h").agg(["sum", "count"]))
        units = chunk[sum(active.values(), [])]
        units = units.mask(units.lt(0) | ~np.isfinite(units))
        unit_parts.append(units.resample("h").agg(["sum", "count"]))
    # An hour can straddle two chunks, so add the sums and counts before dividing
    load = pd.concat(load_parts).groupby(level=0).sum()
    load = load.xs("sum", axis=1, level=1) / load.xs("count", axis=1, level=1)
    units = pd.concat(unit_parts).groupby(level=0).sum()
    counts = units.xs("count", axis=1, level=1)
    means = (units.xs("sum", axis=1, level=1) / counts).where(counts >= MIN_MINUTES_PER_HOUR)
    demand = pd.DataFrame({f: means[a].sum(axis=1, min_count=len(a)) for f, a in active.items()})
    return load, demand


def slope(load, x, end=TEST_START):
    """kW per unit of x above its usual weekday level for that hour, from baseline hours (before `end`) when the AC
    normally runs."""
    base = load[(load.index < end) & ~load.index.normalize().isin(CLOSED_DATES)]
    hour_of_week = base.index.dayofweek * 24 + base.index.hour
    profile = base.groupby(hour_of_week).median()
    expected = pd.Series(hour_of_week.map(profile).to_numpy(), index=base.index)
    xb = x[(x.index < end) & (x.index.dayofweek < 5)]
    anomaly = x.reindex(base.index) - base.index.hour.map(xb.groupby(xb.index.hour).mean()).to_numpy()
    use = (expected > 0.05 * profile.max()) & base.notna() & anomaly.notna()
    r, a = (base - expected)[use], anomaly[use]
    return float((r * a).sum() / (a ** 2).sum()), int(use.sum())


def metrics(actual, prediction, peak_threshold):
    """The notebook's metrics_table columns, for one method."""
    error = prediction - actual
    rmse = np.sqrt(np.mean(error ** 2))
    peak = actual >= peak_threshold
    return {"hours": len(actual), "MAE_kW": error.abs().mean(), "RMSE_kW": rmse,
            "NMBE_pct": 100 * error.mean() / actual.mean(), "CV_RMSE_pct": 100 * rmse / actual.mean(),
            "peak_hour_MAE_kW": error.loc[peak].abs().mean()}


def validate(floor, load, demand, weather, heat_index):
    """Apr–Jun 2019 accuracy of the heat-index method, fitted on Jul 2018 – Mar 2019 only."""
    history = demand.loc[(demand.index >= START) & (demand.index < TEST_START)]
    fit = history.loc[history.index < VALIDATION_START].dropna()
    valid_index = history.index[history.index >= VALIDATION_START]
    medians = fit.groupby(fit.index.dayofweek * 24 + fit.index.hour).median()
    m1 = pd.Series(valid_index.dayofweek * 24 + valid_index.hour, index=valid_index).map(medians)
    hi_slope, _ = slope(load, heat_index, end=VALIDATION_START)
    heat = shift_profile(m1, heat_index, max(0.0, hi_slope), VALIDATION_START)
    # The hours the notebook scores every method on: demand, both lags and all weather inputs present
    common = (history.reindex(valid_index).notna() & m1.notna()
              & history.reindex(valid_index - pd.Timedelta(hours=24)).notna().to_numpy()
              & history.reindex(valid_index - pd.Timedelta(hours=168)).notna().to_numpy()
              & weather[WEATHER_FEATURES].reindex(valid_index).notna().all(axis=1))
    actual = history.reindex(valid_index)[common]
    peak_threshold = fit.quantile(.9)
    check = metrics(actual, m1[common], peak_threshold)
    notebook = pd.read_csv(OUT / f"floor_{floor}" / "forecast_validation_metrics.csv", index_col=0).loc[M1]
    if check["hours"] != notebook["hours"] or not np.isclose(check["MAE_kW"], notebook["MAE_kW"], rtol=1e-6):
        raise SystemExit(f"Floor {floor}: could not reproduce the notebook's validation ({check['hours']} hours, "
                         f"M1 MAE {check['MAE_kW']:.4f}) vs ({int(notebook['hours'])} hours, {notebook['MAE_kW']:.4f}). "
                         "Re-run run_all_floors.py first.")
    return ({"floor": floor, "model": HEAT_MODEL, **metrics(actual, heat[common], peak_threshold),
             "kW_per_degC_heat_index_to_Mar": round(hi_slope, 4)}, heat.rename(f"floor_{floor}"))


def next_day(floor, demand, heat_index, hi_slope):
    """The heat-index method's 24-hour demonstration for the day after the data ends, issued the way the notebook issues
    its own (typical weekly pattern from every hour completed by midnight). The unshifted pattern must match the
    notebook's next_24_hours_forecast.csv wherever that file used the typical weekly pattern."""
    history = demand.loc[demand.index >= START]
    issue_time = history.last_valid_index().normalize() + pd.Timedelta(days=1)
    future = pd.date_range(issue_time, periods=24, freq="h")
    available = history.loc[history.index + pd.Timedelta(hours=1) <= issue_time].dropna()
    medians = available.groupby(available.index.dayofweek * 24 + available.index.hour).median()
    m1 = pd.Series(future.dayofweek * 24 + future.hour, index=future).map(medians)
    notebook = pd.read_csv(OUT / f"floor_{floor}" / "next_24_hours_forecast.csv", index_col=0, parse_dates=True)
    same = notebook["method"].str.startswith(M1)
    if not np.allclose(m1[same], notebook.loc[same, "predicted_kW"]):
        raise SystemExit(f"Floor {floor}: the next-day typical weekly pattern does not match the notebook's.")
    return shift_profile(m1, heat_index, max(0.0, hi_slope), TEST_START).rename(floor)


def main():
    weather = pd.read_csv(OUT / f"floor_{FLOORS[0]}" / "weather_hourly_local.csv", index_col=0, parse_dates=True)
    heat_index = pd.Series(heat_index_c(weather["outdoor_temp_C"], weather["outdoor_rh_pct"]), index=weather.index)
    load, demand = hourly_loads()
    rows, validation, predictions, next24 = [], [], [], []
    for f in FLOORS:
        hi_slope, hours = slope(load[f], heat_index)
        temp_slope, _ = slope(load[f], weather["outdoor_temp_C"])
        rows.append({"floor": f, "kW_per_degC_heat_index": round(hi_slope, 4),
                     "kW_per_degC_temperature": round(temp_slope, 4), "fit_hours": hours})
        score, prediction = validate(f, load[f], demand[f], weather, heat_index)
        validation.append(score)
        predictions.append(prediction)
        next24.append(next_day(f, demand[f], heat_index, hi_slope))
    table = pd.DataFrame(rows).set_index("floor")
    table.to_csv(OUT / "heat_index_sensitivity.csv")
    print(table.to_string())
    scores = pd.DataFrame(validation).set_index("floor")
    scores.to_csv(OUT / "heat_index_validation_metrics.csv")
    print(scores.round(3).to_string())
    valid = pd.concat(predictions, axis=1)
    valid.index.name = "Date"
    valid.to_csv(OUT / "heat_index_validation_predictions.csv")
    demo = pd.concat(next24, axis=1).add_prefix("floor_")
    demo.index.name = "Date"
    demo.to_csv(OUT / "heat_index_next_24_hours.csv")
    print(demo.round(1).to_string())


if __name__ == "__main__":
    main()
