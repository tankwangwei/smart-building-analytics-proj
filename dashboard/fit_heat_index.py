"""Fit each floor's AC sensitivity to heat index ('feels like') for the dashboard's heat-index forecast method.

Run once from the project folder after run_all_floors.py: `python dashboard/fit_heat_index.py` (about 3 minutes; it reads
combined_by_floor.csv). Writes ac_analysis_outputs/heat_index_sensitivity.csv, which the dashboard reads.

Method, baseline year only (Jul 2018 – Jun 2019, so nothing from the Jul–Dec 2019 test period is used):
- hourly floor AC load = sum of the floor's AC units;
- expected load = median for that hour of the week (public holidays and closures left out);
- anomaly = heat index minus the average weekday heat index for that hour;
- slope (kW per °C of heat index) = least squares through the origin of (load − expected) on the anomaly, over hours when
  the AC normally runs (expected above 5 % of its maximum).
The temperature slope is fitted the same way, for comparison.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fm_core import CLOSED_DATES, heat_index_c  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ac_analysis_outputs"
TEST_START = pd.Timestamp("2019-07-01")
FLOORS = [3, 4, 5, 7]


def hourly_floor_load():
    head = pd.read_csv(ROOT / "combined_by_floor.csv", nrows=1)
    cols = {f: [c for c in head.columns if re.match(rf"F{f}_.*AC\d+\(kW\)", c)] for f in FLOORS}
    parts = []
    for chunk in pd.read_csv(ROOT / "combined_by_floor.csv", usecols=["Date"] + sum(cols.values(), []),
                             parse_dates=["Date"], chunksize=300_000):
        chunk = chunk.set_index("Date")
        parts.append(pd.DataFrame({f: chunk[c].sum(axis=1, min_count=len(c)) for f, c in cols.items()}))
    return pd.concat(parts).resample("h").mean()


def slope(load, x):
    """kW per unit of x above its usual weekday level for that hour, from baseline hours when the AC normally runs."""
    base = load[(load.index < TEST_START) & ~load.index.normalize().isin(CLOSED_DATES)]
    hour_of_week = base.index.dayofweek * 24 + base.index.hour
    profile = base.groupby(hour_of_week).median()
    expected = pd.Series(hour_of_week.map(profile).to_numpy(), index=base.index)
    xb = x[(x.index < TEST_START) & (x.index.dayofweek < 5)]
    anomaly = x.reindex(base.index) - base.index.hour.map(xb.groupby(xb.index.hour).mean()).to_numpy()
    use = (expected > 0.05 * profile.max()) & base.notna() & anomaly.notna()
    r, a = (base - expected)[use], anomaly[use]
    return float((r * a).sum() / (a ** 2).sum()), int(use.sum())


def main():
    weather = pd.read_csv(OUT / f"floor_{FLOORS[0]}" / "weather_hourly_local.csv", index_col=0, parse_dates=True)
    heat_index = pd.Series(heat_index_c(weather["outdoor_temp_C"], weather["outdoor_rh_pct"]), index=weather.index)
    load = hourly_floor_load()
    rows = []
    for f in FLOORS:
        hi_slope, hours = slope(load[f], heat_index)
        temp_slope, _ = slope(load[f], weather["outdoor_temp_C"])
        rows.append({"floor": f, "kW_per_degC_heat_index": round(hi_slope, 4),
                     "kW_per_degC_temperature": round(temp_slope, 4), "fit_hours": hours})
    table = pd.DataFrame(rows).set_index("floor")
    table.to_csv(OUT / "heat_index_sensitivity.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
