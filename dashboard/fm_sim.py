"""Setpoint and schedule what-if maths, shared by the simulator tab and the digital twin (plan §10 B). Streamlit-free.

simulate_series() gives, per unit and 15-minute interval, the simulated kW and indoor °C. Both views call it, so
they always agree."""
from dataclasses import dataclass

import numpy as np

from fm_core import CLOSED_DATES, INTERVAL_HOURS

DEFAULTS = {"sim_setpoint": 25.0, "sim_band": (23.0, 26.0), "sim_comfort_target": 70, "sim_window": (0.0, 24.0),
            "sim_weekend_on": True, "sim_holidays_off": False}


@dataclass(frozen=True)
class Settings:
    setpoint: float = 25.0
    band: tuple = (23.0, 26.0)
    start_h: float = 0.0
    stop_h: float = 24.0
    weekend_on: bool = True
    holidays_off: bool = False


def settings_from_state(state):
    """Settings from st.session_state (or any dict), with the simulator's defaults."""
    g = lambda k: state.get(k, DEFAULTS[k])
    return Settings(float(g("sim_setpoint")), tuple(g("sim_band")), float(g("sim_window")[0]), float(g("sim_window")[1]),
                    bool(g("sim_weekend_on")), bool(g("sim_holidays_off")))


def prepare(detail, slope_by_unit, on_kw):
    """The simulator's working frame: valid intervals with each unit's slope and current effective setpoint (the median
    indoor temperature while it runs, over the frame given)."""
    sim = detail.loc[detail["eligible"] & detail["actual_kW"].notna(), ["Date", "unit", "actual_kW", "indoor_temp_C"]].copy()
    sim["running"] = sim["actual_kW"].gt(on_kw)
    sim["slope"] = sim["unit"].map(slope_by_unit).fillna(0)
    sim["current_sp"] = sim["unit"].map(sim.loc[sim["running"]].groupby("unit")["indoor_temp_C"].median())
    return sim


def simulate_series(sim, s, current_sp=None):
    """Adds avoided, active, shift, setpoint_saved_kW, schedule_saved_kW, saved_kW, sim_kW, sim_temp to a copy of `sim`.
    A setpoint below today's level costs energy (negative saving); energy outside the window counts as avoided."""
    out = sim.copy()
    if current_sp is not None:
        out["current_sp"] = out["unit"].map(current_sp)
    hour = out["Date"].dt.hour + out["Date"].dt.minute / 60
    weekend = out["Date"].dt.dayofweek >= 5
    in_window = hour.ge(s.start_h) & hour.lt(s.stop_h)
    outside = ((~in_window & ~weekend) | (weekend & (not s.weekend_on))
               | (s.holidays_off & out["Date"].dt.normalize().isin(CLOSED_DATES)))
    out["avoided"] = out["running"] & outside
    out["active"] = out["running"] & ~out["avoided"] & out["indoor_temp_C"].notna()
    out["shift"] = (s.setpoint - out["current_sp"]).where(out["active"], 0)
    out["setpoint_saved_kW"] = np.minimum(out["slope"] * out["shift"], out["actual_kW"]).where(~out["avoided"], 0.0)
    out["schedule_saved_kW"] = out["actual_kW"].where(out["avoided"], 0.0)
    out["saved_kW"] = out["setpoint_saved_kW"] + out["schedule_saved_kW"]
    out["sim_kW"] = out["actual_kW"] - out["saved_kW"]
    out["sim_temp"] = out["indoor_temp_C"] + out["shift"]
    return out


def summarize(series, s):
    """(setpoint kWh, schedule kWh, comfort % of active running time inside the band)."""
    sp = series["setpoint_saved_kW"].sum() * INTERVAL_HOURS
    sch = series["schedule_saved_kW"].sum() * INTERVAL_HOURS
    temp = series.loc[series["active"], "sim_temp"]
    comfort = 100 * temp.between(*s.band).mean() if len(temp) else np.nan
    return sp, sch, comfort


def twin_arrays(series, slots):
    """{unit: (sim_kW array, sim_temp array)} aligned to the twin's 15-minute slots."""
    out = {}
    for unit, part in series.groupby("unit"):
        p = part.set_index("Date").reindex(slots)
        out[unit] = (p["sim_kW"], p["sim_temp"])
    return out
