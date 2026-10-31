"""Multi-floor AC dashboard: priority list, setpoint/schedule simulator, forecasts and floor comparison.

Run `python run_all_floors.py` first (it writes ac_analysis_outputs/floor_<n>/), then:
    streamlit run dashboard/app.py
"""
import datetime as dt
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fm_maint_ui
import fm_tabs
import twin
from brick_model import SUPPLY_NOTE, get_model
from forecast_checks import ashrae_rows, breakdowns, interval_bounds
from fm_asset import monthly_kw_per_degc
from fm_health import attach_risk, fault_risk_model, fleet_health, likely_causes, peer_weekly, seed_maintenance
from fm_incidents import STATUS_COLOUR, all_events, build_incidents
from fm_learning import exceptions_tuple, history_note, seed_demo, track_record, verification_results
from fm_sim import Settings, prepare, settings_from_state, simulate_series, summarize, twin_arrays
from fm_store import get_store
from fm_twin import unit_forecasts
from fm_core import (CLOSED_DATES, COMFORT_BAND, DEFAULT_TARIFF_THB, INTERVAL_HOURS, PATTERNS, PRIORITY_LEVELS, add_levels,
                     build_events, building_closed, flag_detail, priority_levels)
from thai_holidays import HOLIDAYS, OBSERVED_CLOSURES, closure_name

ROOT = Path(os.environ.get("AC_OUTPUT_FOLDER", Path(__file__).resolve().parents[1] / "ac_analysis_outputs"))
FLOORS = sorted(int(p.name.split("_")[1]) for p in ROOT.glob("floor_*") if (p / "anomaly_15min_detail.csv").exists())
ALL = "all"  # selector value for the whole-building view
DEFAULT_SCOPE = ALL
SCOPE = None  # set from the sidebar; unit labels include the floor in the whole-building view
TEST_START = pd.Timestamp("2019-07-01")
WEATHER_MODEL = "M1 + weather"

# Roles, not ranks: the same colour always means the same thing on every chart.
ACTUAL = "#2a78d6"
EXPECTED = "#eb6834"
FLAG = "#e34948"
MUTED = "#8a8984"
GOOD = "#1baf7a"
UNIT_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MODEL_LABELS = {"M0 yesterday": "Same hour yesterday", "M0 last week": "Same hour last week",
                "M1 median profile": "Typical weekly pattern", "M2 calendar regression": "Time-based regression",
                "M3 regression with lags": "Regression + past demand", "M4 Random Forest": "Random Forest",
                "M5 regression + weather": "Regression + past demand + weather",
                "M6 Random Forest + weather": "Random Forest + weather",
                WEATHER_MODEL: "Typical weekly pattern + weather",
                "Selected per floor": "Best method per floor (combined)"}

st.set_page_config(page_title="Building AC Dashboard", page_icon="❄️", layout="wide")
# Dark "control room" look (theme colours live in .streamlit/config.toml): glow background, rounded glass cards,
# round-icon KPIs, plus the key-finding box
st.markdown("""<style>
.stApp {background: radial-gradient(1100px 650px at 50% -5%, #1a3563 0%, #0d1a30 45%, #0a1322 75%) fixed;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: #0c172b; border-right: 1px solid rgba(140,170,220,.12);}
.block-container {padding-top: 3.25rem; padding-bottom: 3rem; max-width: 1600px;}
h1 {font-size: 1.55rem !important; padding: 0 0 .2rem 0 !important; letter-spacing: -.01em;}
h3 {font-size: 1.05rem !important; padding: 1rem 0 .1rem 0 !important;}
[data-testid="stVerticalBlock"] {gap: .7rem;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
[data-testid="stMetricLabel"] p {font-size: .85rem; color: #9fb0cc;}
[data-testid="stSidebarContent"] h2 {font-size: 1.05rem; padding-top: .6rem;}
[data-baseweb="tab-panel"] {padding-top: 1rem;}
[data-baseweb="tab"] {font-size: .95rem;}
[class*="st-key-card"] {background: rgba(17, 30, 54, .72); border: 1px solid rgba(140,170,220,.18); border-radius: 18px;
                        padding: 16px 18px; backdrop-filter: blur(6px);}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"], iframe {border-radius: 14px;}
.card-h {font-size: 1rem; font-weight: 600; margin-bottom: .4rem;}
.card-f {font-size: .8rem; color: #9fb0cc; text-align: right; margin-top: .3rem;}
.kpis {display: flex; gap: 34px; flex-wrap: wrap; justify-content: center; margin: .3rem 0 .9rem;}
.kpi {display: flex; align-items: center; gap: 12px;}
.ico {width: 46px; height: 46px; flex: none; border-radius: 50%; border: 1.5px solid rgba(230,237,247,.55);
      display: flex; align-items: center; justify-content: center; color: #e6edf7;}
.ico.sm {width: 38px; height: 38px;}
.kpi .v {font-size: 1.5rem; font-weight: 700; line-height: 1.1;}
.kpi .l, .item .s {font-size: .78rem; color: #9fb0cc;}
.item {display: flex; gap: 12px; align-items: center; padding: .55rem 0; border-bottom: 1px solid rgba(140,170,220,.12);}
.item:last-child {border-bottom: none;}
.item .t {font-size: .88rem; line-height: 1.3;}
.floor-row {display: flex; align-items: center; gap: 10px; margin: .25rem 0;}
.floor-row .fl {width: 26px; font-size: .8rem; color: #9fb0cc;}
.dot {display: inline-block; width: 17px; height: 17px; border-radius: 50%; margin: 2px;}
.legend-dots {font-size: .75rem; color: #9fb0cc; margin-top: .4rem;}
.legend-dots .dot {width: 9px; height: 9px; margin: 0 4px 0 10px;}
.load-line {margin-top: .7rem; padding-top: .6rem; border-top: 1px solid rgba(140,170,220,.12); font-size: .88rem;}
/* Smart assistant: hovering an item highlights it and slides open the reasoning ("why") under the action */
.item.ai {border-radius: 12px; padding: .55rem .5rem; margin: 0 -.5rem; transition: background .2s ease;}
.item.ai:hover {background: rgba(90,169,255,.08);}
.item.ai .why {max-height: 0; opacity: 0; overflow: hidden; transform: translateY(-4px);
               transition: max-height .3s ease, opacity .25s ease, transform .25s ease, margin .3s ease;
               font-size: .78rem; line-height: 1.45; color: #c9d6ea; border-left: 2px solid var(--sev);
               padding-left: .6rem; margin-top: 0;}
.item.ai:hover .why {max-height: 220px; opacity: 1; transform: none; margin-top: .45rem;}
.item.ai .why .meta {display: flex; gap: 6px; align-items: center; margin-top: .4rem; flex-wrap: wrap;}
.item.ai .chip {font-size: .68rem; padding: 1px 8px; border-radius: 999px; border: 1px solid rgba(140,170,220,.3);
                color: #9fb0cc;}
.item.ai .chip.sev {color: var(--sev); border-color: var(--sev); font-weight: 600; letter-spacing: .04em;}
.takeaway {border-left: 3px solid #3b8beb; background: rgba(59,139,235,.10); border-radius: 0 .4rem .4rem 0;
           padding: .55rem .9rem; line-height: 1.5;}
/* "Ask the data": a round button floating in the bottom-right corner that opens the chat */
.st-key-chat-fab {position: fixed; right: 28px; bottom: 26px; z-index: 999990; width: auto !important;}
.st-key-chat-fab [data-testid="stPopoverButton"] {width: 58px; height: 58px; border-radius: 50%; padding: 0;
    background: linear-gradient(145deg, #3b8beb, #2563c9); border: 1px solid rgba(255,255,255,.25);
    box-shadow: 0 8px 24px rgba(0,0,0,.45); font-size: 1.5rem; color: #fff; transition: transform .2s ease, box-shadow .2s ease;}
.st-key-chat-fab [data-testid="stPopoverButton"]:hover {transform: translateY(-2px) scale(1.05);
    box-shadow: 0 10px 30px rgba(59,139,235,.45); color: #fff; border-color: rgba(255,255,255,.4);}
.st-key-chat-fab [data-testid="stPopoverButton"] p {font-size: 1.7rem; line-height: 1;}
.st-key-chat-fab [data-testid="stPopoverButton"] [data-testid="stIconMaterial"] {display: none;}  /* no chevron */
[data-testid="stPopoverBody"]:has(.st-key-chat-panel) {width: min(460px, calc(100vw - 32px)); max-width: none;
    background: #0f1d36; border: 1px solid rgba(140,170,220,.22); border-radius: 18px;}
</style>""", unsafe_allow_html=True)

# Line icons for the KPI chips and card items (white outline, like a building-management console)
ICONS = {
    "thermo": '<path d="M10 14V5a2 2 0 1 1 4 0v9a4 4 0 1 1-4 0z"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2'
           'M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "bolt": '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
    "peak": '<path d="M3 20h18M6 16l4-6 4 3 5-8"/>',
    "bell": '<path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9M10 21a2 2 0 0 0 4 0"/>',
    "lens": '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
    "building": '<path d="M4 21V5l8-3v19M12 9h8v12M7 8h2M7 12h2M7 16h2M15 13h2M15 17h2M2 21h20"/>',
    "snow": '<path d="M12 2v20M3.3 7l17.4 10M3.3 17 20.7 7"/>',
}
LEVEL_DOT = {"🔴 High": "#e5533d", "🟠 Medium": "#f0a03c", "🟢 Low": "#3fb27f"}
OFF_DOT = "#5b6577"  # unit did not run in the timeframe (e.g. a Sunday)
STATUS_DOT = {"Flagged": "#e5533d", "Running": "#3fb27f", "Off": OFF_DOT, "No reading": "#2a3347"}  # Day mode


def icon(name, size=22, small=False):
    svg = (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="currentColor" '
           f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')
    return f'<div class="ico{" sm" if small else ""}">{svg}</div>'


def status_chip(status):
    colour = STATUS_COLOUR[status]
    return (f'<span style="font-size:.68rem;padding:0 7px;border-radius:999px;border:1px solid {colour};'
            f'color:{colour}">{status}</span>')


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def unit_label(unit, with_floor=None):
    if with_floor is None:
        with_floor = SCOPE == ALL
    match = re.search(r"F(\d+)_z(\d+)_AC(\d+)", unit)
    if not match:
        return unit
    label = f"Zone {match[2]} · AC{match[3]}"
    return f"Floor {match[1]} · {label}" if with_floor else label


def brick_model():
    """The Brick graph of the building (rdflib/SPARQL; falls back to the JSON copy if rdflib is missing)."""
    return get_model()


def position_of(unit):
    """Same-position key shared by floors with the same layout, e.g. 'z1_AC4'. Read from the Brick model."""
    asset = brick_model().asset_for_column(unit)
    return asset["position"] if asset else unit.split("_", 1)[1].replace("(kW)", "")


def supply_of(unit):
    """Supply group (assumed chilled-water loop) of a unit, from Brick, e.g. 'Floor7_CHW_assumed'."""
    asset = brick_model().asset_for_column(unit)
    return asset["supply_group"] if asset else "unknown"


def supply_label(group):
    return f"Floor {group.removeprefix('Floor').split('_')[0]} chilled-water supply (assumed)"


def to_bool(series):
    return series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False,
                                               "1.0": True, "0.0": False})


def _r_or_none(value, digits=1):
    return None if value is None or pd.isna(value) else round(float(value), digits)


def table_height(rows, cap):
    """Natural height of a table with this many rows, capped so long tables scroll instead of growing the page."""
    return min((rows + 1) * 35 + 3, cap)


def style(fig, height=380, ytitle=None, xtitle=None, margin_top=40):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=margin_top, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, title=xtitle)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", title=ytitle, rangemode="tozero")
    return fig


def day_axis(fig, days):
    """Label dates with the weekday ('Mon 01 Jul') on the ticks and in the hover. One tick per day up to a week."""
    fig.update_xaxes(tickformat="%a %d %b", hoverformat="%a %d %b %Y, %H:%M", **({"dtick": 86400000} if days <= 8 else {}))
    return fig


# ---------------------------------------------------------------- data
def forecast_metrics(table, models):
    """Test-period accuracy on hours where every model has a value (same rules as the notebook)."""
    common = table[["floor_kW"] + models].dropna()
    actual = common["floor_kW"]
    peak = actual >= actual.quantile(.9)
    rows = []
    for m in models:
        error = common[m] - actual
        rmse = np.sqrt(np.mean(error ** 2))
        rows.append({"model": m, "hours": len(common), "MAE_kW": error.abs().mean(), "RMSE_kW": rmse,
                     "NMBE_pct": 100 * error.mean() / actual.mean(), "CV_RMSE_pct": 100 * rmse / actual.mean(),
                     "peak_hour_MAE_kW": error.loc[peak].abs().mean()})
    return pd.DataFrame(rows).set_index("model").sort_values("MAE_kW")


def add_weather_model(forecast, weather, sensitivity, test_metrics):
    """Typical weekly pattern (M1) shifted by outdoor temperature.

    Each hour's M1 forecast moves by (floor kW per °C) × (outdoor temperature − the baseline weekday average for that
    hour of day). The slope is the sum of the floor's per-unit regression slopes and the reference temperatures come
    from the baseline (before Jul 2019), so nothing from the test period is used to fit it. The shift only applies
    while the AC is expected to be running, so closed hours stay at zero. Observed weather stands in for a forecast."""
    temp = weather["outdoor_temp_C"]
    base = temp.loc[temp.index < TEST_START]
    base = base.loc[base.index.dayofweek < 5]
    reference = base.groupby(base.index.hour).mean()
    slope = sensitivity["kW_per_degC_outdoor"].clip(lower=0).sum()
    m1 = forecast["M1 median profile"]
    anomaly = temp.reindex(forecast.index) - forecast.index.hour.map(reference).to_numpy()
    running = m1 > 0.05 * m1.max()
    forecast[WEATHER_MODEL] = (m1 + np.where(running, slope * anomaly, 0)).clip(lower=0)
    models = [c for c in forecast.columns if c != "floor_kW"]
    new_row = forecast_metrics(forecast, models).loc[[WEATHER_MODEL]]
    return forecast, pd.concat([test_metrics, new_row])


@st.cache_data(show_spinner="Combining all floors…")
def load_building():
    parts = {f: load(f) for f in FLOORS}
    first = parts[FLOORS[0]]
    data = {"detail": pd.concat([p["detail"] for p in parts.values()], ignore_index=True),
            "thresholds": pd.concat([p["thresholds"] for p in parts.values()]),
            "sensitivity": pd.concat([p["sensitivity"] for p in parts.values()]),
            "weather": first["weather"]}

    # Building demand = sum of floor totals; an hour counts only when every floor is valid
    n = len(FLOORS)
    models = [c for c in first["forecast"].columns if c != "floor_kW"]
    forecast = pd.DataFrame({c: pd.concat([p["forecast"][c] for p in parts.values()], axis=1).sum(axis=1, min_count=n)
                             for c in ["floor_kW"] + models})
    chosen = {f: p["validation_metrics"]["MAE_kW"].idxmin() for f, p in parts.items()}
    forecast["Selected per floor"] = pd.concat([parts[f]["forecast"][m] for f, m in chosen.items()], axis=1).sum(axis=1, min_count=n)
    data["forecast"] = forecast
    data["test_metrics"] = forecast_metrics(forecast, models + ["Selected per floor"])
    # Only the combination of each floor's validation choice is "selected" for the building
    data["validation_metrics"] = pd.DataFrame({"MAE_kW": [np.nan]}, index=["Selected per floor"])
    data["selected_model"] = "Selected per floor"
    data["next24"] = pd.DataFrame({"predicted_kW": sum(p["next24"]["predicted_kW"] for p in parts.values()),
                                   "method": "Selected per floor"})

    # Floor energies add up; the building peak needs the summed 15-minute load
    k = pd.DataFrame({f: p["floor_kpis"] for f, p in parts.items()}).T
    energy = k["estimated_energy_kWh_valid_intervals"]
    total_15 = data["detail"].pivot_table(index="Date", columns="unit", values="actual_kW").sum(axis=1, min_count=data["detail"]["unit"].nunique())
    data["floor_kpis"] = pd.Series({
        "estimated_energy_kWh_valid_intervals": energy.sum(),
        "after_hours_energy_pct_valid_intervals": (energy * k["after_hours_energy_pct_valid_intervals"]).sum() / energy.sum(),
        "15min_peak_kW": total_15.max()})
    return data


def load(floor):
    """The whole building or one floor. Not cached itself: each loader has its own cache, so a code change to
    either one refreshes it (a cached dispatcher kept serving stale building data after load_building changed)."""
    return load_building() if floor == ALL else load_floor(floor)


@st.cache_data(show_spinner="Loading analysis outputs…")
def load_floor(floor):
    OUT = ROOT / f"floor_{floor}"
    detail = pd.read_csv(OUT / "anomaly_15min_detail.csv", parse_dates=["Date"])
    for column in ["eligible", "hot_day"]:
        detail[column] = to_bool(detail[column]).fillna(False).astype(bool)
    detail["schedule_on"] = to_bool(detail["schedule_on"])  # True / False / NaN (unknown)
    data = {"detail": detail,
            "thresholds": pd.read_csv(OUT / "anomaly_unit_thresholds.csv", index_col=0),
            "sensitivity": pd.read_csv(OUT / "setpoint_sensitivity.csv", index_col=0),
            "forecast": pd.read_csv(OUT / "forecast_hourly_actual_predictions.csv", index_col=0, parse_dates=True),
            "test_metrics": pd.read_csv(OUT / "forecast_test_metrics.csv", index_col=0),
            "validation_metrics": pd.read_csv(OUT / "forecast_validation_metrics.csv", index_col=0),
            "next24": pd.read_csv(OUT / "next_24_hours_forecast.csv", index_col=0, parse_dates=True),
            "weather": pd.read_csv(OUT / "weather_hourly_local.csv", index_col=0, parse_dates=True),
            "floor_kpis": pd.read_csv(OUT / "floor_reporting_15min_kpis.csv", index_col=0).iloc[:, 0],
            # lighting, plug and lux per zone (15-minute, reporting period) for the digital twin
            "zone_loads": (pd.read_csv(OUT / "zone_loads_15min.csv", index_col=0, parse_dates=True)
                           if (OUT / "zone_loads_15min.csv").exists() else None)}
    data["forecast"], data["test_metrics"] = add_weather_model(
        data["forecast"], data["weather"], data["sensitivity"], data["test_metrics"])
    return data


@st.cache_data(show_spinner=False)
def flags_for(floor, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist, open_spec, exceptions=()):
    """Cached flagging for one view. Every input (including the exceptions tuple) is an argument so caching works."""
    data = load(floor)
    return flag_detail(data["detail"], data["thresholds"], open_spec, k_iqr, k_res, on_kw, persist, exceptions)


def unit_priority(d, events, days):
    g = d.groupby("unit")
    table = pd.DataFrame({"valid_h": g["eligible"].sum() * INTERVAL_HOURS,
                          "flagged_h": g["flagged"].sum() * INTERVAL_HOURS,
                          "after_hours_flagged_h": g["after_hours"].sum() * INTERVAL_HOURS})
    table["flagged_pct"] = 100 * table["flagged_h"] / table["valid_h"].replace(0, np.nan)
    table["flagged_excess_kWh"] = d.loc[d["flagged"]].groupby("unit")["excess_kWh_est"].sum().reindex(table.index, fill_value=0)
    table["excess_THB"] = table["flagged_excess_kWh"] * TARIFF_THB
    table["confidence"] = np.nan
    table["effort"] = "–"
    table["comfort_hit"] = False
    if len(events):
        e = events.groupby("unit")
        table["events"] = e.size().reindex(table.index, fill_value=0)
        table["longest_event_h"] = e["duration_h"].max().reindex(table.index, fill_value=0)
        table["hot_day_events"] = e["hot_day"].sum().reindex(table.index, fill_value=0)
        table["triage_score"] = e["triage_score"].sum().reindex(table.index, fill_value=0)
        # Unit confidence = THB-weighted mean of its events' confidence; effort = the pattern that costs the most
        weight = events["excess_THB"].clip(lower=1e-9)
        table["confidence"] = ((events["confidence"] * weight).groupby(events["unit"]).sum()
                               / weight.groupby(events["unit"]).sum()).reindex(table.index)
        main = events.groupby(["unit", "pattern"])["excess_THB"].sum().reset_index().sort_values("excess_THB")
        main = main.drop_duplicates("unit", keep="last").set_index("unit")["pattern"]
        table["pattern"] = main.reindex(table.index)
        table["effort"] = table["pattern"].map(lambda p: PATTERNS[p][0] if isinstance(p, str) else "–")
        table["comfort_hit"] = e["comfort_hit"].mean().reindex(table.index, fill_value=0).ge(.5)
    else:
        table[["events", "longest_event_h", "hot_day_events", "triage_score"]] = 0
        table["pattern"] = None
    table["per_day_THB"] = table["excess_THB"] / max(1, days)
    table = add_levels(table.sort_values("triage_score", ascending=False))
    table.insert(0, "rank", range(1, len(table) + 1))
    table.insert(1, "AC unit", [unit_label(u) for u in table.index])
    return table


CONFIDENCE_HELP = ("0–100 from five signals: how many of the three rules agree (35%), how long it lasted (25%), whether it "
                   "repeats at the same weekday and hour (20%), whether it is not just a hot day (10%) and data quality "
                   "(10%). High ≥ 70 · Medium 40–69 · Low < 40.")
TRIAGE_HELP = ("Triage score = excess THB × confidence ÷ effort. Effort is how hard the fix is: Low = schedule or timer, "
               "Medium = inspect, High = repair. Priority: 🔴 High = the top items that together make up half of the "
               "total score, 🟠 Medium = the next 30 %, 🟢 Low = the rest.")

# Column settings shared by every priority table
PRIORITY_COLUMNS = {
    "rank": st.column_config.NumberColumn("#", width=35),
    "priority_level": st.column_config.TextColumn("Priority", help=TRIAGE_HELP),
    "share_pct": st.column_config.NumberColumn("Share of score", format="%.0f%%",
                                               help="This item's share of the total triage score in the table."),
    "flagged_excess_kWh": st.column_config.NumberColumn("Excess kWh", format="%.0f",
                                                        help="Energy above the unit's normal pattern during flagged periods (estimate)."),
    "excess_THB": st.column_config.NumberColumn("Excess THB", format="%.0f",
                                                help="Excess kWh × the electricity price (sidebar → Advanced). An estimate."),
    "per_day_THB": st.column_config.NumberColumn("THB per day", format="%.0f",
                                                 help="Excess THB ÷ number of days in the timeframe."),
    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.0f",
                                                  help=CONFIDENCE_HELP),
    "effort": st.column_config.TextColumn("Effort", help="Low = schedule or timer fix · Medium = inspect · High = repair."),
    "comfort_hit": st.column_config.CheckboxColumn("Comfort", help="Ticked when the zone was outside 23–26 °C for at "
                                                   "least half of the flagged time."),
    "triage_score": st.column_config.NumberColumn("Triage score", format="%.0f", help=TRIAGE_HELP),
}


def prepare_sim(detail_frame, slope, on_kw, start_h, stop_h, weekend_on, holidays_off):
    """15-minute rows for the what-if simulator (used by the simulator tab and the chat's what-if tool)."""
    sim = detail_frame.loc[detail_frame["eligible"] & detail_frame["actual_kW"].notna(),
                           ["Date", "unit", "actual_kW", "indoor_temp_C"]].copy()
    hour = sim["Date"].dt.hour + sim["Date"].dt.minute / 60
    weekend = sim["Date"].dt.dayofweek >= 5
    in_window = hour.ge(start_h) & hour.lt(stop_h)
    sim["outside_window"] = ((~in_window & ~weekend) | (weekend & (not weekend_on))
                             | (holidays_off & sim["Date"].dt.normalize().isin(CLOSED_DATES)))
    sim["running"] = sim["actual_kW"].gt(on_kw)
    sim["slope"] = sim["unit"].map(slope).fillna(0)
    # Each zone's current "effective setpoint" = median indoor temperature while its unit runs
    current_sp = sim.loc[sim["running"]].groupby("unit")["indoor_temp_C"].median()
    sim["current_sp"] = sim["unit"].map(current_sp)
    return sim


def simulate_setpoint(sim, sp, band):
    """kWh saved by the new setpoint, kWh saved by the schedule window, and % of running time in the comfort band."""
    avoided = sim["running"] & sim["outside_window"]
    active = sim["running"] & ~avoided & sim["indoor_temp_C"].notna()
    shift = (sp - sim["current_sp"]).where(active, 0)
    setpoint_kwh = np.minimum(sim["slope"] * shift, sim["actual_kW"]).sum() * INTERVAL_HOURS
    schedule_kwh = sim.loc[avoided, "actual_kW"].sum() * INTERVAL_HOURS
    new_temp = sim.loc[active, "indoor_temp_C"] + shift.loc[active]
    comfort = 100 * new_temp.between(*band).mean() if len(new_temp) else np.nan
    return setpoint_kwh, schedule_kwh, comfort


@st.cache_data(show_spinner=False)
def period_kpis(floor, start, end, open_spec):
    """AC energy, energy used while the building is closed, and 15-minute peak for the timeframe."""
    det = load(floor)["detail"]
    det = det.loc[det["Date"].ge(start) & det["Date"].lt(end)]
    valid = det.loc[det["eligible"]]
    energy = valid["actual_kW"].sum() * INTERVAL_HOURS
    after = valid.loc[building_closed(valid["Date"], open_spec), "actual_kW"].sum() * INTERVAL_HOURS
    total = det.pivot_table(index="Date", columns="unit", values="actual_kW", aggfunc="first")
    peak = total.sum(axis=1, min_count=total.shape[1]).max()
    return {"energy_kWh": energy, "after_kWh": after, "after_pct": 100 * after / energy if energy else np.nan,
            "peak_kW": peak}


def section(title, help=None):
    st.subheader(title, help=help)


def takeaway(html):
    """The conclusion of a section, written so it leads into the next one."""
    st.markdown(f'<div class="takeaway">→ <b>So:</b> {html}</div>', unsafe_allow_html=True)



# ---------------------------------------------------------------- page
if not FLOORS:
    st.error(f"No analysis outputs found in `{ROOT}/floor_<n>/`. Run `python run_all_floors.py` first, "
             "or set the AC_OUTPUT_FOLDER environment variable.")
    st.stop()

with st.sidebar:
    scopes = [ALL] + FLOORS
    floor = st.selectbox("View", scopes, index=scopes.index(DEFAULT_SCOPE) if DEFAULT_SCOPE in scopes else 0,
                         format_func=lambda f: "Whole building" if f == ALL else f"Floor {f}",
                         help="Whole building combines Floors " + ", ".join(map(str, FLOORS)) + ". "
                              "Applies to every tab except Compare floors, which always shows each floor.")
SCOPE = floor
SCOPE_NAME = "Building" if floor == ALL else f"Floor {floor}"

data = load(floor)
detail_all = data["detail"]
units = list(dict.fromkeys(detail_all["unit"]))
unit_colour = {u: UNIT_COLOURS[i % len(UNIT_COLOURS)] for i, u in enumerate(units)}
data_first, data_last = detail_all["Date"].min().date(), detail_all["Date"].max().date()

# ---------------------------------------------------------------- header with the timeframe picker
# The timeframe drives every total on every tab ('normal' is always learned from Jul 2018 – Jun 2019, and forecast
# accuracy is always scored on Jul–Dec 2019). It sits above the tabs so it is visible wherever you are.
head, tf_mode_col, tf_date_col = st.columns([3.2, 1.7, 1], vertical_alignment="bottom")
head.title("Chamchuri 5 · AC energy monitor",
           help="Where is air-conditioning energy wasted, and what load should we plan for? Flags are potential anomalies "
                "for review, not diagnosed faults. kWh and THB figures are estimates, not verified savings.")
tf_mode = tf_mode_col.segmented_control(
    "Timeframe", ["Day", "Week", "Month", "Jul–Dec"], default="Jul–Dec", key="tf_mode",
    help="Every number, colour and alert on every tab is worked out for this timeframe. Week = Monday to Sunday and "
         "Month = the calendar month around the date you pick.") or "Jul–Dec"
default_day = min(max(pd.Timestamp("2019-07-27").date(), data_first), data_last)  # a Saturday with clear issues
tf_anchor = pd.Timestamp(tf_date_col.date_input("Date", default_day, min_value=data_first, max_value=data_last,
                                                format="DD/MM/YYYY", key="tf_date", disabled=tf_mode == "Jul–Dec"))
if tf_mode == "Day":
    period_start = tf_anchor
    period_end = period_start + pd.Timedelta(days=1)
elif tf_mode == "Week":
    period_start = tf_anchor - pd.Timedelta(days=tf_anchor.dayofweek)
    period_end = period_start + pd.Timedelta(days=7)
elif tf_mode == "Month":
    period_start = tf_anchor.replace(day=1)
    period_end = period_start + pd.offsets.MonthBegin(1)
else:
    period_start, period_end = pd.Timestamp(data_first), pd.Timestamp(data_last) + pd.Timedelta(days=1)
period_start = max(period_start, pd.Timestamp(data_first))
period_end = min(period_end, pd.Timestamp(data_last) + pd.Timedelta(days=1))
PERIOD_DAYS = (period_end - period_start).days
PERIOD = {"Day": f"{period_start:%a %d %b %Y}", "Month": f"{period_start:%B %Y}", "Jul–Dec": "Jul–Dec 2019"}.get(
    tf_mode, f"{period_start:%d %b} – {period_end - pd.Timedelta(days=1):%d %b %Y}")
scope_desc = (f"Floors {', '.join(map(str, FLOORS[:-1]))} and {FLOORS[-1]} · {len(units)} AC units" if floor == ALL
              else f"{SCOPE_NAME} · {len(units)} AC units")
st.caption(f"**{scope_desc}** · **{period_start:%a %d %b %Y}"
           + (f" – {period_end - pd.Timedelta(days=1):%a %d %b %Y}" if PERIOD_DAYS > 1 else "")
           + f"** ({PERIOD_DAYS} day{'s' if PERIOD_DAYS > 1 else ''})"
           + (f" · 🎌 **{closure_name(period_start.date())}**" if PERIOD_DAYS == 1 and closure_name(period_start.date()) else "")
           + " · normal pattern learned from Jul 2018 – Jun 2019 · "
             "data: CU-BEMS, Chulalongkorn University, Bangkok · weather: Open-Meteo")

with st.sidebar:
    with st.expander("Advanced: anomaly thresholds"):
        st.caption("Defaults: k = 1.5, 1.5, 8 intervals (2 h), 0.5 kW. Every tab updates.")
        k_iqr = st.slider("Statistical rule: IQR multiplier k", 0.5, 5.0, 1.5, 0.25,
                          help="How far above the normal range power must go to count as 'unusually high' for that "
                               "time of day and weekday. A higher k means only extreme spikes get flagged. Default 1.5.")
        k_res = st.slider("Persistence rule: residual IQR multiplier", 0.5, 5.0, 1.5, 0.25,
                          help="How far above the expected power a unit must be to count as 'running high'. "
                               "A higher value means only larger, sustained overuse gets flagged. Default 1.5.")
        persist = st.slider("Persistence: minimum consecutive 15-min intervals", 1, 24, 8,
                            help="How long a unit must stay high before it's flagged. "
                                 "8 intervals × 15 minutes = 2 hours. Default 8.")
        on_kw = st.slider("'Running' threshold (kW)", 0.1, 5.0, 0.5, 0.1,
                          help="Below this power the unit counts as 'off', and nothing below it is ever flagged. "
                               "Default 0.5 kW.")
    with st.expander("Building opening hours"):
        st.caption("Not in the dataset, so set these to the building's real hours. Defaults follow the data: on weekdays "
                   "the AC units run from about 07:45 to 16:45.")
        open_days = st.multiselect("Open on", DAYS, default=DAYS[:5], key="open_days")
        open_from, open_to = st.slider("Opening hours", value=(dt.time(7, 30), dt.time(17, 30)), step=dt.timedelta(minutes=15),
                                       format="HH:mm", key="open_hours",
                                       help="While the building is closed, normal AC use counts as zero: any AC running "
                                            "is flagged as after-hours and all its energy counts as waste.")
        use_holidays = st.checkbox("Closed on Thai public holidays", True, key="use_holidays",
                                   help="Official holidays and substitution days, plus closures seen in the data. Also "
                                        "used by the forecast: a holiday is a known closure, so demand is forecast as zero.")
        if use_holidays:
            in_range = {d: n for d, n in {**HOLIDAYS, **OBSERVED_CLOSURES}.items() if data_first <= d <= data_last}
            st.caption("Closure days in the reporting period: " + "; ".join(f"{d:%d %b} {n.split(' (')[0]}"
                                                                           for d, n in sorted(in_range.items())))
OPEN = (tuple(DAYS.index(x) for x in open_days), open_from.hour * 60 + open_from.minute,
        open_to.hour * 60 + open_to.minute, use_holidays)
OPEN_TEXT = (("Mon–Fri" if open_days == DAYS[:5] else ", ".join(open_days) or "never")
             + f" {open_from:%H:%M}–{open_to:%H:%M}" + (", closed on public holidays" if use_holidays else ""))

with st.sidebar:
    with st.expander("Advanced: electricity price"):
        TARIFF_THB = st.number_input("Electricity price (THB per kWh)", 0.0, 20.0, DEFAULT_TARIFF_THB, 0.1, key="tariff",
                                     help="Assumption. Used for THB figures in triage, work orders and the simulator.")

# All three flagging rules are always on. Items are ranked by the triage score (magnitude × confidence ÷ effort).
use_stat = use_after = use_persist = True


# ---------------------------------------------------------------- FM store, exceptions and incidents (plan phases 3-6)
STORE = get_store()
BRICK_UNITS = brick_model().units()
COL_OF = dict(zip(BRICK_UNITS["asset_id"], BRICK_UNITS["power_column"]))
DEFAULT_OPEN = ((0, 1, 2, 3, 4), 7 * 60 + 30, 17 * 60 + 30, True)
OPEN_STATUSES = ("Open", "Acknowledged", "Work order raised", "Reopened")
with st.sidebar:
    st.caption(f"Work-order storage: {STORE.name}")


def floor_forecasts():
    """Hourly forecast per floor from each floor's chosen method (used to place work orders in low-demand hours)."""
    return {f: load(f)["forecast"][load(f)["validation_metrics"]["MAE_kW"].idxmin()] for f in FLOORS}


@st.cache_data(show_spinner="Building incidents…")
def full_data_events(k_iqr, k_res, on_kw, persist, open_spec, tariff, exceptions, track_items):
    """Events for every floor over the whole data, so incident identity does not depend on the timeframe."""
    details = {f: flags_for(f, k_iqr, k_res, on_kw, persist, True, True, True, open_spec, exceptions) for f in FLOORS}
    return all_events(details, BRICK_UNITS, tariff, dict(track_items))


if STORE.is_empty():
    # First run: about 10 close-outs on real incidents, all labelled Simulated
    _defaults = (1.5, 1.5, 0.5, 8)
    _details = {f: flags_for(f, *_defaults, True, True, True, DEFAULT_OPEN, ()) for f in FLOORS}
    _ev = full_data_events(*_defaults, DEFAULT_OPEN, DEFAULT_TARIFF_THB, (), ())
    _inc = build_incidents(_ev)
    seed_demo(STORE, _ev, _inc, _details, BRICK_UNITS, DEFAULT_OPEN, floor_forecasts(), pd.Timestamp(data_last))
    # 18-month simulated maintenance history, consistent with what the data shows (fixed seed)
    _warm = set(_inc.loc[_inc["pattern"].eq("warm_running"), "asset_id"])
    _rising = set()
    for _u in BRICK_UNITS.itertuples():
        _m = monthly_kw_per_degc(_details[int(_u.floor)].query("unit == @_u.power_column"))
        if len(_m) >= 4 and _m["kW_per_degC"].iloc[-1] > 1.15 * _m["kW_per_degC"].iloc[0] > 0:
            _rising.add(_u.asset_id)
    seed_maintenance(STORE, BRICK_UNITS, _warm, _rising, seed=42)

EXCEPTIONS = exceptions_tuple(STORE, COL_OF)  # confirmed legitimate-use exceptions: matching intervals are never flagged
TRACK = track_record(STORE, COL_OF)           # close-out track record: adjusts confidence
TRACK_ITEMS = tuple(sorted(TRACK.items(), key=str))


def in_period(frame):
    return frame.loc[frame["Date"].ge(period_start) & frame["Date"].lt(period_end)]


# ---------------------------------------------------------------- results shared by every tab
d = in_period(flags_for(floor, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist, OPEN, EXCEPTIONS))
detail = in_period(detail_all)
events = build_events(d, TARIFF_THB, track_record=TRACK)
priority = unit_priority(d, events, PERIOD_DAYS)
kpis = period_kpis(floor, period_start, period_end, OPEN)
sens = data["sensitivity"]

# Incidents: one per asset and pattern over the whole data; the timeframe only decides which are active
EV_FULL = full_data_events(k_iqr, k_res, on_kw, persist, OPEN, TARIFF_THB, EXCEPTIONS, TRACK_ITEMS)
DETAILS_FULL = {f: flags_for(f, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist, OPEN, EXCEPTIONS) for f in FLOORS}
VERIF = verification_results(STORE, EV_FULL, DETAILS_FULL, BRICK_UNITS, TARIFF_THB)  # also verifies/reopens resolved ones
INC_ALL = build_incidents(EV_FULL, STORE.table("incident_state"), (period_start, period_end))
INC = INC_ALL if floor == ALL else INC_ALL[INC_ALL["floor"].eq(floor)]
VERIFIED_KWH = sum(v["saved_kWh"] for k, v in VERIF.items() if v["verdict"] == "Verified")

# When the waste happens (flagged excess kWh by type of time)
flag_rows = d.loc[d["flagged"]]
_dow, _hour = flag_rows["Date"].dt.dayofweek, flag_rows["Date"].dt.hour
WHEN_ORDER = ["Weekday 08:00–17:00", "Weekday, other hours", "Saturday", "Sunday", "Public holiday"]
when_kwh = (flag_rows.groupby(np.select([flag_rows["holiday"], _dow.eq(5), _dow.eq(6), _hour.between(8, 16)],
                                        ["Public holiday", "Saturday", "Sunday", "Weekday 08:00–17:00"],
                                        "Weekday, other hours"))["excess_kWh_est"].sum().reindex(WHEN_ORDER, fill_value=0))
when_pct = 100 * when_kwh / max(1e-9, when_kwh.sum())
after_hours_pct = 100 * flag_rows.loc[flag_rows["after_hours"], "excess_kWh_est"].sum() / max(1e-9, flag_rows["excess_kWh_est"].sum())
top = priority.iloc[0] if len(priority) else None

# Every floor, for Compare floors and the twin: same rules, same review period
floor_rows, unit_tables, details, peer_mean = [], [], {}, {}
for f in FLOORS:
    fd = in_period(flags_for(f, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist, OPEN, EXCEPTIONS))
    fev = build_events(fd, TARIFF_THB, track_record=TRACK)
    fpr = unit_priority(fd, fev, PERIOD_DAYS)
    fpr.insert(0, "floor", f)
    unit_tables.append(fpr)
    details[f] = fd
    for u, part in fd.loc[fd["eligible"]].groupby("unit"):
        peer_mean[u] = part["actual_kW"].mean()
    fk = period_kpis(f, period_start, period_end, OPEN)
    s_ = load(f)["sensitivity"]
    floor_rows.append({"Floor": f"Floor {f}", "AC energy kWh": fk["energy_kWh"], "After-hours %": fk["after_pct"],
                       "15-min peak kW": fk["peak_kW"], "Flagged events": len(fev),
                       "Excess kWh (est.)": fpr["flagged_excess_kWh"].sum(),
                       "Excess THB": fpr["excess_THB"].sum(),
                       "Triage score": fpr["triage_score"].sum(),
                       "Median indoor °C (running)": s_["median_indoor_temp_C_running"].median()})
floors_table = pd.DataFrame(floor_rows)
allunits = pd.concat(unit_tables)
allunits["position"] = allunits.index.map(position_of)
allunits["mean_kW"] = allunits.index.map(peer_mean)


def peer_gap(unit):
    """% difference from the median of the same unit position on the other floors. Only meaningful when the peers were
    actually running: on a Saturday or Sunday the peers are near 0 kW and the ratio would explode."""
    group = brick_model().peer_groups().get(position_of(unit), [])
    peers = [peer_mean[u] for u in group if u != unit and u in peer_mean]
    peer = np.nanmedian(peers) if peers else np.nan
    return 100 * (peer_mean[unit] / peer - 1) if pd.notna(peer) and peer >= on_kw else np.nan


def vs_peers(row):
    return peer_gap(row.name)


allunits["vs_peers_pct"] = allunits.apply(vs_peers, axis=1)
allunits = add_levels(allunits.sort_values("triage_score", ascending=False))  # levels across the whole building
allunits["rank"] = range(1, len(allunits) + 1)
allunits["AC unit"] = [unit_label(u, with_floor=True) for u in allunits.index]
worst_floor = floors_table.loc[floors_table["Triage score"].idxmax()]
# Unit furthest above its peers; None when no comparison is possible (e.g. a Sunday, when everything is off)
outlier = allunits.loc[allunits["vs_peers_pct"].idxmax()] if allunits["vs_peers_pct"].notna().any() else None

# Forecast results (fixed test period Jul–Dec 2019)
fc = data["forecast"].copy()
forecast_models = [c for c in fc.columns if c != "floor_kW"]
vm = data["validation_metrics"]
selected_model = data.get("selected_model") or vm["MAE_kW"].idxmin()
mae_without_holidays = forecast_metrics(fc, forecast_models).loc[selected_model, "MAE_kW"]
if use_holidays:
    # A public holiday is known in advance, so a forecast issued the midnight before can expect the building closed
    fc.loc[fc.index.normalize().isin(CLOSED_DATES), forecast_models] = 0.0
# Scored on the same hours for every method, after the holiday adjustment
tm = forecast_metrics(fc, forecast_models).sort_values("MAE_kW")
tm.insert(0, "Rank", range(1, len(tm) + 1))
tm.insert(1, "Model", [("✓ " if m == selected_model else "") + MODEL_LABELS.get(m, m) for m in tm.index])
tm["Validation MAE_kW"] = vm["MAE_kW"].reindex(tm.index)
selected_mae = tm.loc[selected_model, "MAE_kW"]
lastweek_mae = tm["MAE_kW"].get("M0 last week", np.nan)
# "Next day" = the day after the timeframe. Inside Jul–Dec 2019 every day has a genuine day-ahead forecast (made with
# data up to the midnight before) and its actual load; after 31 Dec only the 1 Jan 2020 demonstration exists.
next_day = period_end.normalize()
_next = fc.reindex(pd.date_range(next_day, periods=24, freq="h"))
if next_day.date() <= data_last:
    # A few hours can be missing where a floor's meter had a gap; those hours are simply left blank
    n24 = pd.DataFrame({"predicted_kW": _next[selected_model], "actual_kW": _next["floor_kW"]})
    n24_note = "day-ahead forecast made at the midnight before, shown with the actual load"
    if n24["predicted_kW"].isna().any():
        n24_note += f" ({int(n24['predicted_kW'].isna().sum())} hour(s) missing because of meter gaps)"
        n24["predicted_kW"] = n24["predicted_kW"].fillna(0.0)
else:
    n24 = data["next24"][["predicted_kW"]].assign(actual_kW=np.nan)
    n24_note = "demonstration for the day after the data ends, so there is no actual load to compare"
NEXT_CLOSURE = closure_name(next_day.date()) if use_holidays else None
if NEXT_CLOSURE:
    n24["predicted_kW"] = 0.0
    n24_note = f"public holiday ({NEXT_CLOSURE}), so the building is expected closed"
HAS_ACTUAL = n24["actual_kW"].notna().any()
peak_time, peak_kw = n24["predicted_kW"].idxmax(), n24["predicted_kW"].max()
_common = fc[["floor_kW", selected_model]].dropna()
error_by_hour = (_common[selected_model] - _common["floor_kW"]).abs().groupby(_common.index.hour).mean()
median_indoor = sens["median_indoor_temp_C_running"].median()
overcooled = median_indoor < 23

# ---------------------------------------------------------------- FM context shared by the work-order, maintenance and card views
def incident_chart(key):
    """The incident's latest episode: actual vs expected power with the flagged intervals."""
    row = INC_ALL.set_index("incident_key").loc[key]
    unit_col = COL_OF.get(row["asset_id"]) or brick_model().supply_groups().get(row["asset_id"], [None])[0]
    part = DETAILS_FULL[int(row["floor"])]
    part = part[part["unit"].eq(unit_col)]
    lo, hi = pd.Timestamp(row["latest_start"]) - pd.Timedelta(hours=12), pd.Timestamp(row["latest_end"]) + pd.Timedelta(hours=12)
    view = part[part["Date"].between(lo, hi)].set_index("Date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["actual_kW"], name="Actual", line=dict(color=ACTUAL, width=2)))
    fig.add_trace(go.Scatter(x=view.index, y=view["expected_kW"], name="Expected", line=dict(color=EXPECTED, width=2, dash="dash")))
    fl = view[view["flagged"]]
    fig.add_trace(go.Scatter(x=fl.index, y=fl["actual_kW"], mode="markers", name="Flagged", marker=dict(color=FLAG, size=6)))
    style(fig, 260, "AC power (kW)", None, margin_top=30)
    day_axis(fig, 2)
    st.plotly_chart(fig, width="stretch", key=f"inc_chart_{key}")
    st.caption(f"Latest episode of {row['label']}: {pd.Timestamp(row['latest_start']):%a %d %b %H:%M}")


@st.cache_data(show_spinner="Computing fleet health…")
def health_bundle(k_iqr, k_res, on_kw, persist, open_spec, exceptions, log_version):
    """Health table, weekly indicators and the fault-risk model result (trained on the simulated log)."""
    details = {f: flags_for(f, k_iqr, k_res, on_kw, persist, True, True, True, open_spec, exceptions) for f in FLOORS}
    table, inds = fleet_health(BRICK_UNITS, details, STORE, pd.Timestamp(data_last), None,
                               peer_weekly(details, brick_model().peer_groups()))
    _, _, metrics, risk = fault_risk_model(inds, STORE)
    return attach_risk(table, risk), metrics


_log = STORE.table("maintenance_log")
HEALTH, RISK_METRICS = health_bundle(k_iqr, k_res, on_kw, persist, OPEN, EXCEPTIONS,
                                     (len(_log), str(_log["log_id"].max()) if len(_log) else ""))
FM_CTX = SimpleNamespace(
    store=STORE, inc=INC, inc_all=INC_ALL, events=EV_FULL, period=PERIOD, actor="dashboard user", brick=brick_model(),
    col_of=COL_OF, as_of=period_end - pd.Timedelta(days=1), open_spec=OPEN, forecasts=floor_forecasts(),
    likely_causes=lambda pattern, asset: likely_causes(STORE, pattern, asset), chart_for_incident=incident_chart,
    days=DAYS, k_iqr=k_iqr, tariff=TARIFF_THB, verif=VERIF, verified_kwh=VERIFIED_KWH, units=BRICK_UNITS, details=DETAILS_FULL,
    allunits=allunits, peer_groups=brick_model().peer_groups(), on_kw=on_kw, health=HEALTH, risk_metrics=RISK_METRICS,
    label_of=dict(zip(INC_ALL["asset_id"], INC_ALL["label"])), style=style,
    colours={"muted": MUTED, "actual": ACTUAL, "expected": EXPECTED})


@st.dialog("Asset history", width="large")
def asset_dialog(asset_id):
    fm_maint_ui.asset_card_body(FM_CTX, asset_id)


FM_CTX.open_asset = asset_dialog
FM_CTX.health_as_of = pd.Timestamp(data_last)

# ---------------------------------------------------------------- tabs
tab_overview, tab_twin, tab_priority, tab_orders, tab_maint, tab_sim, tab_forecast, tab_compare = st.tabs(
    ["Overview", "Digital twin", "Priority list", "Work orders", "Maintenance", "Setpoint & schedule simulator", "Forecast",
     "Compare floors"])

PRIORITY_HELP = TRIAGE_HELP


def next_day_chart(height):
    fig = go.Figure(go.Scatter(x=n24.index, y=n24["predicted_kW"], mode="lines+markers",
                               line=dict(color=ACTUAL, width=2), marker=dict(size=6), name="Forecast"))
    if HAS_ACTUAL:
        fig.add_trace(go.Scatter(x=n24.index, y=n24["actual_kW"], mode="lines", name="Actual",
                                 line=dict(color=EXPECTED, width=2, dash="dash")))
    fig.add_annotation(x=peak_time, y=peak_kw, text=f"Peak {peak_time:%H:%M}: {peak_kw:.0f} kW", showarrow=True, arrowhead=2)
    style(fig, height, "Predicted AC demand (kW)")
    fig.update_xaxes(tickformat="%H:%M", hoverformat="%a %d %b %Y, %H:%M")
    return fig


def floor_forecast_series(f):
    """One floor's hourly forecast (its chosen method), holiday-adjusted like the Forecast tab, plus the day after the data."""
    fd = load(f)
    series = fd["forecast"][fd["validation_metrics"]["MAE_kW"].idxmin()].copy()
    nxt = fd["next24"]["predicted_kW"].copy()
    if use_holidays:
        series[series.index.normalize().isin(CLOSED_DATES)] = 0.0
        nxt[nxt.index.normalize().isin(CLOSED_DATES)] = 0.0
    return pd.concat([series, nxt[~nxt.index.isin(series.index)]])


def floor_interval(f):
    """80 % interval bounds for one floor's forecast, learned on Jul–Sep (where available)."""
    fd = load(f)
    series = floor_forecast_series(f)
    lower, upper, _ = interval_bounds(fd["forecast"]["floor_kW"], series.reindex(fd["forecast"].index))
    return lower, upper


def twin_frame(start, days, compact, height, start_time=None, mode="Replay"):
    """The 3D twin for the floors in view. Unit summaries use the review period; the replay uses the full data.
    start_time: the moment the replay opens at (default 14:00)."""
    twin_floors = FLOORS if floor == ALL else [floor]
    details_full, loads, summaries = {}, {}, {}
    for f in twin_floors:
        details_full[f] = flags_for(f, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist, OPEN, EXCEPTIONS)
        loads[f] = load(f)["zone_loads"]
        fpr = next(t for t in unit_tables if (t["floor"] == f).all())
        fsens = load(f)["sensitivity"]
        for u in fpr.index:
            summaries[u] = {"excess": _r_or_none(fpr.loc[u, "flagged_excess_kWh"]),
                            "priority": _r_or_none(fpr.loc[u, "triage_score"]),
                            "confidence": _r_or_none(fpr.loc[u, "confidence"], 0),
                            "level": fpr.loc[u, "priority_level"],
                            "rank": int(fpr.loc[u, "rank"]), "of": len(fpr),
                            "median": _r_or_none(fsens["median_indoor_temp_C_running"].get(u)),
                            "comfort": _r_or_none(fsens["pct_running_hours_in_comfort_band"].get(u)),
                            "peers": _r_or_none(peer_gap(u))}
    if mode != "Forecast next day":
        days = min(days, (pd.Timestamp(data_last) - pd.Timestamp(start)).days + 1)
    slots = pd.date_range(pd.Timestamp(start), periods=96 * days, freq="15min")
    forecast = whatif = None
    if mode == "Forecast next day":
        forecast = {"fc": {}, "lo": {}, "hi": {}}
        for f in twin_floors:
            fc_f = floor_forecast_series(f)
            raw = load(f)["detail"]
            forecast["fc"].update(unit_forecasts(raw, fc_f, slots))
            lo, hi = floor_interval(f)
            forecast["lo"].update(unit_forecasts(raw, lo, slots))
            forecast["hi"].update(unit_forecasts(raw, hi, slots))
    elif mode == "What-if":
        settings = settings_from_state(st.session_state)
        cur_sp = prepare(detail_all.loc[detail_all["Date"].ge(period_start) & detail_all["Date"].lt(period_end)],
                         data["sensitivity"]["kW_per_degC_outdoor"].clip(lower=0), on_kw).groupby("unit")["current_sp"].first()
        frame = pd.concat([details_full[f][details_full[f]["Date"].isin(slots)] for f in twin_floors], ignore_index=True)
        series = simulate_series(prepare(frame, data["sensitivity"]["kW_per_degC_outdoor"].clip(lower=0), on_kw), settings, cur_sp)
        whatif = {"arrays": twin_arrays(series, slots), "tariff": TARIFF_THB, "band": settings.band}
    payload = twin.payload(details_full, loads, summaries, twin_floors, pd.Timestamp(start), days, on_kw, SCOPE_NAME,
                           forecast, whatif, mode if mode != "Replay" else None)
    if start_time is not None:
        payload["startIndex"] = int(min(max((start_time - pd.Timestamp(start)) / pd.Timedelta(minutes=15), 0),
                                        96 * days - 1))
    page = twin.html(payload, compact=compact)
    if hasattr(st, "iframe"):  # st.components.v1.html is deprecated in newer Streamlit
        st.iframe(page, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(page, height=height)


when_top = when_pct.idxmax()
when_phrase = {"Weekday 08:00–17:00": "during weekday working hours", "Weekday, other hours": "on weekdays outside 08:00–17:00",
               "Saturday": "on Saturdays", "Sunday": "on Sundays", "Public holiday": "on public holidays"}[when_top]
outdoor_mean = in_period(data["weather"].reset_index().rename(columns={"index": "Date"}))["outdoor_temp_C"].mean()

# 15-minute detail for the floors in view, within the timeframe, plus the whole building (for "is the rest running?")
building_now = pd.concat([details[f].assign(floor=f) for f in FLOORS], ignore_index=True)
building_now["running"] = building_now["actual_kW"].gt(on_kw)
scope_now = building_now if floor == ALL else building_now.loc[building_now["floor"].eq(floor)]
indoor_now = scope_now.loc[scope_now["running"], "indoor_temp_C"].median()
ran_now = scope_now.groupby("unit")["running"].any()

SEVERITY = {"Critical": ("#e5533d", 0), "Check": ("#f0a03c", 1), "Info": ("#5aa9ff", 2)}


def assistant_incident(x):
    """Incident key and status chip for an assistant item that belongs to an incident."""
    row = INC_ALL.set_index("incident_key").loc[x["key"]] if x.get("key") in set(INC_ALL["incident_key"]) else None
    return "" if row is None else f'<span class="chip">Incident {x["key"]}</span>' + status_chip(row["status"])


def assistant_history(x):
    """'Last cases on this unit: dirty filter (simulated)' from the close-outs."""
    if not x.get("key"):
        return ""
    note = history_note(STORE, x["key"].split("|")[0])
    return f'<div style="margin-top:.3rem;color:#9fb0cc">{note}</div>' if note else ""


def assistant_findings():
    """Rule-based checks on the selected timeframe, written as what to look at next. Suggestions, not diagnoses."""
    days = max(1, PERIOD_DAYS)
    found = []

    def add(severity, icon_name, title, action, detail, where, weight, key=None):
        """title + one-line action are shown; detail (the reasoning) appears on hover. key = the incident it belongs to."""
        found.append({"severity": severity, "icon": icon_name, "title": title, "action": action, "detail": detail,
                      "where": where, "key": key, "rank": (SEVERITY[severity][1], -weight)})

    def short(u):
        m = re.search(r"F(\d+)_z(\d+)_AC(\d+)", u)
        return f"F{m[1]} · Z{m[2]} · AC{m[3]}" if m else u

    valid = scope_now.loc[scope_now["eligible"]].copy()
    valid["supply"] = valid["unit"].map(dict(zip(brick_model().units()["power_column"], brick_model().units()["supply_group"])))
    valid["building_running"] = valid["Date"].map(building_now.groupby("Date")["running"].mean())
    weekday = valid["Date"].dt.dayofweek.lt(5)

    # Units off during their usual hours while most of the building runs (closures excluded)
    down = valid.loc[valid["schedule_on"].eq(True) & ~valid["running"] & valid["building_running"].ge(.5) & weekday]
    together = down.groupby(["supply", "Date"])["unit"].nunique()
    together = together[together.ge(3)]
    if len(together):
        hours = together.groupby(level=0).size() * INTERVAL_HOURS
        f = hours.idxmax()
        add("Critical", "building", f"{together.loc[f].max()} units on {supply_label(f)} off together",
            "check chilled water, AHU or power feed",
            f"{hours[f]:.0f} h from {together.loc[f].index.min():%a %d %b %H:%M} while the rest of the building ran. "
            f"Suspect the shared supply: chilled water, the air-handling unit or the electrical feed ({SUPPLY_NOTE}).",
            "Digital twin", hours[f], key=f"{f}|supply_group_off")
    single = down.groupby("unit").size() * INTERVAL_HOURS
    for u, h in single[single.ge(max(1, .05 * days))].sort_values(ascending=False).head(2).items():
        n = down.loc[down["unit"].eq(u), "Date"].dt.normalize().nunique()
        add("Check", "bell", f"{short(u)} off in its usual hours", f"{h:.0f} h · check breaker or controller",
            f"{h:.0f} h on {n} day{'s' if n > 1 else ''} while other units ran. Check for a tripped breaker, a controller "
            "fault or a manual switch-off.", "Priority list → Inspect a unit", h)

    # Running but the zone stays warm: the classic sign of low refrigerant or restricted airflow
    warm = valid.loc[valid["running"] & valid["indoor_temp_C"].gt(26)].groupby("unit").size() * INTERVAL_HOURS
    for u, h in warm[warm.ge(max(1, .1 * days))].sort_values(ascending=False).head(2).items():
        add("Check", "thermo", f"{short(u)} running, zone stays above 26 °C",
            f"{h:.0f} h · suspect refrigerant leak or dirty filter", f"{h:.0f} h of running with the zone still warm. Suspect low refrigerant (a leak), a blocked filter or a "
            "dirty coil.", "Priority list → Inspect a unit", h,
            key=f"{brick_model().asset_for_column(u)['asset_id']}|warm_running")

    # Several units on one floor above normal at the same moment: a shared cause is more likely than unit faults
    high_now = valid.loc[(valid["statistical"] | valid["persistent"]) & ~valid["after_hours"]]
    clusters = high_now.groupby(["supply", "Date"])["unit"].nunique()
    clusters = clusters[clusters.ge(3)]
    if len(clusters):
        n_days = clusters.reset_index().assign(day=lambda x: x["Date"].dt.normalize()).groupby("supply")["day"].nunique()
        f = n_days.idxmax()
        add("Check", "building", f"{supply_label(f)}: 3+ units high at once",
            f"{n_days[f]} day{'s' if n_days[f] > 1 else ''} · check shared chilled water or AHU", f"3 or more units above normal together on {n_days[f]} day{'s' if n_days[f] > 1 else ''}. Check what they "
            "share before treating them as unit faults: chilled-water temperature, the air-handling unit, or an event "
            "on that floor.", "Compare floors", n_days[f], key=f"{f}|supply_group_high")

    # Frequent on/off switching
    ordered = valid.sort_values(["unit", "Date"])
    starts = ordered["running"] & ~ordered.groupby("unit")["running"].shift(fill_value=False)
    per_day = starts.groupby([ordered["unit"], ordered["Date"].dt.normalize()]).sum()
    ran_day = ordered.groupby(["unit", ordered["Date"].dt.normalize()])["running"].any()
    cycling = per_day[ran_day].groupby(level=0).mean() if ran_day.any() else pd.Series(dtype=float)
    for u, n in cycling[cycling.gt(4)].sort_values(ascending=False).head(1).items():
        add("Check", "peak", f"{short(u)} starts {n:.0f}× a day", "check thermostat or controls",
            "Frequent starts waste energy and wear the compressor. Check the thermostat position, the control "
            "settings, or whether the unit is oversized for the zone.", "Priority list → Inspect a unit", n)

    # Energy outside the usual schedule, and which unit stands out against its peers
    if len(flag_rows) and after_hours_pct >= 25:
        add("Check", "lens", f"{after_hours_pct:.0f}% of waste is outside usual hours",
            f"mostly {when_phrase} · review timers", f"Mostly {when_phrase}. Review timers and weekend bookings: a schedule fix, not a repair.", "Priority list",
            after_hours_pct / 10)
    if (outlier is not None and outlier["vs_peers_pct"] >= 30
            and (floor == ALL or int(outlier["floor"]) == floor)):
        add("Check", "building", f"{short(outlier.name)} uses {outlier['vs_peers_pct']:.0f}% more than its peers",
            "check this unit's schedule", "The same unit on the other floors uses much less, so look at this unit and its schedule.",
            "Compare floors", outlier["vs_peers_pct"] / 10)

    # AC running on a public holiday or observed closure day
    on_holiday = valid.loc[valid["holiday"] & valid["running"]]
    if len(on_holiday):
        kwh = on_holiday["actual_kW"].sum() * INTERVAL_HOURS
        days_on = sorted(on_holiday["Date"].dt.date.unique())
        names = "; ".join(f"{d:%a %d %b} ({closure_name(d).split(' (')[0]})" for d in days_on[:3])
        add("Check", "bell", f"AC ran on {len(days_on)} public holiday{'s' if len(days_on) > 1 else ''}",
            f"{kwh:,.0f} kWh · check bookings or timers",
            f"The building should be closed: {names}. All of this energy counts as waste unless the day was booked.",
            "Priority list", 25 + kwh / 100)

    # Predictive maintenance: efficiency trending the wrong way since the last service (real data; see Maintenance tab)
    for a, r in HEALTH[HEALTH["rul_deg_days"].le(45)].sort_values("rul_deg_days").head(1).iterrows():
        if floor == ALL or int(a[1]) == floor:
            ser = r["deg"]["series"]
            change = 100 * (ser.iloc[-1] / ser.iloc[0] - 1)
            add("Check", "thermo", f"{short(COL_OF[a])} efficiency {change:+.0f}% since last service",
                f"service due in about {int(r['rul_deg_days'])} days", "The unit needs more kW per °C of outdoor temperature than "
                f"just after its last service ({r['last_service']:%d %b}). The trend reaches +15% in about {int(r['rul_deg_days'])} days "
                f"(range {int(r['rul_low'])}–{int(r['rul_high'])}). Suspect a dirty coil or filter.", "Maintenance", 5)

    # Comfort, data quality and tomorrow's load
    if pd.notna(indoor_now) and indoor_now < 23:
        add("Info", "snow", f"Zones overcooled at {indoor_now:.1f} °C", "raise the setpoint · see simulator",
            "Below the 23–26 °C comfort band. Overcooled: raising the setpoint saves energy and improves comfort.", "Setpoint & schedule simulator", 1)
    coverage = scope_now.groupby("unit")["eligible"].mean()
    for u, c in coverage[coverage.lt(.95)].sort_values().head(1).items():
        add("Info", "bell", f"{short(u)} missing {100 * (1 - c):.0f}% of readings", "check the meter",
            "Check the meter and its connection; flags for this unit may be incomplete.", "Priority list", 1 - c)
    actual_peak = f"; the actual peak was {n24['actual_kW'].max():.0f} kW" if HAS_ACTUAL else ""
    if NEXT_CLOSURE:
        add("Info", "peak", f"Next day ({next_day:%a %d %b}) is a public holiday", "expect the building closed",
            f"{NEXT_CLOSURE}. The forecast expects no AC load{actual_peak}. Check that timers are off.", "Forecast", 0)
    else:
        add("Info", "peak", f"Next day ({peak_time:%a %d %b}) peak ≈ {peak_kw:.0f} kW at {peak_time:%H:%M}",
            "stagger the start-up",
            f"Day-ahead forecast for the day after this timeframe{actual_peak}. Stagger the start-up to flatten the peak.",
            "Forecast", 0)
    return sorted(found, key=lambda x: x["rank"])


# ================================================================ overview
with tab_overview:
    chips = [("thermo", f"{indoor_now:.1f} °C" if pd.notna(indoor_now) else "–", "Indoor · median while AC runs"),
             ("sun", f"{outdoor_mean:.1f} °C", "Outdoor, average"),
             ("bolt", f"{kpis['energy_kWh']:,.0f} kWh", "AC energy"),
             ("peak", f"{kpis['peak_kW']:,.0f} kW", "15-min peak"),
             ("bell", f"{len(events):,}", "Flagged events"),
             ("snow", f"{VERIFIED_KWH:,.0f} kWh", "Verified savings")]
    html('<div class="kpis">' + "".join(f'<div class="kpi">{icon(i)}<div><div class="v">{v}</div>'
                                         f'<div class="l">{l}</div></div></div>' for i, v, l in chips) + "</div>")

    left, centre, right = st.columns([1.05, 2.3, 1.05], gap="medium")
    with left:
        with st.container(key="card-floors"):
            day_mode = PERIOD_DAYS == 1
            at_time = None
            if day_mode:
                # One day: each dot shows the unit's status at a moment you pick, like a live board
                slots = list(pd.date_range(period_start, periods=96, freq="15min"))
                first_alert = events["start"].iloc[0] if len(events) else period_start + pd.Timedelta(hours=14)
                html(f'<div class="card-h">Floors <span style="font-weight:400;font-size:.8rem;color:#9fb0cc">· status at'
                     f'</span></div>')
                at_time = st.select_slider("Status at", options=slots, value=first_alert if first_alert in slots else slots[56],
                                           format_func=lambda t: f"{t:%H:%M}", key=f"floors_time_{period_start:%Y%m%d}",
                                           label_visibility="collapsed")
                snap = building_now.loc[building_now["Date"].eq(at_time)].set_index("unit")

                def dot_state(u, name, level):
                    if u not in snap.index or pd.isna(snap.loc[u, "actual_kW"]):
                        return STATUS_DOT["No reading"], f"{name}: no reading"
                    kw = snap.loc[u, "actual_kW"]
                    state = "Flagged" if snap.loc[u, "flagged"] else "Running" if kw > on_kw else "Off"
                    return STATUS_DOT[state], f"{name}: {state.lower()}, {kw:.1f} kW at {at_time:%H:%M}"
                legend_items = list(STATUS_DOT.items())
                legend_note = f"One dot per AC unit · status at {at_time:%H:%M} on {at_time:%a %d %b}"
            else:
                def dot_state(u, name, level):
                    if not ran_now.get(u, True):
                        return OFF_DOT, f"{name}: did not run"
                    return LEVEL_DOT[level], f"{name}: {level}"
                legend_items = [*((k.split()[1], v) for k, v in LEVEL_DOT.items()), ("Off", OFF_DOT)]
                legend_note = "One dot per AC unit, coloured by priority in this timeframe (pick Day for status at a time)"
            rows = []
            for f in FLOORS:
                part = allunits.loc[allunits["floor"].eq(f)].sort_values("position")
                dots = "".join(f'<span class="dot" style="background:{c}" title="{t}"></span>'
                               for c, t in (dot_state(u, name, l) for u, name, l
                                            in zip(part.index, part["AC unit"], part["priority_level"])))
                weight = "font-weight:700;color:#e6edf7" if floor in (ALL, f) else ""
                rows.append(f'<div class="floor-row"><span class="fl" style="{weight}">F{f}</span>{dots}</div>')
            legend = "".join(f'<span style="white-space:nowrap"><span class="dot" style="background:{c}"></span>'
                             f'{l}</span>' for l, c in legend_items)
            html(("" if day_mode else f'<div class="card-h">Floors <span style="font-weight:400;font-size:.8rem;'
                                      f'color:#9fb0cc">· {PERIOD}</span></div>') + "".join(rows)
                 + f'<div class="legend-dots">{legend_note}<br>{legend}</div>'
                 + f'<div class="load-line">AC energy <b>{kpis["energy_kWh"]:,.0f} kWh</b> · '
                   f'≈ <b>{kpis["energy_kWh"] * TARIFF_THB:,.0f} THB</b><br>'
                   f'of which <b>{kpis["after_kWh"]:,.0f} kWh</b>'
                   + (f' ({kpis["after_pct"]:.1f}%)' if pd.notna(kpis["after_pct"]) else "")
                   + ' while the building was closed<br>'
                   f'<span style="color:#9fb0cc;font-size:.78rem">{PERIOD} · at {TARIFF_THB} THB/kWh (assumed) · '
                   f'open {OPEN_TEXT}</span></div>')
        with st.container(key="card-alerts"):
            items = []
            queue = INC[INC["status"].isin(OPEN_STATUSES) & INC["active"]].sort_values("triage_score", ascending=False)
            for _, e in queue.head(4).iterrows():
                items.append(f'<div class="item">{icon("bell", 18, small=True)}<div><div class="t">'
                             f'{e["label"]} {PATTERNS[e["pattern"]][2]}</div><div class="s">{e["priority_level"]} · '
                             f'{e["period_THB"]:,.0f} THB · confidence {e["confidence"]:.0f} · '
                             f'{PATTERNS[e["pattern"]][0].lower()} effort · {status_chip(e["status"])}</div></div></div>')
            html('<div class="card-h">Top open incidents</div>'
                 + ("".join(items) if items else '<div class="item"><div class="t">No open incidents in this timeframe</div></div>')
                 + '<div class="card-f">Full queue: Work orders tab</div>')

    with centre:
        replay_default = events["start"].iloc[0].date() if len(events) else period_start.date()
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        c1.markdown("**Building replay**",
                    help="Red zones are flagged at that moment. "
                         "Drag to rotate, scroll to zoom, hover a zone for readings, press play to replay the day. "
                         "☰ Layers changes what the colours show. The Digital twin tab has the full controls.")
        replay_day = c2.date_input("Day", replay_default, min_value=data_first, max_value=data_last, format="DD/MM/YYYY",
                                   key=f"overview_day_{floor}_{period_start:%Y%m%d}_{PERIOD_DAYS}",
                                   label_visibility="collapsed")
        # On a single day the replay opens at the same moment as the Floors status
        twin_frame(replay_day, 1, compact=True, height=600,
                   start_time=at_time if at_time is not None and replay_day == at_time.date() else None)
        st.caption(f"{pd.Timestamp(replay_day):%A %d %B %Y}" + (" · day of the top alert" if len(events) and replay_day == replay_default else "")
                   + " · red zones are flagged at that moment")

    with right:
        with st.container(key="card-findings"):
            findings = assistant_findings()
            # Compact: coloured severity dot + title + one-line action; hovering slides open the reasoning
            items = [f'<div class="item ai" style="--sev:{SEVERITY[x["severity"]][0]}">'
                     f'{icon(x["icon"], 18, small=True)}<div><div class="t">'
                     f'<span class="dot" style="background:var(--sev);width:8px;height:8px;'
                     f'margin:0 6px 1px 0"></span>{x["title"]}</div><div class="s">→ {x["action"]}</div>'
                     f'<div class="why">{x["detail"]}{assistant_history(x)}<div class="meta"><span class="chip sev">{x["severity"].upper()}'
                     f'</span><span class="chip">Look in: {x["where"]}</span>{assistant_incident(x)}</div></div></div></div>'
                     for x in findings]
            html('<div class="card-h">✦ Smart assistant <span style="font-weight:400;font-size:.75rem;color:#9fb0cc">'
                 '· hover for why</span></div>' + "".join(items[:3]))
            if len(items) > 3:
                with st.expander(f"{len(items) - 3} more"):
                    html("".join(items[3:]))
        with st.container(key="card-load"):
            load_15 = scope_now.groupby("Date")["actual_kW"].sum(min_count=1)
            one_day = PERIOD_DAYS == 1
            hourly = load_15.groupby(load_15.index.hour).mean()
            html(f'<div class="card-h">Load pattern <span style="font-weight:400;font-size:.8rem;color:#9fb0cc">· '
                 f'{"hourly AC load" if one_day else "average day"} · {PERIOD}</span></div>')
            fig = go.Figure(go.Bar(x=[f"{h:02d}:00" for h in hourly.index], y=hourly.values, marker_color="#6fa8ff",
                                   marker_line_width=0, hovertemplate="%{x} · %{y:.0f} kW<extra></extra>"))
            style(fig, 220, None, margin_top=6)
            fig.update_layout(hovermode="closest", bargap=.35)
            fig.update_xaxes(tickvals=["00:00", "06:00", "12:00", "18:00"])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            if len(hourly) and hourly.max() > 0:
                html(f'<div class="card-f" style="text-align:left">Busiest hour <b>{hourly.idxmax():02d}:00 · '
                     f'{hourly.max():.0f} kW</b>{" average" if not one_day else ""}</div>')

# ================================================================ priority list
with tab_priority:
    section(f"Unit priority list ({PERIOD} totals)",
            help="Each unit is compared with its own normal pattern (learned from Jul 2018 – Jun 2019). Readings that are "
                 "unusually high, outside the unit's usual hours, or high for 2 hours or more are flagged. " + PRIORITY_HELP)
    cols = ["rank", "priority_level", "AC unit", "excess_THB", "per_day_THB", "confidence", "effort", "triage_score",
            "comfort_hit", "events", "flagged_pct"]
    with st.container(key="card-triage"):
        scatter = priority.loc[priority["events"].gt(0)].copy()
        if len(scatter):
            thb_cut = scatter["excess_THB"].quantile(.75)
            scatter["label"] = scatter["AC unit"]
            fig = go.Figure()
            for effort, colour in [("Low", GOOD), ("Medium", "#eda100"), ("High", FLAG)]:
                part = scatter.loc[scatter["effort"].eq(effort)]
                if part.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=part["confidence"], y=part["excess_THB"], mode="markers", name=f"{effort} effort",
                    marker=dict(color=colour, size=(10 + 3 * part["events"].clip(upper=30)), opacity=.8,
                                line=dict(width=1, color="rgba(255,255,255,.5)")),
                    text=part["label"], customdata=part[["events", "triage_score"]],
                    hovertemplate="%{text}<br>confidence %{x:.0f} · %{y:,.0f} THB<br>%{customdata[0]} events · "
                                  "triage %{customdata[1]:,.0f}<extra></extra>"))
            fig.add_vline(x=70, line=dict(color=MUTED, dash="dot"))
            fig.add_hline(y=thb_cut, line=dict(color=MUTED, dash="dot"))
            style(fig, 320, "Excess THB", "Confidence · act first = top right (size = number of events)")
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(range=[0, 100])
            st.plotly_chart(fig, width="stretch")
    st.dataframe(priority[cols], hide_index=True, width="stretch", height=table_height(len(priority), 420), column_config={
        **PRIORITY_COLUMNS, "events": st.column_config.NumberColumn("Events"),
        "flagged_pct": st.column_config.NumberColumn("Time flagged", format="%.1f%%")})
    if top is not None and top["events"]:
        st.caption(f"Why **{top['AC unit']}** ranks first: about {top['excess_THB']:,.0f} THB of excess energy "
                   f"({top['per_day_THB']:,.0f} THB/day) × confidence {top['confidence']:.0f} ÷ effort "
                   f"({top['effort'].lower()}) = triage score {top['triage_score']:,.0f}, "
                   f"{top['share_pct']:.0f}% of the total → {top['priority_level']}.")

    section("When flags happen",
            help="Left: the share of flagged extra energy by type of time. Right: how often readings are flagged, by "
                 "weekday and time of day (darker = more often).")
    c1, c2 = st.columns([2, 3], gap="large")
    fig = go.Figure(go.Bar(x=when_pct.values, y=when_pct.index, orientation="h",
                           marker_color=[ACTUAL, FLAG, FLAG, FLAG, FLAG], text=[f"{v:.0f}%" for v in when_pct.values],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1f}% of extra energy<extra></extra>"))
    style(fig, 300, None, "% of flagged extra energy (red = outside weekday working hours)")
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    c1.plotly_chart(fig, width="stretch")
    hm = d.loc[d["eligible"]].assign(dow=lambda x: x["Date"].dt.dayofweek,
                                     slot=lambda x: x["Date"].dt.hour * 4 + x["Date"].dt.minute // 15)
    heat = (100 * hm.groupby(["dow", "slot"])["flagged"].mean()).unstack().reindex(index=range(7), columns=range(96))
    times = [f"{s // 4:02d}:{(s % 4) * 15:02d}" for s in range(96)]
    fig = go.Figure(go.Heatmap(z=heat.values, x=times, y=DAYS, colorscale=[[0, "#13213a"], [0.5, "#2f6fbf"], [1, "#9cccff"]],
                               zmin=0, zmax=max(1, float(np.nanmax(heat.values))) if np.isfinite(heat.values).any() else 1,
                               colorbar=dict(title="% flagged", thickness=10),
                               hovertemplate="%{y} %{x}<br>%{z:.1f}% of valid intervals flagged<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickvals=times[::12], ticktext=times[::12])
    c2.plotly_chart(style(fig, 300, margin_top=10), width="stretch")
    if not events.empty:
        takeaway(f"{when_pct.max():.0f}% of the extra energy is used <b>{when_phrase}</b> and {after_hours_pct:.0f}% falls "
                 "outside the units' usual schedule, so most of it is a schedule or timer fix rather than a repair.")

    section("Event list",
            help="One row per unusual episode, ranked by triage score. " + TRIAGE_HELP)
    if events.empty:
        st.warning("No events with the current thresholds and review period.")
    else:
        f1, f2, f3, f4 = st.columns([2, 2, 2, 3], vertical_alignment="bottom")
        min_hours = f1.number_input("Minimum duration (h)", 0.0, 48.0, 0.0, 0.25)
        only_after = f2.checkbox("After-hours only")
        hide_hot = f3.checkbox("Hide hot-day events")
        ev = events.loc[events["duration_h"].ge(min_hours)]
        if only_after:
            ev = ev.loc[ev["after_hours_h"].gt(0)]
        if hide_hot:
            ev = ev.loc[~ev["hot_day"]]
        ev = ev.assign(**{"AC unit": ev["unit"].map(unit_label)})
        ev.insert(0, "rank", range(1, len(ev) + 1))
        f4.caption(f"{len(ev):,} of {len(events):,} events shown")
        ev_cols = ["rank", "priority_level", "AC unit", "start", "end", "duration_h", "peak_kW", "excess_kWh",
                   "excess_THB", "confidence", "effort", "triage_score", "after_hours_h", "outdoor_C", "reason"]
        st.dataframe(ev[ev_cols], hide_index=True, width="stretch", height=300, column_config={
            "rank": st.column_config.NumberColumn("#", width=35),
            "priority_level": PRIORITY_COLUMNS["priority_level"],
            "start": st.column_config.DatetimeColumn("Start", format="ddd D MMM YYYY, HH:mm"),
            "end": st.column_config.DatetimeColumn("End", format="HH:mm"),
            "duration_h": st.column_config.NumberColumn("Hours", format="%.2f"),
            "peak_kW": st.column_config.NumberColumn("Peak kW", format="%.1f"),
            "excess_kWh": st.column_config.NumberColumn("Excess kWh", format="%.1f"),
            "excess_THB": PRIORITY_COLUMNS["excess_THB"], "confidence": PRIORITY_COLUMNS["confidence"],
            "effort": PRIORITY_COLUMNS["effort"], "triage_score": PRIORITY_COLUMNS["triage_score"],
            "after_hours_h": st.column_config.NumberColumn("After-hours h", format="%.2f"),
            "outdoor_C": st.column_config.NumberColumn("Outdoor °C", format="%.1f"),
            "reason": st.column_config.TextColumn("Why flagged", width="large")})
        st.download_button("Download event list (CSV)", ev[ev_cols].to_csv(index=False).encode("utf-8"),
                           "ac_priority_events.csv", "text/csv")

    section("Inspect a unit",
            help="Blue = actual power, orange dashed = expected, red dots = "
                 "flagged. Lower panel: outdoor and indoor °C.")
    controls, chart = st.columns([1, 3], gap="large")
    with controls:
        unit = st.selectbox("AC unit", priority.index.tolist(), format_func=unit_label, key=f"inspect_unit_{floor}")
        if st.button("Open asset card", key=f"card_priority_{floor}", help="History, timeline, efficiency trend and peers for this unit."):
            asset_dialog(brick_model().asset_for_column(unit)["asset_id"])
        # Follows the timeframe at the top of the page (d is already limited to it)
        view = d.loc[d["unit"].eq(unit)].set_index("Date")
        start, end = period_start, period_end
        st.caption(f"Showing **{PERIOD}** (change it with the timeframe at the top) · flagged excess "
                   f"{view.loc[view['flagged'], 'excess_kWh_est'].sum():,.0f} kWh"
                   + (" · zoom in on the chart for detail" if PERIOD_DAYS > 31 else ""))
    with chart:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=view.index, y=view["actual_kW"], name="Actual", line=dict(color=ACTUAL, width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=view.index, y=view["expected_kW"], name="Expected", line=dict(color=EXPECTED, width=2, dash="dash")), 1, 1)
        fl = view.loc[view["flagged"]]
        fig.add_trace(go.Scatter(x=fl.index, y=fl["actual_kW"], mode="markers", name="Flagged",
                                 marker=dict(color=FLAG, size=7, opacity=.85)), 1, 1)
        fig.add_trace(go.Scatter(x=view.index, y=view["outdoor_temp_C"], name="Outdoor °C", line=dict(color=MUTED, width=2)), 2, 1)
        fig.add_trace(go.Scatter(x=view.index, y=view["indoor_temp_C"], name="Indoor °C", line=dict(color=GOOD, width=2)), 2, 1)
        style(fig, 400)
        day_axis(fig, (end - start).days)
        fig.update_yaxes(title="AC power (kW)", row=1, col=1)
        fig.update_yaxes(title="°C", rangemode="normal", row=2, col=1)
        st.plotly_chart(fig, width="stretch")

with tab_orders:
    fm_tabs.render(FM_CTX)

with tab_maint:
    fm_maint_ui.maintenance_tab(FM_CTX)

# ================================================================ compare floors
with tab_compare:
    floor_colour = {f: UNIT_COLOURS[i % len(UNIT_COLOURS)] for i, f in enumerate(FLOORS)}

    section(f"How do the floors compare? ({PERIOD} totals)",
            help="Floors 3, 4, 5 and 7 share the same AC layout, so the same unit position can be compared across floors. "
                 "Always shows every floor, whatever the View. The sidebar thresholds and review period apply to every "
                 "floor. Equal layout does not prove equal occupancy or equipment condition: treat differences as "
                 "questions, not verdicts.")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Floors analysed", len(FLOORS))
    b2.metric("AC energy, all floors", f"{floors_table['AC energy kWh'].sum():,.0f} kWh")
    b3.metric("Flagged events", f"{int(floors_table['Flagged events'].sum()):,}")
    b4.metric("Floor to review first", worst_floor["Floor"], help="Highest total triage score")
    t, c1, c2 = st.columns([3, 1, 1], gap="medium")
    t.dataframe(floors_table[["Floor", "AC energy kWh", "Excess THB", "Triage score", "After-hours %", "Flagged events",
                              "Median indoor °C (running)"]].round(1),
                hide_index=True, width="stretch", height=table_height(len(floors_table), 300),
                column_config={"AC energy kWh": st.column_config.NumberColumn(format="%.0f"),
                               "Excess THB": st.column_config.NumberColumn(format="%.0f"),
                               "Triage score": st.column_config.NumberColumn(format="%.0f"),
                               "After-hours %": st.column_config.NumberColumn("Used while closed %", format="%.1f"),
                               "Median indoor °C (running)": st.column_config.NumberColumn("Indoor °C", format="%.1f")})
    for col, metric, title in [(c1, "Triage score", "Triage score"),
                               (c2, "After-hours %", "Used while closed (%)")]:
        fig = go.Figure(go.Bar(x=floors_table["Floor"].str.replace("Floor ", "F"), y=floors_table[metric],
                               marker_color=[FLAG if f == worst_floor["Floor"] else ACTUAL for f in floors_table["Floor"]],
                               text=floors_table[metric].map(lambda v: f"{v:,.0f}" if v >= 100 else f"{v:.1f}"),
                               textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:,.1f}<extra></extra>"))
        col.plotly_chart(style(fig, table_height(len(floors_table), 300), title, margin_top=10), width="stretch")
    share_worst = 100 * worst_floor["Triage score"] / max(1, floors_table["Triage score"].sum())

    section("Building-wide priority list and same unit on other floors",
            help="All units on all floors, ranked by triage score. 'vs peers' compares a unit's average power with the "
                 "median of the same unit position on the other floors: +30% means it draws 30% more than its peers.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.dataframe(allunits[["rank", "priority_level", "AC unit", "excess_THB", "confidence", "triage_score", "vs_peers_pct"]],
                     hide_index=True, width="stretch", height=420, column_config={
                         **PRIORITY_COLUMNS,
                         "vs_peers_pct": st.column_config.NumberColumn("vs peers", format="%+.0f%%",
                                                                       help="Average power vs the median of the same unit "
                                                                            "position on the other floors.")})
        st.download_button("Download building-wide list (CSV)",
                           allunits.drop(columns=["position"]).to_csv().encode("utf-8"), "ac_priority_all_floors.csv", "text/csv")
    with right:
        positions = sorted({position_of(u) for u in peer_mean})
        default_pos = (positions.index(position_of(outlier.name))
                       if outlier is not None and position_of(outlier.name) in positions else 0)
        pc1, pc2 = st.columns([1, 2], vertical_alignment="bottom")
        pos = pc1.selectbox("Unit position", positions, index=default_pos,
                            format_func=lambda p: unit_label(f"F0_{p}", with_floor=False))
        day_type = pc2.radio("Days", ["Weekdays", "Saturdays", "Sundays"], horizontal=True)
        fig = go.Figure()
        rows = []
        for f in FLOORS:
            fd = details[f]
            u = f"F{f}_{pos}(kW)"
            part = fd.loc[fd["unit"].eq(u) & fd["eligible"]]
            if part.empty:
                continue
            running = part["actual_kW"].gt(on_kw)
            rows.append({"Floor": f"Floor {f}",
                         "Average kW": part["actual_kW"].mean(),
                         "vs peers": allunits.loc[u, "vs_peers_pct"] if u in allunits.index else np.nan,
                         "After-hours kWh": part.loc[running & part["schedule_on"].eq(False), "actual_kW"].sum() * INTERVAL_HOURS,
                         "Flagged excess kWh": part.loc[part["flagged"], "excess_kWh_est"].sum()})
            dow = part["Date"].dt.dayofweek
            part = part.loc[{"Weekdays": dow < 5, "Saturdays": dow == 5, "Sundays": dow == 6}[day_type]]
            if part.empty:
                continue
            slot = part["Date"].dt.hour * 4 + part["Date"].dt.minute // 15
            profile = part.groupby(slot)["actual_kW"].mean().reindex(range(96))
            fig.add_trace(go.Scatter(x=times, y=profile.values, name=f"Floor {f}",
                                     line=dict(color=floor_colour[f], width=3 if f == floor else 2)))
        style(fig, 250, "Average kW", f"Time of day · average {day_type.lower()}")
        fig.update_xaxes(tickvals=times[::12])
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch",
                     height=table_height(len(rows), 200), column_config={
                         "vs peers": st.column_config.NumberColumn(format="%+.0f%%"),
                         "After-hours kWh": st.column_config.NumberColumn(format="%.0f"),
                         "Flagged excess kWh": st.column_config.NumberColumn(format="%.0f")})
    if outlier is not None and outlier["vs_peers_pct"] > 0:
        takeaway(f"<b>{worst_floor['Floor']}</b> causes {share_worst:.0f}% of the building's triage score, and "
                 f"<b>{outlier['AC unit']}</b> draws {outlier['vs_peers_pct']:.0f}% more power than the same unit on the "
                 "other floors, so the issue is that unit and its schedule, not the layout.")
    else:
        takeaway(f"<b>{worst_floor['Floor']}</b> causes {share_worst:.0f}% of the building's triage score. No unit "
                 "can be compared with its peers in this timeframe (the peer units were not running).")

    with st.expander("Building model (Brick)"):
        bm = brick_model()
        counts = bm.counts()
        cols = st.columns(len(counts))
        for col, (k, v) in zip(cols, counts.items()):
            col.metric(k.capitalize() if k != "AC units" else k, v)
        st.caption(f"Brick Schema 1.3 graph, queried with {bm.backend}. Each AC unit links to its zone, power point and supply "
                   f"group. **Supply groups are {SUPPLY_NOTE}**: confirm with facilities before relying on them.")
        st.dataframe(bm.table(), hide_index=True, width="stretch", height=table_height(len(bm.table()), 380))
        ttl = Path(__file__).resolve().parent / "brick" / "chamchuri5.ttl"
        st.download_button("Download chamchuri5.ttl", ttl.read_bytes(), "chamchuri5.ttl", "text/turtle")

# ================================================================ forecast
with tab_forecast:
    models = [c for c in fc.columns if c != "floor_kW"]
    forecast_help = ("Methods were trained on Jul 2018 – Jun 2019, one was chosen on Apr–Jun 2019, and all were scored "
                     "on Jul–Dec 2019, data they never saw. Not affected by the review period. "
                     "Ranked by average error on Jul–Dec 2019 (lower = better). The ✓ method is the one to use: it was "
                     "chosen on earlier data (Apr–Jun 2019) before this test, which is the fair way to pick one. ")
    if floor == ALL:
        forecast_help += ("For the whole building, 'Best method per floor (combined)' forecasts each floor with its "
                          "own chosen method and adds them up: "
                          + "; ".join(f"Floor {f}: {MODEL_LABELS.get(m, m)}" for f, m in {f: load(f)["validation_metrics"]["MAE_kW"].idxmin() for f in FLOORS}.items())
                          + ". ")
    forecast_help += ("Peak-hour error = average error in the busiest 10 % of hours. Bias: + means it predicts too "
                      "high on average, − too low. Methods with weather use observed Open-Meteo weather in place of a "
                      "weather forecast, so their errors are optimistic.")

    section("Next-day hourly demand forecast: accuracy", help=forecast_help)
    if use_holidays:
        st.caption(f"Uses the Thai public-holiday calendar: a holiday is known in advance, so demand is forecast as zero. "
                   f"This cuts the chosen method's average error from {mae_without_holidays:.1f} to {selected_mae:.1f} kW "
                   "per hour. Turn it off under Building opening hours in the sidebar.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.dataframe(tm[["Rank", "Model", "MAE_kW", "peak_hour_MAE_kW"]].round(2), hide_index=True,
                     width="stretch", height=table_height(len(tm), 420), column_config={
                         "Rank": st.column_config.NumberColumn("#", width=35),
                         "Model": st.column_config.TextColumn("Method (✓ = use this)", width="medium"),
                         "MAE_kW": st.column_config.NumberColumn("Avg error (kW)", format="%.2f",
                                                                 help="Average forecast error per hour. Lower is better."),
                         "peak_hour_MAE_kW": st.column_config.NumberColumn("Peak-hour error", format="%.2f",
                                                                           help="Average error in the busiest 10 % of hours (kW)."),
                         "NMBE_pct": st.column_config.NumberColumn("Bias %", format="%+.1f",
                                                                   help="+ = predicts too high on average, − = too low.")})
        with st.expander("More accuracy measures"):
            extra = (["Rank", "Model", "NMBE_pct"] + ([] if floor == ALL else ["Validation MAE_kW"])
                     + ["RMSE_kW", "CV_RMSE_pct", "hours"])
            st.dataframe(tm[extra].round(2), hide_index=True, width="stretch", column_config={
                "Rank": st.column_config.NumberColumn("#", width=35),
                "NMBE_pct": st.column_config.NumberColumn("Bias %", format="%+.1f",
                                                          help="+ = predicts too high on average, − = too low."),
                "Validation MAE_kW": st.column_config.NumberColumn("Apr–Jun error (kW)", format="%.2f"),
                "RMSE_kW": st.column_config.NumberColumn("RMSE (kW)", format="%.2f"),
                "CV_RMSE_pct": st.column_config.NumberColumn("CV(RMSE) %", format="%.1f"),
                "hours": st.column_config.NumberColumn("Hours scored")})
    with right:
        order = tm.index.tolist()[::-1]
        colours = [ACTUAL if m == selected_model else GOOD if "weather" in m else "#4a5d7e" for m in order]
        fig = go.Figure(go.Bar(y=[MODEL_LABELS.get(m, m) for m in order], x=tm.loc[order, "MAE_kW"], orientation="h",
                               marker_color=colours, text=tm.loc[order, "MAE_kW"].round(2), textposition="outside",
                               cliponaxis=False, hovertemplate="%{y}: %{x:.2f} kW<extra></extra>"))
        style(fig, table_height(len(tm), 420), None, "Avg error (kW) · blue = chosen · green = weather")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch")

    with st.expander("Model checks: how far can this forecast be trusted?"):
        ok = fc[["floor_kW", selected_model]].dropna()
        actual_s, model_s = ok["floor_kW"], ok[selected_model]
        raw = data["forecast"][selected_model].reindex(ok.index)
        st.caption("Chosen method on Jul–Dec 2019. Full write-up, with what could not be checked: "
                   "`docs/VERIFICATION.md`. Re-run the checks with `pytest tests/verify_forecast.py`.")
        a_table = ashrae_rows(actual_s, model_s)
        a_raw = ashrae_rows(actual_s, raw)
        compare = pd.concat([a_raw.assign(Holidays="not adjusted"), a_table.assign(Holidays="adjusted")])
        st.markdown("**ASHRAE Guideline 14**: bias within ±10% and CV(RMSE) ≤ 30%")
        st.dataframe(compare[["Resolution", "Holidays", "MAE_kW", "NMBE_pct", "CV_RMSE_pct", "NMBE ok (±10%)",
                              "CV(RMSE) ok (≤30%)"]].round(1), hide_index=True, width="stretch")
        hourly_ok = bool(a_table.loc[a_table["Resolution"].eq("Hourly"), "CV(RMSE) ok (≤30%)"].iloc[0])
        daily_ok = bool(a_table.loc[a_table["Resolution"].eq("Daily totals"), "CV(RMSE) ok (≤30%)"].iloc[0])
        st.caption(("Passes" if hourly_ok else "Fails") + " at hourly resolution and "
                   + ("passes" if daily_ok else "fails") + " for daily totals. "
                   "Use the forecast for planning (when the peak is, which days are quiet), not as a precise meter.")
        by = breakdowns(actual_s, model_s)
        b1, b2, b3 = st.columns([2, 1, 1])
        for col, name in [(b1, "Hour"), (b2, "Weekday"), (b3, "Month")]:
            fig = go.Figure(go.Bar(x=[str(i) for i in by[name].index], y=by[name]["MAE kW"], marker_color=ACTUAL,
                                   hovertemplate="%{x}: %{y:.1f} kW<extra></extra>"))
            style(fig, 220, "MAE (kW)" if name == "Hour" else None, f"By {name.lower()}", margin_top=10)
            fig.update_layout(hovermode="closest")
            col.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        lower, upper, coverage = interval_bounds(actual_s, model_s)
        st.metric("80% interval coverage (Oct–Dec)", f"{coverage * 100:.0f}%" if pd.notna(coverage) else "–",
                  help="Error quantiles by hour of day are learned on Jul–Sep and checked on Oct–Dec. Above 80% means the "
                       "band is conservative.")

    section("Actual vs forecast")
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    picked = c1.multiselect("Methods to plot", models,
                            default=[selected_model] + ([WEATHER_MODEL] if selected_model != WEATHER_MODEL and WEATHER_MODEL in models else []),
                            format_func=lambda m: MODEL_LABELS.get(m, m), max_selections=3)
    dates = fc.index.normalize().unique()
    start = c2.date_input("Week starting", dates.min().date(), min_value=dates.min().date(), max_value=dates.max().date(),
                          format="DD/MM/YYYY")
    view = fc.loc[pd.Timestamp(start): pd.Timestamp(start) + pd.Timedelta(days=7) - pd.Timedelta(hours=1)]
    fig = go.Figure(go.Scatter(x=view.index, y=view["floor_kW"], name="Actual", line=dict(color=ACTUAL, width=2)))
    for colour, m in zip([EXPECTED, "#1baf7a", "#4a3aa7"], picked):
        fig.add_trace(go.Scatter(x=view.index, y=view[m], name=MODEL_LABELS.get(m, m), line=dict(color=colour, width=2, dash="dash")))
    style(fig, 320, "AC demand (kW)")
    day_axis(fig, 7)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Showing {pd.Timestamp(start):%a %d %b %Y} – {pd.Timestamp(start) + pd.Timedelta(days=6):%a %d %b %Y}. "
               "Public holidays and closures are not modelled, so those days show large misses.")

    section(f"Next-day forecast · {peak_time:%a %d %b %Y}",
            help="The day after the selected timeframe (change it at the top). " + n24_note[0].upper() + n24_note[1:] + ".")
    c1, c2 = st.columns([3, 1], gap="large")
    c1.plotly_chart(next_day_chart(300), width="stretch")
    c1.caption(f"{peak_time:%A %d %B %Y}: {n24_note}.")
    c2.metric("Expected peak", f"{peak_kw:.0f} kW", f"at {peak_time:%H:%M}", delta_color="off")
    if HAS_ACTUAL:
        actual_peak_time = n24["actual_kW"].idxmax()
        c2.metric("Actual peak", f"{n24['actual_kW'].max():.0f} kW", f"at {actual_peak_time:%H:%M}", delta_color="off")
        c2.metric("This day's average error", f"{(n24['predicted_kW'] - n24['actual_kW']).abs().mean():.1f} kW",
                  help=f"Typical for Jul–Dec 2019: {selected_mae:.1f} kW")
    else:
        c2.metric("Expected daily energy", f"{n24['predicted_kW'].sum():,.0f} kWh")
        c2.metric("Typical error at the peak hour", f"±{error_by_hour.get(peak_time.hour, selected_mae):.0f} kW")
    quiet = n24.between_time("08:00", "16:59")
    quiet = [t for t in quiet.index if quiet.loc[t, "predicted_kW"] < 0.2 * peak_kw]
    better = f" ({100 * (1 - selected_mae / lastweek_mae):.0f}% better than repeating last week)" if np.isfinite(lastweek_mae) else ""
    takeaway(f"Plan for about {peak_kw:.0f} kW at {peak_time:%H:%M}, give or take "
             f"{error_by_hour.get(peak_time.hour, selected_mae):.0f} kW: the forecast is off by {selected_mae:.1f} kW per "
             f"hour on average{better}. Demand jumps when units start, so a staggered start-up flattens the peak."
             + (f" The quietest working hour is {quiet[0]:%H:%M}, a good time for inspections." if quiet else ""))

# ================================================================ simulator
with tab_sim:
    slope = sens["kW_per_degC_outdoor"].clip(lower=0)

    section("Comfort today",
            help="Share of running hours (weekdays, Jul–Dec 2019) with indoor temperature in 23–26 °C and humidity in 40–60 %.")
    comfort = sens[["median_indoor_temp_C_running", "pct_running_hours_in_comfort_band", "pct_running_hours_in_RH_band"]].copy()
    comfort.index = comfort.index.map(unit_label).rename("AC unit")
    comfort.columns = ["Median indoor °C", "% hours 23–26 °C", "% hours RH 40–60%"]
    st.dataframe(comfort.round(1), width="stretch", height=table_height(len(comfort), 260))
    takeaway((f"Zones are kept at a median <b>{median_indoor:.1f} °C</b> while the AC runs, below the 23–26 °C comfort band. "
              "They are overcooled, so raising the setpoint should save energy <i>and</i> improve comfort.")
             if overcooled else
             f"Zones sit at a median {median_indoor:.1f} °C while the AC runs, inside the comfort band. The main lever is the schedule.")

    section("What if we change the setpoint or the operating schedule?",
            help="Screening estimates: confirm with a measured trial. Setpoint effect: each unit's baseline regression slope (kW per °C of outdoor temperature) stands in for "
                 "the kW saved per 1 °C setpoint rise, and the zone's temperature profile shifts so its median sits at the "
                 "new setpoint. A setpoint below today's level costs energy (negative saving). Schedule effect: AC energy "
                 "outside the chosen window counts as avoidable.")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        setpoint = c1.slider("New indoor setpoint (°C)", 18.0, 28.0, 25.0, 0.5, key="sim_setpoint")
        band = c1.slider("Comfort band (°C)", 18.0, 30.0, (23.0, 26.0), 0.5, key="sim_band")
        comfort_target = c2.slider("Minimum comfort compliance (%)", 30, 100, 70, key="sim_comfort_target",
                                   help="An assumption, not a standard: the share of running time that must stay in the "
                                        "comfort band. The recommended setpoint is the biggest saving that meets it.")
        tariff = TARIFF_THB  # set under sidebar → Advanced: electricity price, shared with triage
        c2.caption(f"Electricity price: {tariff:g} THB/kWh (sidebar → Advanced)")
        start_h, stop_h = c3.slider("Weekday AC window", 0.0, 24.0, (0.0, 24.0), 0.25, format="%.2f h", key="sim_window",
                                    help="The hours the AC is allowed to run on weekdays. 00:00–24:00 = no change from today. "
                                         "Narrow it (e.g. 07:00–18:00) and all AC energy used outside the window counts as "
                                         "saved. This is the maximum possible saving: some after-hours use may be "
                                         "legitimate (events, cleaning, servers).")
        weekend_on = c4.checkbox("Keep weekends running", True, key="sim_weekend_on",
                                 help="Ticked: weekends are allowed (no change). Unticked: the proposal is no AC at "
                                      "weekends, so all weekend AC energy counts as saved.")
        holidays_off = c4.checkbox("AC off on public holidays", False, key="sim_holidays_off",
                                   help="Proposal: no AC on Thai public holidays and closure days, so any AC energy on "
                                        "those days counts as saved.")

    sim = prepare(detail, slope, on_kw)  # fm_sim: the same maths drives the digital twin's What-if mode
    total_kwh = sim["actual_kW"].sum() * INTERVAL_HOURS

    def sim_settings(sp):
        return Settings(setpoint=sp, band=tuple(band), start_h=start_h, stop_h=stop_h, weekend_on=weekend_on,
                        holidays_off=holidays_off)

    def simulate(sp):
        return summarize(simulate_series(sim, sim_settings(sp)), sim_settings(sp))

    running_now = sim["running"] & sim["indoor_temp_C"].notna()
    comfort_now = 100 * sim.loc[running_now, "indoor_temp_C"].between(*band).mean()
    sp_kwh, sch_kwh, comfort_new = simulate(setpoint)
    saved = sp_kwh + sch_kwh

    sweep_sp = np.arange(18.0, 28.01, 0.5)
    rows = [(sp, *simulate(sp)) for sp in sweep_sp]
    sp_table = pd.DataFrame(rows, columns=["setpoint", "setpoint_kWh", "schedule_kWh", "comfort_pct"])
    sp_table["total_kWh"] = sp_table["setpoint_kWh"] + sp_table["schedule_kWh"]
    ok = sp_table.loc[sp_table["comfort_pct"].ge(comfort_target)]
    best = ok.loc[ok["total_kWh"].idxmax()] if len(ok) else None

    section(f"Estimated savings ({PERIOD})")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estimated energy saved", f"{saved:,.0f} kWh", f"{-100 * saved / max(1, total_kwh):.1f}% of AC energy",
              delta_color="inverse")
    k2.metric("Estimated cost saved", f"{saved * tariff:,.0f} THB", f"≈ {saved * tariff * 365 / max(1, PERIOD_DAYS):,.0f} THB/year",
              delta_color="off")
    k3.metric("Comfort compliance", f"{comfort_new:.0f}%", f"{comfort_new - comfort_now:+.0f} pts vs now")
    k4.metric("From setpoint / schedule", f"{sp_kwh:,.0f} / {sch_kwh:,.0f}", "kWh", delta_color="off",
              help="Saving from raising the setpoint / saving from AC energy outside the chosen operating window.")

    per_unit = []
    series_now = simulate_series(sim, sim_settings(setpoint))
    for unit_name, part in series_now.groupby("unit"):
        sp_unit = part["setpoint_saved_kW"].sum() * INTERVAL_HOURS
        sch_unit = part["schedule_saved_kW"].sum() * INTERVAL_HOURS
        energy = part["actual_kW"].sum() * INTERVAL_HOURS
        per_unit.append({"AC unit": unit_label(unit_name), "AC energy kWh": energy,
                         "Median indoor °C (running)": part.loc[part["running"], "indoor_temp_C"].median(),
                         "kW per °C": slope.get(unit_name, np.nan), "R²": sens["R2"].get(unit_name, np.nan),
                         "Setpoint saving kWh": sp_unit, "Schedule saving kWh": sch_unit,
                         "Saving %": 100 * (sp_unit + sch_unit) / energy if energy else np.nan})

    g1, g2 = st.columns(2, gap="large")
    fig = go.Figure(go.Scatter(x=sp_table["setpoint"], y=sp_table["comfort_pct"], mode="lines+markers",
                               line=dict(color=GOOD, width=2), marker=dict(size=7), name="Comfort compliance"))
    fig.add_hline(y=comfort_target, line=dict(color=MUTED, dash="dot"), annotation_text="target")
    fig.add_vline(x=setpoint, line=dict(color=EXPECTED, dash="dash"))
    style(fig, 300, f"% running time in {band[0]:g}–{band[1]:g} °C", "Setpoint (°C) · step A: which setpoints keep people comfortable?")
    fig.update_yaxes(range=[0, 100])
    g1.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Scatter(x=sp_table["setpoint"], y=sp_table["total_kWh"], mode="lines+markers",
                               line=dict(color=ACTUAL, width=2), marker=dict(size=7), name="Estimated saving"))
    fig.add_vline(x=setpoint, line=dict(color=EXPECTED, dash="dash"), annotation_text="selected")
    if best is not None:
        fig.add_vline(x=best["setpoint"], line=dict(color=GOOD, dash="dot"), annotation_text="recommended",
                      annotation_position="bottom right")
    g2.plotly_chart(style(fig, 300, f"kWh saved ({PERIOD})", "Setpoint (°C) · step B: of those, which saves the most?"),
                    width="stretch")
    if best is not None:
        st.success(f"**Conclusion: recommended setpoint {best['setpoint']:.1f} °C.** Of the setpoints that keep at least "
                   f"{comfort_target}% of running time in {band[0]:g}–{band[1]:g} °C, it saves the most: "
                   f"{best['total_kWh']:,.0f} kWh, {best['total_kWh'] * tariff:,.0f} THB over {PERIOD} "
                   f"({best['comfort_pct']:.0f}% comfortable vs {comfort_now:.0f}% today). Pair it with the schedule fixes "
                   "from the Priority list for the after-hours waste.")
    elif sp_table["comfort_pct"].isna().all():
        st.info(f"No AC ran in {PERIOD}, so there is nothing to simulate. Pick a longer timeframe or a working day.")
    else:
        best_comfort = sp_table.loc[sp_table["comfort_pct"].idxmax()]
        st.warning(f"No setpoint in 18–28 °C reaches {comfort_target}% compliance. The most comfortable setpoint is "
                   f"**{best_comfort['setpoint']:.1f} °C** ({best_comfort['comfort_pct']:.0f}% in band, "
                   f"{best_comfort['total_kWh']:,.0f} kWh estimated saving). Lower the target or widen the band.")
    with st.expander("Per unit at the selected setpoint"):
        st.caption("kW per °C = how much the unit's power changes per 1 °C, learned from the baseline year. "
                   "R² = how reliable that estimate is (closer to 1 = more reliable).")
        st.dataframe(pd.DataFrame(per_unit).round(2), hide_index=True, width="stretch")
    with st.expander("Assumptions and limits"):
        st.markdown(f"""
- Each zone's current effective setpoint = its median indoor temperature while the unit runs. A new setpoint shifts that zone's whole temperature profile by the difference, so normal variability (start-up pull-down, afternoon peaks) is kept.
- kW saved per °C = the unit's outdoor-temperature regression slope (baseline weekdays, unit running; hour-of-day controlled). Typical literature values are 3–10 % per °C. {SCOPE_NAME} units: {sens['pct_per_degC'].min():.1f}–{sens['pct_per_degC'].max():.1f} % per °C.
- Savings are capped at the unit's actual power and ignore humidity/latent load and compressor part-load efficiency.
- The schedule saving assumes all AC energy outside the window is avoidable. Check for legitimate after-hours use (events, cleaning, servers).
- Comfort compliance uses the zone S1 sensor only; it does not measure occupant satisfaction.
""")

# ================================================================ digital twin
with tab_twin:
    c_head, c_date, c_days = st.columns([4, 1, 1], vertical_alignment="bottom")
    c_head.markdown("**Digital twin** · choose the days to replay", help="Interactive 3D model of Chamchuri 5, traced from the CU-BEMS paper's floor plan "
                                          "(Floors 3–7) and building view. Drag to rotate, right-drag to pan, scroll to zoom. "
                                          "Hover a zone for live readings; click it to fly there and open its detail panel. "
                                          "Press play (or Space) to replay the selected days; the sun follows its real path "
                                          "over Bangkok. Schematic to scale, not a surveyed model.")
    # The twin's own clock shows the weekday, so no separate weekday caption here (it pushed the row out of line)
    twin_start = c_date.date_input("From", max(pd.Timestamp("2019-07-22").date(), data_first), min_value=data_first,
                                   max_value=data_last, format="DD/MM/YYYY", key="twin_start")
    twin_days = c_days.selectbox("Days to load", [1, 3, 7], index=2, key="twin_days",
                                 format_func=lambda n: f"{n} day" + ("s" if n > 1 else ""))
    mode = st.segmented_control("Mode", ["Replay", "Forecast next day", "What-if"], default="Replay", key="twin_mode",
                                help="Replay = what happened. Forecast next day = the forecast for the day after the timeframe "
                                     "(including 1 Jan 2020, where no actual exists), split across units by their baseline share. "
                                     "What-if = the Setpoint & schedule simulator's proposal applied to the replayed days.") or "Replay"
    if mode == "Forecast next day":
        twin_frame(period_end.normalize(), 1, compact=False, height=760, mode=mode)
        st.caption(f"Forecast for {period_end:%A %d %B %Y}. Each unit's share of the floor forecast is its baseline share of "
                   "load for that hour of the week (learned from the training year). The 80% band uses error quantiles from Jul–Sep.")
    elif mode == "What-if":
        _s = settings_from_state(st.session_state)
        twin_frame(twin_start, min(twin_days, 3), compact=False, height=760, mode=mode)
        st.caption(f"What-if: setpoint {_s.setpoint:g} °C, AC window {_s.start_h:g}–{_s.stop_h:g} h, weekends "
                   f"{'on' if _s.weekend_on else 'off'}, holidays {'off' if _s.holidays_off else 'unchanged'} (set in the "
                   "Setpoint & schedule simulator tab; at most 3 days load to keep the view fast). Model estimates: the comfort "
                   "shift is a simplification, as in the simulator, and the 'what-if indoor °C' colour uses the 23–26 °C band.")
    else:
        twin_frame(twin_start, twin_days, compact=False, height=760)
    _pick, _btn = st.columns([3, 1], vertical_alignment="bottom")
    _asset = _pick.selectbox("Unit history", BRICK_UNITS["asset_id"].tolist(), key="twin_asset",
                             format_func=lambda a: BRICK_UNITS.set_index("asset_id").loc[a, "label"],
                             help="The 3D view cannot call back into the dashboard, so open a unit's history card from here.")
    if _btn.button("Open asset card", key="twin_card"):
        asset_dialog(_asset)
    st.caption("Zone colours and the glowing ceiling lights come from the 15-minute data. Lights count as 'after hours' when a zone's "
               f"lighting is above {twin.LIGHT_ON_KW} kW while every AC unit on that floor is normally off (stairwells excluded). "
               f"The priority-level colours use the review period ({PERIOD}). "
               "Floors 1, 2 and 6 are shown for context but are not in this analysis.")


# ================================================================ ask the data (chat)
# Read-only tools Gemini can call to answer FM questions. Their names, arguments and docstrings form the tool schema,
# so keep them stable: changing them mid-conversation would invalidate the conversation's history.
import functools
import json

import fm_chat

DATA_FIRST, DATA_LAST = pd.Timestamp(data_first), pd.Timestamp(data_last)


def _json(obj):
    return json.dumps(obj, default=lambda o: None if pd.isna(o) else str(o), ensure_ascii=False)


def _r(value, digits=1):
    return None if value is None or pd.isna(value) else round(float(value), digits)


def _scope_arg(floor_arg):
    """'' = the dashboard's current view; 'all'/'building' = whole building; '7' / 'Floor 7' = that floor."""
    text = str(floor_arg or "").strip().lower()
    if not text:
        return floor
    if text in ("all", "building", "whole building"):
        return ALL
    digits = re.findall(r"\d+", text)
    if digits and int(digits[0]) in FLOORS:
        return int(digits[0])
    raise ValueError(f"Unknown floor '{floor_arg}'. Use one of {FLOORS} or 'all'.")


def _dates(start_date, end_date):
    """Inclusive YYYY-MM-DD dates → (start, exclusive end), defaulting to the dashboard's timeframe."""
    start = pd.Timestamp(start_date) if start_date else period_start
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1) if end_date else period_end
    start, end = max(start.normalize(), DATA_FIRST), min(end.normalize(), DATA_LAST + pd.Timedelta(days=1))
    if end <= start:
        raise ValueError(f"No data in that range. Data covers {data_first} to {data_last}.")
    return start, end


def _scope_name(fl):
    return "Whole building (Floors " + ", ".join(map(str, FLOORS)) + ")" if fl == ALL else f"Floor {fl}"


def _period_text(start, end):
    return f"{start:%a %d %b %Y} to {end - pd.Timedelta(days=1):%a %d %b %Y} ({(end - start).days} days)"


def _scope_results(fl, start, end):
    """Flags, events, unit ranking and KPIs for a scope and date range (same rules as the dashboard)."""
    d_ = flags_for(fl, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist, OPEN, EXCEPTIONS)
    d_ = d_.loc[d_["Date"].ge(start) & d_["Date"].lt(end)]
    ev = build_events(d_, TARIFF_THB, track_record=TRACK)
    pr = unit_priority(d_, ev, (end - start).days)
    return d_, ev, pr, period_kpis(fl, start, end, OPEN)


def _all_units():
    return sorted({u for f in FLOORS for u in load(f)["detail"]["unit"].unique()})


def _find_unit(name):
    """Match 'F7 Z1 AC4', 'Floor 7 Zone 1 AC4' or 'F7_z1_AC4(kW)' to a unit id."""
    nums = re.findall(r"\d+", str(name))
    for u in _all_units():
        m = re.search(r"F(\d+)_z(\d+)_AC(\d+)", u)
        if m and nums[:3] == [m[1], m[2], m[3]]:
            return u
    raise ValueError(f"Unknown AC unit '{name}'. Units: " + ", ".join(unit_label(u, with_floor=True) for u in _all_units()))


def _event_rows(ev, limit):
    return [{"unit": unit_label(e["unit"], with_floor=True), "start": f"{e['start']:%a %d %b %Y %H:%M}",
             "hours": _r(e["duration_h"], 2), "peak_kW": _r(e["peak_kW"]), "excess_kWh": _r(e["excess_kWh"]),
             "excess_THB": _r(e["excess_THB"], 0), "confidence": _r(e["confidence"], 0),
             "pattern": PATTERNS[e["pattern"]][2], "effort": e["effort"], "triage_score": _r(e["triage_score"], 0),
             "priority": e["priority_level"].split()[-1], "why": e["reason"]}
            for _, e in ev.head(limit).iterrows()]


def _forecast_for(fl):
    """(hourly forecast table, chosen model) for a scope, with public holidays forecast as zero."""
    table = load(fl)["forecast"].copy()
    model = load(fl).get("selected_model") or load(fl)["validation_metrics"]["MAE_kW"].idxmin()
    cols = [c for c in table.columns if c != "floor_kW"]
    if use_holidays:
        table.loc[table.index.normalize().isin(CLOSED_DATES), cols] = 0.0
    return table, model


def _tool(fn):
    """Run a tool body, turning bad arguments into a short error the model can read and correct."""
    @functools.wraps(fn)  # keeps the name, docstring, signature and type hints the tool schema is built from
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as error:
            return _json({"error": str(error)})
    return wrapped


@_tool
def dashboard_overview(floor: str = "", start_date: str = "", end_date: str = "") -> str:
    """Headline numbers for a scope and period: AC energy, energy used while the building was closed, 15-minute peak,
    flagged events, excess kWh, the top 3 units to review, opening hours and holidays in the period.

    Args:
        floor: '' for the dashboard's current view, 'all' for the whole building, or a floor number such as '7'.
        start_date: First day, YYYY-MM-DD. Empty = the dashboard's timeframe.
        end_date: Last day (inclusive), YYYY-MM-DD. Empty = the dashboard's timeframe.
    """
    fl = _scope_arg(floor)
    start, end = _dates(start_date, end_date)
    d_, ev, pr, k = _scope_results(fl, start, end)
    closures = [f"{d:%a %d %b}: {closure_name(d.date())}" for d in CLOSED_DATES if start <= d < end]
    return _json({"scope": _scope_name(fl), "period": _period_text(start, end),
                  "ac_energy_kWh": _r(k["energy_kWh"], 0), "used_while_closed_kWh": _r(k["after_kWh"], 0),
                  "used_while_closed_pct": _r(k["after_pct"]), "peak_15min_kW": _r(k["peak_kW"]),
                  "ac_cost_THB": _r(k["energy_kWh"] * TARIFF_THB, 0), "tariff_THB_per_kWh": TARIFF_THB,
                  "flagged_events": len(ev), "excess_kWh_flagged": _r(ev["excess_kWh"].sum() if len(ev) else 0, 0),
                  "top_units": [{"unit": unit_label(u, with_floor=True), "priority": r["priority_level"].split()[-1],
                                 "triage_score": _r(r["triage_score"], 0), "excess_THB": _r(r["excess_THB"], 0),
                                 "confidence": _r(r["confidence"], 0), "share_of_score_pct": _r(r["share_pct"], 0)}
                                for u, r in pr.head(3).iterrows()],
                  "opening_hours": OPEN_TEXT, "closures_in_period": closures})


@_tool
def rank_units(floor: str = "", start_date: str = "", end_date: str = "", limit: int = 10) -> str:
    """AC units ranked by triage score = excess THB x confidence / effort (most worth checking first), with priority
    level, excess kWh and THB, confidence (0-100), fix effort, main pattern, events and the share of time flagged.

    Args:
        floor: '' for the dashboard's current view, 'all' for the whole building, or a floor number such as '7'.
        start_date: First day, YYYY-MM-DD. Empty = the dashboard's timeframe.
        end_date: Last day (inclusive), YYYY-MM-DD. Empty = the dashboard's timeframe.
        limit: How many units to return (1-16).
    """
    fl = _scope_arg(floor)
    start, end = _dates(start_date, end_date)
    _, _, pr, _ = _scope_results(fl, start, end)
    rows = [{"rank": int(r["rank"]), "unit": unit_label(u, with_floor=True), "priority": r["priority_level"].split()[-1],
             "triage_score": _r(r["triage_score"], 0), "excess_kWh": _r(r["flagged_excess_kWh"], 0),
             "excess_THB": _r(r["excess_THB"], 0), "per_day_THB": _r(r["per_day_THB"], 0),
             "confidence": _r(r["confidence"], 0), "effort": r["effort"],
             "main_pattern": PATTERNS[r["pattern"]][2] if isinstance(r["pattern"], str) else None,
             "share_of_score_pct": _r(r["share_pct"], 0), "events": int(r["events"]),
             "time_flagged_pct": _r(r["flagged_pct"])} for u, r in pr.head(max(1, min(int(limit), 16))).iterrows()]
    return _json({"scope": _scope_name(fl), "period": _period_text(start, end), "units": rows})


@_tool
def list_events(floor: str = "", start_date: str = "", end_date: str = "", unit: str = "", limit: int = 10,
                order: str = "priority") -> str:
    """Flagged episodes (consecutive flagged 15-minute readings): unit, start, hours, peak kW, excess kWh and THB,
    confidence, pattern, fix effort, triage score and why it was flagged.

    Args:
        floor: '' for the dashboard's current view, 'all' for the whole building, or a floor number such as '7'.
        start_date: First day, YYYY-MM-DD. Empty = the dashboard's timeframe.
        end_date: Last day (inclusive), YYYY-MM-DD. Empty = the dashboard's timeframe.
        unit: Optional single unit, e.g. 'F7 Z1 AC4'. Empty = all units in scope.
        limit: How many events to return (1-30).
        order: 'priority' (highest triage score first) or 'time' (earliest first).
    """
    fl = _scope_arg(floor)
    if unit:
        u_id = _find_unit(unit)
        fl = int(re.search(r"F(\d+)_", u_id)[1])
    start, end = _dates(start_date, end_date)
    _, ev, _, _ = _scope_results(fl, start, end)
    if len(ev) and unit:
        ev = ev.loc[ev["unit"].eq(u_id)]
    if len(ev) and order == "time":
        ev = ev.sort_values("start")
    return _json({"scope": unit_label(u_id, with_floor=True) if unit else _scope_name(fl),
                  "period": _period_text(start, end), "events_total": len(ev),
                  "events": _event_rows(ev, max(1, min(int(limit), 30))) if len(ev) else []})


@_tool
def unit_profile(unit: str, start_date: str = "", end_date: str = "") -> str:
    """Everything about one AC unit in a period: energy, running hours, energy while the building was closed, flags and
    top events, priority on its floor, indoor temperature and comfort, comparison with the same unit on other floors,
    temperature sensitivity, and its daily (or weekly) energy.

    Args:
        unit: The unit, e.g. 'F7 Z1 AC4' or 'Floor 7 Zone 1 AC4'.
        start_date: First day, YYYY-MM-DD. Empty = the dashboard's timeframe.
        end_date: Last day (inclusive), YYYY-MM-DD. Empty = the dashboard's timeframe.
    """
    u_id = _find_unit(unit)
    fl = int(re.search(r"F(\d+)_", u_id)[1])
    start, end = _dates(start_date, end_date)
    d_, ev, pr, _ = _scope_results(fl, start, end)
    part = d_.loc[d_["unit"].eq(u_id) & d_["eligible"]]
    running = part["actual_kW"].gt(on_kw)
    energy = part.set_index("Date")["actual_kW"] * INTERVAL_HOURS
    series = energy.resample("D" if (end - start).days <= 31 else "W-MON", label="left", closed="left").sum()
    sens_ = load(fl)["sensitivity"]
    position = position_of(u_id)
    peers = {f: load(f)["detail"] for f in FLOORS}
    peer_means = {f: (p.loc[p["unit"].str.contains(f"_{position}") & p["eligible"] & p["Date"].ge(start) & p["Date"].lt(end),
                            "actual_kW"].mean()) for f, p in peers.items()}
    others = [v for f, v in peer_means.items() if f != fl and pd.notna(v)]
    peer_med = float(np.median(others)) if others else np.nan
    u_ev = ev.loc[ev["unit"].eq(u_id)] if len(ev) else ev
    row = pr.loc[u_id] if u_id in pr.index else None
    return _json({
        "unit": unit_label(u_id, with_floor=True), "period": _period_text(start, end),
        "ac_energy_kWh": _r(energy.sum(), 0), "running_hours": _r(running.sum() * INTERVAL_HOURS, 0),
        "used_while_closed_kWh": _r(part.loc[part["closed"] & running, "actual_kW"].sum() * INTERVAL_HOURS, 0),
        "time_flagged_pct": _r(100 * part["flagged"].mean()) if len(part) else None, "events": len(u_ev),
        "excess_kWh": _r(u_ev["excess_kWh"].sum() if len(u_ev) else 0, 0),
        "priority_on_its_floor": None if row is None else f"{row['priority_level'].split()[-1]} (rank {int(row['rank'])} of {len(pr)})",
        "top_events": _event_rows(u_ev, 3) if len(u_ev) else [],
        "median_indoor_C_while_running": _r(part.loc[running, "indoor_temp_C"].median()),
        "pct_running_time_in_23_26C": _r(100 * part.loc[running, "indoor_temp_C"].between(23, 26).mean()) if running.any() else None,
        "mean_kW_vs_same_unit_other_floors_pct": _r(100 * (peer_means[fl] / peer_med - 1), 0)
        if pd.notna(peer_med) and peer_med >= on_kw else None,
        "kW_per_degC_outdoor": _r(sens_["kW_per_degC_outdoor"].get(u_id), 2),
        "pct_per_degC": _r(sens_["pct_per_degC"].get(u_id)),
        ("daily_kWh" if (end - start).days <= 31 else "weekly_kWh"): {f"{t:%Y-%m-%d}": _r(v, 0) for t, v in series.items()},
    })


@_tool
def waste_timing(floor: str = "", start_date: str = "", end_date: str = "") -> str:
    """When the extra energy happens: share of flagged excess kWh on weekday working hours, other weekday hours,
    Saturdays, Sundays and public holidays, plus the weekday/hour slots flagged most often.

    Args:
        floor: '' for the dashboard's current view, 'all' for the whole building, or a floor number such as '7'.
        start_date: First day, YYYY-MM-DD. Empty = the dashboard's timeframe.
        end_date: Last day (inclusive), YYYY-MM-DD. Empty = the dashboard's timeframe.
    """
    fl = _scope_arg(floor)
    start, end = _dates(start_date, end_date)
    d_, _, _, _ = _scope_results(fl, start, end)
    fr = d_.loc[d_["flagged"]]
    dow, hour = fr["Date"].dt.dayofweek, fr["Date"].dt.hour
    cat = np.select([fr["holiday"], dow.eq(5), dow.eq(6), hour.between(8, 16)],
                    ["Public holiday", "Saturday", "Sunday", "Weekday 08:00–17:00"], "Weekday, other hours")
    kwh = fr.groupby(cat)["excess_kWh_est"].sum().reindex(WHEN_ORDER, fill_value=0)
    valid = d_.loc[d_["eligible"]]
    slots = (100 * valid.groupby([valid["Date"].dt.dayofweek, valid["Date"].dt.hour])["flagged"].mean()).nlargest(5)
    return _json({"scope": _scope_name(fl), "period": _period_text(start, end),
                  "excess_share_pct": {k: _r(100 * v / max(1e-9, kwh.sum()), 0) for k, v in kwh.items()},
                  "excess_kWh": {k: _r(v, 0) for k, v in kwh.items()},
                  "most_flagged_slots": [{"slot": f"{DAYS[dw]} {h:02d}:00", "pct_of_readings_flagged": _r(v)}
                                         for (dw, h), v in slots.items()]})


@_tool
def compare_floors(start_date: str = "", end_date: str = "") -> str:
    """Compare Floors 3, 4, 5 and 7 (same AC layout): energy, energy while closed, peak, events, excess THB, triage
    score and median indoor temperature per floor, plus the units that draw most above the same unit on other floors.

    Args:
        start_date: First day, YYYY-MM-DD. Empty = the dashboard's timeframe.
        end_date: Last day (inclusive), YYYY-MM-DD. Empty = the dashboard's timeframe.
    """
    start, end = _dates(start_date, end_date)
    rows, means = [], {}
    for f in FLOORS:
        d_, ev, pr, k = _scope_results(f, start, end)
        rows.append({"floor": f"Floor {f}", "ac_energy_kWh": _r(k["energy_kWh"], 0),
                     "used_while_closed_pct": _r(k["after_pct"]), "peak_15min_kW": _r(k["peak_kW"]),
                     "flagged_events": len(ev), "excess_THB": _r(pr["excess_THB"].sum(), 0),
                     "triage_score": _r(pr["triage_score"].sum(), 0),
                     "median_indoor_C_running": _r(d_.loc[d_["eligible"] & d_["actual_kW"].gt(on_kw), "indoor_temp_C"].median())})
        for u, part in d_.loc[d_["eligible"]].groupby("unit"):
            means[u] = part["actual_kW"].mean()
    gaps = []
    for u, m in means.items():
        others = [v for o, v in means.items() if position_of(o) == position_of(u) and o != u]
        med = np.nanmedian(others) if others else np.nan
        if pd.notna(med) and med >= on_kw:
            gaps.append((u, 100 * (m / med - 1)))
    gaps.sort(key=lambda x: -x[1])
    return _json({"period": _period_text(start, end), "floors": rows,
                  "units_most_above_peers": [{"unit": unit_label(u, with_floor=True), "vs_peers_pct": _r(g, 0)}
                                             for u, g in gaps[:5]]})


@_tool
def energy_timeseries(floor: str = "", start_date: str = "", end_date: str = "", resolution: str = "day") -> str:
    """Measured AC energy over time for a scope: totals per hour (periods up to 7 days), day or week.

    Args:
        floor: '' for the dashboard's current view, 'all' for the whole building, or a floor number such as '7'.
        start_date: First day, YYYY-MM-DD. Empty = the dashboard's timeframe.
        end_date: Last day (inclusive), YYYY-MM-DD. Empty = the dashboard's timeframe.
        resolution: 'hour', 'day' or 'week'.
    """
    fl = _scope_arg(floor)
    start, end = _dates(start_date, end_date)
    hourly = load(fl)["forecast"]["floor_kW"]
    hourly = hourly.loc[(hourly.index >= start) & (hourly.index < end)]
    if resolution == "hour" and (end - start).days <= 7:
        values = {f"{t:%a %d %b %H:00}": _r(v, 0) for t, v in hourly.items()}
    else:
        resolution = "week" if resolution == "week" or (end - start).days > 92 else "day"
        series = hourly.resample("D" if resolution == "day" else "W-MON", label="left", closed="left").sum(min_count=1)
        values = {f"{t:%a %d %b %Y}": _r(v, 0) for t, v in series.items()}
    return _json({"scope": _scope_name(fl), "period": _period_text(start, end), "unit": "kWh",
                  "resolution": resolution, "values": values,
                  "note": "Hours with a meter gap on any unit are left out of the totals."})


@_tool
def day_forecast(date: str, floor: str = "") -> str:
    """The day-ahead hourly AC demand forecast for one day (made with data up to the midnight before), with the actual
    load when the day is in the data: peak, daily energy and that day's error. Covers 2019-07-01 to 2020-01-01.

    Args:
        date: The day, YYYY-MM-DD.
        floor: '' for the dashboard's current view, 'all' for the whole building, or a floor number such as '7'.
    """
    fl = _scope_arg(floor)
    day = pd.Timestamp(date).normalize()
    table, model = _forecast_for(fl)
    hours = pd.date_range(day, periods=24, freq="h")
    part = table.reindex(hours)
    if part[model].notna().any():
        pred, actual = part[model].fillna(0.0), part["floor_kW"]
    elif day == pd.Timestamp(load(fl)["next24"].index[0]).normalize():
        pred, actual = load(fl)["next24"]["predicted_kW"].copy(), pd.Series(np.nan, index=hours)
        if use_holidays and closure_name(day.date()):
            pred[:] = 0.0
    else:
        raise ValueError("Forecasts exist for 2019-07-01 to 2020-01-01 only.")
    has_actual = actual.notna().any()
    return _json({"scope": _scope_name(fl), "date": f"{day:%a %d %b %Y}", "method": MODEL_LABELS.get(model, model),
                  "public_holiday": closure_name(day.date()) if use_holidays else None,
                  "forecast_peak_kW": _r(pred.max()), "forecast_peak_hour": f"{pred.idxmax():%H:00}",
                  "forecast_daily_kWh": _r(pred.sum(), 0),
                  "actual_peak_kW": _r(actual.max()) if has_actual else None,
                  "actual_daily_kWh": _r(actual.sum(), 0) if has_actual else None,
                  "mean_abs_error_kW_this_day": _r((pred - actual).abs().mean()) if has_actual else None,
                  "hourly_kW": {f"{t:%H:00}": {"forecast": _r(p), "actual": _r(a) if has_actual else None}
                                for t, p, a in zip(hours, pred.values, actual.reindex(hours).values)}})


@_tool
def forecast_accuracy(floor: str = "") -> str:
    """How accurate each forecasting method was on Jul–Dec 2019 (data the methods never saw), with public holidays
    forecast as zero: average error, peak-hour error, bias, and which method the dashboard uses.

    Args:
        floor: '' for the dashboard's current view, 'all' for the whole building, or a floor number such as '7'.
    """
    fl = _scope_arg(floor)
    table, model = _forecast_for(fl)
    cols = [c for c in table.columns if c != "floor_kW"]
    m = forecast_metrics(table, cols)
    return _json({"scope": _scope_name(fl), "method_used": MODEL_LABELS.get(model, model),
                  "methods": [{"method": MODEL_LABELS.get(i, i), "mean_abs_error_kW": _r(r["MAE_kW"], 2),
                               "peak_hour_error_kW": _r(r["peak_hour_MAE_kW"], 2), "bias_pct": _r(r["NMBE_pct"]),
                               "cv_rmse_pct": _r(r["CV_RMSE_pct"])} for i, r in m.iterrows()],
                  "hours_scored": int(m["hours"].iloc[0]) if len(m) else 0})


@_tool
def holidays_and_hours(start_date: str = "", end_date: str = "") -> str:
    """The building's opening hours as set in the dashboard, and the Thai public holidays and closure days in a period.

    Args:
        start_date: First day, YYYY-MM-DD. Empty = the dashboard's timeframe.
        end_date: Last day (inclusive), YYYY-MM-DD. Empty = the dashboard's timeframe.
    """
    start = pd.Timestamp(start_date) if start_date else period_start
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1) if end_date else period_end
    return _json({"opening_hours": OPEN_TEXT, "holidays_used_as_closures": use_holidays,
                  "closures": [{"date": f"{d:%a %d %b %Y}", "name": closure_name(d.date())}
                               for d in CLOSED_DATES if start <= d < end]})


@_tool
def smart_assistant_findings() -> str:
    """The dashboard's rule-based checks for its current view and timeframe: units off during usual hours, units
    running while their zone stays warm, several units on one floor high or off together, waste outside usual hours,
    peer outliers, overcooling, missing data and the next day's forecast, each with a suggested action."""
    return _json({"scope": scope_desc, "period": PERIOD,
                  "findings": [{"severity": x["severity"], "finding": x["title"], "action": x["action"],
                                "detail": x["detail"], "where_in_dashboard": x["where"]} for x in assistant_findings()]})


@_tool
def what_if(setpoint_c: float = 0, weekday_start_hour: float = 0, weekday_end_hour: float = 24,
            keep_weekends: bool = True, ac_off_on_public_holidays: bool = False, floor: str = "",
            start_date: str = "", end_date: str = "") -> str:
    """Estimate the energy and cost saved by a setpoint change and/or a stricter AC schedule, and the effect on comfort
    (share of running time inside 23–26 °C). Same model as the dashboard's simulator; screening estimates only.

    Args:
        setpoint_c: New indoor setpoint in °C (e.g. 25). 0 = keep today's setpoints.
        weekday_start_hour: Hour the AC may start on weekdays (0-24, e.g. 7.5). 0 with end 24 = no change.
        weekday_end_hour: Hour the AC must stop on weekdays (0-24, e.g. 18).
        keep_weekends: False = proposal of no AC at weekends.
        ac_off_on_public_holidays: True = proposal of no AC on public holidays and closure days.
        floor: '' for the dashboard's current view, 'all' for the whole building, or a floor number such as '7'.
        start_date: First day, YYYY-MM-DD. Empty = the dashboard's timeframe.
        end_date: Last day (inclusive), YYYY-MM-DD. Empty = the dashboard's timeframe.
    """
    fl = _scope_arg(floor)
    start, end = _dates(start_date, end_date)
    det = load(fl)["detail"]
    det = det.loc[det["Date"].ge(start) & det["Date"].lt(end)]
    slope_ = load(fl)["sensitivity"]["kW_per_degC_outdoor"].clip(lower=0)
    sim = prepare_sim(det, slope_, on_kw, float(weekday_start_hour), float(weekday_end_hour), bool(keep_weekends),
                      bool(ac_off_on_public_holidays))
    if sim.empty:
        raise ValueError("No AC readings in that period.")
    band_ = (23.0, 26.0)
    running_now = sim["running"] & sim["indoor_temp_C"].notna()
    comfort_now = 100 * sim.loc[running_now, "indoor_temp_C"].between(*band_).mean()
    if setpoint_c and setpoint_c > 0:
        sp_kwh, sch_kwh, comfort_new = simulate_setpoint(sim, float(setpoint_c), band_)
    else:  # schedule change only: setpoints and comfort stay as they are
        sp_kwh, sch_kwh, comfort_new = 0.0, simulate_setpoint(sim, 0.0, band_)[1], comfort_now
    total = sim["actual_kW"].sum() * INTERVAL_HOURS
    saved = sp_kwh + sch_kwh
    current_sp = sim.loc[sim["running"]].groupby("unit")["indoor_temp_C"].median()
    return _json({"scope": _scope_name(fl), "period": _period_text(start, end),
                  "saved_kWh": _r(saved, 0), "saved_pct_of_ac_energy": _r(100 * saved / total if total else np.nan),
                  "saved_THB": _r(saved * TARIFF_THB, 0), "tariff_THB_per_kWh": TARIFF_THB,
                  "from_setpoint_kWh": _r(sp_kwh, 0), "from_schedule_kWh": _r(sch_kwh, 0),
                  "comfort_pct_now": _r(comfort_now, 0), "comfort_pct_after": _r(comfort_new, 0),
                  "current_effective_setpoint_C_by_unit": {unit_label(u, with_floor=True): _r(v)
                                                           for u, v in current_sp.items()},
                  "assumptions": "kW saved per °C = each unit's outdoor-temperature slope from the baseline year; "
                                 "schedule savings count all AC energy outside the window as avoidable."})


CHAT_TOOLS = [dashboard_overview, rank_units, list_events, unit_profile, waste_timing, compare_floors,
              energy_timeseries, day_forecast, forecast_accuracy, holidays_and_hours, smart_assistant_findings, what_if]
TOOL_LABELS = {"dashboard_overview": "overview", "rank_units": "unit ranking", "list_events": "events",
               "unit_profile": "unit profile", "waste_timing": "when waste happens", "compare_floors": "floor comparison",
               "energy_timeseries": "energy over time", "day_forecast": "forecast", "forecast_accuracy": "forecast accuracy",
               "holidays_and_hours": "holidays & hours", "smart_assistant_findings": "smart assistant", "what_if": "what-if"}
SUGGESTIONS = ["Which AC units should we check first, and why?",
               "What happened on Floor 7 on Saturday 27 July 2019?",
               "What load should we plan for on Monday 21 October 2019, and how good was the forecast?",
               "How much would raising the setpoint to 25 °C save, and is it still comfortable?",
               "Which floor wastes the most energy while the building is closed?"]

# Floating "Ask the data" button (bottom-right corner, on every tab). Keyed, so it stays open across reruns.
with st.container(key="chat-fab"):
    chat_box = st.popover("✦", key="chat_open")
with chat_box, st.container(key="chat-panel"):
    api_key = fm_chat.get_api_key(st.secrets) or st.session_state.get("gemini_key")
    chat_model = fm_chat.get_model(st.secrets)
    st.session_state.setdefault("chat_api", [])      # the API conversation (append-only)
    st.session_state.setdefault("chat_display", [])  # what the panel shows
    h1, h2 = st.columns([3, 1], vertical_alignment="center")
    h1.markdown("**✦ Ask the data**",
                help=f"Google Gemini ({chat_model}) answers by calling read-only tools over the same data and rules as the "
                     "dashboard, so figures come from the analysis, not from memory. It uses the dashboard's current "
                     "view (floor, timeframe, opening hours) unless you ask about something else. Questions and the "
                     "data needed to answer them are sent to the Gemini API. On the free tier Google limits requests per "
                     "minute and per day, and may use free-tier prompts to improve its products.")
    if h2.button("Clear", type="tertiary", width="stretch", disabled=not st.session_state["chat_display"]):
        st.session_state["chat_api"], st.session_state["chat_display"] = [], []
        st.rerun()

    if not api_key:
        st.markdown("**Add a Gemini API key to start.** Create one free at "
                    "[Google AI Studio](https://aistudio.google.com) (API keys). The best place for it is "
                    "`.streamlit/secrets.toml` in the project folder, with the line\n\n`GEMINI_API_KEY = \"...\"`\n\n"
                    "then restart the app (on Streamlit Cloud: App → Settings → Secrets). Or paste a key below "
                    "for this browser session only.")
        pasted = st.text_input("Gemini API key", type="password")
        if pasted:
            st.session_state["gemini_key"] = pasted.strip()
            st.rerun()
    else:
        msgs = st.container(height=430, border=False)
        with msgs:
            for item in st.session_state["chat_display"]:
                with st.chat_message(item["role"]):
                    st.markdown(item["text"])
                    if item.get("tools"):
                        st.caption("Looked at: " + ", ".join(dict.fromkeys(TOOL_LABELS.get(t, t) for t in item["tools"])))
            if not st.session_state["chat_display"]:
                st.caption("Questions about AC energy, anomalies, comfort and forecasts, answered from the dashboard's "
                           "own data. Try one:")
                for text in SUGGESTIONS:
                    if st.button(text, width="stretch"):
                        st.session_state["chat_pending"] = text
                        st.rerun()
        question = st.chat_input("Ask about the data…") or st.session_state.pop("chat_pending", None)
        if question:
            st.session_state["chat_display"].append({"role": "user", "text": question})
            # The current view travels with the question, so the system prompt never changes mid-conversation
            view = (f"[Dashboard view: {scope_desc} · timeframe {PERIOD} ({period_start:%Y-%m-%d} to "
                    f"{period_end - pd.Timedelta(days=1):%Y-%m-%d}) · opening hours {OPEN_TEXT}]")
            with msgs:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Looking at the data…"):
                        try:
                            answer, new_turns, used = fm_chat.ask(api_key, chat_model, st.session_state["chat_api"],
                                                                  f"{view}\n\n{question}", CHAT_TOOLS)
                        except Exception as error:  # API errors are explained in plain words; anything else is shown as is
                            answer, new_turns, used = "⚠️ " + fm_chat.error_message(error), [], []
                    st.markdown(answer)
                    if used:
                        st.caption("Looked at: " + ", ".join(dict.fromkeys(TOOL_LABELS.get(t, t) for t in used)))
            st.session_state["chat_api"] += new_turns
            st.session_state["chat_display"].append({"role": "assistant", "text": answer, "tools": used})
