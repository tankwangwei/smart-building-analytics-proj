"""Multi-floor AC dashboard: priority list, setpoint/schedule simulator, forecasts and floor comparison.

Run `python run_all_floors.py` first (it writes ac_analysis_outputs/floor_<n>/), then:
    streamlit run dashboard/app.py
"""
import datetime as dt
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import twin
from thai_holidays import HOLIDAYS, OBSERVED_CLOSURES, closure_name

ROOT = Path(os.environ.get("AC_OUTPUT_FOLDER", Path(__file__).resolve().parents[1] / "ac_analysis_outputs"))
FLOORS = sorted(int(p.name.split("_")[1]) for p in ROOT.glob("floor_*") if (p / "anomaly_15min_detail.csv").exists())
ALL = "all"  # selector value for the whole-building view
DEFAULT_SCOPE = ALL
SCOPE = None  # set from the sidebar; unit labels include the floor in the whole-building view
INTERVAL_HOURS = 0.25
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


def position_of(unit):
    """Same-position key shared by floors with the same layout, e.g. 'z1_AC4'."""
    return unit.split("_", 1)[1].replace("(kW)", "")


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


CLOSED_DATES = pd.DatetimeIndex(sorted({*HOLIDAYS, *OBSERVED_CLOSURES}))


def building_closed(dates, open_spec):
    """True where the building is closed: a closed weekday, outside opening hours, or (if enabled) a public holiday or
    observed closure. open_spec = (open weekdays, opening minute, closing minute, use holidays)."""
    days, start_min, end_min, use_holidays = open_spec
    minutes = dates.dt.hour * 60 + dates.dt.minute
    closed = ~dates.dt.dayofweek.isin(days) | minutes.lt(start_min) | minutes.ge(end_min)
    if use_holidays:
        closed |= dates.dt.normalize().isin(CLOSED_DATES)
    return closed


def run_rules(detail, thresholds, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist):
    """Re-apply the notebook's three flagging rules with adjustable thresholds."""
    d = detail
    upper = d["IQR_q3_kW"] + k_iqr * (d["IQR_q3_kW"] - d["IQR_q1_kW"])
    running = d["eligible"] & d["actual_kW"].gt(on_kw)
    stat = running & d["actual_kW"].gt(upper)
    # After hours = running outside the unit's usual schedule, or while the building is closed
    after = running & (d["schedule_on"].eq(False) | d["closed"])
    q1r = d["unit"].map(thresholds["residual_q1_kW"])
    q3r = d["unit"].map(thresholds["residual_q3_kW"])
    above = running & d["residual_kW"].gt(q3r + k_res * (q3r - q1r))
    # Runs of consecutive intervals within one unit (the detail is a regular 15-minute grid per unit)
    run_id = (above.ne(above.shift()) | d["unit"].ne(d["unit"].shift())).cumsum()
    persistent = above & above.groupby(run_id).transform("size").ge(persist)
    flagged = (stat & use_stat) | (after & use_after) | (persistent & use_persist)
    return pd.DataFrame({"upper_kW": upper, "statistical": stat, "after_hours": after,
                         "persistent": persistent, "flagged": flagged}, index=d.index)


@st.cache_data(show_spinner=False)
def flags_for(floor, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist, open_spec):
    data = load(floor)
    detail = data["detail"].copy()
    detail["closed"] = building_closed(detail["Date"], open_spec)
    detail["holiday"] = detail["Date"].dt.normalize().isin(CLOSED_DATES) & open_spec[3]
    # While the building is closed, normal AC use is zero, so all AC energy then counts as excess
    detail.loc[detail["closed"], "expected_kW"] = 0.0
    detail["residual_kW"] = detail["actual_kW"] - detail["expected_kW"]
    detail["excess_kWh_est"] = detail["residual_kW"].clip(lower=0) * INTERVAL_HOURS
    flags = run_rules(detail, data["thresholds"], k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist)
    # Replace the notebook's default-threshold flags with the recomputed ones
    base = detail.drop(columns=[c for c in flags.columns if c in detail.columns])
    return pd.concat([base, flags], axis=1)


def build_events(d, hot_weight, after_weight):
    f = d.loc[d["flagged"]].copy()
    if f.empty:
        return pd.DataFrame()
    run = (d["flagged"].ne(d["flagged"].shift()) | d["unit"].ne(d["unit"].shift())).cumsum()
    f["run"] = run.loc[f.index]
    events = f.groupby("run").agg(unit=("unit", "first"), start=("Date", "min"), last=("Date", "max"),
                                  intervals=("Date", "size"), peak_kW=("actual_kW", "max"),
                                  excess_kWh=("excess_kWh_est", "sum"), after_hours=("after_hours", "sum"),
                                  statistical=("statistical", "any"), persistent=("persistent", "any"),
                                  hot_day=("hot_day", "any"), outdoor_C=("outdoor_temp_C", "mean"),
                                  indoor_C=("indoor_temp_C", "mean"), holiday=("holiday", "any"))
    events["end"] = events["last"] + pd.Timedelta(minutes=15)
    events["duration_h"] = events["intervals"] * INTERVAL_HOURS
    events["after_hours_h"] = events["after_hours"] * INTERVAL_HOURS
    after_share = events["after_hours_h"] / events["duration_h"]
    # Weighted waste (kWh) = excess kWh + an extra for after-hours use (the most actionable)
    # − a discount for hot days (partly explained by the weather). Kept as parts so the dashboard can show the sum.
    events["after_hours_extra_kWh"] = events["excess_kWh"] * (after_weight - 1) * after_share
    events["hot_day_discount_kWh"] = np.where(events["hot_day"], (1 - hot_weight), 0.0) * (
        events["excess_kWh"] + events["after_hours_extra_kWh"])
    events["priority_score"] = events["excess_kWh"] + events["after_hours_extra_kWh"] - events["hot_day_discount_kWh"]

    def why(row):
        parts = []
        if row.holiday:
            parts.append(f"on a closure day: {closure_name(row.start.date())}")
        if row.after_hours_h:
            parts.append(f"after hours {row.after_hours_h:g} h (building closed or unit normally off)")
        if row.persistent:
            parts.append("persistent ≥ threshold")
        if row.statistical:
            parts.append("unusually high for time slot")
        if row.hot_day:
            parts.append(f"hot day ({row.outdoor_C:.1f} °C outside)")
        return "; ".join(parts)

    events["reason"] = [why(r) for r in events.itertuples()]
    events = events.sort_values("priority_score", ascending=False).reset_index(drop=True)
    events["priority_level"] = priority_levels(events["priority_score"])
    return events


PRIORITY_LEVELS = ("🔴 High", "🟠 Medium", "🟢 Low")


def priority_levels(score):
    """High = the top items that together make up the first half of all weighted waste,
    Medium = the items that take it to 80 %, Low = the rest (and anything with no waste)."""
    order = score.sort_values(ascending=False)
    total = order.sum()
    share_above = (order.cumsum() - order) / total if total > 0 else pd.Series(1.0, index=order.index)
    level = pd.Series(np.select([share_above < .5, share_above < .8], PRIORITY_LEVELS[:2], PRIORITY_LEVELS[2]),
                      index=order.index)
    level[order <= 0] = PRIORITY_LEVELS[2]
    return level.reindex(score.index)


def add_levels(table):
    """Priority level and share of all weighted waste, computed over the rows of this table."""
    table["priority_level"] = priority_levels(table["priority_score"])
    table["share_pct"] = 100 * table["priority_score"] / max(1e-9, table["priority_score"].sum())
    return table


def priority_example(row):
    """One unit's weighted waste written out as a sum."""
    return (f"Example: **{row['AC unit']}** used about {row['flagged_excess_kWh']:,.0f} kWh more than its normal pattern "
            f"during flagged periods, + {row['after_hours_extra_kWh']:,.0f} kWh extra because part of it was after hours, "
            f"− {row['hot_day_discount_kWh']:,.0f} kWh because part of it was on hot days = "
            f"**{row['priority_score']:,.0f} weighted kWh**, {row['share_pct']:.0f}% of the total → **{row['priority_level']}**.")


def unit_priority(d, events, days):
    g = d.groupby("unit")
    table = pd.DataFrame({"valid_h": g["eligible"].sum() * INTERVAL_HOURS,
                          "flagged_h": g["flagged"].sum() * INTERVAL_HOURS,
                          "after_hours_flagged_h": g["after_hours"].sum() * INTERVAL_HOURS})
    table["flagged_pct"] = 100 * table["flagged_h"] / table["valid_h"].replace(0, np.nan)
    table["flagged_excess_kWh"] = d.loc[d["flagged"]].groupby("unit")["excess_kWh_est"].sum().reindex(table.index, fill_value=0)
    if len(events):
        e = events.groupby("unit")
        table["events"] = e.size().reindex(table.index, fill_value=0)
        table["longest_event_h"] = e["duration_h"].max().reindex(table.index, fill_value=0)
        table["hot_day_events"] = e["hot_day"].sum().reindex(table.index, fill_value=0)
        for column in ["after_hours_extra_kWh", "hot_day_discount_kWh", "priority_score"]:
            table[column] = e[column].sum().reindex(table.index, fill_value=0)
    else:
        table[["events", "longest_event_h", "hot_day_events", "after_hours_extra_kWh", "hot_day_discount_kWh",
               "priority_score"]] = 0
    table["per_day_kWh"] = table["priority_score"] / max(1, days)
    table = add_levels(table.sort_values("priority_score", ascending=False))
    table.insert(0, "rank", range(1, len(table) + 1))
    table.insert(1, "AC unit", [unit_label(u) for u in table.index])
    return table


# Column settings shared by every priority table
PRIORITY_COLUMNS = {
    "rank": st.column_config.NumberColumn("#", width=35),
    "priority_level": st.column_config.TextColumn("Priority", help="🔴 High = the top units that together cause half of "
                                                  "all weighted waste. 🟠 Medium = the next 30 %. 🟢 Low = the rest."),
    "share_pct": st.column_config.NumberColumn("Share of waste", format="%.0f%%",
                                               help="This unit's share of all weighted waste in the table."),
    "flagged_excess_kWh": st.column_config.NumberColumn("Excess kWh", format="%.0f",
                                                        help="Energy above the unit's normal pattern during flagged periods (estimate)."),
    "after_hours_extra_kWh": st.column_config.NumberColumn("+ After-hours extra", format="%.0f",
                                                           help="After-hours waste counts 1.5×: it is usually a quick timer fix."),
    "hot_day_discount_kWh": st.column_config.NumberColumn("− Hot-day discount", format="%.0f",
                                                          help="Waste on unusually hot days counts 0.5×: the weather partly explains it."),
    "per_day_kWh": st.column_config.NumberColumn("Per day (kWh)", format="%.1f",
                                                 help="Weighted waste ÷ number of days in the review period."),
}


def weighted_column(label="= Weighted waste (kWh)"):
    return st.column_config.NumberColumn(label, format="%.0f",
                                         help="Excess kWh + after-hours extra − hot-day discount, summed over the "
                                              "review period. Ranks the list.")


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

# All three flagging rules are always on, and priorities use fixed weights:
# after-hours waste counts 1.5x (usually a quick timer fix), hot-day waste counts 0.5x (partly weather-driven).
use_stat = use_after = use_persist = True
AFTER_HOURS_WEIGHT, HOT_DAY_WEIGHT = 1.5, 0.5
after_weight, hot_weight = AFTER_HOURS_WEIGHT, HOT_DAY_WEIGHT


def in_period(frame):
    return frame.loc[frame["Date"].ge(period_start) & frame["Date"].lt(period_end)]


# ---------------------------------------------------------------- results shared by every tab
d = in_period(flags_for(floor, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist, OPEN))
detail = in_period(detail_all)
events = build_events(d, hot_weight, after_weight)
priority = unit_priority(d, events, PERIOD_DAYS)
kpis = period_kpis(floor, period_start, period_end, OPEN)
sens = data["sensitivity"]

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
    fd = in_period(flags_for(f, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist, OPEN))
    fev = build_events(fd, hot_weight, after_weight)
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
                       "Weighted waste kWh": fpr["priority_score"].sum(),
                       "Median indoor °C (running)": s_["median_indoor_temp_C_running"].median()})
floors_table = pd.DataFrame(floor_rows)
allunits = pd.concat(unit_tables)
allunits["position"] = allunits.index.map(position_of)
allunits["mean_kW"] = allunits.index.map(peer_mean)


def peer_gap(unit):
    """% difference from the median of the same unit position on the other floors. Only meaningful when the peers were
    actually running: on a Saturday or Sunday the peers are near 0 kW and the ratio would explode."""
    peers = [v for u, v in peer_mean.items() if position_of(u) == position_of(unit) and u != unit]
    peer = np.nanmedian(peers) if peers else np.nan
    return 100 * (peer_mean[unit] / peer - 1) if pd.notna(peer) and peer >= on_kw else np.nan


def vs_peers(row):
    return peer_gap(row.name)


allunits["vs_peers_pct"] = allunits.apply(vs_peers, axis=1)
allunits = add_levels(allunits.sort_values("priority_score", ascending=False))  # levels across the whole building
allunits["rank"] = range(1, len(allunits) + 1)
allunits["AC unit"] = [unit_label(u, with_floor=True) for u in allunits.index]
worst_floor = floors_table.loc[floors_table["Weighted waste kWh"].idxmax()]
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

# ---------------------------------------------------------------- tabs
tab_overview, tab_twin, tab_priority, tab_sim, tab_forecast, tab_compare = st.tabs(
    ["Overview", "Digital twin", "Priority list", "Setpoint & schedule simulator", "Forecast",
     "Compare floors"])

PRIORITY_HELP = ("Units are ranked by weighted waste (kWh) = the estimated extra energy each unit used during flagged "
                 "periods, with after-hours waste counted 1.5× (usually a quick timer fix) and waste on unusually hot "
                 "days counted 0.5× (partly explained by the weather). Priority level: 🔴 High = the top units that "
                 "together cause half of all weighted waste, 🟠 Medium = the next 30 %, 🟢 Low = the rest.")


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


def twin_frame(start, days, compact, height, start_time=None):
    """The 3D twin for the floors in view. Unit summaries use the review period; the replay uses the full data.
    start_time: the moment the replay opens at (default 14:00)."""
    twin_floors = FLOORS if floor == ALL else [floor]
    details_full, loads, summaries = {}, {}, {}
    for f in twin_floors:
        details_full[f] = flags_for(f, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist, OPEN)
        loads[f] = load(f)["zone_loads"]
        fpr = next(t for t in unit_tables if (t["floor"] == f).all())
        fsens = load(f)["sensitivity"]
        for u in fpr.index:
            summaries[u] = {"excess": _r_or_none(fpr.loc[u, "flagged_excess_kWh"]),
                            "priority": _r_or_none(fpr.loc[u, "priority_score"]),
                            "level": fpr.loc[u, "priority_level"],
                            "rank": int(fpr.loc[u, "rank"]), "of": len(fpr),
                            "median": _r_or_none(fsens["median_indoor_temp_C_running"].get(u)),
                            "comfort": _r_or_none(fsens["pct_running_hours_in_comfort_band"].get(u)),
                            "peers": _r_or_none(peer_gap(u))}
    days = min(days, (pd.Timestamp(data_last) - pd.Timestamp(start)).days + 1)
    payload = twin.payload(details_full, loads, summaries, twin_floors, pd.Timestamp(start), days, on_kw, SCOPE_NAME)
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
TARIFF_THB = 4.2  # same default as the simulator; an assumption
outdoor_mean = in_period(data["weather"].reset_index().rename(columns={"index": "Date"}))["outdoor_temp_C"].mean()

# 15-minute detail for the floors in view, within the timeframe, plus the whole building (for "is the rest running?")
building_now = pd.concat([details[f].assign(floor=f) for f in FLOORS], ignore_index=True)
building_now["running"] = building_now["actual_kW"].gt(on_kw)
scope_now = building_now if floor == ALL else building_now.loc[building_now["floor"].eq(floor)]
indoor_now = scope_now.loc[scope_now["running"], "indoor_temp_C"].median()
ran_now = scope_now.groupby("unit")["running"].any()

SEVERITY = {"Critical": ("#e5533d", 0), "Check": ("#f0a03c", 1), "Info": ("#5aa9ff", 2)}


def assistant_findings():
    """Rule-based checks on the selected timeframe, written as what to look at next. Suggestions, not diagnoses."""
    days = max(1, PERIOD_DAYS)
    found = []

    def add(severity, icon_name, title, action, detail, where, weight):
        """title + one-line action are shown; detail (the reasoning) appears on hover."""
        found.append({"severity": severity, "icon": icon_name, "title": title, "action": action, "detail": detail,
                      "where": where, "rank": (SEVERITY[severity][1], -weight)})

    def short(u):
        m = re.search(r"F(\d+)_z(\d+)_AC(\d+)", u)
        return f"F{m[1]} · Z{m[2]} · AC{m[3]}" if m else u

    valid = scope_now.loc[scope_now["eligible"]].copy()
    valid["building_running"] = valid["Date"].map(building_now.groupby("Date")["running"].mean())
    weekday = valid["Date"].dt.dayofweek.lt(5)

    # Units off during their usual hours while most of the building runs (closures excluded)
    down = valid.loc[valid["schedule_on"].eq(True) & ~valid["running"] & valid["building_running"].ge(.5) & weekday]
    together = down.groupby(["floor", "Date"])["unit"].nunique()
    together = together[together.ge(3)]
    if len(together):
        hours = together.groupby(level=0).size() * INTERVAL_HOURS
        f = hours.idxmax()
        add("Critical", "building", f"{together.loc[f].max()} units on Floor {f} off together",
            "check chilled water, AHU or power feed",
            f"{hours[f]:.0f} h from {together.loc[f].index.min():%a %d %b %H:%M} while the rest of the building ran. "
            "Suspect a shared supply: chilled water, the air-handling unit or the electrical feed.", "Digital twin", hours[f])
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
            "dirty coil.", "Priority list → Inspect a unit", h)

    # Several units on one floor above normal at the same moment: a shared cause is more likely than unit faults
    high_now = valid.loc[(valid["statistical"] | valid["persistent"]) & ~valid["after_hours"]]
    clusters = high_now.groupby(["floor", "Date"])["unit"].nunique()
    clusters = clusters[clusters.ge(3)]
    if len(clusters):
        n_days = clusters.reset_index().assign(day=lambda x: x["Date"].dt.normalize()).groupby("floor")["day"].nunique()
        f = n_days.idxmax()
        add("Check", "building", f"Floor {f}: 3+ units high at once",
            f"{n_days[f]} day{'s' if n_days[f] > 1 else ''} · check shared chilled water or AHU", f"3 or more units above normal together on {n_days[f]} day{'s' if n_days[f] > 1 else ''}. Check what they "
            "share before treating them as unit faults: chilled-water temperature, the air-handling unit, or an event "
            "on that floor.", "Compare floors", n_days[f])

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
             ("bell", f"{len(events):,}", "Flagged events")]
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
            for _, e in events.head(4).iterrows():
                what = (f"ran {e['after_hours_h']:g} h outside its schedule" if e["after_hours_h"] >= e["duration_h"] / 2
                        else f"{e['excess_kWh']:.0f} kWh above normal over {e['duration_h']:g} h")
                items.append(f'<div class="item">{icon("bell", 18, small=True)}<div><div class="t">'
                             f'{unit_label(e["unit"])} {what}</div><div class="s">{e["priority_level"]} · '
                             f'{e["start"]:%a %d %b, %H:%M}</div></div></div>')
            html('<div class="card-h">Top alerts</div>'
                 + ("".join(items) if items else '<div class="item"><div class="t">No flagged events in this timeframe</div></div>')
                 + '<div class="card-f">Full list: Priority list tab</div>')

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
                     f'<div class="why">{x["detail"]}<div class="meta"><span class="chip sev">{x["severity"].upper()}'
                     f'</span><span class="chip">Look in: {x["where"]}</span></div></div></div></div>'
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
    cols = ["rank", "priority_level", "AC unit", "flagged_excess_kWh", "after_hours_extra_kWh", "hot_day_discount_kWh",
            "priority_score", "per_day_kWh", "share_pct", "events", "flagged_pct"]
    st.dataframe(priority[cols], hide_index=True, width="stretch", height=table_height(len(priority), 420), column_config={
        **PRIORITY_COLUMNS, "priority_score": weighted_column(),
        "events": st.column_config.NumberColumn("Events"),
        "flagged_pct": st.column_config.NumberColumn("Time flagged", format="%.1f%%")})
    if top is not None:
        st.caption("Read left to right: excess kWh + after-hours extra − hot-day discount = weighted waste. "
                   + priority_example(top))

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
            help="One row per unusual episode, ranked by weighted waste (kWh). Priority level uses the same rule as the "
                 "unit list: 🔴 High events together cause half of all weighted waste.")
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
                   "priority_score", "after_hours_h", "outdoor_C", "reason"]
        st.dataframe(ev[ev_cols], hide_index=True, width="stretch", height=300, column_config={
            "rank": st.column_config.NumberColumn("#", width=35),
            "priority_level": PRIORITY_COLUMNS["priority_level"],
            "start": st.column_config.DatetimeColumn("Start", format="ddd D MMM YYYY, HH:mm"),
            "end": st.column_config.DatetimeColumn("End", format="HH:mm"),
            "duration_h": st.column_config.NumberColumn("Hours", format="%.2f"),
            "peak_kW": st.column_config.NumberColumn("Peak kW", format="%.1f"),
            "excess_kWh": st.column_config.NumberColumn("Excess kWh", format="%.1f"),
            "priority_score": st.column_config.NumberColumn("Weighted kWh", format="%.1f",
                                                            help="Excess kWh, ×1.5 for after-hours time, ×0.5 on hot days."),
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
    b4.metric("Floor to review first", worst_floor["Floor"], help="Highest total weighted waste")
    t, c1, c2 = st.columns([3, 1, 1], gap="medium")
    t.dataframe(floors_table[["Floor", "AC energy kWh", "Weighted waste kWh", "After-hours %", "Flagged events",
                              "Median indoor °C (running)"]].round(1),
                hide_index=True, width="stretch", height=table_height(len(floors_table), 300),
                column_config={"AC energy kWh": st.column_config.NumberColumn(format="%.0f"),
                               "Weighted waste kWh": st.column_config.NumberColumn(format="%.0f"),
                               "After-hours %": st.column_config.NumberColumn("Used while closed %", format="%.1f"),
                               "Median indoor °C (running)": st.column_config.NumberColumn("Indoor °C", format="%.1f")})
    for col, metric, title in [(c1, "Weighted waste kWh", "Weighted waste (kWh)"),
                               (c2, "After-hours %", "Used while closed (%)")]:
        fig = go.Figure(go.Bar(x=floors_table["Floor"].str.replace("Floor ", "F"), y=floors_table[metric],
                               marker_color=[FLAG if f == worst_floor["Floor"] else ACTUAL for f in floors_table["Floor"]],
                               text=floors_table[metric].map(lambda v: f"{v:,.0f}" if v >= 100 else f"{v:.1f}"),
                               textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:,.1f}<extra></extra>"))
        col.plotly_chart(style(fig, table_height(len(floors_table), 300), title, margin_top=10), width="stretch")
    share_worst = 100 * worst_floor["Weighted waste kWh"] / max(1, floors_table["Weighted waste kWh"].sum())

    section("Building-wide priority list and same unit on other floors",
            help="All units on all floors, ranked by weighted waste. 'vs peers' compares a unit's average power with the "
                 "median of the same unit position on the other floors: +30% means it draws 30% more than its peers.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.dataframe(allunits[["rank", "priority_level", "AC unit", "priority_score", "vs_peers_pct"]],
                     hide_index=True, width="stretch", height=420, column_config={
                         **PRIORITY_COLUMNS, "priority_score": weighted_column("Weighted waste (kWh)"),
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
        takeaway(f"<b>{worst_floor['Floor']}</b> causes {share_worst:.0f}% of the building's weighted waste, and "
                 f"<b>{outlier['AC unit']}</b> draws {outlier['vs_peers_pct']:.0f}% more power than the same unit on the "
                 "other floors, so the issue is that unit and its schedule, not the layout.")
    else:
        takeaway(f"<b>{worst_floor['Floor']}</b> causes {share_worst:.0f}% of the building's weighted waste. No unit "
                 "can be compared with its peers in this timeframe (the peer units were not running).")

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
        setpoint = c1.slider("New indoor setpoint (°C)", 18.0, 28.0, 25.0, 0.5)
        band = c1.slider("Comfort band (°C)", 18.0, 30.0, (23.0, 26.0), 0.5)
        comfort_target = c2.slider("Minimum comfort compliance (%)", 30, 100, 70,
                                   help="An assumption, not a standard: the share of running time that must stay in the "
                                        "comfort band. The recommended setpoint is the biggest saving that meets it.")
        tariff = c2.number_input("Electricity price (THB per kWh)", 0.0, 20.0, 4.2, 0.1,
                                 help="Assumption. Change it to your actual tariff.")
        start_h, stop_h = c3.slider("Weekday AC window", 0.0, 24.0, (0.0, 24.0), 0.25, format="%.2f h",
                                    help="The hours the AC is allowed to run on weekdays. 00:00–24:00 = no change from today. "
                                         "Narrow it (e.g. 07:00–18:00) and all AC energy used outside the window counts as "
                                         "saved. This is the maximum possible saving: some after-hours use may be "
                                         "legitimate (events, cleaning, servers).")
        weekend_on = c4.checkbox("Keep weekends running", True,
                                 help="Ticked: weekends are allowed (no change). Unticked: the proposal is no AC at "
                                      "weekends, so all weekend AC energy counts as saved.")
        holidays_off = c4.checkbox("AC off on public holidays", False,
                                   help="Proposal: no AC on Thai public holidays and closure days, so any AC energy on "
                                        "those days counts as saved.")

    sim = detail.loc[detail["eligible"] & detail["actual_kW"].notna(),
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
    total_kwh = sim["actual_kW"].sum() * INTERVAL_HOURS

    def simulate(sp):
        avoided = sim["running"] & sim["outside_window"]
        active = sim["running"] & ~avoided & sim["indoor_temp_C"].notna()
        shift = (sp - sim["current_sp"]).where(active, 0)
        setpoint_kwh = np.minimum(sim["slope"] * shift, sim["actual_kW"]).sum() * INTERVAL_HOURS
        schedule_kwh = sim.loc[avoided, "actual_kW"].sum() * INTERVAL_HOURS
        new_temp = sim.loc[active, "indoor_temp_C"] + shift.loc[active]
        comfort = 100 * new_temp.between(*band).mean() if len(new_temp) else np.nan
        return setpoint_kwh, schedule_kwh, comfort

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
    for unit_name, part in sim.groupby("unit"):
        avoided = part["running"] & part["outside_window"]
        active = part["running"] & ~avoided & part["indoor_temp_C"].notna()
        shift = (setpoint - part["current_sp"]).where(active, 0)
        sp_unit = np.minimum(part["slope"] * shift, part["actual_kW"]).sum() * INTERVAL_HOURS
        sch_unit = part.loc[avoided, "actual_kW"].sum() * INTERVAL_HOURS
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
    twin_frame(twin_start, twin_days, compact=False, height=760)
    st.caption("Zone colours and the glowing ceiling lights come from the 15-minute data. Lights count as 'after hours' when a zone's "
               f"lighting is above {twin.LIGHT_ON_KW} kW while every AC unit on that floor is normally off (stairwells excluded). "
               f"The priority-level colours use the review period ({PERIOD}). "
               "Floors 1, 2 and 6 are shown for context but are not in this analysis.")
