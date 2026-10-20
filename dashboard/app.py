"""Multi-floor AC dashboard: priority list, threshold tuner, setpoint/schedule simulator, forecasts and floor comparison.

Run `python run_all_floors.py` first (it writes ac_analysis_outputs/floor_<n>/), then:
    streamlit run dashboard/app.py
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(os.environ.get("AC_OUTPUT_FOLDER", Path(__file__).resolve().parents[1] / "ac_analysis_outputs"))
FLOORS = sorted(int(p.name.split("_")[1]) for p in ROOT.glob("floor_*") if (p / "anomaly_15min_detail.csv").exists())
ALL = "all"  # selector value for the whole-building view
DEFAULT_SCOPE = ALL
SCOPE = None  # set from the sidebar; unit labels include the floor in the whole-building view
INTERVAL_HOURS = 0.25

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
                "Selected per floor": "Each floor's validation-selected model"}

st.set_page_config(page_title="Building AC Dashboard", page_icon="❄️", layout="wide")
# Compact dashboard spacing: smaller headings and metrics, tighter gaps between blocks
st.markdown("""<style>
.block-container {padding-top: 2.75rem; padding-bottom: 1rem;}
h1 {font-size: 1.6rem !important; padding: 0 !important;}
h3 {font-size: 1.1rem !important; padding: .25rem 0 .35rem 0 !important;}
[data-testid="stVerticalBlock"] {gap: .6rem;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
[data-testid="stSidebarContent"] h2 {font-size: 1.1rem;}
[data-baseweb="tab-panel"] {padding-top: .4rem;}
</style>""", unsafe_allow_html=True)


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


def table_height(rows, cap):
    """Natural height of a table with this many rows, capped so long tables scroll instead of growing the page."""
    return min((rows + 1) * 35 + 3, cap)


def style(fig, height=380, ytitle=None, xtitle=None, margin_top=40):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=margin_top, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    fig.update_xaxes(showgrid=False, title=xtitle)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", title=ytitle, rangemode="tozero")
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


@st.cache_data(show_spinner="Loading analysis outputs…")
def load(floor):
    if floor == ALL:
        return load_building()
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
            "floor_kpis": pd.read_csv(OUT / "floor_reporting_15min_kpis.csv", index_col=0).iloc[:, 0]}
    return data


def run_rules(detail, thresholds, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist):
    """Re-apply the notebook's three flagging rules with adjustable thresholds."""
    d = detail
    upper = d["IQR_q3_kW"] + k_iqr * (d["IQR_q3_kW"] - d["IQR_q1_kW"])
    running = d["eligible"] & d["actual_kW"].gt(on_kw)
    stat = running & d["actual_kW"].gt(upper)
    after = running & d["schedule_on"].eq(False)
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
def flags_for(floor, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist):
    data = load(floor)
    flags = run_rules(data["detail"], data["thresholds"], k_iqr, k_res, on_kw, persist,
                      use_stat, use_after, use_persist)
    # Replace the notebook's default-threshold flags with the recomputed ones
    base = data["detail"].drop(columns=[c for c in flags.columns if c in data["detail"].columns])
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
                                  indoor_C=("indoor_temp_C", "mean"))
    events["end"] = events["last"] + pd.Timedelta(minutes=15)
    events["duration_h"] = events["intervals"] * INTERVAL_HOURS
    events["after_hours_h"] = events["after_hours"] * INTERVAL_HOURS
    after_share = events["after_hours_h"] / events["duration_h"]
    # Priority = magnitude (excess kWh) × context: after-hours use is the most actionable,
    # high use on a hot day is partly explained by the weather.
    events["priority_score"] = (events["excess_kWh"] * (1 + (after_weight - 1) * after_share)
                                * np.where(events["hot_day"], hot_weight, 1.0))

    def why(row):
        parts = []
        if row.after_hours_h:
            parts.append(f"after-hours {row.after_hours_h:g} h")
        if row.persistent:
            parts.append("persistent ≥ threshold")
        if row.statistical:
            parts.append("unusually high for time slot")
        if row.hot_day:
            parts.append(f"hot day ({row.outdoor_C:.1f} °C outside)")
        return "; ".join(parts)

    events["reason"] = [why(r) for r in events.itertuples()]
    return events.sort_values("priority_score", ascending=False).reset_index(drop=True)


def unit_priority(d, events):
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
        table["priority_score"] = e["priority_score"].sum().reindex(table.index, fill_value=0)
    else:
        table[["events", "longest_event_h", "hot_day_events", "priority_score"]] = 0
    table = table.sort_values("priority_score", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    table.insert(1, "AC unit", [unit_label(u) for u in table.index])
    return table


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
detail = data["detail"]
units = list(dict.fromkeys(detail["unit"]))
unit_colour = {u: UNIT_COLOURS[i % len(UNIT_COLOURS)] for i, u in enumerate(units)}

with st.sidebar:
    st.header("Anomaly thresholds")
    st.caption("These settings drive the priority list, heatmap and overview. The notebook defaults are "
               "k = 1.5, 0.5 kW and 8 intervals (2 h).")
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

# All three flagging rules are always on, and priorities use fixed weights:
# after-hours waste counts 1.5x (usually a quick timer fix), hot-day waste counts 0.5x (partly weather-driven).
use_stat = use_after = use_persist = True
AFTER_HOURS_WEIGHT, HOT_DAY_WEIGHT = 1.5, 0.5
after_weight, hot_weight = AFTER_HOURS_WEIGHT, HOT_DAY_WEIGHT

d = flags_for(floor, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist)
events = build_events(d, hot_weight, after_weight)
priority = unit_priority(d, events)

scope_desc = (f"Floors {', '.join(map(str, FLOORS))} · {len(units)} AC units" if floor == ALL
              else f"{len(units)} AC units")
st.title(f"{SCOPE_NAME} AC: anomaly priorities and energy simulator",
         help="Flags are potential anomalies for review, not diagnosed faults. kWh and THB figures are estimates, "
              "not verified savings. Pick a single floor under View in the sidebar to focus on it.")
st.caption(f"CU-BEMS Chamchuri 5, Bangkok · {scope_desc} · reporting {detail['Date'].min():%d %b %Y} – "
           f"{detail['Date'].max():%d %b %Y} · baseline Jul 2018 – Jun 2019 · weather: Open-Meteo")

tab_overview, tab_priority, tab_tuner, tab_sim, tab_forecast, tab_compare = st.tabs(
    ["Overview", "Priority list", "Threshold tuner", "Setpoint & schedule simulator", "Forecast", "Compare floors"])

PRIORITY_HELP = ("Priority score = each unit's estimated excess kWh during flagged periods, where after-hours "
                 "waste counts 1.5× (usually a quick timer fix) and waste on unusually hot days counts 0.5× "
                 "(partly explained by the weather). Higher score = inspect first.")

# ================================================================ overview
with tab_overview:
    kpis = data["floor_kpis"]
    sens = data["sensitivity"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AC energy (valid intervals)", f"{kpis['estimated_energy_kWh_valid_intervals']:,.0f} kWh")
    c2.metric("After-hours share", f"{kpis['after_hours_energy_pct_valid_intervals']:.1f}%")
    c3.metric("15-min peak", f"{kpis['15min_peak_kW']:.1f} kW")
    c4.metric("Flagged events", f"{len(events):,}", help="With the thresholds in the sidebar")
    c5.metric("Indoor temp (running)", f"{sens['median_indoor_temp_C_running'].median():.1f} °C",
              help="Reporting period, weekdays, unit running. Typical comfort band 23–26 °C.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Review these units", help=PRIORITY_HELP)
        show = priority[["rank", "AC unit", "priority_score", "flagged_excess_kWh", "events",
                         "after_hours_flagged_h", "flagged_pct"]]
        st.dataframe(show, hide_index=True, width="stretch", height=table_height(len(show), 460), column_config={
            "priority_score": st.column_config.ProgressColumn("Priority score", format="%.0f",
                                                              min_value=0, max_value=float(max(1, show["priority_score"].max()))),
            "flagged_excess_kWh": st.column_config.NumberColumn("Excess kWh (est.)", format="%.0f"),
            "events": st.column_config.NumberColumn("Events"),
            "after_hours_flagged_h": st.column_config.NumberColumn("After-hours flagged (h)", format="%.1f"),
            "flagged_pct": st.column_config.NumberColumn("Flagged %", format="%.1f%%")})
    with right:
        n24 = data["next24"]
        st.subheader("Next-day forecast" if floor != ALL else "Next-day building forecast",
                     help=f"Hourly AC demand predicted for 1 Jan 2020, the day after the data ends. "
                          f"Method: {MODEL_LABELS.get(n24['method'].iloc[0], n24['method'].iloc[0])}. "
                          "Holidays are not modelled.")
        fig = go.Figure(go.Scatter(x=n24.index, y=n24["predicted_kW"], mode="lines+markers",
                                   line=dict(color=ACTUAL, width=2), marker=dict(size=6), name="Forecast"))
        peak = n24["predicted_kW"].idxmax()
        fig.add_annotation(x=peak, y=n24["predicted_kW"].max(), text=f"Peak {peak:%H:%M}: {n24['predicted_kW'].max():.1f} kW",
                           showarrow=True, arrowhead=2)
        st.plotly_chart(style(fig, 210, "kW"), width="stretch")
        st.caption(f"Predicted daily energy {n24['predicted_kW'].sum():,.0f} kWh")
        comfort = sens[["median_indoor_temp_C_running", "pct_running_hours_in_comfort_band", "pct_running_hours_in_RH_band"]].copy()
        comfort.index = comfort.index.map(unit_label)
        comfort.columns = ["Median indoor °C", "% hours 23–26 °C", "% hours RH 40–60%"]
        overcooled = comfort["Median indoor °C"].median() < 23
        st.subheader("Comfort Level",
                     help="Share of running hours (weekdays, reporting period) with indoor temperature in 23–26 °C "
                          "and humidity in 40–60 %."
                          + (" Rooms are kept below 23 °C, which points to overcooling: see the setpoint simulator."
                             if overcooled else ""))
        st.dataframe(comfort.round(1), width="stretch", height=table_height(len(comfort), 210))

# ================================================================ priority list
with tab_priority:
    top_left, top_right = st.columns([3, 2])
    with top_left:
        st.subheader("Unit priority list", help=PRIORITY_HELP)
        st.dataframe(priority, hide_index=True, width="stretch", height=table_height(len(priority), 300), column_config={
            "priority_score": st.column_config.ProgressColumn("Priority score", format="%.0f", min_value=0,
                                                              max_value=float(max(1, priority["priority_score"].max()))),
            "valid_h": st.column_config.NumberColumn("Valid hours", format="%.0f"),
            "flagged_h": st.column_config.NumberColumn("Flagged hours", format="%.1f"),
            "after_hours_flagged_h": st.column_config.NumberColumn("After-hours flagged (h)", format="%.1f"),
            "flagged_pct": st.column_config.NumberColumn("Flagged %", format="%.1f%%"),
            "flagged_excess_kWh": st.column_config.NumberColumn("Excess kWh (est.)", format="%.0f"),
            "longest_event_h": st.column_config.NumberColumn("Longest event (h)", format="%.2f"),
            "hot_day_events": st.column_config.NumberColumn("Events on hot days")})
    with top_right:
        st.subheader("When are flags most common?",
                     help="Share of valid 15-minute intervals flagged, by weekday and time of day. "
                          "Darker = problems happen more often at that time.")
        hm = d.loc[d["eligible"]].assign(dow=lambda x: x["Date"].dt.dayofweek,
                                         slot=lambda x: x["Date"].dt.hour * 4 + x["Date"].dt.minute // 15)
        heat = (100 * hm.groupby(["dow", "slot"])["flagged"].mean()).unstack().reindex(index=range(7), columns=range(96))
        times = [f"{s // 4:02d}:{(s % 4) * 15:02d}" for s in range(96)]
        fig = go.Figure(go.Heatmap(z=heat.values, x=times, y=DAYS, colorscale=[[0, "#f0efec"], [0.5, "#86b6ef"], [1, "#104281"]],
                                   zmin=0, zmax=max(1, float(np.nanmax(heat.values))) if np.isfinite(heat.values).any() else 1,
                                   colorbar=dict(title="% flagged", thickness=10),
                                   hovertemplate="%{y} %{x}<br>%{z:.1f}% of valid intervals flagged<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickvals=times[::12], ticktext=times[::12])
        st.plotly_chart(style(fig, table_height(len(priority), 300), margin_top=10), width="stretch")

    if events.empty:
        st.warning("No events with the current thresholds.")
    else:
        h, f2, f3, f4 = st.columns([3, 2, 2, 2], vertical_alignment="center")
        h.subheader("Event list", help="One row per unusual episode, ranked by priority score.")
        min_hours = f2.number_input("Min duration (h)", 0.0, 48.0, 0.0, 0.25)
        only_after = f3.checkbox("After-hours only")
        hide_hot = f4.checkbox("Hide hot-day events")
        ev = events.loc[events["duration_h"].ge(min_hours)]
        if only_after:
            ev = ev.loc[ev["after_hours_h"].gt(0)]
        if hide_hot:
            ev = ev.loc[~ev["hot_day"]]
        ev = ev.assign(**{"AC unit": ev["unit"].map(unit_label)})
        ev.insert(0, "rank", range(1, len(ev) + 1))
        cols = ["rank", "AC unit", "start", "end", "duration_h", "peak_kW", "excess_kWh", "after_hours_h",
                "outdoor_C", "indoor_C", "priority_score", "reason"]
        st.dataframe(ev[cols], hide_index=True, width="stretch", height=250, column_config={
            "start": st.column_config.DatetimeColumn("Start", format="ddd D MMM YYYY, HH:mm"),
            "end": st.column_config.DatetimeColumn("End", format="HH:mm"),
            "duration_h": st.column_config.NumberColumn("Hours", format="%.2f"),
            "peak_kW": st.column_config.NumberColumn("Peak kW", format="%.1f"),
            "excess_kWh": st.column_config.NumberColumn("Excess kWh", format="%.1f"),
            "after_hours_h": st.column_config.NumberColumn("After-hours h", format="%.2f"),
            "outdoor_C": st.column_config.NumberColumn("Outdoor °C", format="%.1f"),
            "indoor_C": st.column_config.NumberColumn("Indoor °C", format="%.1f"),
            "priority_score": st.column_config.NumberColumn("Score", format="%.1f")})
        st.download_button("Download event list (CSV)", ev[cols].to_csv(index=False).encode("utf-8"),
                           "ac_priority_events.csv", "text/csv")

    controls, chart = st.columns([1, 3])
    with controls:
        st.subheader("Inspect a unit", help="Blue = actual power, orange dashed = expected, grey dotted = the "
                                            "'unusually high' limit, red dots = flagged. Lower panel: outdoor and indoor °C.")
        unit = st.selectbox("AC unit", priority.index.tolist(), format_func=unit_label, key=f"inspect_unit_{floor}")
        u = d.loc[d["unit"].eq(unit)].set_index("Date")
        first_day, last_day = u.index.min().date(), u.index.max().date()
        # Default: the Monday-to-Sunday week with the most flagged excess energy for this unit
        daily_excess = u["excess_kWh_est"].where(u["flagged"], 0).resample("D").sum()
        weekly_excess = daily_excess.groupby(daily_excess.index.to_period("W-SUN")).sum()
        busiest = weekly_excess.idxmax().start_time.date() if weekly_excess.max() > 0 else first_day
        period = st.radio("Show", ["Week", "Month", "Custom"], horizontal=True, key=f"inspect_period_{floor}",
                          help="Week = Monday to Sunday around the date you pick. Month = the calendar month of the date. "
                               "Custom = any start and end date.")
        if period == "Custom":
            picked = st.date_input("From – to", (busiest, min(busiest + pd.Timedelta(days=6), last_day)),
                                   min_value=first_day, max_value=last_day, format="DD/MM/YYYY",
                                   key=f"inspect_range_{floor}_{unit}")
            # While the user is mid-selection only the start date exists
            start = pd.Timestamp(picked[0])
            end = pd.Timestamp(picked[1] if len(picked) > 1 else picked[0]) + pd.Timedelta(days=1)
        else:
            day = pd.Timestamp(st.date_input("Any date in the " + period.lower(), busiest, min_value=first_day,
                                             max_value=last_day, format="DD/MM/YYYY",
                                             key=f"inspect_date_{floor}_{unit}_{period}",
                                             help="The default is the week with the most flagged excess energy."))
            if period == "Week":
                start = day - pd.Timedelta(days=day.dayofweek)
                end = start + pd.Timedelta(days=7)
            else:
                start = day.replace(day=1)
                end = start + pd.offsets.MonthBegin(1)
        end = min(end, pd.Timestamp(last_day) + pd.Timedelta(days=1))  # data ends 31 Dec 2019
        view = u.loc[start: end - pd.Timedelta(minutes=15)]
        st.caption(f"Showing {start:%a %d %b %Y} – {end - pd.Timedelta(days=1):%a %d %b %Y} · "
                   f"flagged excess {view.loc[view['flagged'], 'excess_kWh_est'].sum():,.0f} kWh")
    with chart:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=view.index, y=view["actual_kW"], name="Actual", line=dict(color=ACTUAL, width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=view.index, y=view["expected_kW"], name="Expected", line=dict(color=EXPECTED, width=2, dash="dash")), 1, 1)
        fig.add_trace(go.Scatter(x=view.index, y=view["upper_kW"], name="Statistical upper limit", line=dict(color=MUTED, width=1, dash="dot")), 1, 1)
        fl = view.loc[view["flagged"]]
        fig.add_trace(go.Scatter(x=fl.index, y=fl["actual_kW"], mode="markers", name="Flagged",
                                 marker=dict(color=FLAG, size=7, opacity=.85)), 1, 1)
        fig.add_trace(go.Scatter(x=view.index, y=view["outdoor_temp_C"], name="Outdoor °C", line=dict(color=MUTED, width=2)), 2, 1)
        fig.add_trace(go.Scatter(x=view.index, y=view["indoor_temp_C"], name="Indoor °C", line=dict(color=GOOD, width=2)), 2, 1)
        style(fig, 380)
        fig.update_yaxes(title="AC power (kW)", row=1, col=1)
        fig.update_yaxes(title="°C", rangemode="normal", row=2, col=1)
        st.plotly_chart(fig, width="stretch")

# ================================================================ threshold tuner
with tab_tuner:
    grids = {"Statistical IQR multiplier k": np.arange(0.5, 5.01, 0.25),
             "Residual IQR multiplier": np.arange(0.5, 5.01, 0.25),
             "Persistence (intervals)": np.arange(1, 25),
             "'Running' threshold (kW)": np.round(np.arange(0.1, 5.01, 0.2), 2)}
    pick, m1, m2, m3 = st.columns([3, 1, 1, 1], vertical_alignment="bottom")
    with pick:
        st.subheader("How sensitive are the results to the threshold?",
                     help="Each curve re-runs the rules across a range of values and keeps the other sidebar settings "
                          "fixed. A good operating point sits just past the 'elbow': beyond it, raising the threshold "
                          "removes few flags, so the remaining flags are the robust ones. Set your chosen value in the "
                          "sidebar; every tab updates. The true optimum needs facilities-manager feedback on which flags were real.")
        param = st.radio("Vary", list(grids), horizontal=True)
    current = {"Statistical IQR multiplier k": k_iqr, "Residual IQR multiplier": k_res,
               "Persistence (intervals)": persist, "'Running' threshold (kW)": on_kw}[param]

    @st.cache_data(show_spinner="Sweeping thresholds…")
    def sweep(floor, param, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist):
        rows = []
        base = load(floor)
        for v in grids[param]:
            args = dict(k_iqr=k_iqr, k_res=k_res, on_kw=on_kw, persist=persist)
            key = {"Statistical IQR multiplier k": "k_iqr", "Residual IQR multiplier": "k_res",
                   "Persistence (intervals)": "persist", "'Running' threshold (kW)": "on_kw"}[param]
            args[key] = int(v) if key == "persist" else float(v)
            fl = run_rules(base["detail"], base["thresholds"], use_stat=use_stat, use_after=use_after,
                           use_persist=use_persist, **args)
            dd = base["detail"]
            run = (fl["flagged"].ne(fl["flagged"].shift()) | dd["unit"].ne(dd["unit"].shift())).cumsum()
            rule = {"k_iqr": "statistical", "k_res": "persistent", "persist": "persistent", "on_kw": "flagged"}[key]
            rows.append({"value": v,
                         "flagged_pct": 100 * fl["flagged"].sum() / max(1, dd["eligible"].sum()),
                         "rule_pct": 100 * fl[rule].sum() / max(1, dd["eligible"].sum()),
                         "flagged_excess_kWh": dd.loc[fl["flagged"], "excess_kWh_est"].sum(),
                         "events": run.loc[fl["flagged"]].nunique()})
        return pd.DataFrame(rows)

    sw = sweep(floor, param, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist)
    # Elbow: point furthest from the straight line joining the ends of the normalised curve
    # The elbow is found on the varied rule's own curve (the combined curve is dominated by the other rules)
    y = sw["rule_pct"].to_numpy()
    x = sw["value"].to_numpy(dtype=float)
    if np.ptp(y) > 0 and len(x) > 2:
        xn, yn = (x - x.min()) / np.ptp(x), (y - y.min()) / np.ptp(y)
        dist = np.abs((yn[-1] - yn[0]) * xn - (xn[-1] - xn[0]) * yn + xn[-1] * yn[0] - yn[-1] * xn[0])
        elbow = x[int(np.argmax(dist))]
    else:
        elbow = None
    m1.metric("Current value", f"{current:g}")
    m2.metric("Elbow suggestion", f"{elbow:g}" if elbow is not None else "n/a")
    cur_row = sw.iloc[int(np.argmin(np.abs(x - current)))]
    m3.metric("Flagged share now", f"{cur_row['flagged_pct']:.1f}%")

    c1, c2, c3 = st.columns(3)
    for col, metric, title in [(c1, "flagged_pct", "Share of valid intervals flagged (%)"),
                               (c2, "flagged_excess_kWh", "Estimated excess kWh inside flags")]:
        fig = go.Figure(go.Scatter(x=sw["value"], y=sw[metric], mode="lines+markers", line=dict(color=ACTUAL, width=2),
                                   marker=dict(size=7), name="All rules"))
        if metric == "flagged_pct" and param != "'Running' threshold (kW)":
            fig.add_trace(go.Scatter(x=sw["value"], y=sw["rule_pct"], mode="lines+markers", name="This rule alone",
                                     line=dict(color=EXPECTED, width=2, dash="dash"), marker=dict(size=7)))
        fig.add_vline(x=current, line=dict(color=EXPECTED, dash="dash"), annotation_text="current")
        if elbow is not None:
            fig.add_vline(x=elbow, line=dict(color=MUTED, dash="dot"), annotation_text="elbow", annotation_position="bottom right")
        col.plotly_chart(style(fig, 330, title, param), width="stretch")
    fig = go.Figure(go.Bar(x=sw["value"], y=sw["events"], marker_color=ACTUAL, name="Events"))
    fig.add_vline(x=current, line=dict(color=EXPECTED, dash="dash"))
    c3.plotly_chart(style(fig, 330, "Number of events", param), width="stretch")

# ================================================================ simulator
with tab_sim:
    sens = data["sensitivity"]
    slope = sens["kW_per_degC_outdoor"].clip(lower=0)
    st.subheader("What if we change the setpoint or the operating schedule?",
                 help="Uses the reporting-period 15-minute data. Setpoint effect: each unit's baseline regression slope "
                      "(kW per °C of outdoor temperature) stands in for the kW saved per 1 °C setpoint rise, and the "
                      "zone's temperature profile shifts so its median sits at the new setpoint. A setpoint below "
                      "today's level costs energy (negative saving). Schedule effect: AC energy outside the chosen "
                      "window counts as avoidable. Screening estimates only: confirm with a measured trial.")

    c1, c2, c3, c4 = st.columns(4)
    setpoint = c1.slider("New indoor setpoint (°C)", 18.0, 28.0, 25.0, 0.5)
    band = c1.slider("Comfort band (°C)", 18.0, 30.0, (23.0, 26.0), 0.5)
    comfort_target = c2.slider("Minimum comfort compliance (%)", 30, 100, 70)
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

    sim = detail.loc[detail["eligible"] & detail["actual_kW"].notna(),
                     ["Date", "unit", "actual_kW", "indoor_temp_C"]].copy()
    hour = sim["Date"].dt.hour + sim["Date"].dt.minute / 60
    weekend = sim["Date"].dt.dayofweek >= 5
    in_window = hour.ge(start_h) & hour.lt(stop_h)
    sim["outside_window"] = (~in_window & ~weekend) | (weekend & ~weekend_on)
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
    days = (detail["Date"].max() - detail["Date"].min()).days + 1

    sweep_sp = np.arange(18.0, 28.01, 0.5)
    rows = [(sp, *simulate(sp)) for sp in sweep_sp]
    sp_table = pd.DataFrame(rows, columns=["setpoint", "setpoint_kWh", "schedule_kWh", "comfort_pct"])
    sp_table["total_kWh"] = sp_table["setpoint_kWh"] + sp_table["schedule_kWh"]
    ok = sp_table.loc[sp_table["comfort_pct"].ge(comfort_target)]
    best = ok.loc[ok["total_kWh"].idxmax()] if len(ok) else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estimated energy saved", f"{saved:,.0f} kWh", f"{-100 * saved / total_kwh:.1f}% of AC energy",
              delta_color="inverse")
    k2.metric("Estimated cost saved", f"{saved * tariff:,.0f} THB", f"≈ {saved * tariff * 365 / days:,.0f} THB/year",
              delta_color="off")
    k3.metric("Comfort compliance", f"{comfort_new:.0f}%", f"{comfort_new - comfort_now:+.0f} pts vs now")
    k4.metric("Split", f"{sp_kwh:,.0f} / {sch_kwh:,.0f}", "setpoint / schedule kWh", delta_color="off")
    if best is not None:
        st.success(f"**Recommended setpoint: {best['setpoint']:.1f} °C.** Largest estimated saving "
                   f"({best['total_kWh']:,.0f} kWh, {best['total_kWh'] * tariff:,.0f} THB over the reporting period) "
                   f"with {best['comfort_pct']:.0f}% of running time in {band[0]:g}–{band[1]:g} °C "
                   f"(target ≥ {comfort_target}%; now {comfort_now:.0f}%).")
    else:
        best_comfort = sp_table.loc[sp_table["comfort_pct"].idxmax()]
        st.warning(f"No setpoint in 18–28 °C reaches {comfort_target}% compliance. The most comfortable setpoint is "
                   f"**{best_comfort['setpoint']:.1f} °C** ({best_comfort['comfort_pct']:.0f}% in band, "
                   f"{best_comfort['total_kWh']:,.0f} kWh estimated saving). Lower the target or widen the band.")

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

    g1, g2, g3 = st.columns([1, 1, 1.4])
    fig = go.Figure(go.Scatter(x=sp_table["setpoint"], y=sp_table["total_kWh"], mode="lines+markers",
                               line=dict(color=ACTUAL, width=2), marker=dict(size=7), name="Estimated saving"))
    fig.add_vline(x=setpoint, line=dict(color=EXPECTED, dash="dash"), annotation_text="selected")
    if best is not None:
        fig.add_vline(x=best["setpoint"], line=dict(color=GOOD, dash="dot"), annotation_text="recommended",
                      annotation_position="bottom right")
    g1.plotly_chart(style(fig, 300, "kWh saved (reporting period)", "Setpoint (°C)"), width="stretch")
    fig = go.Figure(go.Scatter(x=sp_table["setpoint"], y=sp_table["comfort_pct"], mode="lines+markers",
                               line=dict(color=GOOD, width=2), marker=dict(size=7), name="Comfort compliance"))
    fig.add_hline(y=comfort_target, line=dict(color=MUTED, dash="dot"), annotation_text="target")
    fig.add_vline(x=setpoint, line=dict(color=EXPECTED, dash="dash"))
    style(fig, 300, f"% running time in {band[0]:g}–{band[1]:g} °C", "Setpoint (°C)")
    fig.update_yaxes(range=[0, 100])
    g2.plotly_chart(fig, width="stretch")
    with g3:
        st.markdown("**Per unit at the selected setpoint**",
                    help="kW per °C = how much the unit's power changes per 1 °C, learned from the baseline year. "
                         "R² = how reliable that estimate is (closer to 1 = more reliable).")
        st.dataframe(pd.DataFrame(per_unit).round(2), hide_index=True, width="stretch", height=270)
    with st.expander("Assumptions and limits"):
        st.markdown(f"""
- Each zone's current effective setpoint = its median indoor temperature while the unit runs. A new setpoint shifts that zone's whole temperature profile by the difference, so normal variability (start-up pull-down, afternoon peaks) is kept.
- kW saved per °C = the unit's outdoor-temperature regression slope (baseline weekdays, unit running; hour-of-day controlled). Typical literature values are 3–10 % per °C. {SCOPE_NAME} units: {sens['pct_per_degC'].min():.1f}–{sens['pct_per_degC'].max():.1f} % per °C.
- Savings are capped at the unit's actual power and ignore humidity/latent load and compressor part-load efficiency.
- The schedule saving assumes all AC energy outside the window is avoidable. Check for legitimate after-hours use (events, cleaning, servers).
- Comfort compliance uses the zone S1 sensor only; it does not measure occupant satisfaction. Indoor RH (proposal band 40–60 %) is shown on the Overview tab.
""")

# ================================================================ forecast
with tab_forecast:
    fc = data["forecast"]
    tm = data["test_metrics"].copy()
    vm = data["validation_metrics"]
    selected_model = data.get("selected_model") or vm["MAE_kW"].idxmin()
    # Rank by accuracy on the unseen test period (lower average error = better)
    tm = tm.sort_values("MAE_kW")
    tm.insert(0, "Rank", range(1, len(tm) + 1))
    tm.insert(1, "Model", [("✓ " if m == selected_model else "") + MODEL_LABELS.get(m, m) for m in tm.index])
    tm["Validation MAE_kW"] = vm["MAE_kW"].reindex(tm.index)
    selected_rank = int(tm.loc[selected_model, "Rank"])
    best_mae, selected_mae = tm["MAE_kW"].iloc[0], tm.loc[selected_model, "MAE_kW"]
    models = [c for c in fc.columns if c != "floor_kW"]

    forecast_help = ("Ranked by average error on Jul–Dec 2019, data the methods never saw (lower = better). "
                     "The ✓ method is the one to use: it was chosen on earlier data (Apr–Jun 2019) before this "
                     "test, which is the fair way to pick one. ")
    if floor == ALL:
        forecast_help += ("For the whole building, 'Each floor's validation-selected model' adds up the method each "
                          "floor chose. ")
    forecast_help += ("Peak-hour error = average error in the busiest 10 % of hours. Bias: + means it predicts too "
                      "high on average, − too low. Methods with weather use observed Open-Meteo weather in place of a "
                      "weather forecast, so their errors are optimistic.")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Next-day hourly demand forecast", help=forecast_help)
        columns = (["Rank", "Model", "MAE_kW", "peak_hour_MAE_kW", "NMBE_pct"]
                   + ([] if floor == ALL else ["Validation MAE_kW"])  # the building has no validation run of its own
                   + ["RMSE_kW", "CV_RMSE_pct", "hours"])
        st.dataframe(tm[columns].round(2), hide_index=True, width="stretch", height=table_height(len(tm), 400),
                     column_config={
                         "Rank": st.column_config.NumberColumn("#", width=35),
                         "Model": st.column_config.TextColumn("Method (✓ = use this)", width="medium"),
                         "MAE_kW": st.column_config.NumberColumn("Avg error (kW)", format="%.2f",
                                                                 help="Average forecast error per hour. Lower is better."),
                         "peak_hour_MAE_kW": st.column_config.NumberColumn("Peak-hour error (kW)", format="%.2f",
                                                                           help="Average error in the busiest 10 % of hours."),
                         "NMBE_pct": st.column_config.NumberColumn("Bias (%)", format="%+.1f",
                                                                   help="+ = predicts too high on average, − = too low."),
                         "Validation MAE_kW": st.column_config.NumberColumn("Apr–Jun error (kW)", format="%.2f",
                                                                            help="Error on the earlier data used to choose the method."),
                         "RMSE_kW": st.column_config.NumberColumn("RMSE (kW)", format="%.2f"),
                         "CV_RMSE_pct": st.column_config.NumberColumn("CV(RMSE) %", format="%.1f"),
                         "hours": st.column_config.NumberColumn("Hours scored")})
        if selected_rank == 1:
            st.caption(f"Use **{MODEL_LABELS.get(selected_model, selected_model)}**: recommended and the most accurate on Jul–Dec 2019.")
        else:
            st.caption(f"Use **{MODEL_LABELS.get(selected_model, selected_model)}** (rank {selected_rank}). It was chosen "
                       f"before the test period; #1 on Jul–Dec was only {selected_mae - best_mae:.2f} kW per hour better.")
    with right:
        c1, c2 = st.columns([2, 1])
        picked = c1.multiselect("Models to plot", models, default=[selected_model] + [m for m in models if "weather" in m][:1],
                                format_func=lambda m: MODEL_LABELS.get(m, m), max_selections=3)
        dates = fc.index.normalize().unique()
        start = c2.date_input("Week starting", dates.min().date(), min_value=dates.min().date(), max_value=dates.max().date())
        view = fc.loc[pd.Timestamp(start): pd.Timestamp(start) + pd.Timedelta(days=7) - pd.Timedelta(hours=1)]
        fig = go.Figure(go.Scatter(x=view.index, y=view["floor_kW"], name="Actual", line=dict(color=ACTUAL, width=2)))
        for colour, m in zip([EXPECTED, "#1baf7a", "#4a3aa7"], picked):
            fig.add_trace(go.Scatter(x=view.index, y=view[m], name=MODEL_LABELS.get(m, m), line=dict(color=colour, width=2, dash="dash")))
        st.plotly_chart(style(fig, 300, "AC demand (kW)"), width="stretch")


# ================================================================ compare floors
with tab_compare:
    floor_colour = {f: UNIT_COLOURS[i % len(UNIT_COLOURS)] for i, f in enumerate(FLOORS)}

    summary_rows, unit_tables, details = [], [], {}
    for f in FLOORS:
        fdata = load(f)
        fd = flags_for(f, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist)
        fev = build_events(fd, hot_weight, after_weight)
        fpr = unit_priority(fd, fev)
        fpr.insert(0, "floor", f)
        unit_tables.append(fpr)
        details[f] = fd
        k = fdata["floor_kpis"]
        s_ = fdata["sensitivity"]
        vm_, tm_ = fdata["validation_metrics"], fdata["test_metrics"]
        chosen = vm_["MAE_kW"].idxmin()
        summary_rows.append({
            "Floor": f"Floor {f}",
            "AC energy kWh": k["estimated_energy_kWh_valid_intervals"],
            "After-hours %": k["after_hours_energy_pct_valid_intervals"],
            "15-min peak kW": k["15min_peak_kW"],
            "Flagged events": len(fev),
            "Excess kWh (est.)": fpr["flagged_excess_kWh"].sum(),
            "Priority score": fpr["priority_score"].sum(),
            "Median indoor °C (running)": s_["median_indoor_temp_C_running"].median(),
            "% running time 23–26 °C": s_["pct_running_hours_in_comfort_band"].mean(),
            "Forecast MAE kW": tm_.loc[chosen, "MAE_kW"],
            "Forecast model": MODEL_LABELS.get(chosen, chosen)})
    summary = pd.DataFrame(summary_rows)

    b0, b1, b2, b3, b4 = st.columns([2, 1, 1, 1, 1], vertical_alignment="center")
    b0.subheader("How do the floors compare?",
                 help="Floors 3, 4, 5 and 7 share the same AC layout, so units in the same position can be compared "
                      "directly. The sidebar thresholds apply to every floor. Equal layout does not prove equal occupancy "
                      "or equipment condition: treat differences as questions for the facilities team, not verdicts.")
    b1.metric("Floors analysed", len(FLOORS))
    b2.metric("AC energy, all floors", f"{summary['AC energy kWh'].sum():,.0f} kWh")
    b3.metric("Flagged events", f"{int(summary['Flagged events'].sum()):,}")
    b4.metric("Floor to review first", summary.loc[summary["Priority score"].idxmax(), "Floor"],
              help="Highest total priority score")

    t, c1, c2 = st.columns([3, 1, 1])
    t.dataframe(summary.round(1), hide_index=True, width="stretch", height=table_height(len(summary), 300), column_config={
        "AC energy kWh": st.column_config.NumberColumn(format="%.0f"),
        "Excess kWh (est.)": st.column_config.NumberColumn(format="%.0f"),
        "Priority score": st.column_config.ProgressColumn(format="%.0f", min_value=0,
                                                          max_value=float(max(1, summary["Priority score"].max())))})
    for col, metric, title in [(c1, "Excess kWh (est.)", "Excess kWh (est.)"),
                               (c2, "After-hours %", "After-hours share (%)")]:
        fig = go.Figure(go.Bar(x=summary["Floor"].str.replace("Floor ", "F"), y=summary[metric], marker_color=ACTUAL,
                               text=summary[metric].map(lambda v: f"{v:,.0f}" if v >= 100 else f"{v:.1f}"),
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: %{y:,.1f}<extra></extra>"))
        col.plotly_chart(style(fig, table_height(len(summary), 300), title, margin_top=10), width="stretch")

    # ---- building-wide priority list, with a same-position peer benchmark
    allunits = pd.concat(unit_tables)
    mean_kw = {u: details[f].loc[details[f]["unit"].eq(u) & details[f]["eligible"], "actual_kW"].mean()
               for f in FLOORS for u in details[f]["unit"].unique()}
    allunits["position"] = allunits.index.map(position_of)
    allunits["mean_kW"] = allunits.index.map(mean_kw)

    def vs_peers(row):
        peers = [v for u, v in mean_kw.items() if position_of(u) == row["position"] and u != row.name]
        peer = np.nanmedian(peers) if peers else np.nan
        return 100 * (row["mean_kW"] / peer - 1) if peer else np.nan

    allunits["vs_peers_pct"] = allunits.apply(vs_peers, axis=1)
    allunits = allunits.sort_values("priority_score", ascending=False)
    allunits["rank"] = range(1, len(allunits) + 1)
    allunits["AC unit"] = [unit_label(u, with_floor=True) for u in allunits.index]

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Building-wide priority list",
                     help="All units on all floors, ranked by priority score. 'vs peers' compares a unit's average power "
                          "with the median of the same unit on the other floors: +30% means it draws 30% more than its peers.")
        st.dataframe(allunits[["rank", "AC unit", "priority_score", "flagged_excess_kWh", "events",
                               "after_hours_flagged_h", "mean_kW", "vs_peers_pct"]],
                     hide_index=True, width="stretch", height=330, column_config={
                         "priority_score": st.column_config.ProgressColumn("Priority score", format="%.0f", min_value=0,
                                                                           max_value=float(max(1, allunits["priority_score"].max()))),
                         "flagged_excess_kWh": st.column_config.NumberColumn("Excess kWh", format="%.0f"),
                         "events": st.column_config.NumberColumn("Events"),
                         "after_hours_flagged_h": st.column_config.NumberColumn("After-hours h", format="%.1f"),
                         "mean_kW": st.column_config.NumberColumn("Avg kW", format="%.2f"),
                         "vs_peers_pct": st.column_config.NumberColumn("vs peers", format="%+.0f%%")})
        st.download_button("Download building-wide list (CSV)",
                           allunits.drop(columns=["position"]).to_csv().encode("utf-8"), "ac_priority_all_floors.csv", "text/csv")

    with right:
        positions = sorted({position_of(u) for u in mean_kw})
        st.subheader("Same unit, different floors",
                     help="Average daily power of the same AC position on each floor (reporting period). "
                          "Shows whether one floor starts earlier, runs later or runs higher.")
        pc1, pc2 = st.columns([1, 2], vertical_alignment="bottom")
        pos = pc1.selectbox("Unit position", positions, format_func=lambda p: unit_label(f"F0_{p}", with_floor=False))
        day_type = pc2.radio("Days", ["Weekdays", "Saturdays", "Sundays"], horizontal=True)
        times = [f"{s // 4:02d}:{(s % 4) * 15:02d}" for s in range(96)]
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
                         "Flagged excess kWh": part.loc[part["flagged"], "excess_kWh_est"].sum(),
                         "Median indoor °C": part.loc[running, "indoor_temp_C"].median(),
                         "kW per °C": load(f)["sensitivity"]["kW_per_degC_outdoor"].get(u, np.nan)})
            dow = part["Date"].dt.dayofweek
            part = part.loc[{"Weekdays": dow < 5, "Saturdays": dow == 5, "Sundays": dow == 6}[day_type]]
            if part.empty:
                continue
            slot = part["Date"].dt.hour * 4 + part["Date"].dt.minute // 15
            profile = part.groupby(slot)["actual_kW"].mean().reindex(range(96))
            fig.add_trace(go.Scatter(x=times, y=profile.values, name=f"Floor {f}",
                                     line=dict(color=floor_colour[f], width=3 if f == floor else 2)))
        style(fig, 220, "Average kW")
        fig.update_xaxes(tickvals=times[::12])
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch",
                     height=table_height(len(rows), 200), column_config={
                         "vs peers": st.column_config.NumberColumn(format="%+.0f%%"),
                         "After-hours kWh": st.column_config.NumberColumn(format="%.0f"),
                         "Flagged excess kWh": st.column_config.NumberColumn(format="%.0f")})
