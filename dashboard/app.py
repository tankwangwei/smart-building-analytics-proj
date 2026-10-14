"""Floor 7 AC dashboard: priority list, threshold tuner, setpoint/schedule simulator and forecasts.

Run the notebook first (it writes ac_analysis_outputs/), then:
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

OUT = Path(os.environ.get("AC_OUTPUT_FOLDER", Path(__file__).resolve().parents[1] / "ac_analysis_outputs"))
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
                "M6 Random Forest + weather": "Random Forest + weather"}

st.set_page_config(page_title="Floor 7 AC Dashboard", page_icon="❄️", layout="wide")


def unit_label(unit):
    match = re.search(r"z(\d+)_AC(\d+)", unit)
    return f"Zone {match[1]} · AC{match[2]}" if match else unit


def to_bool(series):
    return series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False,
                                               "1.0": True, "0.0": False})


def style(fig, height=380, ytitle=None, xtitle=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    fig.update_xaxes(showgrid=False, title=xtitle)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", title=ytitle, rangemode="tozero")
    return fig


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading analysis outputs…")
def load():
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
def flags_for(k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist):
    data = load()
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
if not (OUT / "anomaly_15min_detail.csv").exists():
    st.error(f"No analysis outputs found in `{OUT}`. Run `AC_Analysis_Floor7_Clear_Charts.ipynb` first, "
             "or set the AC_OUTPUT_FOLDER environment variable.")
    st.stop()

data = load()
detail = data["detail"]
units = list(dict.fromkeys(detail["unit"]))
unit_colour = {u: UNIT_COLOURS[i % len(UNIT_COLOURS)] for i, u in enumerate(units)}

with st.sidebar:
    st.header("Anomaly thresholds")
    st.caption("These settings drive the priority list, heatmap and overview. The notebook defaults are "
               "k = 1.5, 0.5 kW and 8 intervals (2 h).")
    k_iqr = st.slider("Statistical rule: IQR multiplier k", 0.5, 5.0, 1.5, 0.25,
                      help="Flag when power > Q3 + k·IQR for that 15-minute slot and weekday (baseline).")
    k_res = st.slider("Persistence rule: residual IQR multiplier", 0.5, 5.0, 1.5, 0.25,
                      help="Residual = actual − expected. Flag when it stays above Q3 + k·IQR of baseline residuals.")
    persist = st.slider("Persistence: minimum consecutive 15-min intervals", 1, 24, 8,
                        help="8 intervals = 2 hours")
    on_kw = st.slider("'Running' threshold (kW)", 0.1, 5.0, 0.5, 0.1,
                      help="Power must exceed this for any flag; also used as the on/off cut-off.")
    st.markdown("**Rules enabled**")
    use_stat = st.checkbox("Unusually high for time slot", True)
    use_after = st.checkbox("After-hours operation", True)
    use_persist = st.checkbox("Persistent deviation", True)
    st.header("Priority weighting")
    after_weight = st.slider("After-hours multiplier", 1.0, 3.0, 1.5, 0.1,
                             help="After-hours waste is usually the easiest to act on (schedule fix).")
    hot_weight = st.slider("Hot-day multiplier", 0.0, 1.0, 0.5, 0.1,
                           help="Down-weight events on hot days: higher demand is partly explained by weather.")

d = flags_for(k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist)
events = build_events(d, hot_weight, after_weight)
priority = unit_priority(d, events)

st.title("Floor 7 AC: anomaly priorities and energy simulator")
st.caption(f"CU-BEMS Chamchuri 5, Bangkok · reporting period {detail['Date'].min():%d %b %Y} – "
           f"{detail['Date'].max():%d %b %Y} · baseline Jul 2018 – Jun 2019 · weather: Open-Meteo. "
           "Flags are potential anomalies for review, not diagnosed faults; kWh figures are estimates, not verified savings.")

tab_overview, tab_priority, tab_tuner, tab_sim, tab_forecast = st.tabs(
    ["Overview", "Priority list", "Threshold tuner", "Setpoint & schedule simulator", "Forecast"])

# ================================================================ overview
with tab_overview:
    kpis = data["floor_kpis"]
    sens = data["sensitivity"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AC energy (valid intervals)", f"{kpis['estimated_energy_kWh_valid_intervals']:,.0f} kWh")
    c2.metric("After-hours share", f"{kpis['after_hours_energy_pct_valid_intervals']:.1f}%")
    c3.metric("15-min peak", f"{kpis['15min_peak_kW']:.1f} kW")
    c4.metric("Flagged events", f"{len(events):,}", help="With the thresholds in the sidebar")
    c5.metric("Median indoor temp while running", f"{sens['median_indoor_temp_C_running'].median():.1f} °C",
              help="Reporting period, weekdays, unit running. Typical comfort band 23–26 °C.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Review these units first")
        show = priority[["rank", "AC unit", "priority_score", "flagged_excess_kWh", "events",
                         "after_hours_flagged_h", "flagged_pct"]]
        st.dataframe(show, hide_index=True, use_container_width=True, column_config={
            "priority_score": st.column_config.ProgressColumn("Priority score", format="%.0f",
                                                              min_value=0, max_value=float(max(1, show["priority_score"].max()))),
            "flagged_excess_kWh": st.column_config.NumberColumn("Excess kWh (est.)", format="%.0f"),
            "events": st.column_config.NumberColumn("Events"),
            "after_hours_flagged_h": st.column_config.NumberColumn("After-hours flagged (h)", format="%.1f"),
            "flagged_pct": st.column_config.NumberColumn("Flagged %", format="%.1f%%")})
        st.subheader("Top 5 events")
        if len(events):
            top = events.head(5).copy()
            top["AC unit"] = top["unit"].map(unit_label)
            st.dataframe(top[["AC unit", "start", "duration_h", "peak_kW", "excess_kWh", "reason"]],
                         hide_index=True, use_container_width=True, column_config={
                             "start": st.column_config.DatetimeColumn("Start", format="ddd D MMM YYYY, HH:mm"),
                             "duration_h": st.column_config.NumberColumn("Hours", format="%.2f"),
                             "peak_kW": st.column_config.NumberColumn("Peak kW", format="%.1f"),
                             "excess_kWh": st.column_config.NumberColumn("Excess kWh", format="%.1f")})
    with right:
        st.subheader("Next-day forecast (demo: day after dataset)")
        n24 = data["next24"]
        fig = go.Figure(go.Scatter(x=n24.index, y=n24["predicted_kW"], mode="lines+markers",
                                   line=dict(color=ACTUAL, width=2), marker=dict(size=8), name="Forecast"))
        peak = n24["predicted_kW"].idxmax()
        fig.add_annotation(x=peak, y=n24["predicted_kW"].max(), text=f"Peak {peak:%H:%M}: {n24['predicted_kW'].max():.1f} kW",
                           showarrow=True, arrowhead=2)
        st.plotly_chart(style(fig, 300, "kW"), use_container_width=True)
        st.caption(f"Method: {MODEL_LABELS.get(n24['method'].iloc[0], n24['method'].iloc[0])} · "
                   f"predicted daily energy {n24['predicted_kW'].sum():,.0f} kWh · holidays are not modelled.")
        st.subheader("Comfort while running (reporting period)")
        comfort = sens[["median_indoor_temp_C_running", "pct_running_hours_in_comfort_band", "pct_running_hours_in_RH_band"]].copy()
        comfort.index = comfort.index.map(unit_label)
        comfort.columns = ["Median indoor °C", "% hours 23–26 °C", "% hours RH 40–60%"]
        st.dataframe(comfort.round(1), use_container_width=True)
        if comfort["Median indoor °C"].median() < 23:
            st.info("Zones are held **below** the comfort band. This points to overcooling. "
                    "Try the setpoint simulator to see the potential savings.")

# ================================================================ priority list
with tab_priority:
    st.subheader("Unit priority list")
    st.caption("Priority score = Σ events of excess kWh × after-hours multiplier × hot-day multiplier (sidebar). "
               "Ranked by score; use it to decide which unit to inspect first.")
    st.dataframe(priority, hide_index=True, use_container_width=True, column_config={
        "priority_score": st.column_config.ProgressColumn("Priority score", format="%.0f", min_value=0,
                                                          max_value=float(max(1, priority["priority_score"].max()))),
        "valid_h": st.column_config.NumberColumn("Valid hours", format="%.0f"),
        "flagged_h": st.column_config.NumberColumn("Flagged hours", format="%.1f"),
        "after_hours_flagged_h": st.column_config.NumberColumn("After-hours flagged (h)", format="%.1f"),
        "flagged_pct": st.column_config.NumberColumn("Flagged %", format="%.1f%%"),
        "flagged_excess_kWh": st.column_config.NumberColumn("Excess kWh (est.)", format="%.0f"),
        "longest_event_h": st.column_config.NumberColumn("Longest event (h)", format="%.2f"),
        "hot_day_events": st.column_config.NumberColumn("Events on hot days")})

    st.subheader("Event list")
    if events.empty:
        st.warning("No events with the current thresholds.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        pick_units = f1.multiselect("Units", units, default=units, format_func=unit_label)
        min_hours = f2.number_input("Min duration (h)", 0.0, 48.0, 0.0, 0.25)
        only_after = f3.checkbox("After-hours only")
        hide_hot = f4.checkbox("Hide hot-day events")
        ev = events.loc[events["unit"].isin(pick_units) & events["duration_h"].ge(min_hours)]
        if only_after:
            ev = ev.loc[ev["after_hours_h"].gt(0)]
        if hide_hot:
            ev = ev.loc[~ev["hot_day"]]
        ev = ev.assign(**{"AC unit": ev["unit"].map(unit_label)})
        ev.insert(0, "rank", range(1, len(ev) + 1))
        cols = ["rank", "AC unit", "start", "end", "duration_h", "peak_kW", "excess_kWh", "after_hours_h",
                "outdoor_C", "indoor_C", "priority_score", "reason"]
        st.dataframe(ev[cols], hide_index=True, use_container_width=True, height=360, column_config={
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

    st.subheader("Inspect a unit")
    a, b = st.columns([1, 2])
    unit = a.selectbox("AC unit", priority.index.tolist(), format_func=unit_label)
    u = d.loc[d["unit"].eq(unit)].set_index("Date")
    weekly = u["excess_kWh_est"].where(u["flagged"], 0).resample("W-MON", label="left", closed="left").sum()
    weeks = weekly.index.tolist()
    default_week = weekly.idxmax() if weekly.max() > 0 else weeks[0]
    week = b.select_slider("Week starting", options=weeks, value=default_week,
                           format_func=lambda t: f"{t:%d %b %Y}" + (" (most excess)" if t == weekly.idxmax() else ""))
    view = u.loc[week: week + pd.Timedelta(days=7) - pd.Timedelta(minutes=15)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=view.index, y=view["actual_kW"], name="Actual", line=dict(color=ACTUAL, width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=view.index, y=view["expected_kW"], name="Expected", line=dict(color=EXPECTED, width=2, dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=view.index, y=view["upper_kW"], name="Statistical upper limit", line=dict(color=MUTED, width=1, dash="dot")), 1, 1)
    fl = view.loc[view["flagged"]]
    fig.add_trace(go.Scatter(x=fl.index, y=fl["actual_kW"], mode="markers", name="Flagged",
                             marker=dict(color=FLAG, size=7, opacity=.85)), 1, 1)
    fig.add_trace(go.Scatter(x=view.index, y=view["outdoor_temp_C"], name="Outdoor °C", line=dict(color=MUTED, width=2)), 2, 1)
    fig.add_trace(go.Scatter(x=view.index, y=view["indoor_temp_C"], name="Indoor °C", line=dict(color=GOOD, width=2)), 2, 1)
    style(fig, 520)
    fig.update_yaxes(title="AC power (kW)", row=1, col=1)
    fig.update_yaxes(title="°C", rangemode="normal", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("When are flags most common?")
    hm = d.loc[d["eligible"]].assign(dow=lambda x: x["Date"].dt.dayofweek,
                                     slot=lambda x: x["Date"].dt.hour * 4 + x["Date"].dt.minute // 15)
    heat = (100 * hm.groupby(["dow", "slot"])["flagged"].mean()).unstack().reindex(index=range(7), columns=range(96))
    times = [f"{s // 4:02d}:{(s % 4) * 15:02d}" for s in range(96)]
    fig = go.Figure(go.Heatmap(z=heat.values, x=times, y=DAYS, colorscale=[[0, "#f0efec"], [0.5, "#86b6ef"], [1, "#104281"]],
                               zmin=0, zmax=max(1, float(np.nanmax(heat.values))) if np.isfinite(heat.values).any() else 1,
                               colorbar=dict(title="% flagged"),
                               hovertemplate="%{y} %{x}<br>%{z:.1f}% of valid intervals flagged<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickvals=times[::8], ticktext=times[::8])
    st.plotly_chart(style(fig, 320), use_container_width=True)

# ================================================================ threshold tuner
with tab_tuner:
    st.subheader("How sensitive are the results to the threshold?")
    st.caption("Each curve re-runs the rules across a range of values and keeps the other sidebar settings fixed. "
               "A good operating point sits just past the 'elbow'. Beyond it, raising the threshold removes few flags, "
               "so the remaining flags are the robust ones. The true optimum needs facilities-manager feedback on which flags were real.")
    param = st.radio("Vary", ["Statistical IQR multiplier k", "Residual IQR multiplier", "Persistence (intervals)",
                              "'Running' threshold (kW)"], horizontal=True)
    grids = {"Statistical IQR multiplier k": np.arange(0.5, 5.01, 0.25),
             "Residual IQR multiplier": np.arange(0.5, 5.01, 0.25),
             "Persistence (intervals)": np.arange(1, 25),
             "'Running' threshold (kW)": np.round(np.arange(0.1, 5.01, 0.2), 2)}
    current = {"Statistical IQR multiplier k": k_iqr, "Residual IQR multiplier": k_res,
               "Persistence (intervals)": persist, "'Running' threshold (kW)": on_kw}[param]

    @st.cache_data(show_spinner="Sweeping thresholds…")
    def sweep(param, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist):
        rows = []
        base = load()
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

    sw = sweep(param, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist)
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
    m1, m2, m3 = st.columns(3)
    m1.metric("Current value", f"{current:g}")
    m2.metric("Elbow suggestion", f"{elbow:g}" if elbow is not None else "n/a")
    cur_row = sw.iloc[int(np.argmin(np.abs(x - current)))]
    m3.metric("Flagged share at current value", f"{cur_row['flagged_pct']:.1f}%")

    c1, c2 = st.columns(2)
    for col, metric, title in [(c1, "flagged_pct", "Share of valid intervals flagged (%)"),
                               (c2, "flagged_excess_kWh", "Estimated excess kWh inside flags")]:
        fig = go.Figure(go.Scatter(x=sw["value"], y=sw[metric], mode="lines+markers", line=dict(color=ACTUAL, width=2),
                                   marker=dict(size=8), name="All enabled rules"))
        if metric == "flagged_pct" and param != "'Running' threshold (kW)":
            fig.add_trace(go.Scatter(x=sw["value"], y=sw["rule_pct"], mode="lines+markers", name="This rule alone",
                                     line=dict(color=EXPECTED, width=2, dash="dash"), marker=dict(size=8)))
        fig.add_vline(x=current, line=dict(color=EXPECTED, dash="dash"), annotation_text="current")
        if elbow is not None:
            fig.add_vline(x=elbow, line=dict(color=MUTED, dash="dot"), annotation_text="elbow", annotation_position="bottom right")
        col.plotly_chart(style(fig, 340, title, param), use_container_width=True)
    fig = go.Figure(go.Bar(x=sw["value"], y=sw["events"], marker_color=ACTUAL, name="Events"))
    fig.add_vline(x=current, line=dict(color=EXPECTED, dash="dash"))
    st.plotly_chart(style(fig, 280, "Number of events", param), use_container_width=True)
    st.caption("Set the chosen value in the sidebar. The priority list and overview update immediately.")

# ================================================================ simulator
with tab_sim:
    st.subheader("What if we change the setpoint or the operating schedule?")
    st.caption("Uses the reporting-period 15-minute data. Setpoint effect: each unit's baseline regression slope "
               "(kW per °C of outdoor temperature) stands in for the kW saved per 1 °C setpoint rise. "
               "The zone's temperature profile shifts so its median sits at the new setpoint. A setpoint below today's level costs energy (negative saving). Schedule effect: AC energy outside the chosen window "
               "is counted as avoidable. These are screening estimates; confirm with a measured trial.")
    sens = data["sensitivity"]
    slope = sens["kW_per_degC_outdoor"].clip(lower=0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Setpoint**")
        setpoint = st.slider("New indoor setpoint (°C)", 18.0, 28.0, 25.0, 0.5)
        band = st.slider("Comfort band (°C)", 18.0, 30.0, (23.0, 26.0), 0.5)
        comfort_target = st.slider("Minimum comfort compliance (%)", 30, 100, 70)
    with c2:
        st.markdown("**Schedule**")
        start_h, stop_h = st.slider("Weekday AC window", 0.0, 24.0, (7.0, 18.0), 0.25,
                                    format="%.2f h", help="Energy outside this window counts as avoidable.")
        weekend_on = st.checkbox("Keep weekends running", True)
        apply_schedule = st.checkbox("Apply schedule change", False)
    with c3:
        st.markdown("**Cost**")
        tariff = st.number_input("Electricity price (THB per kWh)", 0.0, 20.0, 4.2, 0.1,
                                 help="Assumption. Change it to your actual tariff.")
        st.markdown("**Per-unit sensitivity (from baseline)**")
        st.dataframe(pd.DataFrame({"kW/°C": slope.round(2), "% per °C": sens["pct_per_degC"].round(1),
                                   "R²": sens["R2"].round(2)}).set_index(sens.index.map(unit_label)),
                     use_container_width=True)

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
        avoided = sim["running"] & sim["outside_window"] if apply_schedule else pd.Series(False, index=sim.index)
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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estimated energy saved", f"{saved:,.0f} kWh", f"{-100 * saved / total_kwh:.1f}% of AC energy",
              delta_color="inverse")
    k2.metric("Estimated cost saved", f"{saved * tariff:,.0f} THB", f"≈ {saved * tariff * 365 / days:,.0f} THB/year",
              delta_color="off")
    k3.metric("Comfort compliance", f"{comfort_new:.0f}%", f"{comfort_new - comfort_now:+.0f} pts vs now")
    k4.metric("Split", f"{sp_kwh:,.0f} / {sch_kwh:,.0f}", "setpoint / schedule kWh", delta_color="off")

    sweep_sp = np.arange(18.0, 28.01, 0.5)
    rows = [(sp, *simulate(sp)) for sp in sweep_sp]
    sp_table = pd.DataFrame(rows, columns=["setpoint", "setpoint_kWh", "schedule_kWh", "comfort_pct"])
    sp_table["total_kWh"] = sp_table["setpoint_kWh"] + sp_table["schedule_kWh"]
    ok = sp_table.loc[sp_table["comfort_pct"].ge(comfort_target)]
    best = ok.loc[ok["total_kWh"].idxmax()] if len(ok) else None
    if best is not None:
        st.success(f"**Recommended setpoint: {best['setpoint']:.1f} °C.** It gives the largest estimated saving "
                   f"({best['total_kWh']:,.0f} kWh, {best['total_kWh'] * tariff:,.0f} THB over the reporting period) "
                   f"while {best['comfort_pct']:.0f}% of running time stays within {band[0]:g}–{band[1]:g} °C "
                   f"(target ≥ {comfort_target}%). Current compliance: {comfort_now:.0f}%.")
    else:
        best_comfort = sp_table.loc[sp_table["comfort_pct"].idxmax()]
        st.warning(f"No setpoint in 18–28 °C reaches {comfort_target}% compliance. The most comfortable setpoint is "
                   f"**{best_comfort['setpoint']:.1f} °C** ({best_comfort['comfort_pct']:.0f}% in band, "
                   f"{best_comfort['total_kWh']:,.0f} kWh estimated saving). Lower the target or widen the band.")

    g1, g2 = st.columns(2)
    fig = go.Figure(go.Scatter(x=sp_table["setpoint"], y=sp_table["total_kWh"], mode="lines+markers",
                               line=dict(color=ACTUAL, width=2), marker=dict(size=8), name="Estimated saving"))
    fig.add_vline(x=setpoint, line=dict(color=EXPECTED, dash="dash"), annotation_text="selected")
    if best is not None:
        fig.add_vline(x=best["setpoint"], line=dict(color=GOOD, dash="dot"), annotation_text="recommended",
                      annotation_position="bottom right")
    g1.plotly_chart(style(fig, 340, "kWh saved (reporting period)", "Setpoint (°C)"), use_container_width=True)
    fig = go.Figure(go.Scatter(x=sp_table["setpoint"], y=sp_table["comfort_pct"], mode="lines+markers",
                               line=dict(color=GOOD, width=2), marker=dict(size=8), name="Comfort compliance"))
    fig.add_hline(y=comfort_target, line=dict(color=MUTED, dash="dot"), annotation_text="target")
    fig.add_vline(x=setpoint, line=dict(color=EXPECTED, dash="dash"))
    style(fig, 340, f"% running time in {band[0]:g}–{band[1]:g} °C", "Setpoint (°C)")
    fig.update_yaxes(range=[0, 100])
    g2.plotly_chart(fig, use_container_width=True)

    st.markdown("**Per unit at the selected setpoint**")
    per_unit = []
    for unit_name, part in sim.groupby("unit"):
        avoided = part["running"] & part["outside_window"] if apply_schedule else pd.Series(False, index=part.index)
        active = part["running"] & ~avoided & part["indoor_temp_C"].notna()
        shift = (setpoint - part["current_sp"]).where(active, 0)
        sp_unit = np.minimum(part["slope"] * shift, part["actual_kW"]).sum() * INTERVAL_HOURS
        sch_unit = part.loc[avoided, "actual_kW"].sum() * INTERVAL_HOURS
        energy = part["actual_kW"].sum() * INTERVAL_HOURS
        per_unit.append({"AC unit": unit_label(unit_name), "AC energy kWh": energy,
                         "Median indoor °C (running)": part.loc[part["running"], "indoor_temp_C"].median(),
                         "Setpoint saving kWh": sp_unit, "Schedule saving kWh": sch_unit,
                         "Saving %": 100 * (sp_unit + sch_unit) / energy if energy else np.nan})
    st.dataframe(pd.DataFrame(per_unit).round(1), hide_index=True, use_container_width=True)
    with st.expander("Assumptions and limits"):
        st.markdown(f"""
- Each zone's current effective setpoint = its median indoor temperature while the unit runs. A new setpoint shifts that zone's whole temperature profile by the difference, so normal variability (start-up pull-down, afternoon peaks) is kept.
- kW saved per °C = the unit's outdoor-temperature regression slope (baseline weekdays, unit running; hour-of-day controlled). Typical literature values are 3–10 % per °C. Floor 7 units: {sens['pct_per_degC'].min():.1f}–{sens['pct_per_degC'].max():.1f} % per °C.
- Savings are capped at the unit's actual power and ignore humidity/latent load and compressor part-load efficiency.
- The schedule saving assumes all AC energy outside the window is avoidable. Check for legitimate after-hours use (events, cleaning, servers).
- Comfort compliance uses the zone S1 sensor only; it does not measure occupant satisfaction. Indoor RH (proposal band 40–60 %) is shown on the Overview tab.
""")

# ================================================================ forecast
with tab_forecast:
    fc = data["forecast"]
    tm = data["test_metrics"].copy()
    vm = data["validation_metrics"]
    selected_model = vm["MAE_kW"].idxmin()
    st.subheader("Next-day hourly demand forecast (RQ2)")
    st.caption(f"Model chosen on Apr–Jun 2019 validation: **{MODEL_LABELS.get(selected_model, selected_model)}**. "
               "Test period Jul–Dec 2019. M5/M6 use observed Open-Meteo weather as a stand-in for a weather forecast, "
               "so their errors are optimistic.")
    tm.insert(0, "Model", tm.index.map(lambda m: MODEL_LABELS.get(m, m)))
    tm["Validation MAE_kW"] = vm["MAE_kW"].reindex(tm.index)
    st.dataframe(tm[["Model", "Validation MAE_kW", "MAE_kW", "RMSE_kW", "NMBE_pct", "CV_RMSE_pct", "peak_hour_MAE_kW", "hours"]]
                 .round(2), hide_index=True, use_container_width=True)

    models = [c for c in fc.columns if c != "floor_kW"]
    c1, c2 = st.columns([2, 1])
    picked = c1.multiselect("Models to plot", models, default=[selected_model] + [m for m in models if "weather" in m][:1],
                            format_func=lambda m: MODEL_LABELS.get(m, m), max_selections=3)
    dates = fc.index.normalize().unique()
    start = c2.date_input("Week starting", dates.min().date(), min_value=dates.min().date(), max_value=dates.max().date())
    view = fc.loc[pd.Timestamp(start): pd.Timestamp(start) + pd.Timedelta(days=7) - pd.Timedelta(hours=1)]
    fig = go.Figure(go.Scatter(x=view.index, y=view["floor_kW"], name="Actual", line=dict(color=ACTUAL, width=2)))
    for colour, m in zip([EXPECTED, "#1baf7a", "#4a3aa7"], picked):
        fig.add_trace(go.Scatter(x=view.index, y=view[m], name=MODEL_LABELS.get(m, m), line=dict(color=colour, width=2, dash="dash")))
    st.plotly_chart(style(fig, 380, "Floor AC demand (kW)"), use_container_width=True)

    c1, c2 = st.columns(2)
    err_model = picked[0] if picked else selected_model
    common = fc[["floor_kW", err_model]].dropna()
    by_hour = (common[err_model] - common["floor_kW"]).abs().groupby(common.index.hour).mean()
    fig = go.Figure(go.Bar(x=by_hour.index, y=by_hour.values, marker_color=ACTUAL,
                           hovertemplate="%{x}:00 · MAE %{y:.2f} kW<extra></extra>"))
    c1.plotly_chart(style(fig, 320, "MAE (kW)", f"Hour of day · {MODEL_LABELS.get(err_model, err_model)}"), use_container_width=True)

    w = data["weather"]
    hourly = fc[["floor_kW"]].join(w)
    daily = pd.DataFrame({"kWh": hourly["floor_kW"].resample("D").sum(min_count=24),
                          "temp": hourly.between_time("08:00", "16:59")["outdoor_temp_C"].resample("D").mean()}).dropna()
    daily = daily.loc[daily.index.dayofweek < 5]
    fig = go.Figure(go.Scatter(x=daily["temp"], y=daily["kWh"], mode="markers", marker=dict(color=ACTUAL, size=8, opacity=.7),
                               text=daily.index.strftime("%a %d %b"),
                               hovertemplate="%{text}<br>%{x:.1f} °C · %{y:,.0f} kWh<extra></extra>", name="Weekday"))
    c2.plotly_chart(style(fig, 320, "Daily AC energy (kWh)", "Daytime outdoor temperature (°C)"), use_container_width=True)
