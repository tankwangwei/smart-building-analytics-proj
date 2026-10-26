"""Streamlit rendering of the asset history card (Phase 7) and the Maintenance tab (Phase 9)."""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fm_asset import monthly_comfort, monthly_kw_per_degc, monthly_peer_gap, recurrence_rate, timeline
from fm_health import DEMO_LABEL, FAULT_TYPES, LOG_TYPES, likely_causes
from fm_learning import precision_tables

from fm_tabs import SIM_BADGE, status_chip

KIND_COLOUR = {"Flagged episode": "#e5533d", "Work order": "#5aa9ff", "Close-out": "#1baf7a", "Maintenance": "#eda100"}
TEMPLATE_COLS = ["asset_id", "date", "type", "fault_type", "severity", "downtime_h", "cost_THB", "parts", "technician", "notes"]


# ================================================================ asset card
def asset_card_body(c, asset_id):
    u = c.units[c.units["asset_id"].eq(asset_id)].iloc[0]
    col = u["power_column"]
    d = c.details[int(u["floor"])]
    d = d[d["unit"].eq(col)]
    st.markdown(f"### {u['label']}")
    st.caption(f"Brick id `{asset_id}` · Floor {u['floor']} · Zone {u['zone']} · supply group {u['supply_group']} (assumed)")
    row = c.allunits.loc[col] if col in c.allunits.index else None
    ev = c.events[c.events["asset_id"].eq(asset_id)]
    weeks = max(1, int(np.ceil(len(d["Date"].dt.normalize().unique()) / 7)))
    inc = c.inc_all[c.inc_all["asset_id"].eq(asset_id)]
    log = c.store.table("maintenance_log")
    log = log[log["asset_id"].eq(asset_id)]
    last_service = pd.to_datetime(log["date"]).max() if len(log) else None
    prec = precision_tables(c.store, c.events)["unit"]
    p = prec["precision"].get(asset_id, np.nan) if len(prec) else np.nan
    k = st.columns(5)
    k[0].metric(f"Excess ({c.period})", f"{row['excess_THB']:,.0f} THB" if row is not None else "–")
    k[1].metric("Recurrence", f"{100 * recurrence_rate(ev, weeks):.0f}% of weeks", help="Weeks with an incident ÷ weeks in the data.")
    k[2].metric("Open incidents", int(inc["status"].isin(["Open", "Acknowledged", "Work order raised", "Reopened"]).sum()))
    k[3].metric("Last service", f"{last_service:%d %b %Y}" if last_service is not None else "–")
    k[4].metric("Flag precision", f"{100 * p:.0f}%" if pd.notna(p) else "–", help="Fault confirmed ÷ (confirmed + no fault) from close-outs.")
    if len(log) and log["simulated"].fillna(0).astype(int).any():
        st.markdown(SIM_BADGE + " Service and fault entries for this unit include simulated history.", unsafe_allow_html=True)
    if len(inc):
        st.markdown("**Incidents** " + " ".join(f"{r.pattern} {status_chip(r.status)}" for r in inc.itertuples()), unsafe_allow_html=True)

    tl = timeline(asset_id, c.events, c.store)
    if len(tl):
        fig = go.Figure()
        for kind, g in tl.groupby("kind"):
            real, sim = g[~g["simulated"]], g[g["simulated"]]
            for part, sym, suffix in [(real, "circle", ""), (sim, "x", " (simulated)")]:
                if len(part):
                    fig.add_trace(go.Scatter(x=part["time"], y=[kind] * len(part), mode="markers", name=kind + suffix,
                                             text=part["label"], marker=dict(color=KIND_COLOUR[kind], size=9, symbol=sym),
                                             hovertemplate="%{text}<br>%{x|%a %d %b %Y}<extra></extra>"))
        c.style(fig, 260, None, None, margin_top=10)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch", key=f"tl_{asset_id}")
    a, b = st.columns(2)
    m = monthly_kw_per_degc(d, c.on_kw)
    fig = go.Figure(go.Scatter(x=m["month"], y=m["kW_per_degC"], mode="lines+markers", line=dict(color=c.colours["actual"])))
    c.style(fig, 240, "kW per °C", "Monthly efficiency (weekday running hours)", margin_top=10)
    a.plotly_chart(fig, width="stretch", key=f"eff_{asset_id}")
    peers = c.peer_groups.get(u["position"], [])
    gap = monthly_peer_gap(c.details, col, peers)
    comfort = monthly_comfort(d, c.on_kw)
    fig = go.Figure()
    if len(gap):
        fig.add_trace(go.Bar(x=gap["month"], y=gap["gap_pct"], name="Gap vs peers %", marker_color=c.colours["expected"]))
    if len(comfort):
        fig.add_trace(go.Scatter(x=comfort["month"], y=comfort["comfort_pct"], name="Comfort % (23–26 °C)", mode="lines+markers",
                                 line=dict(color="#1baf7a"), yaxis="y2"))
    c.style(fig, 240, "% vs peers", "Monthly gap against peers and comfort", margin_top=10)
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100], title="% in comfort band"))
    fig.update_yaxes(rangemode="normal", selector=dict(title_text="% vs peers"))
    b.plotly_chart(fig, width="stretch", key=f"gap_{asset_id}")
    if len(log):
        st.markdown("**Maintenance log**")
        st.dataframe(log.sort_values("date", ascending=False)[["date", "type", "fault_type", "notes", "simulated"]],
                     hide_index=True, width="stretch", height=200)


# ================================================================ Maintenance tab
def log_import_template():
    return pd.DataFrame([["F7_Z1_AC4", "2019-09-01", "fault repair", "dirty filter", "Low", 2, 1500, "", "name", "notes"]],
                        columns=TEMPLATE_COLS).to_csv(index=False).encode("utf-8-sig")


def import_csv(store, raw, actor):
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    missing = [col for col in ["asset_id", "date", "type"] if col not in df.columns]
    if missing:
        raise ValueError(f"missing column(s): {', '.join(missing)}")
    bad = sorted(set(df["type"]) - set(LOG_TYPES))
    if bad:
        raise ValueError(f"unknown type(s): {', '.join(map(str, bad))}. Use: {', '.join(LOG_TYPES)}")
    n = 0
    for r in df.to_dict("records"):
        store.upsert("maintenance_log", {**{k: r.get(k) for k in TEMPLATE_COLS}, "log_id": store.next_id("maintenance_log"),
                                         "date": str(pd.Timestamp(r["date"]).date()), "source": "import", "wo_id": None,
                                         "simulated": 0}, actor, audit=False)
        n += 1
    store.log(actor, "maintenance_log", "import", f"imported {n} rows")
    return n


def maintenance_tab(c):
    log = c.store.table("maintenance_log")
    sim = int(log["simulated"].fillna(0).astype(int).sum()) if len(log) else 0
    if sim:
        st.markdown(SIM_BADGE + f" {sim} of {len(log)} log entries are simulated (an 18-month demo history, fixed random seed). "
                    "Import real records below to replace them.", unsafe_allow_html=True)
    st.subheader("Fleet health", help="Health score: 100 = as normal for the unit. Built from real sensor data (efficiency "
                                      "index, drift against baseline, warm-while-running hours, peer gap, starts). Fault risk "
                                      "and likely fault come from simulated maintenance entries.")
    h = c.health
    show = pd.DataFrame({"AC unit": h["label"], "Health": h["health"].round(0), "30-day fault risk": h["risk_band"] + " (demo)",
                         "Service due": h["service_due"].dt.strftime("%d %b %Y"), "Days": h["days_to_service"],
                         "Efficiency trend": [f"{r.degradation}" + (f", {r.slope_pct_per_week:+.1f}%/wk" if pd.notna(r.slope_pct_per_week) else "")
                                              for r in h.itertuples()],
                         "RUL by trend (days)": [f"{int(r.rul_deg_days)} ({int(r.rul_low)}–{int(r.rul_high)})" if pd.notna(r.rul_deg_days) else "–"
                                                 for r in h.itertuples()],
                         "Likely fault (demo)": [", ".join(f"{k} {v:.0%}" for k, v in likely_causes(c.store, "warm_running", a, 2).items())
                                                 for a in h.index]})
    st.dataframe(show.sort_values("Health"), hide_index=True, width="stretch", height=35 * len(show) + 38,
                 column_config={"Health": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                                "Days": st.column_config.NumberColumn(help="Days until the earlier of: 90 days / 500 run hours since "
                                                                           "the last service (assumed intervals) and the efficiency "
                                                                           "trend crossing +15 % kW per °C. Never negative.")})
    m = c.risk_metrics
    k = st.columns(4)
    k[0].metric("Risk model AUC (time split)", f"{m['auc']:.2f}" if pd.notna(m["auc"]) else "–",
                help=f"Trained on earlier weeks, scored on later weeks. {DEMO_LABEL}. AUC near 0.5 means no skill: expected, "
                     "because the simulated faults are not generated from the sensor data.")
    k[1].metric("Precision, top 3 units", f"{m['precision_top3']:.0%}" if pd.notna(m["precision_top3"]) else "–",
                help=f"Share of the 3 highest-risk units per test week that had a fault within 30 days (base rate {m['base_rate']:.0%}).")
    k[2].metric("Train / test weeks", f"{m['train_weeks']} / {m['test_weeks']}")
    k[3].markdown(f"<div style='margin-top:1.4rem'>{SIM_BADGE} {DEMO_LABEL}</div>", unsafe_allow_html=True)

    st.subheader("Degradation of one unit")
    asset = st.selectbox("AC unit", h.index.tolist(), format_func=lambda a: h.loc[a, "label"], key="maint_unit")
    deg = h.loc[asset, "deg"]
    s = deg["series"]
    if len(s) >= 2:
        fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name="Efficiency index", line=dict(color=c.colours["actual"])))
        if "slope" in deg:
            xs = np.arange(len(s))
            fig.add_trace(go.Scatter(x=s.index, y=deg["intercept"] + deg["slope"] * xs, name="Trend", line=dict(color=c.colours["expected"], dash="dash")))
            fig.add_hline(y=deg["threshold"], line=dict(color="#e34948", dash="dot"), annotation_text="+15% threshold")
            if deg["status"] == "degrading":
                cross = c.health_as_of + pd.Timedelta(days=int(deg["rul_days"]))
                fig.add_vline(x=cross, line=dict(color="#e34948", dash="dot"), annotation_text=f"projected {cross:%d %b}")
        c.style(fig, 300, "kW per °C of outdoor (running, weekdays)", None, margin_top=10)
        fig.update_yaxes(rangemode="normal")
        st.plotly_chart(fig, width="stretch", key="deg_chart")
    st.caption(f"Since the last service ({h.loc[asset, 'last_service']:%d %b %Y}): {deg['status']}. "
               "Real sensor data; the 90 days / 500 run hours / +15% limits are assumptions to confirm with facilities.")
    if st.button("Open asset card", key="maint_card"):
        c.open_asset(asset)

    st.subheader("Maintenance log")
    f1, f2, f3, f4 = st.columns(4)
    assets = sorted(log["asset_id"].unique()) if len(log) else []
    a_f = f1.multiselect("Asset", assets, key="log_asset")
    fl_f = f2.multiselect("Floor", sorted({int(a[1]) for a in assets}), key="log_floor")
    t_f = f3.multiselect("Type", LOG_TYPES, key="log_type")
    dr = f4.date_input("Dates", (pd.Timestamp("2018-07-01").date(), pd.Timestamp("2019-12-31").date()), format="DD/MM/YYYY", key="log_dates")
    view = log.copy()
    if len(view):
        view["date"] = pd.to_datetime(view["date"])
        if a_f:
            view = view[view["asset_id"].isin(a_f)]
        if fl_f:
            view = view[view["asset_id"].str[1].astype(int).isin(fl_f)]
        if t_f:
            view = view[view["type"].isin(t_f)]
        if len(dr) == 2:
            view = view[view["date"].between(pd.Timestamp(dr[0]), pd.Timestamp(dr[1]))]
        fig = go.Figure()
        colours = {"fault repair": "#e5533d", "preventive service": "#1baf7a", "inspection": "#5aa9ff", "replacement": "#e87ba4"}
        for typ, g in view.groupby("type"):
            fig.add_trace(go.Scatter(x=g["date"], y=g["asset_id"], mode="markers", name=typ, text=g["fault_type"].fillna(""),
                                     marker=dict(color=colours.get(typ, "#8a8984"), size=8, symbol=["x" if s_ else "circle" for s_ in g["simulated"].fillna(0)]),
                                     hovertemplate="%{y} · %{text}<br>%{x|%d %b %Y}<extra></extra>"))
        c.style(fig, 420, None, "× = simulated entry", margin_top=10)
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(rangemode="normal", categoryorder="category descending")
        st.plotly_chart(fig, width="stretch", key="log_timeline")
        st.dataframe(view.sort_values("date", ascending=False).assign(Source=lambda x: np.where(x["simulated"].fillna(0).astype(int) == 1, "Simulated", "Real"))
                     [["log_id", "asset_id", "date", "type", "fault_type", "severity", "downtime_h", "cost_THB", "notes", "Source"]],
                     hide_index=True, width="stretch", height=300)
    with st.expander("Add an entry"):
        with st.form("log_form"):
            a, b = st.columns(2)
            asset_new = a.selectbox("Asset", sorted(c.units["asset_id"]))
            date_new = b.date_input("Date", pd.Timestamp(c.health_as_of).date(), format="DD/MM/YYYY")
            a, b, d_ = st.columns(3)
            typ = a.selectbox("Type", LOG_TYPES)
            fault = b.selectbox("Fault type", [""] + FAULT_TYPES)
            sev = d_.selectbox("Severity", ["Low", "Medium", "High"])
            a, b, d_ = st.columns(3)
            down, cost, tech = a.number_input("Downtime (h)", 0.0), b.number_input("Cost (THB)", 0.0), d_.text_input("Technician")
            parts, notes = st.text_input("Parts"), st.text_area("Notes")
            if st.form_submit_button("Save entry"):
                c.store.upsert("maintenance_log", {"log_id": c.store.next_id("maintenance_log"), "asset_id": asset_new, "date": str(date_new),
                                                   "type": typ, "fault_type": fault or None, "severity": sev, "downtime_h": down,
                                                   "cost_THB": cost, "parts": parts, "technician": tech, "notes": notes,
                                                   "source": "manual", "wo_id": None, "simulated": 0}, c.actor)
                st.rerun()
    with st.expander("Import real records (CSV)"):
        st.download_button("Download template", log_import_template(), "maintenance_log_template.csv", "text/csv")
        up = st.file_uploader("CSV file", type="csv", key="log_upload")
        if up is not None and st.button("Import", key="log_import"):
            try:
                n = import_csv(c.store, up.getvalue(), c.actor)
                st.success(f"Imported {n} entries.")
            except Exception as e:  # show a readable message for a bad file
                st.error(str(e))
