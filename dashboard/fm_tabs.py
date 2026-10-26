"""Streamlit rendering of the FM workflow: the Work orders tab (incident queue, work orders, close-out, learning,
verification). Called from app.py with a context object `c` holding the shared state (store, incidents, helpers)."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fm_core import PATTERNS
from fm_incidents import STATUS_COLOUR, STATUSES, allowed_next, set_status
from fm_learning import (OUTCOMES, ROOT_CAUSES, add_exception, close_out, deactivate_exception, overall_precision_pct,
                         precision_tables, suggest_exception, threshold_suggestions)
from fm_workorders import (WO_STATUSES, draft_work_order, printable_html, raise_work_order, to_csv, to_json,
                           update_work_order)

SIM_BADGE = '<span style="font-size:.68rem;padding:1px 7px;border-radius:999px;border:1px solid #f0a03c;color:#f0a03c">Simulated</span>'
ACTIONS = ["—", "Raise work order", "Inspect unit"]
PRIORITY_ORDER = {"🔴 High": 0, "🟠 Medium": 1, "🟢 Low": 2}


def status_chip(status):
    return (f'<span style="font-size:.68rem;padding:1px 8px;border-radius:999px;border:1px solid {STATUS_COLOUR[status]};'
            f'color:{STATUS_COLOUR[status]}">{status}</span>')


def pattern_text(p):
    return PATTERNS[p][2]


def _bump(name):
    st.session_state[name] = st.session_state.get(name, 0) + 1


def _loads(v):
    if isinstance(v, (dict, list)) or v is None:
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return None


# ================================================================ queue
def incident_queue(c):
    inc = c.inc
    f1, f2, f3, f4 = st.columns(4)
    default_status = [s for s in STATUSES if s not in ("Verified", "Dismissed", "Resolved")]
    status_f = f1.multiselect("Status", STATUSES, default_status, key="q_status")
    prio_f = f2.multiselect("Priority", list(PRIORITY_ORDER), list(PRIORITY_ORDER), key="q_prio")
    band_f = f3.multiselect("Confidence", ["High", "Medium", "Low"], ["High", "Medium", "Low"], key="q_band",
                            help="High ≥ 70 · Medium 40–69 · Low < 40")
    floors = sorted(inc["floor"].unique()) if len(inc) else []
    floor_f = f4.multiselect("Floor", floors, floors, key="q_floor")
    active_only = st.checkbox(f"Only incidents active in {c.period}", True, key="q_active",
                              help="An incident is active when at least one of its episodes falls in the selected timeframe. "
                                   "Its status and history cover the whole Jul–Dec data.")
    band = inc["confidence"].map(lambda v: "High" if v >= 70 else "Medium" if v >= 40 else "Low")
    view = inc[inc["status"].isin(status_f) & inc["priority_level"].isin(prio_f) & band.isin(band_f) & inc["floor"].isin(floor_f)]
    if active_only:
        view = view[view["active"]]
    view = view.sort_values("triage_score", ascending=False)
    if view.empty:
        st.info("No incidents match these filters in this timeframe.")
        return
    table = pd.DataFrame({
        "Priority": view["priority_level"].values, "Incident": view["label"].values,
        "What": view["pattern"].map(pattern_text).values, "Episodes": view["period_occurrences"].astype(int).values,
        "Excess THB": view["period_THB"].values, "Confidence": view["confidence"].values,
        "Effort": view["pattern"].map(lambda p: PATTERNS[p][0]).values, "Status": view["status"].values,
        "Action": "—"}, index=view["incident_key"].values)
    ver = st.session_state.get("queue_version", 0)
    edited = st.data_editor(
        table, key=f"queue_{ver}", width="stretch", hide_index=True, height=min(35 * len(table) + 38, 420),
        disabled=["Priority", "Incident", "What", "Episodes", "Excess THB", "Confidence", "Effort"],
        column_config={
            "Status": st.column_config.SelectboxColumn("Status", options=STATUSES, required=True,
                                                       help="Change it here: the change is saved and logged. "
                                                            "Dismissing asks for a reason."),
            "Action": st.column_config.SelectboxColumn("Action", options=ACTIONS, required=True),
            "Excess THB": st.column_config.NumberColumn(format="%.0f", help="In the selected timeframe."),
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Episodes": st.column_config.NumberColumn(help="Episodes in the selected timeframe.")})
    changed = False
    for key in edited.index:
        old, new = table.loc[key, "Status"], edited.loc[key, "Status"]
        act = edited.loc[key, "Action"]
        if new != old:
            if new == "Dismissed":
                st.session_state["pending_dismiss"] = key
            else:
                try:
                    set_status(c.store, key, new, actor=c.actor)
                    changed = True
                except ValueError as e:
                    st.warning(f"{c.inc.set_index('incident_key').loc[key, 'label']}: {e}. Allowed next: "
                               + ", ".join(allowed_next(old)))
                    changed = True
        if act == "Raise work order":
            row = c.inc.set_index("incident_key", drop=False).loc[key]
            raise_wo(c, row)
            changed = True
        elif act == "Inspect unit":
            st.session_state["asset_card"] = row_asset(c, key)
    if changed:
        _bump("queue_version")
        st.rerun()
    pending = st.session_state.get("pending_dismiss")
    if pending is not None:
        label = c.inc.set_index("incident_key").loc[pending, "label"]
        with st.form("dismiss_form"):
            reason = st.text_input(f"Reason for dismissing {label}", key="dismiss_reason")
            ok, cancel = st.columns(2)
            go_, no_ = ok.form_submit_button("Dismiss"), cancel.form_submit_button("Cancel")
        if go_:
            try:
                set_status(c.store, pending, "Dismissed", reason, c.actor)
                st.session_state.pop("pending_dismiss")
                _bump("queue_version")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        elif no_:
            st.session_state.pop("pending_dismiss")
            _bump("queue_version")
            st.rerun()
    if st.session_state.get("asset_card"):
        c.open_asset(st.session_state.pop("asset_card"))


def row_asset(c, key):
    return c.inc_all.set_index("incident_key").loc[key, "asset_id"]


def raise_wo(c, row):
    floor = int(row["floor"])
    n_days = max(1, c.events[c.events["incident_key"] == row["incident_key"]]["start"].dt.normalize().nunique())
    unit = c.brick.asset_for_column(c.col_of.get(row["asset_id"], "")) if row["asset_id"] in c.col_of else None
    causes = c.likely_causes(row["pattern"], row["asset_id"]) if c.likely_causes else None
    draft = draft_work_order(row, c.as_of, c.open_spec, c.forecasts[floor], n_days, unit, causes)
    wo = raise_work_order(c.store, draft, c.actor)
    st.toast(f"Work order {wo['wo_id']} raised for {row['label']}")


# ================================================================ work orders
def work_orders(c):
    wos = c.store.table("work_orders")
    if wos.empty:
        st.caption("No work orders yet. Pick an incident above and choose the action **Raise work order**.")
        return
    wos = wos.sort_values("wo_id", ascending=False)
    show = pd.DataFrame({"Work order": wos["wo_id"], "Asset": wos["asset_id"], "Priority": wos["priority"], "Status": wos["status"],
                         "Assignee": wos["assignee"].fillna(""), "Due": wos["due_date"],
                         "Slot": [f"{pd.Timestamp(a):%a %d %b %H:%M}–{pd.Timestamp(b):%H:%M}"
                                  for a, b in zip(wos["slot_start"], wos["slot_end"])],
                         "Source": ["Simulated" if s else "Real" for s in wos["simulated"].fillna(0)]})
    st.dataframe(show, hide_index=True, width="stretch", height=min(35 * len(show) + 38, 300))
    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button("Export CSV (Excel)", to_csv(wos), "work_orders.csv", "text/csv")
    d2.download_button("Export JSON", to_json(wos), "work_orders.json", "application/json")

    pick = st.selectbox("Open a work order", wos["wo_id"].tolist(), key="wo_pick")
    wo = c.store.get("work_orders", pick)
    sim = bool(wo.get("simulated"))
    st.markdown(f"**{pick}** · {wo['asset_id']} · {wo['priority']} priority "
                + (SIM_BADGE if sim else ""), unsafe_allow_html=True)
    ev = _loads(wo["evidence_json"]) or {}
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.write(wo["symptom"])
        slot = f"{pd.Timestamp(wo['slot_start']):%A %d %b, %H:%M}–{pd.Timestamp(wo['slot_end']):%H:%M}"
        st.caption(f"Due {wo['due_date']} · suggested slot {slot} (lowest forecast demand in opening hours, "
                   f"about {ev.get('slot_forecast_kW', '–')} kW) · cost of delay ≈ {ev.get('cost_of_delay_THB_per_day', '–')} THB/day")
        likely = ev.get("likely_cause")
        if likely:
            st.caption("Likely cause (demo: trained on simulated maintenance history): "
                       + ", ".join(f"{k} {v:.0%}" for k, v in likely.items()))
        checklist = _loads(wo["checklist_json"]) or []
        new_list = []
        for i, item in enumerate(checklist):
            done = st.checkbox(item["item"], bool(item.get("done")), key=f"chk_{pick}_{i}", disabled=wo["status"] == "Done")
            new_list.append({"item": item["item"], "done": done})
        if new_list != checklist and wo["status"] != "Done":
            update_work_order(c.store, pick, c.actor, checklist_json=json.dumps(new_list))
        if wo["status"] != "Done":
            s1, s2, s3 = st.columns([1, 2, 1], vertical_alignment="bottom")
            status = s1.selectbox("Status", WO_STATUSES[:3], WO_STATUSES[:3].index(wo["status"]), key=f"wo_status_{pick}")
            assignee = s2.text_input("Assignee", wo.get("assignee") or "", key=f"wo_assignee_{pick}")
            if s3.button("Save", key=f"wo_save_{pick}"):
                update_work_order(c.store, pick, c.actor, status=status, assignee=assignee)
                st.rerun()
        st.download_button("Printable one-page view (HTML)", printable_html(wo), f"{pick}.html", "text/html", key=f"print_{pick}")
    with right:
        if c.chart_for_incident:
            c.chart_for_incident(wo["incident_key"])
    if wo["status"] != "Done":
        closeout_form(c, wo)
    else:
        co = c.store.get("closeouts", pick)
        if co:
            st.markdown(f"Closed: **{co['outcome']}** · {co['root_cause']} · fixed {co['date_fixed'] if pd.notna(co['date_fixed']) else '–'} · {co['technician'] or ''} "
                        + (SIM_BADGE if co.get("simulated") else ""), unsafe_allow_html=True)


def closeout_form(c, wo):
    pattern = wo["incident_key"].split("|")[1]
    with st.expander("Close out this work order"):
        with st.form(f"closeout_{wo['wo_id']}"):
            a, b = st.columns(2)
            outcome = a.selectbox("Outcome", OUTCOMES)
            cause = b.selectbox("Root cause", ROOT_CAUSES[pattern])
            action = st.text_input("Action taken")
            d1, d2 = st.columns(2)
            fixed = d1.date_input("Date fixed", pd.Timestamp(c.as_of).date(), format="DD/MM/YYYY",
                                  help="Used to verify the fix: 28 days before and after.")
            tech = d2.text_input("Technician")
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("Close out")
        if submitted:
            close_out(c.store, wo["wo_id"], outcome, cause, action, fixed if outcome == "Fault confirmed" else None, tech, notes, c.actor)
            if outcome == "Legitimate use":
                ev = c.events[c.events["incident_key"] == wo["incident_key"]]
                if len(ev):
                    st.session_state["pending_exception"] = suggest_exception(c.store.get("work_orders", wo["wo_id"]), ev)
            _bump("queue_version")
            st.rerun()


def exception_form(c):
    sug = st.session_state.get("pending_exception")
    if not sug:
        return
    st.markdown("**Legitimate use: confirm an exception** so these intervals are no longer flagged.")
    with st.form("exception_form"):
        a, b, d_ = st.columns(3)
        days = a.multiselect("Weekdays", range(7), [int(x) for x in sug["weekdays"].split(",") if x != ""],
                             format_func=lambda i: c.days[i])
        t_from = b.text_input("From (HH:MM)", sug["time_from"])
        t_to = d_.text_input("To (HH:MM)", sug["time_to"])
        e1, e2, e3 = st.columns(3)
        d_from = e1.date_input("From date", pd.Timestamp(sug["date_from"]).date(), format="DD/MM/YYYY")
        d_to = e2.date_input("To date", pd.Timestamp(sug["date_to"]).date(), format="DD/MM/YYYY")
        reason = e3.text_input("Reason", sug["reason"])
        ok, skip = st.columns(2)
        confirm, cancel = ok.form_submit_button("Confirm exception"), skip.form_submit_button("Skip")
    if confirm:
        add_exception(c.store, {**sug, "weekdays": ",".join(map(str, days)), "time_from": t_from, "time_to": t_to,
                                "date_from": str(d_from), "date_to": str(d_to), "reason": reason}, c.actor)
        st.session_state.pop("pending_exception")
        st.rerun()
    elif cancel:
        st.session_state.pop("pending_exception")
        st.rerun()


# ================================================================ learning
def learning(c):
    pct, confirmed, no_fault = overall_precision_pct(c.store)
    k1, k2, k3 = st.columns(3)
    k1.metric("Flags verified as real issues", f"{pct:.0f}%" if pd.notna(pct) else "–",
              help="Fault confirmed ÷ (fault confirmed + no fault found), from close-outs. 'Legitimate use' is counted "
                   "separately as expected use, not as a detection error.")
    k2.metric("Close-outs", confirmed + no_fault + int((c.store.table("closeouts")["outcome"] == "Legitimate use").sum()))
    sim_n = int(c.store.table("closeouts")["simulated"].fillna(0).sum())
    k3.metric("of which simulated", sim_n, help="Demo close-outs seeded on real incidents. Replace them with real ones.")
    if sim_n:
        st.markdown(SIM_BADGE + " The close-outs behind these figures are demo data, not real technician findings.",
                    unsafe_allow_html=True)
    tables = precision_tables(c.store, c.events)
    with st.expander("Precision by rule, pattern and unit"):
        st.caption("Precision = confirmed ÷ (confirmed + no fault). The confidence of new flags blends the base score "
                   "(80%) with this record, smoothed so one or two close-outs cannot swing it: "
                   "final = 0.8 × base + 0.2 × 100 × (confirmed + 1) ÷ (confirmed + no fault + 2).")
        for name in ["rule", "pattern", "unit"]:
            t = tables[name].reset_index()
            if len(t):
                t["precision"] = t["precision"] * 100
                st.markdown(f"**By {name}**")
                st.dataframe(t, hide_index=True, width="stretch",
                             column_config={"precision": st.column_config.NumberColumn("Precision %", format="%.0f")})
    for text in threshold_suggestions(c.store, c.k_iqr):
        st.info(text)
    exc = c.store.table("exceptions")
    if len(exc):
        st.markdown("**Exceptions** (matching intervals are never flagged)")
        for r in exc.itertuples():
            cols = st.columns([4, 1])
            active = str(r.active) in ("1", "True", "true")
            days = ",".join(c.days[int(x)] for x in str(r.weekdays).split(",") if x != "") if r.weekdays else "all days"
            cols[0].markdown(f"{'✅' if active else '⏸'} **{r.exc_id}** · {r.asset_id} · {days} {r.time_from}–{r.time_to} · "
                             f"{r.date_from} → {r.date_to} · {r.reason} " + (SIM_BADGE if r.simulated else ""),
                             unsafe_allow_html=True)
            if active and cols[1].button("Deactivate", key=f"deact_{r.exc_id}"):
                deactivate_exception(c.store, r.exc_id, c.actor)
                st.rerun()


# ================================================================ verification
def verification(c):
    if not c.verif:
        st.caption("Verification appears once an incident has a fixed close-out.")
        return
    verified = {k: v for k, v in c.verif.items() if v["verdict"] == "Verified"}
    kwh = sum(v["saved_kWh"] for v in verified.values())
    k1, k2, k3 = st.columns(3)
    k1.metric("Verified savings", f"{kwh:,.0f} kWh", f"≈ {kwh * c.tariff:,.0f} THB", delta_color="off",
              help="Measured before vs after on the same data: the incident's metric over 28 days before the fix date "
                   "against up to 28 days after. Separate from the estimates elsewhere in the dashboard.")
    k2.metric("Verified", len(verified))
    k3.metric("Reopened", sum(v["verdict"] == "Reopened" for v in c.verif.values()))
    colours = {"Verified": "#1baf7a", "Reopened": "#e87ba4", "Inconclusive": "#8a8984", "Not verified": "#f0a03c"}
    for key, v in sorted(c.verif.items(), key=lambda kv: kv[1]["fix_date"]):
        label = c.label_of.get(key.split("|")[0], key)
        with st.expander(f"{label} · {key.split('|')[1]} · fixed {v['fix_date']:%d %b %Y} · {v['verdict']}"):
            sim = " " + SIM_BADGE if v["simulated"] else ""
            st.markdown(f'<span style="color:{colours[v["verdict"]]};font-weight:600">{v["verdict"]}</span>{sim} · '
                        f'{v["metric"]}: {v["pre_per_day"]:.1f} before → {v["post_per_day"]:.1f} after '
                        f'({v["drop_pct"]:.0f}% change) · {v["post_days"]} days after · {v["events_last_14d"]} recurrence(s) in the '
                        f'last 14 days' + (f' · saved {v["saved_kWh"]:,.0f} kWh ≈ {v["saved_THB"]:,.0f} THB' if v["saved_kWh"] else ""),
                        unsafe_allow_html=True)
            if v["notes"]:
                st.caption(v["notes"])
            b1, b2 = st.columns([1, 2])
            fig = go.Figure(go.Bar(x=["Before (28 d)", "After"], y=[v["pre_per_day"], v["post_per_day"]],
                                   marker_color=[c.colours["muted"], colours[v["verdict"]]],
                                   hovertemplate="%{x}: %{y:.1f} per day<extra></extra>"))
            c.style(fig, 240, "per day", None, margin_top=10)
            fig.update_layout(hovermode="closest")
            b1.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"ba_{key}")
            series = pd.concat([v["pre_series"], v["post_series"]]).resample("W-MON", label="left", closed="left").sum()
            fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines+markers", line=dict(color=c.colours["actual"])))
            fig.add_vline(x=v["fix_date"], line=dict(color=c.colours["expected"], dash="dash"), annotation_text="fix date")
            c.style(fig, 240, "weekly total", None, margin_top=10)
            fig.update_layout(hovermode="x unified")
            b2.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"wk_{key}")


# ================================================================ the tab
def render(c):
    open_inc = c.inc[c.inc["status"].isin(["Open", "Acknowledged", "Work order raised", "Reopened"]) & c.inc["active"]]
    wos = c.store.table("work_orders")
    pct, confirmed, no_fault = overall_precision_pct(c.store)
    k = st.columns(5)
    k[0].metric("Open incidents", len(open_inc), help=f"Active in {c.period}, not yet resolved or dismissed.")
    k[1].metric("Excess THB at stake", f"{open_inc['period_THB'].sum():,.0f}", help=f"Open incidents, {c.period}.")
    k[2].metric("Work orders open", int((wos["status"] != "Done").sum()) if len(wos) else 0)
    k[3].metric("Flags verified as real issues", f"{pct:.0f}%" if pd.notna(pct) else "–",
                help="From close-outs: fault confirmed ÷ (fault confirmed + no fault found).")
    k[4].metric("Verified savings", f"{c.verified_kwh:,.0f} kWh", f"≈ {c.verified_kwh * c.tariff:,.0f} THB", delta_color="off")
    st.subheader("Incident queue", help="One row per asset and pattern (for example, 'runs while the building is closed'), "
                                        "not one per episode. Sorted by triage score. Change a status or pick an action in the table.")
    incident_queue(c)
    exception_form(c)
    st.subheader("Work orders")
    work_orders(c)
    st.subheader("Learning from close-outs")
    learning(c)
    st.subheader("Verification of fixes")
    verification(c)
