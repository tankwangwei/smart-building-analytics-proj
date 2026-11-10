"""Streamlit rendering of the FM workflow: the Work orders tab (incident queue, work orders, close-out, learning,
verification). Called from app.py with a context object `c` holding the shared state (store, incidents, helpers)."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fm_core import CONFIDENCE_WEIGHTS, PATTERNS, confidence_components
from fm_incidents import STATUS_COLOUR, STATUSES, allowed_next, set_status
from fm_learning import (OUTCOMES, ROOT_CAUSES, add_exception, close_out, deactivate_exception, overall_precision_pct,
                         precision_tables, suggest_exception, threshold_suggestions)
from fm_workorders import (WO_STATUSES, draft_work_order, printable_html, raise_work_order, symptom_text, to_csv,
                           to_json,
                           update_work_order)

SIM_BADGE = '<span style="font-size:.68rem;padding:1px 7px;border-radius:999px;border:1px solid #f0a03c;color:#f0a03c">Simulated</span>'
SHORT_PATTERN = {"closed_running": "Runs while closed", "off_schedule": "Off its schedule",
                 "high_while_open": "High while open", "warm_running": "Zone stays warm",
                 "supply_group_off": "Units off together", "supply_group_high": "Units high together"}
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
def latest_work_orders(c):
    """{incident_key: its most recent work order (dict)}."""
    wos = c.store.table("work_orders")
    if wos.empty:
        return {}
    wos = wos.sort_values("wo_id")
    return {k: g.iloc[-1].to_dict() for k, g in wos.groupby("incident_key")}


def wo_cell(wo):
    if not wo:
        return "—"
    if wo["status"] == "Done":
        return f"{wo['wo_id']} · done"
    return f"{wo['wo_id']} · {wo['status']} · due {pd.Timestamp(wo['due_date']):%d %b}"


def incident_workbench(c):
    """One place to work an incident: the queue on the left; click a row to see its evidence, raise or update its work
    order, close it out and read its history on the right."""
    inc = c.inc
    f1, f2, f3, f4 = st.columns([1.3, 1, .8, 1.1], vertical_alignment="bottom")
    default_status = [s for s in STATUSES if s not in ("Verified", "Dismissed", "Resolved")]
    status_f = f1.multiselect("Status", STATUSES, default_status, key="q_status")
    prio_f = f2.multiselect("Priority", list(PRIORITY_ORDER), list(PRIORITY_ORDER), key="q_prio")
    floors = sorted(inc["floor"].unique()) if len(inc) else []
    floor_f = f3.multiselect("Floor", floors, floors, key="q_floor")
    active_only = f4.checkbox("Only active today" if c.operate else f"Only active in {c.period}", True, key="q_active",
                              help="Active = at least one episode on this day / in this period. An incident's status and "
                                   "history cover the whole Jul–Dec data.")
    view = inc[inc["status"].isin(status_f) & inc["priority_level"].isin(prio_f) & inc["floor"].isin(floor_f)]
    if active_only:
        view = view[view["active"]]
    view = view.sort_values("triage_score", ascending=False)
    if view.empty:
        st.info("No incidents match these filters. Untick 'Only active' to see every open incident.")
        return
    latest = latest_work_orders(c)
    table = pd.DataFrame({
        "": view["priority_level"].str[:1].values, "Incident": view["label"].values,
        "Problem": view["pattern"].map(SHORT_PATTERN).values, "THB": view["period_THB"].values,
        "Conf.": view["confidence"].values, "Status": view["status"].values,
        "Work order": [wo_cell(latest.get(k)) for k in view["incident_key"]]})
    keys = view["incident_key"].tolist()
    left, right = st.columns([1.3, 1], gap="medium")
    with left:
        picked = st.dataframe(
            table, hide_index=True, width="stretch", height=min(35 * len(table) + 38, 560), on_select="rerun",
            selection_mode="single-row", key=f"queue_{st.session_state.get('queue_version', 0)}",
            column_config={
                "": st.column_config.TextColumn(width=28, help="Priority: 🔴 high · 🟠 medium · 🟢 low"),
                "THB": st.column_config.NumberColumn(format="%.0f", width="small",
                                                     help="Excess cost on this day / in this period."),
                "Conf.": st.column_config.NumberColumn(format="%.0f", width="small",
                                                       help="Confidence 0-100. The Evidence tab shows the breakdown."),
                "Status": st.column_config.TextColumn(width="small"),
                "Work order": st.column_config.TextColumn(help="The incident's latest work order, if any.")})
        st.caption("Click a row to open it. Sorted by triage score (excess THB × confidence ÷ effort).")
    rows = picked.selection.rows if picked is not None and hasattr(picked, "selection") else []
    if rows:
        st.session_state["sel_incident"] = keys[rows[0]]
    key = st.session_state.get("sel_incident")
    if key not in keys:
        key = keys[0]
    with right, st.container(key="card-incident"):
        incident_panel(c, inc.set_index("incident_key", drop=False).loc[key], latest.get(key))


def confidence_breakdown(c, key):
    """The five confidence checks for an incident, averaged over its episodes weighted by excess THB."""
    ev = c.events[c.events["incident_key"] == key]
    need = {"statistical", "after_hours_h", "persistent", "duration_h", "unit", "start", "hot_day", "quality_share"}
    if ev.empty or not need <= set(ev.columns):
        return None
    comp = confidence_components(ev, ev)
    w = ev["excess_THB"].clip(lower=1e-9)
    avg = comp.mul(w, axis=0).sum() / w.sum()
    names = {"rules": "Flag rules that fired (of 3)", "duration": "Duration (2 h+ = full)",
             "recurrence": "Recurs at the same weekday and hour", "not_weather": "Not on a hot day (open-hours flags only)",
             "quality": "Complete readings"}
    return pd.DataFrame({"Check": [names[k] for k in CONFIDENCE_WEIGHTS], "Weight": list(CONFIDENCE_WEIGHTS.values()),
                         "Score": [avg[k] for k in CONFIDENCE_WEIGHTS],
                         "Points": [100 * w_ * avg[k] for k, w_ in CONFIDENCE_WEIGHTS.items()]})


def incident_panel(c, row, wo):
    key = row["incident_key"]
    st.markdown(f'<div class="card-h">{row["label"]} · {pattern_text(row["pattern"])}</div>'
                f'{status_chip(row["status"])} <span style="font-size:.8rem;color:#9fb0cc">{row["priority_level"]} · '
                f'effort {PATTERNS[row["pattern"]][0].lower()} · first seen {pd.Timestamp(row["first_seen"]):%d %b}, '
                f'last {pd.Timestamp(row["last_seen"]):%d %b %Y}</span>', unsafe_allow_html=True)
    t_ev, t_wo, t_hist = st.tabs(["Evidence", "Work order" + (f" · {wo['wo_id']}" if wo else ""), "History"])
    with t_ev:
        st.write(symptom_text(row))
        m1, m2, m3 = st.columns(3)
        m1.metric("Episodes (all data)", int(row["occurrences"]))
        m2.metric("Excess THB (all data)", f"{row['excess_THB']:,.0f}")
        m3.metric("Confidence", f"{row['confidence']:.0f}")
        bd = confidence_breakdown(c, key)
        if bd is not None:
            base = bd["Points"].sum()
            st.dataframe(bd, hide_index=True, width="stretch",
                         column_config={"Weight": st.column_config.NumberColumn(format="%.2f", width="small"),
                                        "Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f",
                                                                                 width="small"),
                                        "Points": st.column_config.NumberColumn(format="%.0f")})
            st.caption(f"Confidence = Σ weight × score × 100 = {base:.0f}"
                       + (f"; adjusted to {row['confidence']:.0f} by the close-out track record "
                          "(0.8 × base + 0.2 × confirmed rate)" if abs(base - row["confidence"]) >= 1 else "")
                       + ". The weights are our judgement, not fitted: there are no real labelled faults to fit them on.")
        if c.chart_for_incident:
            c.chart_for_incident(key)
        if row["asset_id"] in c.col_of and st.button("Open asset card", key=f"inc_card_{key}"):
            c.open_asset(row["asset_id"])
    with t_wo:
        if wo and wo["status"] != "Done":
            wo_detail(c, wo)
            closeout_form(c, wo)
        else:
            if wo:  # the last order is done: show how it was closed
                co = c.store.get("closeouts", wo["wo_id"])
                if co:
                    st.markdown(f"Last work order **{wo['wo_id']}** closed: **{co['outcome']}** · {co['root_cause']} · "
                                f"fixed {co['date_fixed'] if pd.notna(co['date_fixed']) else '–'} "
                                + (SIM_BADGE if co.get("simulated") else ""), unsafe_allow_html=True)
            if row["status"] in ("Open", "Acknowledged", "Reopened"):
                st.caption("Raising a work order drafts the symptom, evidence, checklist, due date and a time slot "
                           "(the quietest forecast hour of the next working day).")
                if st.button("Raise work order", type="primary", key=f"raise_{key}"):
                    raise_wo(c, row)
                    _bump("queue_version")
                    st.rerun()
            elif not wo:
                st.caption(f"No work order: the incident is {row['status'].lower()}.")
        status_control(c, row)
    with t_hist:
        history(c, key)


def status_control(c, row):
    key = row["incident_key"]
    with st.expander("Change incident status"):
        options = allowed_next(row["status"])
        a, b = st.columns([1, 2], vertical_alignment="bottom")
        new = a.selectbox("Status", options, options.index(row["status"]), key=f"st_{key}")
        reason = b.text_input("Reason" + (" (required)" if new == "Dismissed" else ""), key=f"st_reason_{key}")
        if st.button("Save status", key=f"st_save_{key}", disabled=new == row["status"]):
            try:
                set_status(c.store, key, new, reason, c.actor)
                _bump("queue_version")
                st.rerun()
            except ValueError as e:
                st.error(str(e))


def history(c, key):
    wos = c.store.table("work_orders")
    wo_ids = set(wos.loc[wos["incident_key"] == key, "wo_id"]) if len(wos) else set()
    log = c.store.table("audit_log")
    log = log[log["entity_id"].astype(str).isin({key, *wo_ids})] if len(log) else log
    if log.empty:
        st.caption("No changes recorded yet.")
        return
    log = log.sort_values("ts", ascending=False)
    st.dataframe(pd.DataFrame({"When": log["ts"].astype(str).str.replace("T", " ").str[:16], "Who": log["actor"],
                               "What": log["entity"].str.replace("_", " ") + " " + log["entity_id"].astype(str),
                               "Change": log["change"]}),
                 hide_index=True, width="stretch", height=min(35 * len(log) + 38, 320))


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
def wo_detail(c, wo):
    """The open work order: slot, likely cause, checklist, status and assignee."""
    pick = wo["wo_id"]
    sim = bool(wo.get("simulated"))
    st.markdown(f"**{pick}** · {wo['priority']} priority · {wo['status']} " + (SIM_BADGE if sim else ""), unsafe_allow_html=True)
    ev = _loads(wo["evidence_json"]) or {}
    slot = f"{pd.Timestamp(wo['slot_start']):%A %d %b, %H:%M}–{pd.Timestamp(wo['slot_end']):%H:%M}"
    st.caption(f"Due {wo['due_date']} · slot {slot} · delay ≈ {ev.get('cost_of_delay_THB_per_day', '–')} THB/day",
               help=f"Suggested slot = the lowest forecast demand in opening hours (about {ev.get('slot_forecast_kW', '–')} kW), "
                    "so an inspection disturbs occupants least. Due: 2 / 5 / 10 working days for High / Medium / Low. "
                    "Delay = excess cost per day the fix waits.")
    likely = ev.get("likely_cause")
    if likely:
        st.caption("Likely cause: " + ", ".join(f"{k} {v:.0%}" for k, v in likely.items()),
                   help="Demo: counted from the simulated maintenance history.")
    st.markdown("**Checklist**")
    checklist = _loads(wo["checklist_json"]) or []
    new_list = []
    for i, item in enumerate(checklist):
        done = st.checkbox(item["item"], bool(item.get("done")), key=f"chk_{pick}_{i}")
        new_list.append({"item": item["item"], "done": done})
    if new_list != checklist:
        update_work_order(c.store, pick, c.actor, checklist_json=json.dumps(new_list))
    s1, s2, s3 = st.columns([1.2, 2, .8], vertical_alignment="bottom")
    status = s1.selectbox("Work order status", WO_STATUSES[:3], WO_STATUSES[:3].index(wo["status"]), key=f"wo_status_{pick}",
                          help="Draft → Issued (sent to a technician) → In progress. Close it out below when the job is done.")
    assignee = s2.text_input("Assignee", wo.get("assignee") or "", key=f"wo_assignee_{pick}")
    if s3.button("Save", key=f"wo_save_{pick}"):
        update_work_order(c.store, pick, c.actor, status=status, assignee=assignee)
        _bump("queue_version")
        st.rerun()
    st.download_button("Printable one-page view (HTML)", printable_html(wo), f"{pick}.html", "text/html", key=f"print_{pick}")


def all_work_orders(c):
    wos = c.store.table("work_orders")
    if wos.empty:
        st.caption("No work orders yet.")
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
    st.markdown("**Confirm an exception**", help="Legitimate use: once confirmed, these intervals are no longer flagged.")
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
        st.markdown(SIM_BADGE + " Demo close-outs", unsafe_allow_html=True,
                    help="The close-outs behind these figures are demo data, not real technician findings.")
    tables = precision_tables(c.store, c.events)
    with st.expander("Precision by rule, pattern and unit"):
        st.caption("How precision is used",
                   help="Precision = confirmed ÷ (confirmed + no fault). The confidence of new flags blends the base score "
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
        st.markdown("**Exceptions**", help="Matching intervals are never flagged.")
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
    with st.expander("How verified savings are calculated"):
        st.markdown(
            "Each fixed incident is checked against its own meter data, following the IPMVP *adjusted baseline* idea:\n\n"
            "1. **Before:** the incident's daily measure (e.g. kWh used while the building is closed) over the **28 days "
            "before the fix date**.\n"
            "2. **Expected without the fix:** a small model fitted on those 28 days predicts each day after the fix: the "
            "pre-fix average for that **day type** (open day, or weekend / public holiday). For faults that happen while "
            "the building is open, it is also moved by the **daytime heat index**, so a hotter month after the fix does not "
            "hide a saving.\n"
            "3. **Measured after:** the same measure over **up to 28 days after** the fix (fewer if the data ends; under "
            "14 days is *Inconclusive*).\n"
            "4. **Verdict:** *Verified* if measured is at least **50 % below expected**, the problem did not come back in "
            "the last 14 days and comfort did not drop by more than 10 points. Recurrence → *Reopened*.\n"
            "5. **Saving** = Σ expected − Σ measured over the days after (kWh), × tariff for THB. Faults measured in hours "
            "(zone too warm, units off together) can be verified but add 0 kWh.\n\n"
            f"Counted here: fixes **dated in {c.period}** on **{c.scope_name}**. A fix's 28-day window can run past the "
            "timeframe; its saving is credited to the fix date. **The demo's fixes are simulated** (placed at real drops "
            "in the data); the before and after readings are real. Not adjusted for occupancy, so treat it as evidence a "
            "fix worked, not a contractual saving.")
    if not c.verif:
        st.info(f"No fixes verified in this period ({c.period}, {c.scope_name}). Pick a longer timeframe, e.g. Jul–Dec.")
        return
    verified = {k: v for k, v in c.verif.items() if v["verdict"] == "Verified"}
    kwh = sum(v["saved_kWh"] for v in verified.values())
    k1, k2, k3 = st.columns(3)
    k1.metric("Verified savings", f"{kwh:,.0f} kWh" if verified else "None",
              f"≈ {kwh * c.tariff:,.0f} THB" if verified else None, delta_color="off",
              help=f"Expected without the fix − measured, summed over the verified fixes dated in {c.period} on "
                   f"{c.scope_name}. Separate from the estimates elsewhere in the dashboard.")
    k2.metric("Verified", len(verified))
    k3.metric("Reopened", sum(v["verdict"] == "Reopened" for v in c.verif.values()))
    colours = {"Verified": "#1baf7a", "Reopened": "#e87ba4", "Inconclusive": "#8a8984", "Not verified": "#f0a03c"}
    for key, v in sorted(c.verif.items(), key=lambda kv: kv[1]["fix_date"]):
        label = c.label_of.get(key.split("|")[0], key)
        with st.expander(f"{label} · {key.split('|')[1]} · fixed {v['fix_date']:%d %b %Y} · {v['verdict']}"):
            sim = " " + SIM_BADGE if v["simulated"] else ""
            st.markdown(f'<span style="color:{colours[v["verdict"]]};font-weight:600">{v["verdict"]}</span>{sim} · '
                        f'{v["metric"]}: {v["pre_per_day"]:.1f} before · expected without the fix '
                        f'{v["baseline_per_day"]:.1f}{" (heat-index adjusted)" if v["weather_adjusted"] else ""} → '
                        f'{v["post_per_day"]:.1f} measured ({v["drop_pct"]:.0f}% below expected) · '
                        f'{v["post_days"]} days after · {v["events_last_14d"]} recurrence(s) in the '
                        f'last 14 days' + (f' · saved {v["saved_kWh"]:,.0f} kWh ≈ {v["saved_THB"]:,.0f} THB' if v["saved_kWh"] else ""),
                        unsafe_allow_html=True)
            if v["saved_kWh"]:
                exp_total, meas_total = float(v["baseline_series"].sum()), float(v["post_series"].sum())
                st.caption(f"Saving = expected {exp_total:,.0f} kWh − measured {meas_total:,.0f} kWh over {v['post_days']} days "
                           f"= {v['saved_kWh']:,.0f} kWh × {c.tariff} THB/kWh ≈ {v['saved_THB']:,.0f} THB")
            if v["notes"]:
                st.caption(v["notes"])
            b1, b2 = st.columns([1, 2])
            fig = go.Figure(go.Bar(x=["Before (28 d)", "Expected without fix", "Measured after"],
                                   y=[v["pre_per_day"], v["baseline_per_day"], v["post_per_day"]],
                                   marker_color=[c.colours["muted"], c.colours["expected"], colours[v["verdict"]]],
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
    when = "today" if c.operate else c.period
    if c.operate:
        c.page_note('Work the queue: open an incident, raise its work order, close it out. Fix verification and learning:',
                    "Review", more="Simulated = seeded demo work orders and close-outs. Verification of fixes and "
                                   "learning from close-outs look back over a period, so they live in Review.")
        k = st.columns(3)
        k[0].metric("Open incidents active today", len(open_inc), help="At least one episode today, not yet resolved or dismissed.")
        k[1].metric("Excess THB at stake today", f"{open_inc['period_THB'].sum():,.0f}", help="Open incidents, today's episodes.")
        k[2].metric("Work orders open", int((wos["status"] != "Done").sum()) if len(wos) else 0,
                    help="All open work orders, whatever day they were raised.")
        st.subheader("Incidents and work orders",
                     help="One row per asset and problem (for example, 'runs while the building is closed'), not one per "
                          "episode. Its lifecycle: Open → Acknowledged → Work order raised → Resolved → Verified (or "
                          "Reopened / Dismissed). Every change is saved and logged in History.")
        incident_workbench(c)
        exception_form(c)
        with st.expander("All work orders · export"):
            all_work_orders(c)
        return
    c.page_note(f"Look back over {c.period}: did fixes work, and how reliable are the flags? To work the incident queue,",
                "Operate")
    k = st.columns(3)
    if c.n_verified:
        k[1].metric("Verified savings", f"{c.verified_kwh:,.0f} kWh", f"≈ {c.verified_kwh * c.tariff:,.0f} THB", delta_color="off",
                    help=f"Fixes dated in {when} on {c.scope_name}. See Verification below for how it is calculated.")
    else:
        k[1].metric("Verified savings", "None", help="No fixes verified in this period.")
    k[0].metric("Fixes dated in the period", len(c.verif), help="Close-outs with a fix date in the period, any verdict.")
    k[2].metric("Incidents active in the period", int(c.inc["active"].sum()))
    st.subheader("Verification of fixes")
    verification(c)
    st.subheader("Learning from close-outs")
    learning(c)
    with st.expander("All work orders · export"):
        all_work_orders(c)
