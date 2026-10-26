"""Work orders: draft from an incident, schedule into a forecast low-demand window, export (plan §6). Streamlit-free."""
import html as html_lib
import json

import numpy as np
import pandas as pd

from fm_core import CLOSED_DATES, PATTERNS
from fm_incidents import set_status

WO_STATUSES = ["Draft", "Issued", "In progress", "Done"]
DUE_WORKING_DAYS = {"High": 2, "Medium": 5, "Low": 10}
PLAYBOOK = {
    "closed_running": ["Check the timer and BMS schedule", "Check the room booking calendar", "Look for a manual override left on"],
    "off_schedule": ["Check the timer and BMS schedule", "Check the room booking calendar", "Look for a manual override left on"],
    "high_while_open": ["Clean or replace the filter", "Inspect the coil", "Check the setpoint", "Check the thermostat location",
                        "Ask whether occupancy changed"],
    "warm_running": ["Check refrigerant pressure (leak)", "Clean or replace the filter", "Inspect the coil", "Check the fan",
                     "Calibrate the zone sensor"],
    "supply_group_off": ["Check chilled-water supply temperature", "Check AHU status", "Check the electrical feed"],
    "supply_group_high": ["Check chilled-water supply temperature", "Check AHU status", "Check the electrical feed"],
}
PRIORITY_WORD = {"🔴 High": "High", "🟠 Medium": "Medium", "🟢 Low": "Low"}
LOW_SIGNAL = "Low"


def is_working_day(day, open_spec):
    """Open weekday that is not a holiday/closure (when the holiday calendar is on)."""
    days, _, _, use_holidays = open_spec
    day = pd.Timestamp(day).normalize()
    return day.dayofweek in days and not (use_holidays and day in CLOSED_DATES)


def add_working_days(start, n, open_spec):
    """The date n working days after `start` (start itself does not count)."""
    day = pd.Timestamp(start).normalize()
    while n > 0:
        day += pd.Timedelta(days=1)
        if is_working_day(day, open_spec):
            n -= 1
    return day


def next_working_day(after, open_spec):
    return add_working_days(after, 1, open_spec)


def forecast_at(forecast, ts):
    """Forecast kW for an hour: the stored day-ahead value, else the average for that weekday and hour (typical week)."""
    if ts in forecast.index and pd.notna(forecast.loc[ts]):
        return float(forecast.loc[ts])
    typical = forecast.groupby([forecast.index.dayofweek, forecast.index.hour]).mean()
    return float(typical.get((ts.dayofweek, ts.hour), np.nan))


def suggest_slot(forecast, after_date, open_spec, hours=1):
    """The next working day after `after_date` and its lowest-forecast `hours`-long window inside opening hours.
    Returns (start, end, forecast_kW)."""
    day = next_working_day(after_date, open_spec)
    _, open_min, close_min, _ = open_spec
    first = int(np.ceil(open_min / 60))
    last = int(close_min // 60) - hours  # last start hour so the slot ends within opening hours
    candidates = []
    for h in range(first, last + 1):
        ts = day + pd.Timedelta(hours=h)
        kw = np.mean([forecast_at(forecast, ts + pd.Timedelta(hours=i)) for i in range(hours)])
        candidates.append((kw, ts))
    kw, ts = min(candidates, key=lambda c: (np.inf if np.isnan(c[0]) else c[0], c[1]))
    return ts, ts + pd.Timedelta(hours=hours), float(kw)


def symptom_text(inc):
    what = PATTERNS[inc["pattern"]][2]
    return (f"{inc['label']} {what}: {int(inc['occurrences'])} episode(s) between {pd.Timestamp(inc['first_seen']):%d %b} and "
            f"{pd.Timestamp(inc['last_seen']):%d %b %Y}, about {inc['excess_kWh']:,.0f} kWh ({inc['excess_THB']:,.0f} THB) "
            f"above normal; detection confidence {inc['confidence']:.0f}/100.")


def draft_work_order(inc, as_of, open_spec, forecast, active_days, unit_info=None, likely_causes=None):
    """A Draft work order (a dict ready for store.upsert) from an incident row. `as_of` = the date the order is raised
    (the end of the selected timeframe: the dataset ends in 2019)."""
    priority = PRIORITY_WORD[inc["priority_level"]]
    due = add_working_days(as_of, DUE_WORKING_DAYS[priority], open_spec)
    start, end, kw = suggest_slot(forecast, as_of, open_spec)
    if due < start.normalize():
        due = start.normalize()
    per_day = float(inc["excess_THB"]) / max(1, active_days)
    evidence = {"occurrences": int(inc["occurrences"]), "first_seen": str(inc["first_seen"]), "last_seen": str(inc["last_seen"]),
                "latest_episode": f"{pd.Timestamp(inc['latest_start']):%a %d %b %H:%M}–{pd.Timestamp(inc['latest_end']):%H:%M}",
                "excess_kWh": round(float(inc["excess_kWh"]), 1), "excess_THB": round(float(inc["excess_THB"]), 0),
                "confidence": round(float(inc["confidence"]), 0), "triage_score": round(float(inc["triage_score"]), 0),
                "active_days": int(active_days), "cost_of_delay_THB_per_day": round(per_day, 0),
                "slot_forecast_kW": round(kw, 1), "pattern": inc["pattern"], "as_of": str(pd.Timestamp(as_of).date())}
    if unit_info:
        evidence["location"] = {"floor": unit_info.get("floor"), "zone": unit_info.get("zone"),
                                "supply_group": f"{unit_info.get('supply_group')} (assumed)"}
    if likely_causes:
        evidence["likely_cause"] = likely_causes
    return {"wo_id": None, "incident_key": inc["incident_key"], "asset_id": inc["asset_id"],
            "created_at": pd.Timestamp.now().isoformat(timespec="seconds"), "status": "Draft", "assignee": "",
            "priority": priority, "due_date": str(due.date()), "slot_start": start.isoformat(), "slot_end": end.isoformat(),
            "symptom": symptom_text(inc), "checklist_json": [{"item": i, "done": False} for i in PLAYBOOK[inc["pattern"]]],
            "evidence_json": evidence, "simulated": 0}


def open_work_order(store, incident_key):
    wo = store.table("work_orders")
    wo = wo[(wo["incident_key"] == incident_key) & (wo["status"] != "Done")]
    return wo.iloc[0].to_dict() if len(wo) else None


def raise_work_order(store, draft, actor="user"):
    """Save a draft and move the incident to 'Work order raised'. One open work order per incident."""
    existing = open_work_order(store, draft["incident_key"])
    if existing:
        return existing
    draft = {**draft, "wo_id": store.next_id("work_orders")}
    saved = store.upsert("work_orders", draft, actor)
    state = store.get("incident_state", draft["incident_key"])
    old = state["status"] if state else "Open"
    if old in ("Open", "Reopened", "Acknowledged"):
        set_status(store, draft["incident_key"], "Work order raised", f"{saved['wo_id']}", actor)
    return saved


def update_work_order(store, wo_id, actor="user", **changes):
    row = store.get("work_orders", wo_id)
    if row is None:
        raise KeyError(wo_id)
    if "status" in changes and changes["status"] not in WO_STATUSES:
        raise ValueError(changes["status"])
    store.upsert("work_orders", {**row, **changes}, actor,
                 change="updated " + ", ".join(f"{k}={v}" for k, v in changes.items() if k not in ("checklist_json",)))


# ---------------------------------------------------------------- exports
def flatten(wo):
    """One work order as flat text fields for a CMMS import / spreadsheet."""
    checklist = _loads(wo.get("checklist_json")) or []
    evidence = _loads(wo.get("evidence_json")) or {}
    return {"WO ID": wo["wo_id"], "Incident": wo["incident_key"], "Asset": wo["asset_id"], "Status": wo["status"],
            "Priority": wo["priority"], "Assignee": wo.get("assignee") or "", "Due date": wo["due_date"],
            "Slot start": wo["slot_start"], "Slot end": wo["slot_end"], "Symptom": wo["symptom"],
            "Checklist": "; ".join(c["item"] + (" [x]" if c.get("done") else "") for c in checklist),
            "Cost of delay THB/day": evidence.get("cost_of_delay_THB_per_day", ""),
            "Confidence": evidence.get("confidence", ""), "Simulated": bool(wo.get("simulated"))}


def _loads(value):
    if isinstance(value, (dict, list)) or value is None:
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def to_csv(df):
    """UTF-8 with a byte-order mark, so Excel opens Thai text and symbols correctly."""
    flat = pd.DataFrame([flatten(r) for r in df.to_dict("records")])
    return flat.to_csv(index=False).encode("utf-8-sig")


def to_json(df):
    return json.dumps([{**flatten(r), "checklist": _loads(r.get("checklist_json")), "evidence": _loads(r.get("evidence_json"))}
                       for r in df.to_dict("records")], indent=1, ensure_ascii=False, default=str).encode("utf-8")


def printable_html(wo):
    """A printable one-page view of a work order."""
    e = html_lib.escape
    ev = _loads(wo.get("evidence_json")) or {}
    items = "".join(f"<li>&#9744; {e(c['item'])}</li>" for c in (_loads(wo.get("checklist_json")) or []))
    sim = '<p style="color:#b45309"><b>SIMULATED record (demo data)</b></p>' if wo.get("simulated") else ""
    slot = f"{pd.Timestamp(wo['slot_start']):%a %d %b %Y, %H:%M}–{pd.Timestamp(wo['slot_end']):%H:%M}"
    rows = "".join(f"<tr><td>{e(str(k))}</td><td>{e(str(v))}</td></tr>" for k, v in ev.items() if not isinstance(v, (dict, list)))
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{e(wo['wo_id'])}</title><style>"
            "body{font:14px Arial,sans-serif;max-width:760px;margin:24px auto}h1{font-size:20px}td{padding:2px 12px 2px 0}"
            "@media print{body{margin:0}}</style></head><body>"
            f"<h1>Work order {e(wo['wo_id'])} · {e(wo['priority'])} priority</h1>{sim}"
            f"<p><b>Asset:</b> {e(wo['asset_id'])} &nbsp; <b>Status:</b> {e(wo['status'])} &nbsp; "
            f"<b>Assignee:</b> {e(str(wo.get('assignee') or '____________'))}</p>"
            f"<p><b>Due:</b> {e(str(wo['due_date']))} &nbsp; <b>Suggested slot:</b> {e(slot)} (lowest forecast demand)</p>"
            f"<p><b>Symptom:</b> {e(wo['symptom'])}</p><h3>Checklist</h3><ul style='list-style:none;padding:0'>{items}</ul>"
            f"<h3>Evidence</h3><table>{rows}</table>"
            "<p style='margin-top:40px'>Technician: ____________________ &nbsp; Date fixed: ____________</p></body></html>")
