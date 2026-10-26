"""Close-out and the learning loop (plan §7): precision, track record, exceptions, threshold hints, demo seed.
Streamlit-free."""
import json

import numpy as np
import pandas as pd

from fm_incidents import set_status
from fm_verify import find_natural_changes, verify_incident
from fm_workorders import draft_work_order, raise_work_order, update_work_order

OUTCOMES = ["Fault confirmed", "No fault found", "Legitimate use"]
ROOT_CAUSES = {
    "closed_running": ["timer schedule left on", "manual override not reset", "booking/event", "BMS schedule error", "other"],
    "off_schedule": ["timer schedule left on", "manual override not reset", "booking/event", "BMS schedule error", "other"],
    "high_while_open": ["dirty filter", "thermostat placed near a window", "low refrigerant", "occupancy change",
                        "sensor error", "other"],
    "warm_running": ["dirty filter", "low refrigerant", "thermostat fault", "sensor error", "other"],
    "supply_group_off": ["chilled-water supply fault", "AHU fault", "electrical trip", "other"],
    "supply_group_high": ["chilled-water supply fault", "AHU fault", "exam event, high occupancy", "other"],
}
# root cause -> fault type of the maintenance log (Phase 9)
FAULT_TYPE = {"dirty filter": "dirty filter", "low refrigerant": "refrigerant leak", "thermostat fault": "thermostat/controls",
              "thermostat placed near a window": "thermostat/controls", "timer schedule left on": "thermostat/controls",
              "manual override not reset": "thermostat/controls", "BMS schedule error": "thermostat/controls",
              "sensor error": "sensor fault", "electrical trip": "electrical trip", "AHU fault": "fan motor",
              "chilled-water supply fault": "other"}
RULES = ["statistical", "after_hours", "persistent"]
THRESHOLD_MIN_CLOSEOUTS, THRESHOLD_MAX_PRECISION = 3, 0.30


def _loads(v):
    if isinstance(v, (dict, list)) or v is None:
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return None


def close_out(store, wo_id, outcome, root_cause, action_taken="", date_fixed=None, technician="", notes="", actor="user",
              simulated=0, closed_at=None):
    """Record the close-out, finish the work order, update the incident and the maintenance log. Returns the suggested
    exception (a dict to confirm) for 'Legitimate use', else None."""
    if outcome not in OUTCOMES:
        raise ValueError(outcome)
    wo = store.get("work_orders", wo_id)
    if wo is None:
        raise KeyError(wo_id)
    when = closed_at or pd.Timestamp.now().isoformat(timespec="seconds")
    store.upsert("closeouts", {"wo_id": wo_id, "outcome": outcome, "root_cause": root_cause, "action_taken": action_taken,
                               "date_fixed": str(pd.Timestamp(date_fixed).date()) if date_fixed is not None else None,
                               "technician": technician, "notes": notes, "closed_at": when, "simulated": simulated}, actor,
                 change=f"close-out: {outcome} ({root_cause})")
    update_work_order(store, wo_id, actor, status="Done")
    key = wo["incident_key"]
    state = store.get("incident_state", key)
    if (state["status"] if state else "Open") != "Work order raised":
        set_status(store, key, "Work order raised", "close-out", actor, simulated)  # keep the lifecycle path valid
    if outcome == "Fault confirmed":
        set_status(store, key, "Resolved", f"{root_cause}", actor, simulated)
        store.upsert("maintenance_log", {
            "log_id": store.next_id("maintenance_log"), "asset_id": wo["asset_id"], "date": str(pd.Timestamp(date_fixed or when).date()),
            "type": "fault repair", "fault_type": FAULT_TYPE.get(root_cause, "other"), "severity": wo["priority"],
            "downtime_h": None, "cost_THB": None, "parts": "", "technician": technician,
            "notes": f"{root_cause}: {action_taken}".strip(": "), "source": "work order", "wo_id": wo_id, "simulated": simulated},
            actor, change="from close-out")
    elif outcome == "No fault found":
        set_status(store, key, "Dismissed", f"No fault found. {root_cause}".strip(), actor, simulated)
    else:
        set_status(store, key, "Dismissed", f"Legitimate use. {root_cause}".strip(), actor, simulated)
    return None


def suggest_exception(wo, incident_events):
    """Pre-fill for 'Legitimate use' (asset, weekday(s), time window, date range, reason): the user confirms it."""
    ev = incident_events.sort_values("start")
    last = ev.iloc[-1]
    weekdays = sorted(ev["start"].dt.dayofweek.value_counts().loc[lambda s: s >= s.max() / 2].index.tolist())
    return {"asset_id": wo["asset_id"], "weekdays": ",".join(map(str, weekdays)),
            "time_from": f"{last['start']:%H:%M}", "time_to": f"{last['end']:%H:%M}",
            "date_from": str(last["start"].date()), "date_to": str(last["start"].date()),
            "reason": "Legitimate use", "source_wo_id": wo["wo_id"]}


def add_exception(store, exc, actor="user", simulated=0):
    row = {**exc, "exc_id": store.next_id("exceptions"), "active": 1, "simulated": simulated}
    return store.upsert("exceptions", row, actor)


def deactivate_exception(store, exc_id, actor="user"):
    row = store.get("exceptions", exc_id)
    store.upsert("exceptions", {**row, "active": 0}, actor, change="deactivated")


def _minutes(hhmm):
    h, m = str(hhmm).split(":")
    return int(h) * 60 + int(m)


def exceptions_tuple(store, column_of_asset):
    """Active exceptions as the hashable tuple flag_detail() takes: (unit column, weekdays, from min, to min, from, to)."""
    t = store.table("exceptions")
    out = []
    for r in t[t["active"].astype(str).isin(["1", "True", "true"])].itertuples():
        col = column_of_asset.get(r.asset_id)
        if col is None:
            continue
        days = tuple(int(x) for x in str(r.weekdays).split(",") if str(x).strip().isdigit())
        out.append((col, days, _minutes(r.time_from or "00:00"), _minutes(r.time_to or "23:59") or 24 * 60,
                    r.date_from or None, r.date_to or None))
    return tuple(out)


# ---------------------------------------------------------------- precision and track record
def closeout_table(store, events):
    """One row per close-out with its incident, asset, pattern and which rules were present in that incident."""
    co, wo = store.table("closeouts"), store.table("work_orders")
    if co.empty:
        return pd.DataFrame(columns=["wo_id", "outcome", "asset_id", "pattern", "unit", "simulated", *RULES])
    t = co.merge(wo[["wo_id", "incident_key", "asset_id"]], on="wo_id", how="left")
    t["pattern"] = t["incident_key"].str.split("|").str[1]
    if events is not None and len(events):
        rules = events.groupby("incident_key").agg(
            statistical=("statistical", "any"), persistent=("persistent", "any"),
            after_hours=("after_hours_h", lambda s: bool((s > 0).any()))) if "statistical" in events else None
        t = t.merge(rules, left_on="incident_key", right_index=True, how="left") if rules is not None else t
    for r in RULES:
        if r not in t:
            t[r] = False
        t[r] = t[r].fillna(False).astype(bool)
    return t


def precision_tables(store, events):
    """Precision = confirmed / (confirmed + no fault). 'Legitimate use' is counted separately as expected use."""
    t = closeout_table(store, events)
    out = {}
    for name, key in [("pattern", "pattern"), ("unit", "asset_id")]:
        out[name] = _precision(t, key)
    rows = []
    for rule in RULES:
        rows.append({"rule": rule, **_counts(t[t[rule]])})
    out["rule"] = pd.DataFrame(rows).set_index("rule").assign(precision=lambda x: _prec(x))
    out["overall"] = _counts(t)
    return out


def _counts(t):
    return {"confirmed": int((t["outcome"] == "Fault confirmed").sum()), "no_fault": int((t["outcome"] == "No fault found").sum()),
            "expected_use": int((t["outcome"] == "Legitimate use").sum())}


def _prec(df):
    denom = (df["confirmed"] + df["no_fault"]).replace(0, np.nan)
    return df["confirmed"] / denom


def _precision(t, key):
    if t.empty:
        return pd.DataFrame(columns=["confirmed", "no_fault", "expected_use", "precision"])
    g = t.groupby(key)["outcome"]
    df = pd.DataFrame({"confirmed": g.apply(lambda s: (s == "Fault confirmed").sum()),
                       "no_fault": g.apply(lambda s: (s == "No fault found").sum()),
                       "expected_use": g.apply(lambda s: (s == "Legitimate use").sum())})
    df["precision"] = _prec(df)
    return df


def overall_precision_pct(store):
    co = store.table("closeouts")
    c, n = int((co["outcome"] == "Fault confirmed").sum()), int((co["outcome"] == "No fault found").sum())
    return (100 * c / (c + n)) if c + n else float("nan"), c, n


def track_record(store, column_of_asset):
    """{(unit column, pattern): (confirmed, no_fault)} plus {(None, pattern): ...} fleet-wide, for build_events()."""
    t = closeout_table(store, None)
    rec = {}
    for r in t.itertuples():
        if r.outcome == "Legitimate use":
            continue
        idx = 0 if r.outcome == "Fault confirmed" else 1
        for k in [(column_of_asset.get(r.asset_id), r.pattern), (None, r.pattern)]:
            cur = list(rec.get(k, (0, 0)))
            cur[idx] += 1
            rec[k] = tuple(cur)
    # a unit's own record wins; (unit, pattern) pairs that have none fall back to the pattern-wide record in build_events
    return rec


def threshold_suggestions(store, k_iqr=1.5):
    t = precision_tables(store, None)["unit"]
    out = []
    for asset, r in t.iterrows():
        if r["confirmed"] + r["no_fault"] >= THRESHOLD_MIN_CLOSEOUTS and r["precision"] < THRESHOLD_MAX_PRECISION:
            out.append(f"{asset}: only {r['precision'] * 100:.0f}% of {int(r['confirmed'] + r['no_fault'])} close-outs were real "
                       f"faults. Consider raising k for this unit from {k_iqr:g} to {k_iqr + 0.5:g}. (Suggestion only: not applied.)")
    return out


def history_note(store, asset_id, n=2):
    """'Last 2 similar cases on this unit: dirty filter (simulated)' for the Smart assistant."""
    co = store.table("closeouts").merge(store.table("work_orders")[["wo_id", "asset_id"]], on="wo_id")
    co = co[(co["asset_id"] == asset_id) & (co["outcome"] == "Fault confirmed")].sort_values("closed_at").tail(n)
    if co.empty:
        return ""
    return "Last cases on this unit: " + ", ".join(f"{r.root_cause}" + (" (simulated)" if r.simulated else "") for r in co.itertuples())


# ---------------------------------------------------------------- demo seed
# (incident key, outcome, root cause, note). Keys follow the plan's table, moved to the nearest incident that exists in the
# Jul-Dec data; the plan's F7 Z2 AC1 warm_running and F3 Z5 AC1 off_schedule incidents do not occur in this data.
SEED_ROWS = [
    ("F7_Z1_AC4|closed_running", "Fault confirmed", "timer schedule left on", "Saturday timer"),
    ("F7_Z2_AC1|closed_running", "Legitimate use", "booking/event", "Saturday booking"),
    ("F7_Z5_AC1|warm_running", "Fault confirmed", "dirty filter", ""),
    ("F7_Z1_AC4|warm_running", "Fault confirmed", "low refrigerant", ""),
    ("F5_Z2_AC1|off_schedule", "Fault confirmed", "timer schedule left on", ""),
    ("Floor4_CHW_assumed|supply_group_high", "No fault found", "exam event, high occupancy", ""),
    ("F4_Z4_AC1|high_while_open", "No fault found", "other", ""),
    ("F5_Z2_AC1|high_while_open", "Fault confirmed", "thermostat placed near a window", ""),
    ("F7_Z4_AC1|closed_running", "Fault confirmed", "manual override not reset", ""),
    ("F4_Z1_AC4|high_while_open", "No fault found", "other", ""),
]


def seed_demo(store, events, incidents, details, units, open_spec, forecast_by_floor, as_of):
    """About 10 close-outs on real incidents, all stored with simulated = 1. Runs only when the store is empty.
    Fix dates use an observed natural change where one exists ('simulated fix at an observed change'); otherwise the
    fix has no visible effect, so verification will say so."""
    if not store.is_empty():
        return 0
    inc = incidents.set_index("incident_key")
    col_of = dict(zip(units["asset_id"], units["power_column"]))
    made = 0
    for key, outcome, cause, note in SEED_ROWS:
        if key not in inc.index:
            continue
        row = inc.loc[key].copy()
        row["incident_key"] = key
        floor = int(row["floor"])
        wo = draft_work_order(row, pd.Timestamp(row["last_seen"]).normalize(), open_spec,
                              forecast_by_floor[floor], max(1, events[events["incident_key"] == key]["start"].dt.normalize().nunique()))
        wo["simulated"] = 1
        wo["status"] = "In progress"
        saved = raise_work_order(store, wo, "seed")
        fix, basis = None, ""
        if outcome == "Fault confirmed":
            if not row["pattern"].startswith("supply"):
                d = details[floor][details[floor]["unit"] == col_of[row["asset_id"]]]
                changes = find_natural_changes(d, row["pattern"])
                if changes:
                    fix, basis = changes[0], "simulated fix at an observed change"
            if fix is None:
                fix = (pd.Timestamp(row["last_seen"]).normalize() + pd.Timedelta(days=1)
                       if row["occurrences"] < 5 else pd.Timestamp(row["first_seen"]).normalize() + pd.Timedelta(days=60))
                basis = "simulated fix; no change in the data around this date"
        close_out(store, saved["wo_id"], outcome, cause, action_taken=basis or note, date_fixed=fix,
                  technician="(simulated)", notes=(note + " " + basis).strip(), actor="seed", simulated=1,
                  closed_at=str((fix or pd.Timestamp(row["last_seen"])).date()) + "T12:00:00")
        if outcome == "Legitimate use":
            sug = suggest_exception(store.get("work_orders", saved["wo_id"]), events[events["incident_key"] == key])
            ev = events[events["incident_key"] == key].sort_values("start")
            sats = sorted(ev.loc[ev["start"].dt.dayofweek == 5, "start"].dt.normalize().unique())[-2:]  # 2 Saturdays
            if len(sats):
                sug.update(weekdays="5", date_from=str(pd.Timestamp(sats[0]).date()), date_to=str(pd.Timestamp(sats[-1]).date()),
                           time_from="00:00", time_to="24:00")
            add_exception(store, sug, "seed", simulated=1)
        made += 1
    return made


# ---------------------------------------------------------------- verification of resolved incidents (Phase 6)
def verification_results(store, events, details, units, tariff, sync=True, actor="verification"):
    """{incident_key: verify_incident(...) + wo_id, root_cause, simulated, basis} for every confirmed, fixed close-out.
    With sync=True, a Verified verdict moves a Resolved incident to Verified, and recurrence reopens it."""
    co = store.table("closeouts").merge(store.table("work_orders")[["wo_id", "incident_key", "asset_id"]], on="wo_id")
    col_of = dict(zip(units["asset_id"], units["power_column"]))
    supply_cols = units.groupby("supply_group")["power_column"].apply(list).to_dict()
    out = {}
    for r in co[(co["outcome"] == "Fault confirmed") & co["date_fixed"].notna()].itertuples():
        pattern = r.incident_key.split("|")[1]
        ev = events[events["incident_key"] == r.incident_key] if len(events) else events
        if pattern.startswith("supply_group"):
            floor = int(r.asset_id.removeprefix("Floor").split("_")[0])
            d = details[floor][details[floor]["unit"].isin(supply_cols.get(r.asset_id, []))]
            grp = ev
        else:
            floor = int(r.asset_id[1])
            d = details[floor][details[floor]["unit"] == col_of[r.asset_id]]
            grp = None
        if d.empty:
            continue
        v = verify_incident(d, pattern, r.date_fixed, ev, tariff=tariff, group_events=grp)
        out[r.incident_key] = {**v, "wo_id": r.wo_id, "root_cause": r.root_cause, "simulated": bool(r.simulated), "notes": r.notes}
        if sync:
            state = store.get("incident_state", r.incident_key)
            status = state["status"] if state else "Open"
            if v["verdict"] == "Verified" and status == "Resolved":
                set_status(store, r.incident_key, "Verified", f"metric fell {v['drop_pct']:.0f}%", actor, int(r.simulated))
            elif v["verdict"] == "Reopened" and status in ("Resolved", "Verified"):
                set_status(store, r.incident_key, "Reopened", f"{v['events_last_14d']} recurrence(s) after {v['fix_date']:%d %b}",
                           actor, int(r.simulated))
    return out
