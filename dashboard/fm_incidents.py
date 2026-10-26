"""Incidents: one stateful item per problem instead of hundreds of events (plan §5). Streamlit-free."""
import pandas as pd

from fm_core import DEFAULT_TARIFF_THB, INTERVAL_HOURS, PATTERNS, build_events, priority_levels

STATUSES = ["Open", "Acknowledged", "Work order raised", "Resolved", "Verified", "Reopened", "Dismissed"]
STATUS_COLOUR = {"Open": "#e5533d", "Acknowledged": "#f0a03c", "Work order raised": "#5aa9ff", "Resolved": "#3fb27f",
                 "Verified": "#1baf7a", "Reopened": "#e87ba4", "Dismissed": "#5b6577"}
NEXT = {"Open": {"Acknowledged", "Work order raised"}, "Acknowledged": {"Open", "Work order raised"},
        "Work order raised": {"Acknowledged", "Resolved"}, "Resolved": {"Verified", "Reopened"},
        "Verified": {"Reopened"}, "Reopened": {"Acknowledged", "Work order raised"}, "Dismissed": {"Open"}}
CLOSED_STATUSES = {"Resolved", "Verified", "Dismissed"}


def allowed_next(status):
    """States reachable from `status`, including itself. Any state can go to Dismissed (with a reason)."""
    return sorted(NEXT.get(status, set()) | {status} | ({"Dismissed"} if status != "Dismissed" else set()),
                  key=STATUSES.index)


def incident_key(asset_id, pattern):
    return f"{asset_id}|{pattern}"


def group_events(details, supply_by_column, on_kw=0.5, min_units=3):
    """Supply-group patterns (the assistant's shared-cause rules, over the whole data):
    supply_group_off  = >= min_units units of one supply off in their usual hours while the building runs;
    supply_group_high = >= min_units units above normal together (not after-hours).
    `details` = {floor: flagged 15-minute detail}. Returns one row per run of consecutive intervals."""
    allrows = pd.concat([d.assign(floor=f) for f, d in details.items()], ignore_index=True)
    allrows["running"] = allrows["actual_kW"].gt(on_kw)
    allrows["supply"] = allrows["unit"].map(supply_by_column)
    elig = allrows[allrows["eligible"]]
    building_running = elig.groupby("Date")["running"].mean()
    elig = elig.assign(bld=elig["Date"].map(building_running))
    weekday = elig["Date"].dt.dayofweek.lt(5)
    off = elig["schedule_on"].eq(True) & ~elig["running"] & elig["bld"].ge(.5) & weekday & ~elig["closed"]
    high = (elig["statistical"] | elig["persistent"]) & ~elig["after_hours"]
    rows = []
    for pattern, mask in [("supply_group_off", off), ("supply_group_high", high)]:
        counts = elig[mask].groupby(["supply", "Date"])["unit"].nunique()
        counts = counts[counts.ge(min_units)].reset_index()
        for supply, g in counts.groupby("supply"):
            g = g.sort_values("Date")
            run = (g["Date"].diff() != pd.Timedelta(minutes=15)).cumsum()
            for _, r in g.groupby(run):
                start, end = r["Date"].min(), r["Date"].max() + pd.Timedelta(minutes=15)
                inside = elig[mask & elig["supply"].eq(supply) & elig["Date"].between(start, end - pd.Timedelta(minutes=15))]
                hours = len(r) * INTERVAL_HOURS
                n = int(r["unit"].max())
                rows.append({"asset_id": supply, "pattern": pattern, "start": start, "end": end, "duration_h": hours,
                             "units": n, "excess_kWh": float(inside["excess_kWh_est"].sum()) if pattern.endswith("high") else 0.0,
                             "confidence": float(min(90, 50 + 10 * (n - 3) + 5 * min(hours, 4)))})
    return pd.DataFrame(rows, columns=["asset_id", "pattern", "start", "end", "duration_h", "units", "excess_kWh", "confidence"])


def all_events(details, brick_units, tariff=DEFAULT_TARIFF_THB, track_record=None):
    """Unit events for every floor (whole data), plus group events, in one table keyed by asset and pattern."""
    asset_of = dict(zip(brick_units["power_column"], brick_units["asset_id"]))
    label_of = dict(zip(brick_units["asset_id"], brick_units["label"]))
    supply_of = dict(zip(brick_units["power_column"], brick_units["supply_group"]))
    parts = []
    for f, d in details.items():
        ev = build_events(d, tariff, track_record=track_record)
        if len(ev):
            parts.append(ev.assign(floor=f, asset_id=ev["unit"].map(asset_of)))
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    grp = group_events(details, supply_of)
    if len(grp):
        grp["excess_THB"] = grp["excess_kWh"] * tariff
        grp["effort_factor"] = grp["pattern"].map(lambda p: PATTERNS[p][1])
        grp["triage_score"] = grp["excess_THB"] * grp["confidence"] / 100 / grp["effort_factor"]
        grp["floor"] = grp["asset_id"].str.extract(r"Floor(\d+)")[0].astype(int)
        grp["reason"] = grp["units"].astype(str) + " units together on one supply"
        label_of.update({a: f"Floor {a.removeprefix('Floor').split('_')[0]} chilled-water supply (assumed)" for a in grp["asset_id"]})
    ev = pd.concat([ev, grp], ignore_index=True) if len(grp) else ev
    if len(ev):
        ev["incident_key"] = ev["asset_id"] + "|" + ev["pattern"]
        ev["label"] = ev["asset_id"].map(label_of)
    return ev


def build_incidents(events, state=None, period=None):
    """One row per (asset, pattern). Built on the whole data so identity and status stay stable; `period` =
    (start, end) only decides which incidents are active in the timeframe. `state` = incident_state table."""
    cols = ["incident_key", "asset_id", "label", "floor", "pattern", "first_seen", "last_seen", "occurrences", "excess_kWh",
            "excess_THB", "confidence", "triage_score", "priority_level", "latest_start", "latest_end", "latest_hours",
            "status", "status_reason", "active", "period_occurrences", "period_THB"]
    if events is None or events.empty:
        return pd.DataFrame(columns=cols)
    ev = events.copy()
    weight = ev["excess_THB"].clip(lower=1e-9)
    ev["_cw"] = ev["confidence"] * weight
    ev["_w"] = weight
    g = ev.groupby("incident_key")
    inc = pd.DataFrame({"asset_id": g["asset_id"].first(), "label": g["label"].first(), "floor": g["floor"].first(),
                        "pattern": g["pattern"].first(), "first_seen": g["start"].min(), "last_seen": g["end"].max(),
                        "occurrences": g.size(), "excess_kWh": g["excess_kWh"].sum(), "excess_THB": g["excess_THB"].sum(),
                        "confidence": g["_cw"].sum() / g["_w"].sum(), "triage_score": g["triage_score"].sum()})
    latest = ev.sort_values("start").groupby("incident_key").tail(1).set_index("incident_key")
    inc["latest_start"], inc["latest_end"] = latest["start"], latest["end"]
    inc["latest_hours"] = latest["duration_h"]
    inc["priority_level"] = priority_levels(inc["triage_score"])
    # a supply-group outage is critical whatever its energy cost
    inc.loc[inc["pattern"].eq("supply_group_off"), "priority_level"] = "🔴 High"
    if period is not None:
        in_p = ev[ev["start"].lt(period[1]) & ev["end"].gt(period[0])]
        inc["period_occurrences"] = in_p.groupby("incident_key").size().reindex(inc.index, fill_value=0)
        inc["period_THB"] = in_p.groupby("incident_key")["excess_THB"].sum().reindex(inc.index, fill_value=0.0)
    else:
        inc["period_occurrences"], inc["period_THB"] = inc["occurrences"], inc["excess_THB"]
    inc["active"] = inc["period_occurrences"].gt(0)
    inc["status"], inc["status_reason"] = "Open", ""
    if state is not None and len(state):
        st_ = state.set_index("incident_key")
        known = inc.index.intersection(st_.index)
        inc.loc[known, "status"] = st_.loc[known, "status"]
        inc.loc[known, "status_reason"] = st_.loc[known, "status_reason"].fillna("")
    inc = inc.sort_values("triage_score", ascending=False).reset_index()
    return inc[cols]


def set_status(store, key, status, reason="", actor="user", simulated=0):
    """Write a status change (and an audit entry). Dismissed needs a reason; the transition must be allowed."""
    current = store.get("incident_state", key)
    old = current["status"] if current else "Open"
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if status != old and status not in allowed_next(old):
        raise ValueError(f"cannot go from {old} to {status}")
    if status == "Dismissed" and not str(reason).strip():
        raise ValueError("a reason is required to dismiss an incident")
    store.upsert("incident_state", {"incident_key": key, "status": status, "status_reason": reason,
                                    "updated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                                    "updated_by": actor, "simulated": simulated}, actor,
                 change=f"status {old} -> {status}" + (f" ({reason})" if reason else ""))
