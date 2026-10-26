"""Data for the asset history card (plan §9). Streamlit-free."""
import numpy as np
import pandas as pd

from fm_core import COMFORT_BAND


def running_open_weekday(d, on_kw=0.5):
    """Running, open, weekday, valid intervals of one unit (where kW per °C is meaningful)."""
    return d[d["eligible"] & d["actual_kW"].gt(on_kw) & ~d["closed"] & d["Date"].dt.dayofweek.lt(5)
             & d["outdoor_temp_C"].notna()]


def monthly_kw_per_degc(d, on_kw=0.5, min_hours=20):
    """Per month: slope of hourly kW on outdoor °C for weekday running hours. A creeping rise hints at coil or
    refrigerant problems."""
    r = running_open_weekday(d, on_kw)
    hourly = r.groupby(r["Date"].dt.floor("h")).agg(kw=("actual_kW", "mean"), t=("outdoor_temp_C", "mean"))
    rows = []
    for month, g in hourly.groupby(hourly.index.to_period("M")):
        if len(g) >= min_hours and g["t"].std() > 0.5:
            rows.append({"month": month.to_timestamp(), "kW_per_degC": float(np.polyfit(g["t"], g["kw"], 1)[0]), "hours": len(g)})
    return pd.DataFrame(rows, columns=["month", "kW_per_degC", "hours"])


def monthly_peer_gap(details, column, peer_columns):
    """Monthly mean kW (valid intervals) vs the median of the same unit position on other floors, in %."""
    def monthly(col):
        for d in details.values():
            part = d[d["unit"].eq(col) & d["eligible"]]
            if len(part):
                return part.groupby(part["Date"].dt.to_period("M").dt.to_timestamp())["actual_kW"].mean()
        return pd.Series(dtype=float)
    own = monthly(column)
    peers = pd.concat([monthly(c) for c in peer_columns if c != column], axis=1)
    peer = peers.median(axis=1)
    gap = 100 * (own / peer.where(peer > 0.5) - 1)
    return gap.dropna().rename("gap_pct").reset_index().rename(columns={"Date": "month", "index": "month"})


def monthly_comfort(d, on_kw=0.5):
    r = d[d["actual_kW"].gt(on_kw) & d["indoor_temp_C"].notna()]
    g = r.groupby(r["Date"].dt.to_period("M").dt.to_timestamp())["indoor_temp_C"].apply(lambda s: 100 * s.between(*COMFORT_BAND).mean())
    return g.rename("comfort_pct").reset_index().rename(columns={"Date": "month"})


def recurrence_rate(events, weeks_total):
    """Share of weeks with at least one episode."""
    if events is None or events.empty:
        return 0.0
    return events["start"].dt.to_period("W").nunique() / max(1, weeks_total)


def timeline(asset_id, events, store):
    """Rows of (time, kind, label, simulated) for flags, work orders, close-outs and maintenance entries."""
    rows = []
    for e in events[events["asset_id"].eq(asset_id)].itertuples():
        rows.append((e.start, "Flagged episode", f"{e.pattern}: {e.excess_kWh:.0f} kWh", False))
    wo = store.table("work_orders")
    for w in wo[wo["asset_id"].eq(asset_id)].itertuples():
        rows.append((pd.Timestamp(w.slot_start), "Work order", f"{w.wo_id} ({w.status})", bool(w.simulated)))
        co = store.get("closeouts", w.wo_id)
        if co:
            rows.append((pd.Timestamp(co["closed_at"]), "Close-out", f"{co['outcome']}: {co['root_cause']}", bool(co["simulated"])))
    log = store.table("maintenance_log")
    for m in log[log["asset_id"].eq(asset_id)].itertuples():
        rows.append((pd.Timestamp(m.date), "Maintenance", f"{m.type}: {m.fault_type or ''}", bool(m.simulated)))
    return pd.DataFrame(rows, columns=["time", "kind", "label", "simulated"])
