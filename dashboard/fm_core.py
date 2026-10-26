"""Pure (Streamlit-free) logic for the FM workflow: closures, flagging rules, events, patterns and triage.

app.py imports from here so the same code is unit-tested (tests/) and used by the dashboard.
"""
import numpy as np
import pandas as pd

from thai_holidays import HOLIDAYS, OBSERVED_CLOSURES, closure_name

INTERVAL_HOURS = 0.25
CLOSED_DATES = pd.DatetimeIndex(sorted({*HOLIDAYS, *OBSERVED_CLOSURES}))
PRIORITY_LEVELS = ("🔴 High", "🟠 Medium", "🟢 Low")
DEFAULT_TARIFF_THB = 4.2
COMFORT_BAND = (23.0, 26.0)

# Pattern -> (effort label, effort factor). One place, reused by incidents and work orders.
PATTERNS = {
    "closed_running": ("Low", 1.0, "runs while the building is closed"),
    "off_schedule": ("Low", 1.0, "runs outside its usual schedule"),
    "high_while_open": ("Medium", 1.5, "runs high while the building is open"),
    "warm_running": ("High", 2.5, "runs but the zone stays above 26 °C"),
    # group-level patterns from the shared (assumed) supply of a floor
    "supply_group_off": ("Medium", 1.5, "several units off together on one supply"),
    "supply_group_high": ("Medium", 1.5, "several units high together on one supply"),
}
CONFIDENCE_WEIGHTS = {"rules": .35, "duration": .25, "recurrence": .20, "not_weather": .10, "quality": .10}


def building_closed(dates, open_spec):
    """True where the building is closed: a closed weekday, outside opening hours, or (if enabled) a public holiday or
    observed closure. open_spec = (open weekdays, opening minute, closing minute, use holidays)."""
    days, start_min, end_min, use_holidays = open_spec
    minutes = dates.dt.hour * 60 + dates.dt.minute
    closed = ~dates.dt.dayofweek.isin(days) | minutes.lt(start_min) | minutes.ge(end_min)
    if use_holidays:
        closed |= dates.dt.normalize().isin(CLOSED_DATES)
    return closed


def run_rules(detail, thresholds, k_iqr, k_res, on_kw, persist, use_stat=True, use_after=True, use_persist=True):
    """Apply the three flagging rules (statistical, after_hours, persistent) with adjustable thresholds."""
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


def apply_closures(detail, open_spec, exceptions=()):
    """Add closed/holiday/excepted columns; while closed, expected power is 0 so all energy then counts as excess.

    exceptions: hashable tuple of (unit, weekdays tuple, time_from minute, time_to minute, date_from, date_to) for
    confirmed legitimate use. Matching intervals are marked `excepted` and are never flagged."""
    detail = detail.copy()
    detail["closed"] = building_closed(detail["Date"], open_spec)
    detail["holiday"] = detail["Date"].dt.normalize().isin(CLOSED_DATES) & open_spec[3]
    detail.loc[detail["closed"], "expected_kW"] = 0.0
    detail["residual_kW"] = detail["actual_kW"] - detail["expected_kW"]
    detail["excess_kWh_est"] = detail["residual_kW"].clip(lower=0) * INTERVAL_HOURS
    detail["excepted"] = exception_mask(detail, exceptions)
    return detail


def exception_mask(detail, exceptions):
    mask = pd.Series(False, index=detail.index)
    if not exceptions:
        return mask
    minutes = detail["Date"].dt.hour * 60 + detail["Date"].dt.minute
    day = detail["Date"].dt.normalize()
    for unit, weekdays, t_from, t_to, d_from, d_to in exceptions:
        m = detail["unit"].eq(unit) & minutes.ge(t_from) & minutes.lt(t_to)
        if weekdays:
            m &= detail["Date"].dt.dayofweek.isin(weekdays)
        if d_from:
            m &= day.ge(pd.Timestamp(d_from))
        if d_to:
            m &= day.le(pd.Timestamp(d_to))
        mask |= m
    return mask


def flag_detail(detail, thresholds, open_spec, k_iqr, k_res, on_kw, persist, exceptions=()):
    """Closures + rules, replacing any default-threshold flag columns already in the detail."""
    detail = apply_closures(detail, open_spec, exceptions)
    flags = run_rules(detail, thresholds, k_iqr, k_res, on_kw, persist)
    flags.loc[detail["excepted"], ["statistical", "after_hours", "persistent", "flagged"]] = False
    base = detail.drop(columns=[c for c in flags.columns if c in detail.columns])
    return pd.concat([base, flags], axis=1)


# ---------------------------------------------------------------- events, patterns, triage
def classify_pattern(row):
    """One pattern per event (see plan §3). row needs: closed_share, holiday, after_hours_share, warm_share."""
    if row["closed_share"] >= .5 or row["holiday"]:
        return "closed_running"
    if row["after_hours_share"] > 0:
        return "off_schedule"
    if row["warm_share"] >= .5:
        return "warm_running"
    return "high_while_open"


def confidence_components(events, all_events):
    """Component scores (0-1) per event; `all_events` is the unit's events over the timeframe, for recurrence."""
    rules = (events["statistical"].astype(int) + (events["after_hours_h"] > 0).astype(int)
             + events["persistent"].astype(int)) / 3
    duration = (events["duration_h"] / 2).clip(upper=1)
    slot = events["unit"] + "|" + events["start"].dt.dayofweek.astype(str) + "|" + events["start"].dt.hour.astype(str)
    counts = slot.map(slot.value_counts())
    recurrence = counts.map({1: 0.0, 2: 0.5}).fillna(1.0)
    not_weather = np.where(events["hot_day"], 0.4, 1.0)
    return pd.DataFrame({"rules": rules, "duration": duration, "recurrence": recurrence,
                         "not_weather": not_weather, "quality": events["quality_share"]}, index=events.index)


def confidence_band(score):
    return "High" if score >= 70 else "Medium" if score >= 40 else "Low"


def build_events(d, tariff=DEFAULT_TARIFF_THB, comfort_band=COMFORT_BAND, track_record=None):
    """Merge consecutive flagged intervals per unit into events and score them for triage.

    triage_score = excess_THB * confidence / 100 / effort_factor   (replaces the old weighted-kWh score)
    track_record: optional {(unit column, pattern): (confirmed, no_fault)}, with (None, pattern) as the fleet-wide fallback
    (Phase 5), applied to confidence."""
    flagged = d["flagged"]
    f = d.loc[flagged].copy()
    if f.empty:
        return pd.DataFrame()
    run = (flagged.ne(flagged.shift()) | d["unit"].ne(d["unit"].shift())).cumsum()
    f["run"] = run.loc[f.index]
    f["running"] = f["actual_kW"].gt(0.0)
    f["warm"] = f["indoor_temp_C"].gt(comfort_band[1]) & f["actual_kW"].gt(0.5)
    f["out_of_band"] = ~f["indoor_temp_C"].between(*comfort_band) & f["indoor_temp_C"].notna()
    f["good_data"] = f["observed_minutes"].ge(14) & ~f["has_interpolation"].astype(bool)
    events = f.groupby("run").agg(unit=("unit", "first"), start=("Date", "min"), last=("Date", "max"),
                                  intervals=("Date", "size"), peak_kW=("actual_kW", "max"),
                                  excess_kWh=("excess_kWh_est", "sum"), after_hours=("after_hours", "sum"),
                                  closed_n=("closed", "sum"), warm_n=("warm", "sum"), band_n=("out_of_band", "sum"),
                                  quality_share=("good_data", "mean"),
                                  statistical=("statistical", "any"), persistent=("persistent", "any"),
                                  hot_day=("hot_day", "any"), outdoor_C=("outdoor_temp_C", "mean"),
                                  indoor_C=("indoor_temp_C", "mean"), holiday=("holiday", "any"))
    events["end"] = events["last"] + pd.Timedelta(minutes=15)
    events["duration_h"] = events["intervals"] * INTERVAL_HOURS
    events["after_hours_h"] = events["after_hours"] * INTERVAL_HOURS
    events["closed_share"] = events["closed_n"] / events["intervals"]
    events["after_hours_share"] = events["after_hours"] / events["intervals"]
    events["warm_share"] = events["warm_n"] / events["intervals"]
    events["comfort_hit"] = events["band_n"] / events["intervals"] >= .5
    events["pattern"] = events.apply(classify_pattern, axis=1)
    events["effort"] = events["pattern"].map(lambda p: PATTERNS[p][0])
    events["effort_factor"] = events["pattern"].map(lambda p: PATTERNS[p][1])

    comp = confidence_components(events, events)
    events["confidence"] = 100 * sum(comp[k] * w for k, w in CONFIDENCE_WEIGHTS.items())
    if track_record:
        events["confidence"] = [apply_track_record(c, track_record.get((u, p), track_record.get((None, p))))
                                for c, u, p in zip(events["confidence"], events["unit"], events["pattern"])]
    events["confidence_band"] = events["confidence"].map(confidence_band)
    events["excess_THB"] = events["excess_kWh"] * tariff
    events["triage_score"] = events["excess_THB"] * events["confidence"] / 100 / events["effort_factor"]

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
    events = events.sort_values("triage_score", ascending=False).reset_index(drop=True)
    events["priority_level"] = priority_levels(events["triage_score"])
    return events


def apply_track_record(base, record):
    """Phase 5: final = 0.8 * base + 0.2 * 100 * (confirmed + 1) / (confirmed + no_fault + 2)."""
    if not record:
        return base
    confirmed, no_fault = record
    return 0.8 * base + 0.2 * 100 * (confirmed + 1) / (confirmed + no_fault + 2)


def priority_levels(score):
    """High = the top items that together make up the first half of the total score,
    Medium = the items that take it to 80 %, Low = the rest (and anything with no score)."""
    order = score.sort_values(ascending=False)
    total = order.sum()
    share_above = (order.cumsum() - order) / total if total > 0 else pd.Series(1.0, index=order.index)
    level = pd.Series(np.select([share_above < .5, share_above < .8], PRIORITY_LEVELS[:2], PRIORITY_LEVELS[2]),
                      index=order.index)
    level[order <= 0] = PRIORITY_LEVELS[2]
    return level.reindex(score.index)


def add_levels(table):
    """Priority level and share of the total triage score, computed over the rows of this table."""
    table["priority_level"] = priority_levels(table["triage_score"])
    table["share_pct"] = 100 * table["triage_score"] / max(1e-9, table["triage_score"].sum())
    return table
