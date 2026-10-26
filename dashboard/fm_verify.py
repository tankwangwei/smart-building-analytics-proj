"""Verification (measurement and verification, M&V) of resolved incidents (plan §8). Streamlit-free.

A fix is verified when the incident's own metric fell by at least half in the 28 days after the fix date, nothing
recurred in the last 14 days of that window and comfort did not get worse. Nothing here proves a saving: it compares
before and after on the same data."""
import numpy as np
import pandas as pd

from fm_core import COMFORT_BAND, DEFAULT_TARIFF_THB, INTERVAL_HOURS

WINDOW_DAYS = 28
RECUR_DAYS = 14
MIN_POST_DAYS = 14
DROP_TO_VERIFY = 0.5
COMFORT_WORSE_PTS = 10
VERDICTS = ["Verified", "Reopened", "Inconclusive", "Not verified"]
METRIC_LABEL = {"closed_running": "kWh while closed or off-schedule per day", "off_schedule": "kWh while closed or off-schedule per day",
                "high_while_open": "kWh above expected per day", "warm_running": "hours running with the zone > 26 °C per day",
                "supply_group_off": "hours with units off together per day", "supply_group_high": "flagged kWh per day"}


def daily_metric(detail, pattern, start, end, group_events=None):
    """The incident's metric per day over [start, end) for the given flagged detail (one unit, or a supply group's units).
    For supply_group_off the metric needs the group events instead of the detail."""
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() - pd.Timedelta(days=1), freq="D")
    if pattern == "supply_group_off":
        s = pd.Series(0.0, index=days)
        for _, e in (group_events if group_events is not None else pd.DataFrame()).iterrows():
            day = e["start"].normalize()
            if day in s.index:
                s[day] += e["duration_h"]
        return s
    d = detail[detail["Date"].ge(start) & detail["Date"].lt(end)]
    day = d["Date"].dt.normalize()
    if pattern in ("closed_running", "off_schedule"):
        v = (d["actual_kW"] * INTERVAL_HOURS).where(d["after_hours"], 0.0)
    elif pattern == "high_while_open":
        v = d["residual_kW"].clip(lower=0) * INTERVAL_HOURS
        v = v.where(~d["closed"], 0.0)
    elif pattern == "warm_running":
        v = (d["indoor_temp_C"].gt(COMFORT_BAND[1]) & d["actual_kW"].gt(0.5)).astype(float) * INTERVAL_HOURS
    elif pattern == "supply_group_high":
        v = d["excess_kWh_est"].where(d["flagged"], 0.0)
    else:
        raise ValueError(pattern)
    return v.groupby(day).sum().reindex(days, fill_value=0.0)


def comfort_pct(detail, start, end, on_kw=0.5):
    d = detail[detail["Date"].ge(start) & detail["Date"].lt(end) & detail["actual_kW"].gt(on_kw) & detail["indoor_temp_C"].notna()]
    return float(100 * d["indoor_temp_C"].between(*COMFORT_BAND).mean()) if len(d) else float("nan")


def verify_incident(detail, pattern, fix_date, events=None, data_end=None, tariff=DEFAULT_TARIFF_THB, group_events=None):
    """Compare the 28 days before `fix_date` with up to 28 days after. `detail` = flagged detail for the unit (or the
    supply group's units, concatenated); `events` = the incident's events (start column) for recurrence."""
    fix = pd.Timestamp(fix_date).normalize()
    data_end = pd.Timestamp(data_end if data_end is not None else detail["Date"].max()).normalize() + pd.Timedelta(days=1)
    data_start = detail["Date"].min().normalize()
    pre_start, post_end = max(fix - pd.Timedelta(days=WINDOW_DAYS), data_start), min(fix + pd.Timedelta(days=WINDOW_DAYS), data_end)
    post_days = max(0, (post_end - fix).days)
    pre = daily_metric(detail, pattern, pre_start, fix, group_events)
    post = daily_metric(detail, pattern, fix, post_end, group_events) if post_days else pd.Series(dtype=float)
    ev_post = (events[events["start"].ge(fix) & events["start"].lt(post_end)] if events is not None else pd.DataFrame({"start": []}))
    last_window = ev_post[ev_post["start"].ge(post_end - pd.Timedelta(days=RECUR_DAYS))] if len(ev_post) else ev_post
    c_pre, c_post = comfort_pct(detail, pre_start, fix), (comfort_pct(detail, fix, post_end) if post_days else np.nan)
    pre_day, post_day = float(pre.mean()), (float(post.mean()) if post_days else np.nan)
    drop = 1 - post_day / pre_day if pre_day > 0 and post_days else np.nan
    comfort_ok = not (pd.notna(c_pre) and pd.notna(c_post) and c_post < c_pre - COMFORT_WORSE_PTS)
    if post_days < MIN_POST_DAYS:
        verdict = "Inconclusive"
    elif len(last_window):
        verdict = "Reopened"
    elif pd.notna(drop) and drop >= DROP_TO_VERIFY and comfort_ok:
        verdict = "Verified"
    else:
        verdict = "Not verified"
    saved_kwh = max(0.0, (pre_day - post_day) * post_days) if verdict == "Verified" and pattern not in ("warm_running", "supply_group_off") else 0.0
    return {"verdict": verdict, "fix_date": fix, "pre_start": pre_start, "post_end": post_end, "post_days": post_days,
            "pre_per_day": pre_day, "post_per_day": post_day, "drop_pct": 100 * drop if pd.notna(drop) else np.nan,
            "events_post": int(len(ev_post)), "events_last_14d": int(len(last_window)),
            "comfort_pre": c_pre, "comfort_post": c_post, "saved_kWh": saved_kwh, "saved_THB": saved_kwh * tariff,
            "metric": METRIC_LABEL[pattern], "pre_series": pre, "post_series": post}


def find_natural_changes(detail, pattern, min_drop=0.7, stay_weeks=4, group_events=None):
    """Weeks where the weekly metric falls by >= min_drop against the average of the previous 4 weeks and stays that low
    for >= stay_weeks. Returns the week-start dates. Used to choose realistic seed fix dates."""
    start, end = detail["Date"].min().normalize(), detail["Date"].max().normalize() + pd.Timedelta(days=1)
    daily = daily_metric(detail, pattern, start, end, group_events)
    weekly = daily.resample("W-MON", label="left", closed="left").sum()
    out = []
    for i in range(4, len(weekly) - stay_weeks + 1):
        before = weekly.iloc[i - 4:i].mean()
        if before <= 0:
            continue
        after = weekly.iloc[i:i + stay_weeks]
        if (after <= (1 - min_drop) * before).all():
            out.append(weekly.index[i])
    return out
