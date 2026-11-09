"""Verification (measurement and verification, M&V) of resolved incidents (plan §8). Streamlit-free.

Follows the IPMVP adjusted-baseline idea (Option B, one piece of equipment): a small model of the incident's daily
metric is fitted on the 28 days before the fix (open vs closed day, plus daytime heat index for patterns that happen
while the building is open) and predicts what the metric would have been in the days after had nothing been fixed.
Saving = that adjusted baseline − what was measured. A fix is verified when the metric fell by at least half against
the adjusted baseline, nothing recurred in the last 14 days of the window and comfort did not get worse. The fixes in
the demo are simulated, so these are not proven savings."""
import numpy as np
import pandas as pd

from fm_core import CLOSED_DATES, COMFORT_BAND, DEFAULT_TARIFF_THB, INTERVAL_HOURS

WINDOW_DAYS = 28
RECUR_DAYS = 14
MIN_POST_DAYS = 14
DROP_TO_VERIFY = 0.5
COMFORT_WORSE_PTS = 10
VERDICTS = ["Verified", "Reopened", "Inconclusive", "Not verified"]
METRIC_LABEL = {"closed_running": "kWh while closed or off-schedule per day", "off_schedule": "kWh while closed or off-schedule per day",
                "high_while_open": "kWh above expected per day", "warm_running": "hours running with the zone > 26 °C per day",
                "supply_group_off": "hours with units off together per day", "supply_group_high": "flagged kWh per day"}
# Patterns that happen while the building is open, so their metric also moves with the weather
WEATHER_PATTERNS = {"high_while_open", "warm_running", "supply_group_high"}


def day_features(detail, days):
    """Per day: closed (weekend or public holiday / closure) and the daytime (08:00-17:00) mean outdoor heat index."""
    closed = pd.Series((days.dayofweek >= 5) | days.isin(CLOSED_DATES), index=days)
    if "heat_index_C" in detail:
        d = detail[(detail["Date"].dt.hour >= 8) & (detail["Date"].dt.hour < 17)]
        hi = d.groupby(d["Date"].dt.normalize())["heat_index_C"].mean().reindex(days)
    else:
        hi = pd.Series(np.nan, index=days)
    return closed, hi


def adjusted_baseline(pre, post_days, detail, pattern):
    """Predicted daily metric for `post_days` had nothing been fixed, from a model fitted on the `pre` days: the pre-fix
    average for that day type (open or closed), moved by slope × (heat index − its pre-fix average) for weather-driven
    patterns. Returns (prediction per day, fit CV(RMSE) % on the pre days in ASHRAE form, whether heat index was used)."""
    closed_pre, hi_pre = day_features(detail, pre.index)
    closed_post, hi_post = day_features(detail, post_days)
    means = pre.groupby(closed_pre).mean()
    level = lambda closed: closed.map(lambda c: means.get(c, pre.mean())).astype(float)
    fitted, pred = level(closed_pre), level(closed_post)
    use_hi = (pattern in WEATHER_PATTERNS and hi_pre.notna().all() and hi_post.notna().all() and hi_pre.std() > 0)
    if use_hi:
        x = (hi_pre - hi_pre.mean()).to_numpy()
        slope = float(np.dot(x, (pre - fitted).to_numpy()) / np.dot(x, x))
        fitted = fitted + slope * x
        pred = pred + slope * (hi_post - hi_pre.mean())
    p = len(means) + int(use_hi)
    resid = pre - fitted
    cv = (100 * np.sqrt((resid ** 2).sum() / max(len(pre) - p, 1)) / pre.mean()) if pre.mean() > 0 else np.nan
    return pred.clip(lower=0), float(cv), use_hi


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
    if post_days:
        baseline, fit_cv, used_hi = adjusted_baseline(pre, post.index, detail, pattern)
        base_day = float(baseline.mean())
    else:
        baseline, fit_cv, used_hi, base_day = pd.Series(dtype=float), np.nan, False, np.nan
    drop = 1 - post_day / base_day if post_days and base_day > 0 else np.nan
    comfort_ok = not (pd.notna(c_pre) and pd.notna(c_post) and c_post < c_pre - COMFORT_WORSE_PTS)
    if post_days < MIN_POST_DAYS:
        verdict = "Inconclusive"
    elif len(last_window):
        verdict = "Reopened"
    elif pd.notna(drop) and drop >= DROP_TO_VERIFY and comfort_ok:
        verdict = "Verified"
    else:
        verdict = "Not verified"
    saved_kwh = (max(0.0, float(baseline.sum() - post.sum()))
                 if verdict == "Verified" and pattern not in ("warm_running", "supply_group_off") else 0.0)
    return {"verdict": verdict, "fix_date": fix, "pre_start": pre_start, "post_end": post_end, "post_days": post_days,
            "pre_per_day": pre_day, "post_per_day": post_day, "baseline_per_day": base_day, "baseline_series": baseline,
            "baseline_cv_rmse_pct": fit_cv, "weather_adjusted": used_hi,
            "drop_pct": 100 * drop if pd.notna(drop) else np.nan,
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
