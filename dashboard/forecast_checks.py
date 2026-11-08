"""Forecast accuracy checks (Streamlit-free): error metrics, ASHRAE Guideline 14, error breakdowns, intervals."""
import numpy as np
import pandas as pd

# Guideline 14 gives hourly (NMBE ±10 %, CV(RMSE) ≤ 30 %) and monthly (±5 %, ≤ 15 %) criteria. It has none for daily
# totals, so the hourly limits are used there too: the lenient reading.
NMBE_LIMIT, CV_LIMIT = 10.0, 30.0


def error_metrics(actual, forecast):
    err = forecast - actual
    rmse = np.sqrt((err ** 2).mean())
    return {"MAE_kW": err.abs().mean(), "RMSE_kW": rmse, "NMBE_pct": 100 * err.mean() / actual.mean(),
            "CV_RMSE_pct": 100 * rmse / actual.mean()}


def ashrae_rows(actual, forecast):
    """Hourly and daily-total accuracy against the Guideline 14 limits."""
    ok = actual.notna() & forecast.notna()
    a, f = actual[ok], forecast[ok]
    rows = []
    for label, aa, ff in [("Hourly", a, f), ("Daily totals", a.resample("D").sum(), f.resample("D").sum())]:
        m = error_metrics(aa, ff)
        rows.append({"Resolution": label, **m, "NMBE ok (±10%)": abs(m["NMBE_pct"]) <= NMBE_LIMIT,
                     "CV(RMSE) ok (≤30%)": m["CV_RMSE_pct"] <= CV_LIMIT})
    return pd.DataFrame(rows)


def breakdowns(actual, forecast):
    """MAE and bias by hour, weekday and month."""
    ok = actual.notna() & forecast.notna()
    err = (forecast - actual)[ok]
    out = {}
    for name, key in [("Hour", err.index.hour), ("Weekday", err.index.strftime("%a")), ("Month", err.index.strftime("%Y-%m"))]:
        g = pd.DataFrame({"abs": err.abs(), "err": err}).groupby(key)
        out[name] = pd.DataFrame({"MAE kW": g["abs"].mean(), "Bias kW": g["err"].mean()})
    out["Weekday"] = out["Weekday"].reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    return out


def interval_bounds(cal_actual, cal_forecast, forecast, actual=None, level=0.8):
    """Empirical error quantiles by hour of day, learned from the calibration pair (the Apr–Jun 2019 validation
    predictions, made by methods fitted on Jul 2018 – Mar 2019, so the test period is never used), applied to `forecast`.
    Returns (lower, upper) series for every hour of `forecast`, and the share of `actual` inside the band."""
    ok = cal_actual.notna() & cal_forecast.notna()
    cal = (cal_actual - cal_forecast)[ok]
    lo = cal.groupby(cal.index.hour).quantile((1 - level) / 2)
    hi = cal.groupby(cal.index.hour).quantile(1 - (1 - level) / 2)
    lower = (forecast + forecast.index.hour.map(lo)).clip(lower=0)
    upper = forecast + forecast.index.hour.map(hi)
    if actual is None:
        return lower, upper, float("nan")
    test = (actual - forecast)[actual.notna() & forecast.notna()]
    inside = (test >= test.index.hour.map(lo)) & (test <= test.index.hour.map(hi))
    return lower, upper, float(inside.mean()) if len(test) else float("nan")
