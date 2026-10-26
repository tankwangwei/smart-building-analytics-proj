"""Per-unit forecast for the digital twin's Forecast mode (plan §10 A). Streamlit-free."""
import pandas as pd


def baseline_shares(detail):
    """Each unit's share of its floor's baseline load for each hour of the week, from `expected_kW`, which the analysis
    learned from the training year only (Jul 2018 – Jun 2019). Returns a DataFrame indexed by hour-of-week 0..167 with
    one column per unit; every row sums to 1 (equal shares when the baseline is zero)."""
    d = detail[["Date", "unit", "expected_kW"]].dropna()
    how = d["Date"].dt.dayofweek * 24 + d["Date"].dt.hour
    table = d.groupby([how.rename("how"), "unit"])["expected_kW"].mean().unstack().reindex(range(168))
    table = table.fillna(0.0)
    total = table.sum(axis=1)
    shares = table.div(total.where(total > 1e-9), axis=0)
    return shares.fillna(1 / table.shape[1])


def unit_forecasts(detail, floor_forecast, slots):
    """{unit: kW array on `slots`} = floor forecast (hourly, repeated over its four 15-minute slots) × the unit's baseline
    share for that hour of the week. The units sum to the floor forecast."""
    shares = baseline_shares(detail)
    hourly = floor_forecast.reindex(slots.floor("h")).to_numpy()
    how = (slots.dayofweek * 24 + slots.hour).to_numpy()
    return {u: pd.Series(hourly * shares[u].to_numpy()[how], index=slots) for u in shares.columns}


def interval_arrays(fc_series, lower, upper, slots):
    """Hourly 80% interval bounds repeated over 15-minute slots."""
    return (lower.reindex(slots.floor("h")).to_numpy(), upper.reindex(slots.floor("h")).to_numpy())
