"""Maintenance log seed, health indicators, fault prediction and remaining useful life (plan §11). Streamlit-free.

Health indicators and the degradation trend come from REAL sensor data. The fault-risk model and the likely-cause
ranking are trained on SIMULATED maintenance entries: they demonstrate the method and are not evidence."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fm_asset import running_open_weekday
from fm_core import INTERVAL_HOURS

FAULT_TYPES = ["dirty filter", "refrigerant leak", "compressor fault", "fan motor", "thermostat/controls", "electrical trip",
               "sensor fault", "other"]
LOG_TYPES = ["fault repair", "preventive service", "inspection", "replacement"]
SERVICE_TYPES = {"preventive service", "replacement", "fault repair"}
DEMO_LABEL = "demo: trained on simulated maintenance history"
SERVICE_INTERVAL_DAYS, SERVICE_INTERVAL_RUN_HOURS = 90, 500   # assumptions until facilities confirm them
DEGRADATION_THRESHOLD = 0.15                                    # +15 % kW per °C above the post-service level
PATTERN_FAULTS = {"closed_running": ["thermostat/controls", "other"], "off_schedule": ["thermostat/controls", "other"],
                  "high_while_open": ["dirty filter", "thermostat/controls", "refrigerant leak", "sensor fault"],
                  "warm_running": ["dirty filter", "refrigerant leak", "compressor fault", "fan motor"],
                  "supply_group_off": ["electrical trip", "fan motor", "other"],
                  "supply_group_high": ["fan motor", "other"]}
HEALTH_WEIGHTS = {"efficiency": .30, "drift": .25, "warm": .20, "peer": .15, "starts": .10}


# ---------------------------------------------------------------- simulated maintenance history
def seed_maintenance(store, units, warm_assets=(), rising_assets=(), seed=42, start="2018-07-01", end="2019-12-31"):
    """An 18-month simulated history per unit (fixed seed, labelled Simulated): preventive service about every 3 months,
    faults consistent with the data (warm-running units get filter/refrigerant faults, rising kW per °C gets coil
    cleaning), plus a few random faults. Adds rows only if no 'simulated history' entries exist yet."""
    log = store.table("maintenance_log")
    if len(log) and (log["source"] == "simulated history").any():
        return 0
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    n = 0

    def add(asset, date, typ, fault, severity, downtime, cost, notes):
        nonlocal n
        store.upsert("maintenance_log", {
            "log_id": store.next_id("maintenance_log"), "asset_id": asset, "date": str(pd.Timestamp(date).date()), "type": typ,
            "fault_type": fault, "severity": severity, "downtime_h": downtime, "cost_THB": cost, "parts": "",
            "technician": "(simulated)", "notes": notes, "source": "simulated history", "wo_id": None, "simulated": 1},
            "seed", audit=False)
        n += 1

    for asset in sorted(units["asset_id"]):
        day = start + pd.Timedelta(days=int(rng.integers(10, 60)))
        while day < end:
            add(asset, day, "preventive service", None, "Low", float(rng.integers(1, 4)), int(rng.integers(800, 2500)), "filter and coil check")
            day += pd.Timedelta(days=int(rng.normal(90, 12)))
        if asset in warm_assets:
            for fault in ["dirty filter", "refrigerant leak"]:
                add(asset, start + pd.Timedelta(days=int(rng.integers(120, 500))), "fault repair", fault, "Medium",
                    float(rng.integers(2, 8)), int(rng.integers(2500, 9000)), "zone stayed warm while running")
        if asset in rising_assets:
            add(asset, start + pd.Timedelta(days=int(rng.integers(200, 520))), "fault repair", "dirty filter", "Low", 2.0,
                int(rng.integers(1500, 3500)), "coil cleaning after efficiency drift")
        for _ in range(rng.poisson(2.0)):
            fault = FAULT_TYPES[int(rng.integers(0, len(FAULT_TYPES)))]
            # most faults fall in the period the sensors cover (Jul-Dec 2019), so the risk model has labels to learn from
            when = (pd.Timestamp("2019-07-08") + pd.Timedelta(days=int(rng.integers(0, 160))) if rng.random() < .6
                    else start + pd.Timedelta(days=int(rng.integers(30, 540))))
            add(asset, when, "fault repair", fault,
                ["Low", "Medium", "High"][int(rng.integers(0, 3))], float(rng.integers(1, 12)), int(rng.integers(1500, 15000)),
                "random fault (simulated)")
    return n


# ---------------------------------------------------------------- weekly health indicators (real data)
def weekly_indicators(d, on_kw=0.5):
    """Weekly run hours, starts, efficiency index (kW per outdoor °C on weekday running hours), residual drift,
    warm-while-running hours, mean kW. `d` = flagged detail of one unit."""
    d = d.sort_values("Date")
    week = d["Date"].dt.to_period("W").dt.start_time
    running = d["actual_kW"].gt(on_kw) & d["eligible"]
    starts = running & ~running.shift(fill_value=False)
    r = running_open_weekday(d, on_kw)
    rw = r["Date"].dt.to_period("W").dt.start_time
    out = pd.DataFrame({
        "run_hours": running.groupby(week).sum() * INTERVAL_HOURS,
        "starts": starts.groupby(week).sum(),
        "warm_hours": (running & d["indoor_temp_C"].gt(26)).groupby(week).sum() * INTERVAL_HOURS,
        "mean_kW": d[d["eligible"]].groupby(week)["actual_kW"].mean()})
    out["efficiency"] = (r.groupby(rw)["actual_kW"].mean() / r.groupby(rw)["outdoor_temp_C"].mean().clip(lower=1))
    out["drift"] = r.groupby(rw)["residual_kW"].mean()
    return out.dropna(subset=["mean_kW"])


def add_peer_gap(ind, peers_mean):
    """peers_mean: weekly mean kW of the peer median (Series indexed like ind)."""
    ind = ind.copy()
    ind["peer_gap"] = 100 * (ind["mean_kW"] / peers_mean.reindex(ind.index).where(lambda s: s > 0.5) - 1)
    return ind


def health_score(ind, baseline_weeks=8):
    """0-100 (100 = as normal for this unit): weighted positive z-scores of the indicators against the unit's own first
    `baseline_weeks` running weeks. Higher efficiency index, drift, warm hours, peer gap and starts all mean worse."""
    ind = ind.copy()
    base = ind[ind["run_hours"] > 5].head(baseline_weeks)
    cols = {"efficiency": "efficiency", "drift": "drift", "warm": "warm_hours", "peer": "peer_gap", "starts": "starts"}
    bad = pd.Series(0.0, index=ind.index)
    for key, col in cols.items():
        if col not in ind or base[col].dropna().empty:
            continue
        sd = max(base[col].std(ddof=0), 1e-6 + 0.05 * abs(base[col].mean()))
        z = ((ind[col] - base[col].mean()) / sd).clip(lower=0, upper=4).fillna(0)
        bad += HEALTH_WEIGHTS[key] * z
    return (100 - 25 * bad).clip(0, 100).rename("health")


# ---------------------------------------------------------------- fault risk and likely cause (simulated labels)
def risk_features(ind):
    f = ind[["efficiency", "drift", "warm_hours", "peer_gap", "starts", "run_hours"]].copy()
    f["eff_trend"] = f["efficiency"].diff(4)
    return f.fillna(0).assign(_valid=ind["run_hours"].gt(5))


def fault_risk_model(indicators, store, train_frac=0.7, horizon_days=30):
    """Logistic regression on weekly features; label = a fault entry in the log within the next 30 days. Time split:
    train on earlier weeks, test on later. Returns (model, scaler stats, metrics, risk table of the latest week).
    Trained on simulated entries unless a real log is imported."""
    log = store.table("maintenance_log")
    faults = log[log["type"].eq("fault repair")].assign(date=lambda x: pd.to_datetime(x["date"]))
    rows = []
    for asset, ind in indicators.items():
        f = risk_features(ind)
        f["asset"] = asset
        dates = faults.loc[faults["asset_id"].eq(asset), "date"].to_numpy()
        f["y"] = [int(((dates > w.to_datetime64()) & (dates <= (w + pd.Timedelta(days=horizon_days)).to_datetime64())).any())
                  for w in f.index]
        rows.append(f)
    data = pd.concat(rows)
    weeks = sorted(data.index.unique())
    cut = weeks[int(len(weeks) * train_frac)]
    horizon_weeks = int(np.ceil(horizon_days / 7))
    train = data[data.index < cut - pd.Timedelta(weeks=horizon_weeks)]   # labels of train rows end before the test period
    test = data[data.index >= cut]
    feats = [c for c in data.columns if c not in ("asset", "y", "_valid")]
    mu, sd = train[feats].mean(), train[feats].std(ddof=0).replace(0, 1)
    metrics = {"train_weeks": len(train.index.unique()), "test_weeks": len(test.index.unique()), "auc": np.nan,
               "precision_top3": np.nan, "base_rate": float(test["y"].mean()) if len(test) else np.nan, "label": DEMO_LABEL}
    if train["y"].nunique() < 2:
        return None, (mu, sd, feats), metrics, pd.DataFrame()
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit((train[feats] - mu) / sd, train["y"])
    if test["y"].nunique() == 2:
        proba = model.predict_proba((test[feats] - mu) / sd)[:, 1]
        metrics["auc"] = float(roc_auc_score(test["y"], proba))
        t = test.assign(p=proba)
        top = t.sort_values("p", ascending=False).groupby(level=0).head(3)
        metrics["precision_top3"] = float(top["y"].mean())
    latest = data[data["_valid"]].groupby("asset").tail(1)  # the latest week the unit actually ran
    latest = latest.assign(risk=model.predict_proba((latest[feats] - mu) / sd)[:, 1])
    table = latest.set_index("asset")[["risk"]]
    table["band"] = pd.cut(table["risk"], [-1, .33, .66, 2], labels=["Low", "Medium", "High"])
    return model, (mu, sd, feats), metrics, table


def likely_causes(store, pattern, asset_id=None, top=3):
    """Naive Bayes style frequency table over the maintenance log with Laplace smoothing: counts of each fault type among
    the faults a pattern can produce, with this unit's own history counted double. {fault type: probability}."""
    allowed = PATTERN_FAULTS.get(pattern, FAULT_TYPES)
    log = store.table("maintenance_log")
    log = log[log["type"].eq("fault repair") & log["fault_type"].isin(allowed)]
    counts = pd.Series({f: 1.0 for f in allowed})
    for r in log.itertuples():
        counts[r.fault_type] += 2.0 if asset_id and r.asset_id == asset_id else 1.0
    p = (counts / counts.sum()).sort_values(ascending=False).head(top)
    return {k: round(float(v), 3) for k, v in p.items()}


# ---------------------------------------------------------------- remaining useful life
def last_service(store, asset_id, default):
    log = store.table("maintenance_log")
    log = log[log["asset_id"].eq(asset_id) & log["type"].isin(SERVICE_TYPES)]
    return pd.to_datetime(log["date"]).max() if len(log) else pd.Timestamp(default)


def rul_usage(ind, last_service_date, as_of):
    """Days until the service interval (90 days or 500 run hours since the last service) is reached; 0 if overdue."""
    since = ind[ind.index >= last_service_date.to_period("W").start_time]
    run_h = float(since["run_hours"].sum())
    days_since = (as_of - last_service_date).days
    per_day = run_h / max(days_since, 1)
    by_days = SERVICE_INTERVAL_DAYS - days_since
    by_hours = (SERVICE_INTERVAL_RUN_HOURS - run_h) / per_day if per_day > 0 else np.inf
    return max(0, int(round(min(by_days, by_hours)))), run_h


def rul_degradation(ind, last_service_date, as_of):
    """Robust (Theil–Sen) trend of the efficiency index since the last service; RUL = days until it crosses
    +15 % above its post-service level. Returns dict(status, rul_days, low, high, slope_pct_per_week, series...)."""
    since = ind[(ind.index >= last_service_date.to_period("W").start_time) & ind["efficiency"].notna()]["efficiency"]
    if len(since) < 6:
        # serviced recently: judge the trend over the last 10 weeks with data instead of the few weeks since the service
        since = ind[ind["efficiency"].notna()]["efficiency"].tail(10)
    if len(since) < 6:
        return {"status": "not enough data", "rul_days": np.nan, "low": np.nan, "high": np.nan, "series": since}
    x = np.arange(len(since), dtype=float)
    slope, intercept, lo, hi = stats.theilslopes(since.values, x)
    tau, p = stats.kendalltau(x, since.values)
    level = float(np.median(since.values[:4]))
    if slope <= 0 or p > 0.10:
        return {"status": "no degradation detected", "rul_days": np.nan, "low": np.nan, "high": np.nan, "series": since,
                "slope_pct_per_week": 100 * slope / level, "level": level, "threshold": level * (1 + DEGRADATION_THRESHOLD),
                "intercept": intercept, "slope": slope}
    threshold = level * (1 + DEGRADATION_THRESHOLD)
    now_x = len(since) - 1
    current = intercept + slope * now_x
    weeks = (threshold - current) / slope
    spread = float(np.std(since.values - (intercept + slope * x)))
    # interval from the residual spread: how many weeks the crossing moves if the level is off by +-1.28 sigma
    lo_w, hi_w = (threshold - current - 1.28 * spread) / slope, (threshold - current + 1.28 * spread) / slope
    to_days = lambda w: max(0, int(round(7 * w)))
    return {"status": "degrading", "rul_days": to_days(weeks), "low": to_days(min(lo_w, hi_w)), "high": to_days(max(lo_w, hi_w)),
            "series": since, "slope_pct_per_week": 100 * slope / level, "level": level, "threshold": threshold,
            "intercept": intercept, "slope": slope}


def fleet_health(units, details, store, as_of, risk_table, peer_weekly):
    """One row per unit for the Maintenance tab. details = {floor: flagged detail}."""
    rows, ind_all = [], {}
    for u in units.itertuples():
        d = details[int(u.floor)]
        d = d[d["unit"].eq(u.power_column)]
        ind = add_peer_gap(weekly_indicators(d), peer_weekly[u.position])
        ind["health"] = health_score(ind)
        ind_all[u.asset_id] = ind
        service = last_service(store, u.asset_id, d["Date"].min())
        usage_days, run_h = rul_usage(ind, service, as_of)
        deg = rul_degradation(ind, service, as_of)
        candidates = [usage_days] + ([deg["rul_days"]] if np.isfinite(deg["rul_days"]) else [])
        due = min(candidates)
        rows.append({"asset_id": u.asset_id, "label": u.label, "health": float(ind.loc[ind["run_hours"] > 5, "health"].tail(4).mean()),
                     "last_service": service, "run_h_since": run_h, "rul_usage_days": usage_days,
                     "degradation": deg["status"], "rul_deg_days": deg["rul_days"], "rul_low": deg["low"], "rul_high": deg["high"],
                     "slope_pct_per_week": deg.get("slope_pct_per_week", np.nan), "service_due": as_of + pd.Timedelta(days=int(due)),
                     "days_to_service": int(due), "deg": deg})
    table = pd.DataFrame(rows).set_index("asset_id")
    if risk_table is not None and len(risk_table):
        table["risk"] = risk_table["risk"].reindex(table.index)
        table["risk_band"] = risk_table["band"].reindex(table.index).astype(str)
    else:
        table["risk"], table["risk_band"] = np.nan, "n/a"
    return table, ind_all


def peer_weekly(details, peer_groups):
    """{position: weekly mean kW of the median unit at that position across floors}."""
    out = {}
    for pos, cols in peer_groups.items():
        parts = []
        for d in details.values():
            for col in cols:
                g = d[d["unit"].eq(col) & d["eligible"]]
                if len(g):
                    parts.append(g.groupby(g["Date"].dt.to_period("W").dt.start_time)["actual_kW"].mean())
        out[pos] = pd.concat(parts, axis=1).median(axis=1)
    return out


def attach_risk(table, risk_table):
    table = table.copy()
    table["risk"] = risk_table["risk"].reindex(table.index) if len(risk_table) else np.nan
    table["risk_band"] = risk_table["band"].reindex(table.index).astype(str) if len(risk_table) else "n/a"
    return table
