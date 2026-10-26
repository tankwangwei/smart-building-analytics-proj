import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Tests (including the headless app) use a throw-away store, never the real dashboard/fm.db
TEST_DB = Path(tempfile.mkdtemp(prefix="fm_test_")) / "fm.db"
os.environ["FM_DB"] = str(TEST_DB)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ac_analysis_outputs"
sys.path.insert(0, str(ROOT / "dashboard"))

FLOORS = sorted(int(p.name.split("_")[1]) for p in OUT.glob("floor_*") if (p / "anomaly_15min_detail.csv").exists())
OPEN_DEFAULT = ((0, 1, 2, 3, 4), 7 * 60 + 30, 17 * 60 + 30, True)


def to_bool(series):
    return series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False,
                                               "1.0": True, "0.0": False})


def read_floor(floor):
    d = pd.read_csv(OUT / f"floor_{floor}" / "anomaly_15min_detail.csv", parse_dates=["Date"])
    for c in ["eligible", "hot_day", "has_interpolation"]:
        d[c] = to_bool(d[c]).fillna(False).astype(bool)
    d["schedule_on"] = to_bool(d["schedule_on"])
    th = pd.read_csv(OUT / f"floor_{floor}" / "anomaly_unit_thresholds.csv", index_col=0)
    return d, th


@pytest.fixture(scope="session")
def flagged_floors():
    """{floor: flagged 15-minute detail} with the dashboard's default thresholds and opening hours."""
    from fm_core import flag_detail
    out = {}
    for f in FLOORS:
        d, th = read_floor(f)
        out[f] = flag_detail(d, th, OPEN_DEFAULT, 1.5, 1.5, 0.5, 8)
    return out
