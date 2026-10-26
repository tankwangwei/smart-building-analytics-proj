"""Phase 2: the Brick model gives the same 16 units and peer groups as the old regex logic; the JSON fallback agrees."""
import re

import pandas as pd

from brick_model import BrickModel
from conftest import FLOORS, OUT


def regex_units():
    cols = []
    for f in FLOORS:
        d = pd.read_csv(OUT / f"floor_{f}" / "anomaly_15min_detail.csv", usecols=["unit"])
        cols += list(dict.fromkeys(d["unit"]))
    return sorted(cols)


def regex_peers(cols):
    out = {}
    for c in cols:
        out.setdefault(c.split("_", 1)[1].replace("(kW)", ""), []).append(c)
    return {k: sorted(v) for k, v in out.items()}


def test_same_units_as_the_data():
    m = BrickModel.load()
    assert sorted(m.units()["power_column"]) == regex_units()
    assert len(m.units()) == 16


def test_same_peer_groups_as_regex():
    m = BrickModel.load()
    assert m.peer_groups() == regex_peers(regex_units())


def test_supply_groups_are_per_floor_and_assumed():
    m = BrickModel.load()
    groups = m.supply_groups()
    assert len(groups) == len(FLOORS)
    for loop, cols in groups.items():
        floor = int(re.search(r"Floor(\d+)", loop)[1])
        assert all(c.startswith(f"F{floor}_") for c in cols)
    assert (m.units()["supply_group"].str.contains("assumed")).all()


def test_json_fallback_matches_sparql():
    a, b = BrickModel.load(), BrickModel.load(force_json=True)
    assert a.backend != b.backend
    cols = ["asset_id", "floor", "zone", "position", "power_column", "temp_column", "rh_column", "supply_group"]
    pd.testing.assert_frame_equal(a.units()[cols], b.units()[cols])
    assert a.peer_groups() == b.peer_groups() and a.supply_groups() == b.supply_groups()


def test_asset_lookup():
    m = BrickModel.load()
    a = m.asset_for_column("F7_z1_AC4(kW)")
    assert a["asset_id"] == "F7_Z1_AC4" and a["floor"] == 7 and a["zone"] == 1
    assert m.asset_for_column("nope") is None
