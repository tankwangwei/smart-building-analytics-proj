"""Phase 1: triage (magnitude x confidence / effort)."""
import numpy as np
import pandas as pd

from fm_core import PATTERNS, build_events


def all_events(flagged_floors):
    return pd.concat([build_events(d) for d in flagged_floors.values()], ignore_index=True)


def test_no_nan_scores(flagged_floors):
    ev = all_events(flagged_floors)
    assert len(ev) > 50
    for col in ["excess_THB", "confidence", "triage_score", "effort_factor"]:
        assert ev[col].notna().all(), col
    assert ev["confidence"].between(0, 100).all()
    assert (ev["triage_score"] >= 0).all()


def test_confidence_spread_across_bands(flagged_floors):
    ev = all_events(flagged_floors)
    bands = ev["confidence_band"].value_counts()
    assert len(bands) >= 2, bands
    assert bands.max() / len(ev) < 0.95, bands


def test_triage_formula(flagged_floors):
    ev = all_events(flagged_floors)
    expected = ev["excess_THB"] * ev["confidence"] / 100 / ev["effort_factor"]
    assert np.allclose(ev["triage_score"], expected)
    assert set(ev["pattern"]) <= set(PATTERNS)


def test_events_sorted_by_triage(flagged_floors):
    ev = build_events(next(iter(flagged_floors.values())))
    assert ev["triage_score"].is_monotonic_decreasing
