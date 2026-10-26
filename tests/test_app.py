"""Headless smoke tests: run the whole app in several views and timeframes and assert nothing raises."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")

# (view, timeframe mode, date)  - weekends, a holiday and a week have broken things before
CASES = [
    ("all", "Jul–Dec", None),
    (7, "Jul–Dec", None),
    ("all", "Day", "2019-07-27"),   # Saturday
    ("all", "Day", "2019-07-28"),   # Sunday
    (7, "Day", "2019-10-23"),       # holiday (Chulalongkorn Day)
    ("all", "Week", "2019-10-14"),  # week 14-20 Oct
]


def make(view, mode, date):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.selectbox[0].set_value(view)
    at.session_state["tf_mode"] = mode
    if date:
        import datetime as dt
        at.session_state["tf_date"] = dt.date.fromisoformat(date)
    at.run()
    return at


@pytest.mark.parametrize("view,mode,date", CASES)
def test_app_runs(view, mode, date):
    at = make(view, mode, date)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("mode", ["Forecast next day", "What-if"])
@pytest.mark.parametrize("view,tf,date", [("all", "Jul–Dec", None), (7, "Day", "2019-12-31"), ("all", "Day", "2019-10-22")])
def test_twin_modes_run(view, tf, date, mode):
    """Forecast mode includes the day after the data (1 Jan 2020, no actuals) and a day before a holiday."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.selectbox[0].set_value(view)
    at.session_state["tf_mode"] = tf
    if date:
        import datetime as dt
        at.session_state["tf_date"] = dt.date.fromisoformat(date)
    at.session_state["twin_mode"] = mode
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_asset_card_and_sim_settings_shared():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    btn = next(b for b in at.button if b.key and b.key.startswith("card_priority"))
    btn.click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.session_state["sim_setpoint"] = 27.0
    at.session_state["twin_mode"] = "What-if"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    twin_btn = next(b for b in at.button if b.key == "twin_card")
    twin_btn.click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
