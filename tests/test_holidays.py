"""Future holidays come from the `holidays` package without changing the checked study-period list."""
import builtins
import datetime as dt

import thai_holidays as th


def test_study_period_uses_only_the_checked_list():
    last = max(th.CHECKED_HOLIDAYS)
    study = {d: n for d, n in th.HOLIDAYS.items() if d <= last}
    assert study == th.CHECKED_HOLIDAYS


def test_future_years_include_holidays_but_not_working_days():
    future = th.future_holidays()
    assert dt.date(2027, 1, 1) in future                # New Year's Day
    assert dt.date(2027, 5, 20) in future               # Visakha Bucha (lunar)
    assert dt.date(2027, 5, 3) not in future            # Labour Day substitution
    assert dt.date(2027, 1, 9) not in future            # Children's Day
    assert min(future) > max(th.CHECKED_HOLIDAYS)
    assert th.closure_name(dt.date(2027, 1, 1)) == "New Year's Day"


def test_missing_package_falls_back_to_the_checked_list(monkeypatch):
    real_import = builtins.__import__

    def no_holidays(name, *args, **kwargs):
        if name == "holidays":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_holidays)
    assert th.future_holidays() == {}
