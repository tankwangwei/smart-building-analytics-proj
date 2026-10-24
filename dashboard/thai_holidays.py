"""Thai public holidays (government offices) covering the CU-BEMS study period, Jul 2018 – Jan 2020.

Substitution days (when a holiday falls on a weekend, the next working day is off) are included because offices close
on those days. Sources: Thai Cabinet holiday announcements for 2018–2020; cross-checked against the dataset, where
these weekdays show near-zero AC use. Extend the list before using the dashboard on other years.
"""
import datetime as dt

HOLIDAYS = {
    # 2018
    dt.date(2018, 7, 27): "Asahna Bucha Day",
    dt.date(2018, 7, 30): "King's Birthday (substitution for Sat 28 Jul)",
    dt.date(2018, 8, 13): "Queen Mother's Birthday (substitution for Sun 12 Aug)",
    dt.date(2018, 10, 15): "King Bhumibol Memorial Day (substitution for Sat 13 Oct)",
    dt.date(2018, 10, 23): "Chulalongkorn Day",
    dt.date(2018, 12, 5): "King Bhumibol's Birthday / Father's Day",
    dt.date(2018, 12, 10): "Constitution Day",
    dt.date(2018, 12, 31): "New Year's Eve",
    # 2019
    dt.date(2019, 1, 1): "New Year's Day",
    dt.date(2019, 2, 19): "Makha Bucha Day",
    dt.date(2019, 4, 8): "Chakri Day (substitution for Sat 6 Apr)",
    dt.date(2019, 4, 12): "Songkran (special holiday)",
    dt.date(2019, 4, 15): "Songkran",
    dt.date(2019, 4, 16): "Songkran (substitution)",
    dt.date(2019, 5, 6): "Coronation Day (special holiday)",
    dt.date(2019, 5, 20): "Visakha Bucha Day (substitution for Sat 18 May)",
    dt.date(2019, 6, 3): "Queen Suthida's Birthday",
    dt.date(2019, 7, 16): "Asahna Bucha Day",
    dt.date(2019, 7, 17): "Buddhist Lent Day",
    dt.date(2019, 7, 29): "King's Birthday (substitution for Sun 28 Jul)",
    dt.date(2019, 8, 12): "Queen Mother's Birthday",
    dt.date(2019, 10, 14): "King Bhumibol Memorial Day (substitution for Sun 13 Oct)",
    dt.date(2019, 10, 23): "Chulalongkorn Day",
    dt.date(2019, 12, 5): "King Bhumibol's Birthday / Father's Day",
    dt.date(2019, 12, 10): "Constitution Day",
    dt.date(2019, 12, 31): "New Year's Eve",
    # 2020
    dt.date(2020, 1, 1): "New Year's Day",
}

# Weekdays the whole building was closed in the data but that are not official public holidays. Treated as closures,
# labelled separately so they are not mistaken for gazetted holidays.
OBSERVED_CLOSURES = {
    dt.date(2019, 12, 30): "Building closed (seen in the data; likely a university bridge day before New Year)",
}


def closure_name(day):
    """Holiday or closure name for a date, or None."""
    return HOLIDAYS.get(day) or OBSERVED_CLOSURES.get(day)
