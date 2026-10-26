"""Storage for the FM workflow: incident status, work orders, close-outs, exceptions, audit log, maintenance log.

One interface, two backends. SqliteStore (default, dashboard/fm.db, or the path in $FM_DB) needs nothing installed.
SheetsStore is used when st.secrets has a "gsheets" section (see docs/FM_WORKFLOW_PLAN.md §12).
Everything here is Streamlit-free except get_store(), which reads st.secrets and caches the store.
"""
import datetime as dt
import json
import os
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(os.environ.get("FM_DB", Path(__file__).resolve().parent / "fm.db"))

TABLES = {
    "incident_state": (["incident_key", "status", "status_reason", "updated_at", "updated_by", "simulated"], "incident_key"),
    "work_orders": (["wo_id", "incident_key", "asset_id", "created_at", "status", "assignee", "priority", "due_date",
                     "slot_start", "slot_end", "symptom", "checklist_json", "evidence_json", "simulated"], "wo_id"),
    "closeouts": (["wo_id", "outcome", "root_cause", "action_taken", "date_fixed", "technician", "notes", "closed_at",
                   "simulated"], "wo_id"),
    "exceptions": (["exc_id", "asset_id", "weekdays", "time_from", "time_to", "date_from", "date_to", "reason",
                    "source_wo_id", "active", "simulated"], "exc_id"),
    "audit_log": (["ts", "actor", "entity", "entity_id", "change"], None),
    "maintenance_log": (["log_id", "asset_id", "date", "type", "fault_type", "severity", "downtime_h", "cost_THB", "parts",
                         "technician", "notes", "source", "wo_id", "simulated"], "log_id"),
}
ID_PREFIX = {"work_orders": "WO-", "exceptions": "EXC-", "maintenance_log": "ML-"}


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


class Store:
    """Backend-independent behaviour on top of five primitives: read_all, write_row, delete_row, init, clear."""

    name = "base"

    def table(self, table):
        cols = TABLES[table][0]
        df = self.read_all(table)
        return df.reindex(columns=cols) if len(df) else pd.DataFrame(columns=cols)

    def get(self, table, key):
        pk = TABLES[table][1]
        df = self.table(table)
        row = df[df[pk].astype(str) == str(key)]
        return row.iloc[0].to_dict() if len(row) else None

    def upsert(self, table, row, actor="user", audit=True, change=None):
        cols, pk = TABLES[table]
        clean = {c: _plain(row.get(c)) for c in cols}
        old = self.get(table, clean[pk]) if pk else None
        self.write_row(table, clean)
        if audit and table != "audit_log":
            self.log(actor, table, clean.get(pk), change or ("updated" if old else "created"))
        return clean

    def delete(self, table, key, actor="user"):
        self.delete_row(table, key)
        self.log(actor, table, key, "deleted")

    def log(self, actor, entity, entity_id, change):
        self.write_row("audit_log", {"ts": now(), "actor": actor, "entity": entity, "entity_id": str(entity_id),
                                     "change": change})

    def next_id(self, table, width=4):
        pk = TABLES[table][1]
        prefix = ID_PREFIX[table]
        ids = self.table(table)[pk].dropna().astype(str)
        nums = [int(i[len(prefix):]) for i in ids if i.startswith(prefix) and i[len(prefix):].isdigit()]
        return f"{prefix}{(max(nums) + 1 if nums else 1):0{width}d}"

    def is_empty(self):
        return all(len(self.table(t)) == 0 for t in TABLES if t != "audit_log")


def _plain(v):
    """Values the backends can store: str/int/float/bool/None; everything else becomes JSON text or ISO string."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v)
    if hasattr(v, "item"):
        return v.item()
    return v


class SqliteStore(Store):
    name = "SQLite"

    def __init__(self, path=DB_PATH):
        self.path = str(path)
        self.init()

    def _conn(self):
        return sqlite3.connect(self.path)

    def init(self):
        with self._conn() as c:
            for table, (cols, pk) in TABLES.items():
                defs = ", ".join(f'"{col}"' + (" PRIMARY KEY" if col == pk else "") for col in cols)
                c.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({defs})')

    def read_all(self, table):
        with self._conn() as c:
            return pd.read_sql_query(f'SELECT * FROM "{table}"', c)

    def write_row(self, table, row):
        cols = TABLES[table][0]
        with self._conn() as c:
            c.execute(f'INSERT OR REPLACE INTO "{table}" ({", ".join(chr(34) + k + chr(34) for k in cols)}) '
                      f'VALUES ({", ".join("?" * len(cols))})', [row.get(k) for k in cols])

    def delete_row(self, table, key):
        pk = TABLES[table][1]
        with self._conn() as c:
            c.execute(f'DELETE FROM "{table}" WHERE "{pk}" = ?', [str(key)])

    def clear(self):
        with self._conn() as c:
            for table in TABLES:
                c.execute(f'DELETE FROM "{table}"')


class SheetsStore(Store):
    """Google Sheets backend (gspread). One worksheet per table with a header row; fine for dozens of rows, not built for
    many simultaneous editors. Not covered by the automated tests (needs credentials): see §12 of the plan."""
    name = "Google Sheets"

    def __init__(self, secrets):
        import gspread  # imported here so SQLite users do not need it
        info = {k: v for k, v in dict(secrets).items() if k != "spreadsheet_url"}
        self.book = gspread.service_account_from_dict(info).open_by_url(secrets["spreadsheet_url"])
        self.init()

    def init(self):
        existing = {w.title for w in self.book.worksheets()}
        for table, (cols, _) in TABLES.items():
            if table not in existing:
                ws = self.book.add_worksheet(title=table, rows=200, cols=len(cols))
                ws.update([cols], "A1")
            else:
                ws = self.book.worksheet(table)
                if ws.row_values(1) != cols:
                    ws.update([cols], "A1")

    def read_all(self, table):
        rows = self.book.worksheet(table).get_all_records(default_blank=None)
        return pd.DataFrame(rows)

    def write_row(self, table, row):
        ws = self.book.worksheet(table)
        cols, pk = TABLES[table]
        values = [("" if row.get(c) is None else row.get(c)) for c in cols]
        if pk:
            keys = ws.col_values(cols.index(pk) + 1)
            if str(row[pk]) in keys[1:]:
                ws.update([values], f"A{keys.index(str(row[pk])) + 1}")
                return
        ws.append_row(values, value_input_option="USER_ENTERED")

    def delete_row(self, table, key):
        ws = self.book.worksheet(table)
        cols, pk = TABLES[table]
        keys = ws.col_values(cols.index(pk) + 1)
        if str(key) in keys[1:]:
            ws.delete_rows(keys.index(str(key)) + 1)

    def clear(self):
        for table, (cols, _) in TABLES.items():
            ws = self.book.worksheet(table)
            ws.clear()
            ws.update([cols], "A1")


def get_store():
    """SheetsStore when st.secrets has a [gsheets] section, otherwise SqliteStore. Cached per process."""
    import streamlit as st
    return _cached_store(_secrets_key(st))


def _secrets_key(st):
    try:
        return json.dumps(dict(st.secrets["gsheets"]), sort_keys=True, default=str) if "gsheets" in st.secrets else ""
    except Exception:  # no secrets file at all
        return ""


def _make(secrets_json):
    if secrets_json:
        return SheetsStore(json.loads(secrets_json))
    return SqliteStore()


_STORES = {}


def _cached_store(key):
    if key not in _STORES:
        _STORES[key] = _make(key)
    return _STORES[key]
