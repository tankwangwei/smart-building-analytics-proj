# Building AC dashboard (Floors 3, 4, 5 and 7)

1. Run `python run_all_floors.py` (about 8 minutes for four floors). It writes `ac_analysis_outputs/floor_<n>/`.
   To run one floor only: `python run_all_floors.py 7`.
2. Install the extra packages once: `pip install -r dashboard/requirements.txt`
3. From the project folder: `python -m streamlit run dashboard/app.py`

**Timeframe** (top right, every tab): Day / Week / Month / Jul–Dec plus a date. Every total, colour, alert and finding is worked out for that timeframe. "Normal" is always learned from Jul 2018 – Jun 2019, and forecast accuracy is always scored on Jul–Dec 2019.

**✦ Ask the data** (round button, bottom-right corner of every tab): a chat assistant for the facilities team, using Google Gemini (`gemini-3.8-flash` by default; set `GEMINI_MODEL` in secrets to change it). The code is in `dashboard/fm_chat.py`.
- **How it answers:** it calls 12 read-only tools over the dashboard's own data and rules, so every figure comes from the analysis. The tools are: overview KPIs, unit ranking, events, unit profile, when waste happens, floor comparison, energy over time, day-ahead forecast, forecast accuracy, holidays and hours, the Smart assistant's findings, and a what-if setpoint/schedule estimate (the same maths as the simulator tab).
- **Scope:** it uses the dashboard's current view (floor, timeframe, opening hours) unless the question names another.
- **API key (free):** create one at [Google AI Studio](https://aistudio.google.com) → API keys. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (git-ignored) and paste the key as `GEMINI_API_KEY`. You can also set the `GEMINI_API_KEY` environment variable, or paste a key in the chat panel for the current browser session only. On Streamlit Cloud, put it under App → Settings → Secrets. Never commit a key.
- **Free tier:** Google limits requests per minute and per day, and one question can take several requests (one per data lookup). Check Google's current terms on whether free-tier prompts are used to improve its products.

**Building opening hours and Thai public holidays** (sidebar → Building opening hours). Defaults: open Mon–Fri 07:30–17:30, closed on public holidays. Those defaults follow the data, since the dataset has no official hours.
- **Holiday list:** `dashboard/thai_holidays.py` has the official holidays and substitution days for Jul 2018 – Jan 2020, plus one closure seen in the data (Mon 30 Dec 2019). Every listed holiday in Jul–Dec 2019 shows almost no AC use.
- **When the building is closed,** normal AC use counts as zero. Any AC running is flagged as after hours, and all its energy counts as waste.
- **Forecast:** a public holiday is a known closure, so demand is forecast as zero. This cuts the building forecast's average error from 13.5 to 8.5 kW per hour.
- **Simulator:** has an "AC off on public holidays" option.
- **Academic calendar:** not used. Chamchuri 5 is an administrative and office building, so semester breaks matter little.

**Smart assistant** (Overview): rule-based checks on the timeframe, written as what to look at next. It flags:
- units off during their usual hours while the building runs (tripped breaker or controller);
- 3 or more units on one floor off together (shared chilled water, air-handling unit or electrical supply);
- units running while their zone stays above 26 °C (low refrigerant, blocked filter or dirty coil);
- 3 or more units on one floor running high together (shared cause);
- frequent on/off cycling;
- waste outside usual schedules;
- peer outliers;
- overcooling;
- missing meter data;
- tomorrow's peak.

These are suggestions to investigate, not diagnoses.

**Overview** is a building control-room view (dark theme from `.streamlit/config.toml`, so launch Streamlit from the project folder): round-icon KPIs; a Floors card with one dot per AC unit coloured by priority in the timeframe (grey = did not run), plus energy and cost; the top alerts; a compact 3D replay of the top alert's day (red zones = flagged); the smart assistant; and the load pattern (the day's hourly load, or the average day). The 3D twin colours zones by anomaly status, AC power vs expected, indoor temperature, lights on after hours (all at the replay time), or priority level for the timeframe. Each analysis tab shows one "So:" box with its key finding.

"Outside usual schedule" energy is recalculated from the 15-minute data using each unit's own usual schedule, so it can differ slightly from the notebook's figure.

The sidebar **View** opens on **Whole building** (Floors 3, 4, 5 and 7 combined); pick a single floor to focus on it. Tabs:
- **Overview**: KPIs, top units/events to review, next-day forecast, comfort check.
- **Digital twin**: interactive 3D model of Chamchuri 5 (three.js), traced from the CU-BEMS paper's floor plan and building view. Replay any 1–7 days at 15-minute steps with the sun following its real path over Bangkok; colour zones by AC status, power, power vs expected, indoor temperature, lighting, lights on after hours, plug loads, daylight, or a Jul–Dec summary. Ceiling lights glow with each zone's actual lighting power (heavily used zones bloom); hover for live readings, click a zone for its detail panel and power chart; isolate or explode floors, X-ray the facade, auto-rotate. Schematic to scale, not a surveyed model. Needs internet access to load three.js from jsDelivr.
- **Priority list**: units and events ranked by the **triage score** = excess THB × confidence ÷ effort. Confidence (0–100) blends rule agreement, duration, recurrence, weather and data quality; effort is Low (schedule/timer), Medium (inspect) or High (repair). A magnitude-vs-confidence chart shows the "act first" corner. Priority level: 🔴 High = the top items that together make up half of the total score, 🟠 Medium = the next 30 %, 🟢 Low = the rest. Includes filters, CSV export, a one-unit drill-down and an asset history card.
- **Setpoint & schedule simulator**: estimates kWh/THB saved and comfort compliance for a new setpoint and AC operating window, and recommends a setpoint.
- **Compare floors**: building-wide summary and priority list, each unit vs the same unit on other floors, and daily profiles side by side.
- **Forecast**: model comparison (including the weather-aware M5/M6 and "Typical weekly pattern + weather"), and actual vs forecast for any week. "Typical weekly pattern + weather" shifts the typical weekly forecast by the floor's baseline kW-per-°C slope × how much warmer or cooler the hour is than a normal baseline weekday hour (Open-Meteo, observed weather standing in for a forecast).

The anomaly thresholds sit in the sidebar under **Advanced** and apply to every tab. The notebook defaults are k = 1.5, 0.5 kW and 8 intervals.
All figures are screening estimates: flags are not diagnosed faults, and savings need a measured trial.


## FM workflow (alert → triage → context → work order → execute → verify)

Plan: `docs/FM_WORKFLOW_PLAN.md`. Accuracy checks: `docs/VERIFICATION.md`.

- **Work orders tab**: incidents (one per asset and pattern, e.g. "runs while the building is closed") with a status lifecycle (Open → Acknowledged → Work order raised → Resolved → Verified, or Reopened / Dismissed). Raise a work order in one click: symptom, evidence, checklist per pattern, due date in working days, and a suggested slot in the lowest-forecast hour of opening hours. Close-outs (fault confirmed / no fault found / legitimate use) feed the learning loop: precision, confidence track record and exceptions. Fixes are verified against 28 days before and after.
- **Maintenance tab**: maintenance log (manual entry, CSV import), fleet health score, 30-day fault risk, remaining useful life and a per-unit degradation chart.
- **Digital twin modes**: Replay, Forecast next day, What-if (uses the simulator's settings).
- **Building model**: `dashboard/brick/chamchuri5.ttl` (Brick 1.3), regenerate with `python dashboard/brick/build_brick.py`. Supply groups are assumed (one chilled-water loop per floor).
- **Storage**: SQLite `dashboard/fm.db` (created on first run, git-ignored). Add a `[gsheets]` section to `.streamlit/secrets.toml` to use Google Sheets instead (plan §12; the Sheets backend is not covered by automated tests).
- **Simulated data**: the first run seeds about 10 close-outs and an 18-month maintenance history, all stored with `simulated = 1` and shown with a "Simulated" badge. The fault-risk model and likely-cause ranking are trained on that simulated log, so they demonstrate the method only. Delete `dashboard/fm.db` to start empty-then-reseeded, or import real records.
- **Tests**: `pip install -r requirements-dev.txt`, then `python -m pytest` from the project folder (about a minute; uses a throw-away database).
