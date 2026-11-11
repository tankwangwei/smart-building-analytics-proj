# Building AC dashboard (Floors 3, 4, 5 and 7)

**To install and start it,** follow the numbered steps in the [main README](../README.md#run-the-dashboard-on-your-own-computer). In short, from the project folder:

```bash
python -m pip install -r dashboard/requirements.txt
```

```bash
python -m streamlit run dashboard/app.py
```

Then open http://localhost:8501.

## Using the dashboard, step by step

1. **Pick what to look at.** In the left sidebar, **View** opens on **Whole building** (Floors 3, 4, 5 and 7 combined). Pick a floor to focus on it.
2. **Pick a mode** with the switch at the top right:
   - **Operate** replays one past day as if it were today: that day's alerts, work orders and tomorrow's forecast. It opens on Day.
   - **Review** looks back over a week, a month or Jul–Dec 2019: trends, rankings and whether fixes worked.
3. **Pick the date or period** with the controls next to the mode switch. The small banner under them always shows what you are looking at.
4. **Start on the Overview tab.** It shows the key numbers, the top problems and the smart assistant's suggestions.
5. **Open the Priority list** to see which AC units to check first, ranked by likely waste, confidence and effort. Click a unit to drill down.
6. **Raise and close work orders** in the **Work orders** tab (Operate): click an incident, read its evidence, then raise a work order. Switch to Review to check whether fixes worked.
7. **Explore the rest:**
   - **Digital twin** for the building in 3D (drag to rotate, scroll to zoom, click a zone).
   - **Simulator** to test a new setpoint or AC schedule.
   - **Compare floors**.
   - **Forecast** for tomorrow's demand and how accurate each method is.
   - **Maintenance**.
8. **Ask a question** with the round **✦ Ask the data** button at the bottom right. It needs a Gemini key (step 5 of the main README).
9. **Hover any ⓘ or ? icon** for an explanation of that number or chart.

Optional settings in the sidebar:
- **Building Opening Hours:** opening times and public holidays.
- **Advanced: Anomaly Thresholds:** how sensitive the anomaly flags are.

## Reference: how each part works

**Mode** (top right, every tab): **Operate** replays one past day as if it were today — what an FM would see that morning: today's alerts and incidents, the work-order queue and tomorrow's forecast (the actual next-day load stays hidden unless 'Show what actually happened' is switched on). With a live building-management feed it would simply track the current date. **Review** looks back over a week, a month or Jul–Dec: rank problems, compare floors, verify fixes and check flag precision. A banner under the title always says which mode and period you are in, and each tab has a one-line note on what it does in that mode. "Normal" is always learned from Jul 2018 – Jun 2019, and forecast accuracy is always scored on Jul–Dec 2019.

**✦ Ask the data** (round button, bottom-right corner of every tab): a chat assistant for the facilities team, using Google Gemini (`gemini-3.5-flash-lite` by default; set `GEMINI_MODEL` in secrets to change it). The code is in `dashboard/fm_chat.py`.
- **How it answers:** it calls 12 read-only tools over the dashboard's own data and rules, so every figure comes from the analysis. The tools are: overview KPIs, unit ranking, events, unit profile, when waste happens, floor comparison, energy over time, day-ahead forecast, forecast accuracy, holidays and hours, the Smart assistant's findings, and a what-if setpoint/schedule estimate (the same maths as the simulator tab).
- **Scope:** it uses the dashboard's current view (floor, timeframe, opening hours) unless the question names another.
- **API key (free):** create one at [Google AI Studio](https://aistudio.google.com) → API keys. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (git-ignored) and paste the key as `GEMINI_API_KEY`. You can also set the `GEMINI_API_KEY` environment variable, or paste a key in the chat panel for the current browser session only. On Streamlit Cloud, put it under App → Settings → Secrets. Never commit a key.
- **Free tier:** Google limits requests per minute and per day, and one question can take several requests (one per data lookup). Check Google's current terms on whether free-tier prompts are used to improve its products.

**Building opening hours and Thai public holidays** (sidebar → Building opening hours). Defaults: open Mon–Fri 08:00–17:00 (the building's listed hours on Google, assumed accurate), closed on public holidays. The dataset has no official hours. A pre-cooling allowance (default 15 min) lets the AC start before opening without being flagged: 92% of weekday switch-ons in the data are at 07:45.
- **Holiday list:** `dashboard/thai_holidays.py` has the official holidays and substitution days for Jul 2018 – Jan 2020, plus one closure seen in the data (Mon 30 Dec 2019). Every listed holiday in Jul–Dec 2019 shows almost no AC use. Dates after that come from the `holidays` package (offline, no API key), up to two years ahead and without Labour Day or Children's Day, which offices work through. It cannot know special holidays announced at short notice: add those to `CHECKED_HOLIDAYS`.
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

**Hot days and heat index:** a hot day is one whose daytime (08:00–17:00) mean heat index ("feels like", from temperature and humidity, NOAA method) is in the top 10% for its month in the baseline year. The notebook uses temperature only; the dashboard switches to heat index because it explains daily AC energy better (R² 0.59 vs 0.32, Jul–Dec 2019 weekdays). Hot days lower an event's confidence, since extra AC may be justified. The Overview shows outdoor temperature, humidity and feels-like temperature for the timeframe, and the load pattern averages working days unless weekends or public holidays are picked.

**Overview** is a building control-room view (dark theme from `.streamlit/config.toml`, so launch Streamlit from the project folder): round-icon KPIs labelled with their period ("today", "this month"); a Floors card (Operate: each unit's status at a chosen time; Review: priority over the period); the top open incidents; the smart assistant (tomorrow's peak only in Operate); and the load pattern. The centre shows a compact 3D replay of today in Operate, and in Review a day-by-day chart of AC energy with the flagged excess, the normal level, hot days and closures; clicking a day replays it in Operate. The 3D twin colours zones by anomaly status, AC power vs expected, indoor temperature, lights on after hours, or priority level.

"Outside usual schedule" energy is recalculated from the 15-minute data using each unit's own usual schedule, so it can differ slightly from the notebook's figure.

The sidebar **View** opens on **Whole building** (Floors 3, 4, 5 and 7 combined); pick a single floor to focus on it. Tabs:
- **Overview**: KPIs, top units/events to review, next-day forecast, comfort check.
- **Digital twin**: interactive 3D model of Chamchuri 5 (three.js), traced from the CU-BEMS paper's floor plan and building view. Replay any 1–7 days at 15-minute steps with the sun following its real path over Bangkok; colour zones by AC status, power, power vs expected, indoor temperature, lighting, lights on after hours, plug loads, daylight, or a Jul–Dec summary. Ceiling lights glow with each zone's actual lighting power (heavily used zones bloom); hover for live readings, click a zone for its detail panel and power chart; isolate or explode floors, X-ray the facade, auto-rotate. Schematic to scale, not a surveyed model. Needs internet access to load three.js from jsDelivr.
- **Priority list**: units and events ranked by the **triage score** = excess THB × confidence ÷ effort. Confidence (0–100) blends rule agreement, duration, recurrence, weather and data quality; effort is Low (schedule/timer), Medium (inspect) or High (repair). A magnitude-vs-confidence chart shows the "act first" corner. Priority level: 🔴 High = the top items that together make up half of the total score, 🟠 Medium = the next 30 %, 🟢 Low = the rest. Includes filters, CSV export, a one-unit drill-down and an asset history card.
- **Setpoint & schedule simulator**: estimates kWh/THB saved and comfort compliance for a new setpoint and AC operating window, and recommends a setpoint.
- **Compare floors**: building-wide summary and priority list, each unit vs the same unit on other floors, and daily profiles side by side.
- **Forecast**: model comparison and actual vs forecast for any week, with plain method names (no M-numbers). Every method is fitted, chosen and scored in the notebook, as in the proposal: trained on Jul 2018 – Mar 2019, the method with the lowest Apr–Jun 2019 error is chosen (✓), every method is refitted on Jul 2018 – Jun 2019, and all are scored on Jul–Dec 2019, which is never used for fitting. The weather-aware methods use **heat index** ("feels like", from temperature and humidity): "Regression + past demand + heat index" and "Random Forest + heat index" take it as an input, and "Typical weekly pattern + heat index" shifts the typical week by the floor's kW per °C of heat index. "Typical weekly pattern + temperature" (M7) does the same with outdoor temperature alone; it is shown beside the heat-index version to justify choosing heat index. Weather methods use observed weather in place of a forecast, so their errors are optimistic; `forecast_weather_error_sensitivity.csv` shows how much a day-ahead forecast error would cost. Public holidays are left out of every fit and forecast as zero. Bias (NMBE) and CV(RMSE) use the ASHRAE Guideline 14 form: residual = measured − predicted, so a positive bias means the forecast is too low. "Peak hours" are hours at or above the baseline's 90th percentile, and the 80% forecast band is learned from the Apr–Jun 2019 errors (the notebook's `forecast_validation_predictions.csv`). All forecast errors are for AC load only.

The anomaly thresholds sit in the sidebar under **Advanced** and apply to every tab. The notebook defaults are k = 1.5, 0.5 kW and 8 intervals.
All figures are screening estimates: flags are not diagnosed faults, and savings need a measured trial.


## FM workflow (alert → triage → context → work order → execute → verify)

Plan: `docs/FM_WORKFLOW_PLAN.md`. Accuracy checks: `docs/VERIFICATION.md`.

- **Work orders tab** (Operate: one table of incidents with their work-order status; click a row for an Evidence / Work order / History panel with the confidence breakdown, one-click raise, checklist, status, close-out and audit trail. Review: verification and learning): incidents (one per asset and pattern, e.g. "runs while the building is closed") with a status lifecycle (Open → Acknowledged → Work order raised → Resolved → Verified, or Reopened / Dismissed). Raise a work order in one click: symptom, evidence, checklist per pattern, due date in working days, and a suggested slot in the lowest-forecast hour of opening hours. Close-outs (fault confirmed / no fault found / legitimate use) feed the learning loop: precision, confidence track record and exceptions. Fixes are verified with an IPMVP-style adjusted baseline: a small model fitted on the 28 days before the fix (open vs closed day, plus heat index for faults that happen while open) predicts the up to 28 days after had nothing been fixed; saving = prediction − measured. Error metrics follow ASHRAE Guideline 14; neither is a compliance claim.
- **Maintenance tab**: maintenance log (manual entry, CSV import), fleet health score, 30-day fault risk, remaining useful life and a per-unit degradation chart.
- **Digital twin modes**: Replay, Forecast next day, What-if (uses the simulator's settings).
- **Building model**: `dashboard/brick/chamchuri5.ttl` (Brick 1.3), regenerate with `python dashboard/brick/build_brick.py`. Supply groups are assumed (one chilled-water loop per floor).
- **Storage**: SQLite `dashboard/fm.db` (created on first run, git-ignored). Add a `[gsheets]` section to `.streamlit/secrets.toml` to use Google Sheets instead (plan §12; the Sheets backend is not covered by automated tests).
- **Simulated data**: the first run seeds about 10 close-outs and an 18-month maintenance history, all stored with `simulated = 1` and shown with a "Simulated" badge. The fault-risk model and likely-cause ranking are trained on that simulated log, so they demonstrate the method only. Delete `dashboard/fm.db` to start empty-then-reseeded, or import real records.
- **Tests**: `pip install -r requirements-dev.txt`, then `python -m pytest` from the project folder (about a minute; uses a throw-away database).
