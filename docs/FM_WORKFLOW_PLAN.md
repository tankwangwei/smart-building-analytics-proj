# Smart FM workflow – implementation plan

Turn the Chamchuri 5 AC dashboard from "show alerts" into a full facilities-management (FM) loop:

**Alert → Triage (severity) → Context (asset history) → Work order → Execute → Verify**, with each close-out improving the next diagnosis. Building and equipment data follow **Brick Schema**.

There are 7 phases. Build them in order, test each one before starting the next, and commit per phase.

| Phase | Adds | Workflow step | Size |
|---|---|---|---|
| 1 | Triage: magnitude (THB) · confidence · effort, replacing the current weighted score | Triage | M |
| 2 | Brick model of the building; peer and shared-supply rules read from it | (foundation) | M |
| 3 | Incidents with a status lifecycle, plus a storage layer | Alert | M–L |
| 4 | Work orders generated from incidents, scheduled into forecast low-demand windows | Work order | M |
| 5 | Close-out form and the learning loop (precision, exceptions, confidence) | Execute → learning | L |
| 6 | Verification (measurement and verification, M&V) of resolved incidents | Verify | M |
| 7 | Asset history card per AC unit | Context | M |

Rough total: 8–10 developer days. S ≈ ½ day, M ≈ 1–1.5 days, L ≈ 2 days.

---

## 0. Where the code is today

| File | What it does |
|---|---|
| `AC_Analysis_Floor7_Clear_Charts.ipynb` | The analysis. Run per floor by `run_all_floors.py`; writes `ac_analysis_outputs/floor_<n>/*.csv`. |
| `dashboard/app.py` | The Streamlit app (one file). Reads only the CSVs; never runs the notebook. |
| `dashboard/twin.py` + `dashboard/twin3d.html` | 3D digital twin (three.js), embedded as an iframe. `twin.payload()` builds its data; `twin.html(data, compact)` renders it. |
| `dashboard/thai_holidays.py` | Thai public holidays (Jul 2018 – Jan 2020) and observed closures; `closure_name(date)`. |
| `.streamlit/config.toml` | Dark theme. Run Streamlit **from the project folder**: `python -m streamlit run dashboard/app.py`. |

**Data flow in `app.py`, top to bottom** (helpers first, then the page):

1. **Loading:** `load(floor)` → `load_floor()` / `load_building()`. These read the CSVs and add the weather forecast model (`add_weather_model`).
2. **Flagging:** `flags_for(floor, k_iqr, k_res, on_kw, persist, use_stat, use_after, use_persist, OPEN)` (cached).
   - Copies the 15-minute detail.
   - Adds `closed` (from `building_closed()`: opening hours and holidays) and `holiday`.
   - Sets expected = 0 while closed.
   - Re-runs the three rules via `run_rules()`: **statistical** (IQR limit), **after_hours** (unit's usual schedule off *or* building closed) and **persistent** (above expected ≥ 2 h).
3. **Events and ranking:**
   - `build_events(d, hot_weight, after_weight)` merges consecutive flagged intervals into events and computes today's `priority_score` (weighted waste).
   - `priority_levels()` / `add_levels()` assign High/Medium/Low: the units causing the first 50% of waste are High, up to 80% Medium.
   - `unit_priority()` sums per unit.
4. **Shared state, built once per run:**
   - Timeframe: `period_start`, `period_end`, `PERIOD`, `PERIOD_DAYS`, from the Day/Week/Month/Jul–Dec picker in the header.
   - Opening hours: `OPEN` from the sidebar.
   - Results: `d`, `events`, `priority`, `kpis` (`period_kpis()`), and all-floor tables (`floors_table`, `allunits`, `details`, `peer_mean`).
   - Peers: `peer_gap()` / `outlier`.
   - Forecast: `fc`, `tm`, `selected_model`, `n24`, `NEXT_CLOSURE`.
   - Overview data: `building_now`, `scope_now`.
5. **Tabs:** Overview (KPI chips, Floors card, Top alerts, compact twin, Smart assistant `assistant_findings()`, Load pattern) · Digital twin · Priority list · Setpoint & schedule simulator · Forecast · Compare floors.

**Conventions to keep:**
- **Cards:** `st.container(key="card-<name>")` (the CSS styles any `st-key-card*` container); small HTML via `html()`; icons via `icon()`.
- **Totals:** follow the timeframe; label them with `PERIOD`.
- **Explanations:** go in `help=` tooltips, not on-page text.
- **One finding per tab:** at most one `takeaway()` box per tab.

---

## 1. Decisions already made

| Topic | Decision |
|---|---|
| Storage | SQLite file `dashboard/fm.db` by default (built into Python). Switches automatically to **Google Sheets** when `st.secrets` contains a `gsheets` section. See §9. |
| Maintenance data | None exists in CU-BEMS. Seed **about 10 made-up close-outs**, stored with `simulated = 1` and always shown with a **"Simulated"** badge. |
| Building topology | Unknown (only `combined_by_floor.csv`). Assume **one chilled-water/AHU supply group per floor**, labelled "assumed" in the Brick model and everywhere it is used. |
| Scope | All 7 phases. |
| Scoring | Phase 1 **replaces** the current 1.5× after-hours and 0.5× hot-day weights. It does not add on top of them (that would double-count). |
| UI footprint | One new tab, **Work orders**. Everything else goes into existing tabs or a dialog. |

---

## 2. Phase 1 – Triage: magnitude · confidence · effort

**Goal:** answer the feedback "rank by magnitude *and* confidence" transparently. Each item shows *why* it ranks where it does, e.g. "1,200 THB · confidence 85 · low effort".

**Build (in `build_events` / `unit_priority`):**

1. **Magnitude**
   - `excess_THB = excess_kWh × TARIFF_THB`. Move `TARIFF_THB` (4.2) into the sidebar "Advanced" expander so the simulator and triage share it.
   - Per day: `excess_THB / PERIOD_DAYS`.
   - **Comfort impact:** share of the event's running intervals with indoor temperature outside 23–26 °C. A boolean `comfort_hit` when it is ≥ 50%.
2. **Confidence, 0–100**, as a weighted sum of components scored 0–1:

   | Component | Weight | Score |
   |---|---|---|
   | Rules agree | 0.35 | (statistical + after_hours + persistent) / 3 |
   | Duration | 0.25 | min(duration_h / 2, 1) |
   | Recurrence | 0.20 | events for this unit at the same weekday and hour in the timeframe: 1 → 0, 2 → 0.5, ≥ 3 → 1 |
   | Not weather | 0.10 | 1 if not a hot day, else 0.4 |
   | Data quality | 0.10 | share of the event's intervals with `observed_minutes ≥ 14` and no interpolation |

   Bands: **High ≥ 70 · Medium 40–69 · Low < 40.** Phase 5 adds a track-record factor.
3. **Pattern and effort.** Classify each event into a pattern; Phase 3 reuses it. Put it in one function, `classify_pattern(event)`:

   | Pattern | Rule | Effort | Factor |
   |---|---|---|---|
   | `closed_running` | ≥ 50% of the event while the building is closed, or on a holiday | Low (schedule/timer) | 1.0 |
   | `off_schedule` | after-hours by the unit's own schedule, building open | Low | 1.0 |
   | `high_while_open` | statistical or persistent, building open | Medium (inspect) | 1.5 |
   | `warm_running` | running and zone > 26 °C for ≥ 50% of the event | High (repair) | 2.5 |
4. **Triage score** = `excess_THB × confidence / 100 ÷ effort_factor`. Priority levels (High/Medium/Low) keep the existing Pareto rule in `priority_levels()`, but run on the triage score.
5. **Units:** triage score = sum of the unit's events. Unit confidence = the THB-weighted mean of its event confidences.
6. **Remove** `AFTER_HOURS_WEIGHT`, `HOT_DAY_WEIGHT`, the `after_hours_extra_kWh` / `hot_day_discount_kWh` columns and their wording in `priority_example()`, `PRIORITY_HELP` and the README.

**UI:**
- **Priority list table:** Priority · Unit · Excess THB · Per day · Confidence (badge) · Effort · Triage score · Comfort.
- **New magnitude-vs-confidence scatter** above the table: x = confidence, y = excess THB, colour = effort, size = events. Dotted lines at confidence 70 and the top-25% THB mark the "act first" corner.
- **Top alerts, twin summary and Compare floors** read the triage score.

**Done when:** the Priority list and Top alerts sort by triage score, every row shows its priority and confidence badges, and confidence is spread across bands (not all High) on Jul–Dec.

---

## 3. Phase 2 – Brick model of the building

**Goal:** describe Chamchuri 5 as a Brick graph, so rules work by equipment type and relationships instead of hard-coded column names like `F7_z1_AC4(kW)`.

**Build:**

1. **Generator:** `dashboard/brick/build_brick.py` writes `dashboard/brick/chamchuri5.ttl` from the header of `combined_by_floor.csv`. It also writes `chamchuri5.json`, a fallback for when `rdflib` is missing.
   - **Building and floors:** `cu:Chamchuri5 a brick:Building`, `brick:hasPart` Floors 1–7.
   - **Zones:** each floor `brick:hasPart` its `brick:HVAC_Zone` 1–5. Zone 3 is the stairwell/corridor, with no AC.
   - **AC units:** each AC column becomes `cu:F7_Z1_AC4 a brick:Terminal_Unit`. Each unit `brick:feeds` and `brick:hasLocation` its zone, and `brick:hasPoint` an electric power sensor.
   - **Zone sensors:** each zone `brick:hasPoint` a `brick:Zone_Air_Temperature_Sensor` (`S1(degC)`), a `brick:Zone_Air_Humidity_Sensor` (`S1(RH%)`) and a `brick:Illuminance_Sensor` (`S1(lux)`).
   - **Lighting and plug loads:** one equipment entity each per zone, with power points.
   - **Link to data:** every point carries `ref:hasExternalReference [ a ref:TimeseriesReference ; ref:hasTimeseriesId "F7_z1_AC4(kW)" ]`, so code finds the data column through the graph.
   - **Assumed supply:** one `brick:Chilled_Water_Loop` per floor (`cu:Floor7_CHW_assumed`) that `brick:feeds` that floor's AC units, with `rdfs:comment "ASSUMED – topology not in the dataset; confirm with facilities"`.
   - **Verify the class names** against the Brick 1.3 ontology (brickschema.org/ontology) before finalising, especially the power-sensor and loop classes.
2. **Loader:** `dashboard/brick_model.py` loads the TTL with `rdflib`, cached with `st.cache_resource`. Add `rdflib` to `dashboard/requirements.txt`. It exposes SPARQL-backed helpers:
   - `units()` → asset id, label, floor, zone, power column, zone temperature/RH columns.
   - `peer_groups()` → units grouped by equipment type and zone position.
   - `supply_groups()` → units fed by the same loop.
   - `asset_for_column(col)` → the asset for a data column.
3. **Replace the hard-coded logic:**
   - `position_of()` and the peer list in `peer_gap()` use `peer_groups()`.
   - The Smart assistant's "3+ units high/off together" rules group by `supply_groups()` instead of by floor, and their text names the loop ("assumed").
   - Asset ids from Brick become the keys for Phase 3–7 storage.
4. **UI:** a "Building model (Brick)" expander in Compare floors. It shows counts (floors, zones, AC units, points), a small table of unit → zone → points → supply group, and the "assumed" note. Offer `chamchuri5.ttl` as a download.

**Done when:** Brick queries return the same 16 units and the same peer groups as today, the assistant's shared-cause rule reads supply groups from Brick, and the app still runs when `rdflib` is missing (JSON fallback).

---

## 4. Phase 3 – Incidents and storage

**Goal:** one stateful item per problem instead of hundreds of events. For example, 50 Saturday events on one unit become one "F7 Z1 AC4 runs while the building is closed" incident.

**Storage layer**, `dashboard/fm_store.py`:
- **One interface, two backends.** `SqliteStore` is the default (`dashboard/fm.db`, add it to `.gitignore`). `SheetsStore` is used when `"gsheets" in st.secrets` (§9). Choose the backend in `get_store()`, cached with `st.cache_resource`.
- **Tables** (one worksheet each in Sheets):

| Table | Columns |
|---|---|
| `incident_state` | incident_key (PK) · status · status_reason · updated_at · updated_by · simulated |
| `work_orders` | wo_id (PK, `WO-0001`) · incident_key · asset_id · created_at · status · assignee · priority · due_date · slot_start · slot_end · symptom · checklist_json · evidence_json · simulated |
| `closeouts` | wo_id (PK) · outcome · root_cause · action_taken · date_fixed · technician · notes · closed_at · simulated |
| `exceptions` | exc_id (PK) · asset_id · weekdays · time_from · time_to · date_from · date_to · reason · source_wo_id · active |
| `audit_log` | ts · actor · entity · entity_id · change |

**Incident detection**, `build_incidents(all_events)`:
- Run on the **whole Jul–Dec data**, not just the timeframe, so incident identity and status stay stable. The timeframe only filters which incidents count as "active".
- **Key** = `f"{asset_id}|{pattern}"`, using the Phase 1 patterns. Add group-level patterns from the assistant rules (`supply_group_off`, `supply_group_high`) keyed on the supply group.
- **Fields:** first_seen · last_seen · occurrences · total excess kWh/THB · confidence (THB-weighted) · priority level · latest event · status (from `incident_state`, default Open).

**Lifecycle:**
- Open → Acknowledged → Work order raised → Resolved → **Verified**, or **Reopened**.
- Any state → **Dismissed**, with a reason.
- Reopened is set automatically in Phase 6 when the pattern recurs after its fix date.

**UI:**
- **New tab "Work orders"**, placed after Priority list. It has filters (status, priority, confidence band, floor) and an incident queue: `st.data_editor` with a Status column (selectbox) and an "Action" column. Status changes write to the store and the audit log.
- **Overview "Top alerts"** shows the top **open incidents** with status chips.
- **Smart assistant** items carry their incident key and status.

**Done when:** the Saturday pattern on F7 Z1 AC4 shows as one incident, status changes survive a browser refresh and an app restart, and the app works with the storage file deleted (it starts empty).

---

## 5. Phase 4 – Work orders

**Goal:** go from an incident to an actionable, schedulable job in one click.

**Build:**

1. **"Raise work order"** on an incident creates a draft:
   - **Asset and location** from Brick (floor, zone, supply group).
   - **Symptom:** one sentence from the pattern plus key numbers (hours, kWh, THB, confidence).
   - **Evidence:** the numbers behind the symptom, stored in `evidence_json`, plus the incident's chart rendered in the WO view.
   - **Checklist:** from a `PLAYBOOK` dict per pattern:

     | Pattern | Checklist |
     |---|---|
     | `closed_running` / `off_schedule` | timer and BMS schedule · booking calendar · manual override |
     | `high_while_open` | filter · coil · setpoint · thermostat location · occupancy change |
     | `warm_running` | refrigerant pressure (leak) · filter · coil · fan · sensor calibration |
     | `supply_group_*` | chilled-water supply temperature · AHU status · electrical feed |
   - **Priority** from triage. **Due date:** High = 2 working days, Medium = 5, Low = 10. Working days skip weekends, `closure_name()` holidays and closed days in `OPEN`.
   - **Cost of delay** = the incident's THB ÷ its active days, shown as THB/day.
   - **Suggested slot:** `suggest_slot(asset_id, after_date)` picks the next working day, then the **lowest-forecast hour within opening hours**. It uses the floor's day-ahead forecast in `fc` (the same logic as `n24`), e.g. "Tue 12:00–13:00, forecast 14 kW".
2. **Status:** Draft → Issued → In progress → Done, plus an assignee text field.
3. **Exports:** CSV and JSON for all WOs or one (`st.download_button`), shaped for import into a maintenance management system (CMMS), and a printable one-page view (HTML).

**Done when:** a WO can be raised from any incident, it carries all fields above, the slot falls inside opening hours on a working day, and the CSV export opens cleanly in Excel.

---

## 6. Phase 5 – Close-out and the learning loop

**Goal:** what technicians find improves the next diagnosis.

**Close-out form**, shown when a WO moves to Done:
- **Outcome:** Fault confirmed · No fault found · Legitimate use.
- **Root cause:** a dropdown filtered by pattern, e.g. dirty filter, low refrigerant, thermostat fault, timer schedule, booking/event, sensor error, other.
- **Action taken, date fixed, technician, notes.**

**Learning (all visible, nothing silent):**
1. **Rule precision** = confirmed ÷ (confirmed + no fault). Computed per rule, per pattern and per unit. Legitimate use is counted separately as "expected use", not as a detection error. Show it as the slide KPI **"% of flags verified as real issues"** on the Work orders tab.
2. **Confidence track record:** final confidence = 0.8 × base + 0.2 × 100 × (confirmed + 1) / (confirmed + no fault + 2). The Laplace smoothing (+1/+2) keeps 1–2 close-outs from swinging it.
3. **Exceptions:** "Legitimate use" pre-fills an exception (asset, weekday(s), time window, date or range, reason) for the user to confirm. Active exceptions pass into `flags_for(…, exceptions)` as a hashable tuple; matching intervals are marked `excepted` and never flagged. List them with an "Deactivate" button.
4. **Threshold suggestions:** a unit with ≥ 3 close-outs and precision < 30% gets "consider raising k for this unit to X". It is shown, never applied automatically.
5. **Smart assistant:** cites history, e.g. "Last 2 similar cases on this unit: dirty filter (simulated)".

**Seed data:** `fm_store.seed_demo()`, run once when the store is empty and labelled **Simulated**. About 10 close-outs on real incidents from the data:

| Asset / pattern | Outcome | Root cause |
|---|---|---|
| F7 Z1 AC4 · closed_running (Saturdays) | Fault confirmed | timer schedule left on Saturdays |
| F7 Z2 AC1 · closed_running | Legitimate use | Saturday booking (creates an exception for 2 Saturdays) |
| F7 Z5 AC1 · warm_running | Fault confirmed | dirty filter |
| F7 Z2 AC1 · warm_running | Fault confirmed | low refrigerant |
| F3 Z5 AC1 · off_schedule (off in usual hours) | Fault confirmed | tripped breaker |
| Floor 4 supply group · supply_group_high | No fault found | exam event, high occupancy |
| F4 Z4 AC1 · high_while_open | No fault found | – |
| F5 Z2 AC1 · high_while_open | Fault confirmed | thermostat placed near a window |
| F7 Z4 AC1 · closed_running | Fault confirmed | manual override not reset |
| F4 Z1 AC4 · high_while_open | No fault found | – |

Use the data's own incident dates. Phase 6 picks the fix dates (natural before/after changes where they exist).

**Done when:** closing WOs changes the precision KPI, a "No fault found" visibly lowers that unit's or pattern's confidence, a confirmed exception removes those flags, and every simulated record shows the badge.

---

## 7. Phase 6 – Verification (M&V)

**Goal:** separate **verified** savings from estimated ones, and reopen fixes that didn't hold.

**Build**, `verify_incident(incident, closeout)`:
- **Windows:** pre = 28 days before `date_fixed`; post = 28 days after, capped at the data end of 31 Dec 2019.
- **Metric per pattern:**
  - `closed_running` / `off_schedule`: kWh used while closed per day.
  - `high_while_open`: positive residual kWh per day.
  - `warm_running`: warm-while-running hours per day.
  - `supply_group_*`: the group's flagged kWh per day.
- **Recurrence:** count of the pattern's events in the post window, plus comfort (% of running hours inside 23–26 °C) pre vs post.
- **Verdict:**
  - **Verified:** the metric fell ≥ 50%, no events in the last 14 days of the post window, and comfort is not more than 10 points worse.
  - **Reopened:** events recur. This also sets the incident status.
  - **Inconclusive:** the post window is under 14 days.
- **Verified savings** = (pre per day − post per day) × post days, in kWh and THB. Report it separately from estimates.
- **Honest demo:** nobody actually fixed anything in this dataset. Add `find_natural_changes()`: for each unit and pattern, find where weekly metric values drop ≥ 70% and stay low for ≥ 4 weeks. Use those dates as the seed fix dates, and label them "simulated fix at an observed change".

**UI:** in the Work orders tab, each resolved incident shows a before/after bar chart, a weekly line with the fix date marked, and a verdict chip. The Overview adds a KPI chip, **"Verified savings"**.

**Done when:** each resolved incident shows a verdict, at least one seed case verifies on real data, a recurring one reopens, and the savings maths matches a hand calculation for one case.

---

## 8. Phase 7 – Asset history card

**Goal:** the "Context" step: everything about one AC unit in one place.

**Build:** `asset_card(asset_id)`, rendered with `st.dialog` and opened from the Priority list ("Inspect a unit"), the incident queue and a unit picker in the Digital twin tab. (The twin iframe can't call back into Streamlit.)
- **Header:** label, floor, zone, Brick id, supply group (assumed).
- **KPIs:** excess THB (timeframe), recurrence rate (weeks with an incident ÷ weeks), open incidents, last service date (latest close-out), precision of its flags.
- **Timeline** (plotly, time axis): flags, incidents, WOs and close-outs, coloured by type, simulated ones hatched or badged.
- **Monthly kW-per-°C trend:** a regression per month of hourly kW on outdoor temperature, for weekday running hours (cached). A creeping rise hints at coil or refrigerant problems.
- **Monthly gap against peers** (Brick peer group) and monthly comfort compliance.

**Done when:** every incident and WO links to its asset card, and the card renders for all 16 units in every timeframe.

---

## 9. Storage setup

**SQLite (default):** nothing to install. The app creates `dashboard/fm.db` on first run. Add `dashboard/fm.db` to `.gitignore`. On Streamlit Cloud this file **resets on every redeploy**: fine for demos, not for real use.

**Google Sheets** (persistent; about 20–30 min, someone with a Google account):
1. In Google Cloud Console, create a project → enable **Google Sheets API** (and Google Drive API).
2. **IAM & Admin → Service accounts** → create one → **Keys → Add key → JSON**, and download it.
3. Create a Google Sheet (e.g. "Chamchuri5 FM store") and **share it with the service account's email** as Editor.
4. Add the secrets, locally in `.streamlit/secrets.toml` (never commit it; add it to `.gitignore`) and on Streamlit Cloud under **App → Settings → Secrets**:

   ```toml
   [gsheets]
   spreadsheet_url = "https://docs.google.com/spreadsheets/d/<id>"
   type = "service_account"
   project_id = "..."
   private_key_id = "..."
   private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
   client_email = "...@....iam.gserviceaccount.com"
   client_id = "..."
   token_uri = "https://oauth2.googleapis.com/token"
   ```
5. Add `gspread` and `google-auth` to `dashboard/requirements.txt`.
6. `SheetsStore` opens the sheet, creates one worksheet per table with a header row if it's missing, and appends or updates rows. It's fine for this volume (dozens of rows); it isn't built for many people editing at once.

---

## 10. Testing (every phase)

- **Headless tests:** add `tests/test_app.py` using `streamlit.testing.v1.AppTest`. Run the app in the Whole building view, the Floor 7 view, and the timeframes **Jul–Dec, Day Sat 27 Jul 2019, Day Sun 28 Jul 2019, Day Wed 23 Oct 2019 (holiday), Week 14–20 Oct 2019**. Assert no exceptions. These days cover weekends, closures and holidays, which have broken things before (division by zero in peer comparisons, empty simulator).
- **Phase checks:**
  - **P1:** confidence spread across bands; no NaN scores.
  - **P2:** Brick gives the same 16 units and peer groups as the regex; the JSON fallback works.
  - **P3:** status survives a restart; one incident per asset+pattern.
  - **P4:** the slot sits in opening hours on a working day.
  - **P5:** precision and confidence move after a close-out; an exception removes its flags.
  - **P6:** one hand-checked verification.
  - **P7:** the card renders for all units.
- **Visual check:** a screenshot per phase (Playwright against `python -m streamlit run dashboard/app.py --server.port 8599 --server.headless true`).
- **Performance:** `flags_for` keeps every input in its arguments, including the exceptions tuple, so caching still works. The page should rerun in under ~5 s after the first load.

---

## 11. Risks and open questions

- **Opening hours:** the defaults (Mon–Fri 07:30–17:30) follow the data, not an official source. Confirm them, and whether Saturdays are meant to be open; that changes how much Saturday running counts as waste.
- **Topology:** the per-floor supply groups are an assumption. Correcting them later means editing the generator or the TTL.
- **Brick class names:** check against Brick 1.3 before presenting.
- **Holiday list:** covers Jul 2018 – Jan 2020 only. Extend `thai_holidays.py` for other years.
- **Simulated data:** must stay visibly labelled everywhere, so it isn't mistaken for real maintenance records in the presentation.
- **Divergence from the notebook:** the notebook and slides use the older logic (no opening hours, holidays, triage or incidents). Decide whether to port Phases 1–2 into the notebook or present the dashboard as the newer version.
