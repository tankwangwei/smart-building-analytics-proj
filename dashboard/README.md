# Building AC dashboard (Floors 3, 4, 5 and 7)

1. Run `python run_all_floors.py` (about 8 minutes for four floors). It writes `ac_analysis_outputs/floor_<n>/`.
   To run one floor only: `python run_all_floors.py 7`.
2. Install the extra packages once: `pip install -r dashboard/requirements.txt`
3. From the project folder: `python -m streamlit run dashboard/app.py`

**Timeframe** (top right, every tab): Day / Week / Month / Jul–Dec plus a date. Every total, colour, alert and finding is worked out for that timeframe. "Normal" is always learned from Jul 2018 – Jun 2019, and forecast accuracy is always scored on Jul–Dec 2019.

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
- **Priority list**: units and events ranked by weighted waste (kWh) = excess kWh + an extra for after-hours waste (counts 1.5×) − a discount for hot-day waste (counts 0.5×), shown as a sum with a worked example. Priority level: 🔴 High = the top units that together cause half of all weighted waste, 🟠 Medium = the next 30 %, 🟢 Low = the rest. Includes filters, CSV export, a one-unit drill-down for the selected timeframe and a heatmap.
- **Setpoint & schedule simulator**: estimates kWh/THB saved and comfort compliance for a new setpoint and AC operating window, and recommends a setpoint.
- **Compare floors**: building-wide summary and priority list, each unit vs the same unit on other floors, and daily profiles side by side.
- **Forecast**: model comparison (including the weather-aware M5/M6 and "Typical weekly pattern + weather"), and actual vs forecast for any week. "Typical weekly pattern + weather" shifts the typical weekly forecast by the floor's baseline kW-per-°C slope × how much warmer or cooler the hour is than a normal baseline weekday hour (Open-Meteo, observed weather standing in for a forecast).

The anomaly thresholds sit in the sidebar under **Advanced** and apply to every tab. The notebook defaults are k = 1.5, 0.5 kW and 8 intervals.
All figures are screening estimates: flags are not diagnosed faults, and savings need a measured trial.
