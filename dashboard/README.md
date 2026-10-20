# Building AC dashboard (Floors 3, 4, 5 and 7)

1. Run `python run_all_floors.py` (about 8 minutes for four floors). It writes `ac_analysis_outputs/floor_<n>/`.
   To run one floor only: `python run_all_floors.py 7`.
2. Install the extra packages once: `pip install -r dashboard/requirements.txt`
3. From the project folder: `python -m streamlit run dashboard/app.py`

The sidebar **View** opens on **Whole building** (Floors 3, 4, 5 and 7 combined); pick a single floor to focus on it. Tabs:
- **Overview**: KPIs, top units/events to review, next-day forecast, comfort check.
- **Digital twin**: interactive 3D model of Chamchuri 5 (three.js), traced from the CU-BEMS paper's floor plan and building view. Replay any 1–7 days at 15-minute steps with the sun following its real path over Bangkok; colour zones by AC status, power, power vs expected, indoor temperature, lighting, lights on after hours, plug loads, daylight, or a Jul–Dec summary. Animated cold-air flow scales with each unit's power; hover for live readings, click a zone for its detail panel and power chart; isolate or explode floors, X-ray the facade, auto-rotate. Schematic to scale, not a surveyed model. Needs internet access to load three.js from jsDelivr.
- **Priority list**: units and events ranked by excess kWh (after-hours waste counts 1.5×, hot-day waste 0.5×), with filters, CSV export, a weekly drill-down and a heatmap.
- **Threshold tuner**: sweeps one rule parameter and shows how flags change, with an elbow suggestion.
- **Setpoint & schedule simulator**: estimates kWh/THB saved and comfort compliance for a new setpoint and AC operating window, and recommends a setpoint.
- **Compare floors**: building-wide summary and priority list, each unit vs the same unit on other floors, and daily profiles side by side.
- **Forecast**: model comparison (including the weather-aware M5/M6), and actual vs forecast for any week.

The sidebar thresholds apply to every tab. The notebook defaults are k = 1.5, 0.5 kW and 8 intervals.
All figures are screening estimates: flags are not diagnosed faults, and savings need a measured trial.
