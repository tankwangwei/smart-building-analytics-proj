# Building AC dashboard (Floors 3, 4, 5 and 7)

1. Run `python run_all_floors.py` (about 8 minutes for four floors). It writes `ac_analysis_outputs/floor_<n>/`.
   To run one floor only: `python run_all_floors.py 7`.
2. Install the extra packages once: `pip install -r dashboard/requirements.txt`
3. From the project folder: `python -m streamlit run dashboard/app.py`

Pick a floor in the sidebar (Floor 7 by default). Tabs:
- **Overview**: KPIs, top units/events to review, next-day forecast, comfort check.
- **Priority list**: units and events ranked by excess kWh × after-hours × hot-day weighting, with filters, CSV export, a weekly drill-down and a heatmap.
- **Threshold tuner**: sweeps one rule parameter and shows how flags change, with an elbow suggestion.
- **Setpoint & schedule simulator**: estimates kWh/THB saved and comfort compliance for a new setpoint and AC operating window, and recommends a setpoint.
- **Compare floors**: building-wide summary and priority list, each unit vs the same unit on other floors, and daily profiles side by side.
- **Forecast**: model comparison (including the weather-aware M5/M6), actual vs forecast by week, error by hour, and weather vs daily energy.

The sidebar thresholds apply to every tab. The notebook defaults are k = 1.5, 0.5 kW and 8 intervals.
All figures are screening estimates: flags are not diagnosed faults, and savings need a measured trial.
