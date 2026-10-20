# Building AC dashboard (Floors 3, 4, 5 and 7)

1. Run `python run_all_floors.py` (about 8 minutes for four floors). It writes `ac_analysis_outputs/floor_<n>/`.
   To run one floor only: `python run_all_floors.py 7`.
2. Install the extra packages once: `pip install -r dashboard/requirements.txt`
3. From the project folder: `python -m streamlit run dashboard/app.py`

The sidebar **View** opens on **Whole building** (Floors 3, 4, 5 and 7 combined); pick a single floor to focus on it. Tabs:
- **Overview**: KPIs, top units/events to review, next-day forecast, comfort check.
- **Priority list**: units and events ranked by excess kWh (after-hours waste counts 1.5×, hot-day waste 0.5×), with filters, CSV export, a weekly drill-down and a heatmap.
- **Threshold tuner**: sweeps one rule parameter and shows how flags change, with an elbow suggestion.
- **Setpoint & schedule simulator**: estimates kWh/THB saved and comfort compliance for a new setpoint and AC operating window, and recommends a setpoint.
- **Compare floors**: building-wide summary and priority list, each unit vs the same unit on other floors, and daily profiles side by side.
- **Forecast**: model comparison (including the weather-aware M5/M6), and actual vs forecast for any week.

The sidebar thresholds apply to every tab. The notebook defaults are k = 1.5, 0.5 kW and 8 intervals.
All figures are screening estimates: flags are not diagnosed faults, and savings need a measured trial.

## Data assistant (chat button, bottom-right)

The **💬 Ask the data** button opens a chat assistant (Claude Opus 5.5) that answers business questions only from this
dashboard's data. It calls the same calculations the tabs use (priority list, events, simulator, forecast, floor comparison,
weather) and, when the data can't answer a question, replies "Sorry, I can't answer that from the dashboard's data."

It needs an Anthropic API key (usage is billed to that key):
- **Streamlit Cloud:** app menu ⋮ → Settings → Secrets, add `ANTHROPIC_API_KEY = "sk-ant-..."`.
- **Locally:** set the `ANTHROPIC_API_KEY` environment variable, or create `.streamlit/secrets.toml` with the same line
  (that file is git-ignored; never commit a key).

Without a key the button still appears and explains how to set it up.
