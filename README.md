# Smart Building Analytics: Floor 7 AC (IS4001 Group 5)

AC anomaly detection and next-day demand forecasting for CU-BEMS (Chamchuri 5, Chulalongkorn University, Bangkok), with Open-Meteo weather.

- `AC_Analysis_Floor7_Clear_Charts.ipynb`: the full analysis. It writes charts and tables to `ac_analysis_outputs/`.
- `dashboard/`: Streamlit dashboard with a priority list, threshold tuner, setpoint/schedule simulator and forecasts. See [dashboard/README.md](dashboard/README.md).

The raw data file `combined_by_floor.csv` (618 MB) is too large for GitHub and is not in this repo. The dashboard only needs `ac_analysis_outputs/`. To re-run the notebook, download the [CU-BEMS dataset](https://www.nature.com/articles/s41597-020-00582-3), combine it into `combined_by_floor.csv` and place the file in this folder.

Run the dashboard locally:

```
pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```
