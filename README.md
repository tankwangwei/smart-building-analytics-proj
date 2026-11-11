# Smart Building Analytics: Floors 3, 4, 5 and 7 AC (IS4001 Group 5)

AC anomaly detection and next-day demand forecasting for CU-BEMS (Chamchuri 5, Chulalongkorn University, Bangkok), with Open-Meteo weather.

**Live dashboard:** https://cu-bems-smart.streamlit.app/ (no install needed). If it shows "This app has gone to sleep", click **Yes, get this app back up** and wait about a minute.

## What's in this repo

| Path | What it is |
|---|---|
| `dashboard/` | The Streamlit dashboard. How to use it: [dashboard/README.md](dashboard/README.md). |
| `ac_analysis_outputs/` | The analysis results (charts and tables per floor). The dashboard reads these, so it runs without the raw data. |
| `AC_Analysis_Floor7_Clear_Charts.ipynb` | The full analysis for one floor (Floor 7 by default). |
| `run_all_floors.py` | Runs the notebook for Floors 3, 4, 5 and 7. |
| `tests/`, `docs/VERIFICATION.md` | Automated accuracy checks and their report. |

## Run the dashboard on your own computer

You only need to do steps 1–4 once. After that, start from step 5.

**Step 1. Install Python.** Get Python 3.12 from [python.org](https://www.python.org/downloads/). On Windows, tick **"Add python.exe to PATH"** in the installer. Check it worked by opening a terminal (Windows: PowerShell; Mac: Terminal) and typing:

```bash
python --version
```

It should print `Python 3.12.x` (3.11–3.13 also work).

**Step 2. Download the project.** Either:
- with Git:
  ```bash
  git clone https://github.com/tankwangwei/smart-building-analytics-proj.git
  ```
- or on the GitHub page, click **Code → Download ZIP** and unzip it.

**Step 3. Open a terminal in the project folder** (the folder that contains this README):

```bash
cd smart-building-analytics-proj
```

**Step 4. Install the packages** (takes a few minutes the first time):

```bash
python -m pip install -r dashboard/requirements.txt
```

**Step 5 (optional). Turn on the chatbot.** The dashboard works without this; only the **✦ Ask the data** chatbot needs a key.
1. Go to [Google AI Studio](https://aistudio.google.com), sign in and click **Get API key → Create API key**. It is free.
2. In the project, open the `.streamlit` folder and copy `secrets.toml.example` to a new file called `secrets.toml` in the same folder.
3. Open `secrets.toml` and paste your key between the quotes: `GEMINI_API_KEY = "your-key"`. Save it.

`secrets.toml` is git-ignored. Never commit or share your key.

**Step 6. Start the dashboard:**

```bash
python -m streamlit run dashboard/app.py
```

Your browser opens at http://localhost:8501. If it doesn't, open that address yourself. The first load takes about 30 seconds while the data is prepared.

**Step 7. Stop it** by going back to the terminal and pressing **Ctrl + C**.

## Run the accuracy tests (optional)

```bash
python -m pip install -r requirements-dev.txt
```

```bash
python -m pytest
```

It takes about a minute. The results are explained in [docs/VERIFICATION.md](docs/VERIFICATION.md).

## Re-run the analysis from the raw data (optional)

The dashboard does not need this: the results are already in `ac_analysis_outputs/`.

1. Download the [CU-BEMS dataset](https://www.nature.com/articles/s41597-020-00582-3). It is 618 MB, too large for GitHub.
2. Combine it into one file called `combined_by_floor.csv` and put it in the project folder.
3. Run all four floors (about 8 minutes):
   ```bash
   python run_all_floors.py
   ```
   Or run one floor only, for example `python run_all_floors.py 7`.
4. Restart the dashboard (steps 6–7) to see the new results.

## Put it online with Streamlit Community Cloud (optional)

1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **Create app**.
3. Pick the repo, branch `main` and main file `dashboard/app.py`.
4. Under **Advanced settings**, choose Python **3.12**. Newer versions break one of the packages.
5. In the same place, paste `GEMINI_API_KEY = "your-key"` into **Secrets**, then click **Deploy**.

To change the key later, go to your app → **⋮ → Settings → Secrets**.
