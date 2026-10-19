"""Run the AC analysis notebook once per floor.

Usage:  python run_all_floors.py            (Floors 3, 4, 5 and 7)
        python run_all_floors.py 3 7        (selected floors only)

Each floor writes its charts and tables to ac_analysis_outputs/floor_<n>/, plus an executed copy of
the notebook (executed_notebook.ipynb) so the printed results can be reviewed. The main notebook is not changed.
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "AC_Analysis_Floor7_Clear_Charts.ipynb"
FLOORS = [int(f) for f in sys.argv[1:]] or [3, 4, 5, 7]

for floor in FLOORS:
    started = time.time()
    print(f"Floor {floor}: running...", flush=True)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(f"FLOOR_TO_RUN = {floor}"))
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    out = HERE / "ac_analysis_outputs" / f"floor_{floor}"
    nbformat.write(nb, out / "executed_notebook.ipynb")
    print(f"Floor {floor}: done in {time.time() - started:.0f} s -> {out}", flush=True)
