"""Interactive 3D digital twin of Chamchuri 5 (three.js scene embedded in the dashboard).

Geometry is traced from the CU-BEMS paper (Pipattanasomporn et al., Scientific Data 2020): Fig. 2 gives the
Floors 3-7 plan with Zones 1-5 and the sensor positions, Fig. 1 the 7-storey massing with balcony bands, a
taller ground floor and rooftop PV. It is a schematic to scale (about 52 x 33 m per floor, ~11,700 m2 in
total), not a surveyed BIM model.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Plan of Floors 3-7 in pixels of the paper's Fig. 2 (x to the right, y downwards), as axis-aligned rectangles.
PX_TO_M = 0.1405  # 370 px across the plan = 52 m
PLAN_X0, PLAN_Y0, PLAN_H = 425, 183, 235
ZONE_RECTS = {
    1: [(575, 778, 185, 258), (632, 778, 258, 280)],                          # top right, 4 AC units
    2: [(652, 778, 280, 418)],                                                # bottom right
    3: [(520, 608, 305, 418), (608, 652, 258, 418),                           # stair core and corridor (no AC)
        (425, 445, 220, 380), (778, 795, 220, 380)],                          # end stairs / balconies
    4: [(445, 575, 185, 258), (445, 520, 258, 282), (520, 608, 258, 305)],    # top left, L-shaped
    5: [(445, 520, 282, 418)],                                                # bottom left
}
SENSORS_PX = {1: (718, 231), 2: (711, 349), 4: (480, 226), 5: (482, 341)}     # red dots in Fig. 2
AC_UNITS_PX = {1: [(600, 220), (650, 220), (700, 220), (750, 220)], 2: [(715, 380)], 4: [(510, 215)], 5: [(482, 380)]}
FOOTPRINT_PX = (425, 795, 185, 418)
FLOOR_COUNT = 7
GROUND_HEIGHT, STOREY_HEIGHT = 4.5, 3.8
GAP = 0.25  # visual gap between neighbouring zones, metres
TEMPLATE = Path(__file__).with_name("twin3d.html")


def floor_base(floor):
    return 0.0 if floor == 1 else GROUND_HEIGHT + (floor - 2) * STOREY_HEIGHT


def floor_height(floor):
    return GROUND_HEIGHT if floor == 1 else STOREY_HEIGHT


def _pt(x, y):
    """Plan pixel -> metres (x along the building, y across it; the plan's top edge is the far side)."""
    return round((x - PLAN_X0) * PX_TO_M, 2), round((PLAN_Y0 + PLAN_H - y) * PX_TO_M, 2)


def _rect(x0, x1, y0, y1, gap=GAP):
    ax, ay = _pt(x0, y1)
    bx, by = _pt(x1, y0)
    return [round(ax + gap, 2), round(bx - gap, 2), round(ay + gap, 2), round(by - gap, 2)]


def geometry():
    return {
        "footprint": _rect(*FOOTPRINT_PX, gap=0),
        "zones": {z: [_rect(*r) for r in rects] for z, rects in ZONE_RECTS.items()},
        "sensors": {z: _pt(*p) for z, p in SENSORS_PX.items()},
        "acUnits": {z: [_pt(*p) for p in pts] for z, pts in AC_UNITS_PX.items()},
        "floors": {f: {"base": floor_base(f), "height": floor_height(f)} for f in range(1, FLOOR_COUNT + 1)},
    }


def _num(series, digits=2):
    return [None if pd.isna(v) else round(float(v), digits) for v in series]


LIGHT_ON_KW = 0.2


def payload(details, zone_loads, summaries, analysed_floors, start, days, running_kw, scope_name):
    """details: {floor: flagged 15-min AC detail}; zone_loads: {floor: 15-min lighting/plug/lux per zone};
    summaries: {unit: {...Jul-Dec metrics...}}."""
    slots = pd.date_range(start, periods=96 * days, freq="15min")
    units, zones, outdoor = [], [], None
    for floor in analysed_floors:
        fd = details[floor]
        for unit in dict.fromkeys(fd["unit"]):
            part = fd.loc[fd["unit"].eq(unit)].set_index("Date").reindex(slots)
            zone = int(re.search(r"_z(\d+)_", unit)[1])
            running = part["actual_kW"].gt(running_kw)
            status = np.where(part["flagged"].fillna(False).astype(bool), 2, np.where(running, 1, 0))
            status = np.where(part["actual_kW"].isna(), -1, status)
            reason = (part["statistical"].fillna(False).astype(int) + 2 * part["after_hours"].fillna(False).astype(int)
                      + 4 * part["persistent"].fillna(False).astype(int))
            if outdoor is None:
                outdoor = _num(part["outdoor_temp_C"], 1)
            units.append({"floor": floor, "zone": zone, "unit": unit.replace("(kW)", ""),
                          "label": f"Floor {floor} · Zone {zone} · AC{unit.split('_AC')[1].replace('(kW)', '')}",
                          "kw": _num(part["actual_kW"]), "exp": _num(part["expected_kW"]),
                          "temp": _num(part["indoor_temp_C"], 1), "rh": _num(part["indoor_rh_pct"], 0),
                          "status": status.astype(int).tolist(), "reason": reason.astype(int).tolist(),
                          "summary": summaries.get(unit, {})})
        # "Normally off" for a floor = no AC unit on that floor is scheduled on (unknown schedule counts as on)
        sched = fd.pivot_table(index="Date", columns="unit", values="schedule_on", aggfunc="first").reindex(slots)
        floor_off = sched.eq(False).all(axis=1) & sched.notna().any(axis=1)
        loads = zone_loads.get(floor)
        if loads is not None:
            loads = loads.reindex(slots)
            for zone in range(1, 6):
                light, plug, lux = (f"z{zone}_Light(kW)", f"z{zone}_Plug(kW)", f"z{zone}_S1(lux)")
                if light not in loads:
                    continue
                zones.append({"floor": floor, "zone": zone, "light": _num(loads[light]), "plug": _num(loads[plug]),
                              "lux": _num(loads[lux], 0) if lux in loads else None,
                              # stair lighting (Zone 3) stays on for safety, so it is never "after hours" waste
                              "offHours": (floor_off.astype(int) * (zone != 3)).tolist()})
    return {"scope": scope_name, "analysed": analysed_floors, "geometry": geometry(), "zoneLoads": zones,
            "times": [t.strftime("%Y-%m-%dT%H:%M") for t in slots], "outdoor": outdoor or [],
            "units": units, "runningKW": running_kw,
            "startIndex": int(np.argmax(slots.strftime("%H:%M") == "14:00")) if days else 0}


def html(data, compact=False):
    """compact=True hides the KPI, layer and legend panels behind one button, for embedding in a narrow column."""
    data = {**data, "compact": compact}
    # "</" inside the JSON would end the <script> block early
    return TEMPLATE.read_text(encoding="utf-8").replace("__PAYLOAD__", json.dumps(data).replace("</", "<\\/"))
