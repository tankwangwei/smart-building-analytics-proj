"""Generate the Brick model of Chamchuri 5: dashboard/brick/chamchuri5.ttl and chamchuri5.json.

The building structure is read from the analysis outputs (ac_analysis_outputs/floor_<n>/unit_quality.csv lists every AC
column, zone_loads_15min.csv lists lighting/plug/lux per zone). Class names were checked against Brick 1.3
(https://brickschema.org/schema/1.3/Brick.ttl): Building, Floor, HVAC_Zone, Terminal_Unit, Electric_Power_Sensor,
Zone_Air_Temperature_Sensor, Zone_Air_Humidity_Sensor, Illuminance_Sensor, Chilled_Water_Loop, Lighting_Equipment,
Electrical_Equipment.

Run from the project folder:  python dashboard/brick/build_brick.py
"""
import json
import re
from pathlib import Path

import pandas as pd
from rdflib import RDF, RDFS, Graph, Literal, Namespace

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "ac_analysis_outputs"
BRICK = Namespace("https://brickschema.org/schema/Brick#")
REF = Namespace("https://brickschema.org/schema/Brick/ref#")
CU = Namespace("https://example.org/chamchuri5#")
ALL_FLOORS = range(1, 8)
ASSUMED = "ASSUMED – topology not in the dataset; confirm with facilities"


def analysed_floors():
    return sorted(int(p.name.split("_")[1]) for p in OUT.glob("floor_*") if (p / "unit_quality.csv").exists())


def build():
    g = Graph()
    for prefix, ns in [("brick", BRICK), ("ref", REF), ("cu", CU), ("rdfs", RDFS)]:
        g.bind(prefix, ns)
    model = {"floors": [], "zones": [], "units": [], "points": 0, "loops": []}

    building = CU["Chamchuri5"]
    g.add((building, RDF.type, BRICK.Building))
    g.add((building, RDFS.label, Literal("Chamchuri 5")))
    floors = analysed_floors()
    for f in ALL_FLOORS:
        floor = CU[f"Floor{f}"]
        g.add((floor, RDF.type, BRICK.Floor))
        g.add((floor, RDFS.label, Literal(f"Floor {f}")))
        g.add((building, BRICK.hasPart, floor))
        if f not in floors:
            g.add((floor, RDFS.comment, Literal("Shown for context; not in the AC analysis")))

    def point(parent, cls, column, label):
        pt = CU[re.sub(r"[^A-Za-z0-9_]", "_", column)]
        g.add((pt, RDF.type, cls))
        g.add((pt, RDFS.label, Literal(label)))
        g.add((parent, BRICK.hasPoint, pt))
        ref = CU[f"ref_{pt.split('#')[1]}"]
        g.add((ref, RDF.type, REF.TimeseriesReference))
        g.add((ref, REF.hasTimeseriesId, Literal(column)))
        g.add((pt, REF.hasExternalReference, ref))
        model["points"] += 1
        return pt

    for f in floors:
        floor = CU[f"Floor{f}"]
        model["floors"].append(f)
        quality = pd.read_csv(OUT / f"floor_{f}" / "unit_quality.csv", index_col=0)
        loads = pd.read_csv(OUT / f"floor_{f}" / "zone_loads_15min.csv", nrows=1).columns
        zones = sorted({int(m[1]) for c in list(quality.index) + list(loads) if (m := re.search(r"z(\d+)_", c))})
        loop = CU[f"Floor{f}_CHW_assumed"]
        g.add((loop, RDF.type, BRICK.Chilled_Water_Loop))
        g.add((loop, RDFS.label, Literal(f"Floor {f} chilled-water supply (assumed)")))
        g.add((loop, RDFS.comment, Literal(ASSUMED)))
        model["loops"].append({"loop": f"Floor{f}_CHW_assumed", "floor": f, "assumed": True})
        for z in zones:
            zone = CU[f"F{f}_Z{z}"]
            g.add((zone, RDF.type, BRICK.HVAC_Zone))
            g.add((zone, RDFS.label, Literal(f"Floor {f} · Zone {z}" + (" (stairwell/corridor, no AC)" if z == 3 else ""))))
            g.add((floor, BRICK.hasPart, zone))
            model["zones"].append({"zone": f"F{f}_Z{z}", "floor": f, "number": z})
            for suffix, cls, label in [("S1(degC)", BRICK.Zone_Air_Temperature_Sensor, "Zone air temperature"),
                                       ("S1(RH%)", BRICK.Zone_Air_Humidity_Sensor, "Zone air humidity"),
                                       ("S1(lux)", BRICK.Illuminance_Sensor, "Illuminance")]:
                if suffix == "S1(lux)" and f"z{z}_S1(lux)" not in loads:
                    continue
                point(zone, cls, f"F{f}_z{z}_{suffix}", label)
            for kind, col in [("Lighting", f"z{z}_Light(kW)"), ("Plug", f"z{z}_Plug(kW)")]:
                if col in loads:
                    eq = CU[f"F{f}_Z{z}_{kind}"]
                    g.add((eq, RDF.type, BRICK.Lighting_Equipment if kind == "Lighting" else BRICK.Electrical_Equipment))
                    g.add((eq, RDFS.label, Literal(f"Floor {f} · Zone {z} {kind.lower()}")))
                    g.add((eq, BRICK.hasLocation, zone))
                    point(eq, BRICK.Electric_Power_Sensor, f"F{f}_{col}", f"{kind} power")
        for column, row in quality.iterrows():
            m = re.match(r"F(\d+)_z(\d+)_AC(\d+)\(kW\)", column)
            if not m:
                continue
            z, ac = int(m[2]), int(m[3])
            asset = f"F{f}_Z{z}_AC{ac}"
            unit, zone = CU[asset], CU[f"F{f}_Z{z}"]
            g.add((unit, RDF.type, BRICK.Terminal_Unit))
            g.add((unit, RDFS.label, Literal(f"Floor {f} · Zone {z} · AC{ac}")))
            g.add((unit, BRICK.feeds, zone))
            g.add((unit, BRICK.hasLocation, zone))
            g.add((loop, BRICK.feeds, unit))
            g.add((unit, CU.analysed, Literal(bool(row["active_selected"]))))
            point(unit, BRICK.Electric_Power_Sensor, column, "AC electric power")
            model["units"].append({"asset_id": asset, "label": f"Floor {f} · Zone {z} · AC{ac}", "floor": f, "zone": z,
                                   "ac": ac, "position": f"z{z}_AC{ac}", "power_column": column,
                                   "temp_column": f"F{f}_z{z}_S1(degC)", "rh_column": f"F{f}_z{z}_S1(RH%)",
                                   "supply_group": f"Floor{f}_CHW_assumed", "analysed": bool(row["active_selected"])})
    return g, model


def main():
    g, model = build()
    g.serialize(HERE / "chamchuri5.ttl", format="turtle")
    (HERE / "chamchuri5.json").write_text(json.dumps(model, indent=1))
    analysed = [u for u in model["units"] if u["analysed"]]
    print(f"floors {model['floors']}, zones {len(model['zones'])}, AC units {len(model['units'])} "
          f"({len(analysed)} analysed), points {model['points']}, triples {len(g)}")


if __name__ == "__main__":
    main()
