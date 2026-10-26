"""Read the Brick model of Chamchuri 5 (dashboard/brick/chamchuri5.ttl).

Queries go through rdflib/SPARQL when it is installed. If it is missing, the same answers come from
chamchuri5.json (written next to the TTL by build_brick.py), so the dashboard still runs.
"""
import functools
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent / "brick"
TTL, JSON = HERE / "chamchuri5.ttl", HERE / "chamchuri5.json"
SUPPLY_NOTE = "assumed: one chilled-water supply per floor (topology is not in the dataset)"

try:
    import rdflib
except ImportError:  # JSON fallback
    rdflib = None

_UNITS_QUERY = """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ref: <https://brickschema.org/schema/Brick/ref#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX cu: <https://example.org/chamchuri5#>
SELECT ?unit ?label ?zone ?floor ?power ?loop ?analysed WHERE {
  ?unit a brick:Terminal_Unit ; rdfs:label ?label ; brick:hasLocation ?zone ; cu:analysed ?analysed ;
        brick:hasPoint/ref:hasExternalReference/ref:hasTimeseriesId ?power .
  ?floor brick:hasPart ?zone .
  ?loop a brick:Chilled_Water_Loop ; brick:feeds ?unit .
  ?zone a brick:HVAC_Zone .
}"""
_ZONE_POINTS_QUERY = """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ref: <https://brickschema.org/schema/Brick/ref#>
SELECT ?zone ?cls ?column WHERE {
  ?zone a brick:HVAC_Zone ; brick:hasPoint ?pt .
  ?pt a ?cls ; ref:hasExternalReference/ref:hasTimeseriesId ?column .
}"""


def _local(iri):
    return str(iri).split("#")[-1]


def load_graph():
    """The rdflib graph, or None when rdflib is missing."""
    if rdflib is None:
        return None
    g = rdflib.Graph()
    g.parse(TTL, format="turtle")
    return g


def load_json():
    return json.loads(JSON.read_text())


def _units_from_graph(g):
    zone_cols = {}
    for zone, cls, column in g.query(_ZONE_POINTS_QUERY):
        zone_cols.setdefault(_local(zone), {})[_local(cls)] = str(column)
    rows = []
    for unit, label, zone, floor, power, loop, analysed in g.query(_UNITS_QUERY):
        asset, zone_id = _local(unit), _local(zone)
        _, z, ac = asset.split("_")
        cols = zone_cols.get(zone_id, {})
        rows.append({"asset_id": asset, "label": str(label), "floor": int(_local(floor).removeprefix("Floor")),
                     "zone": int(z[1:]), "ac": int(ac[2:]), "position": f"{z.lower()}_{ac}", "power_column": str(power),
                     "temp_column": cols.get("Zone_Air_Temperature_Sensor"), "rh_column": cols.get("Zone_Air_Humidity_Sensor"),
                     "supply_group": _local(loop), "analysed": bool(analysed.toPython())})
    return rows


class BrickModel:
    """Query helpers over the Brick graph (or its JSON fallback)."""

    def __init__(self, graph=None, data=None):
        self.graph, self.data = graph, data
        self.backend = "rdflib/SPARQL" if graph is not None else "JSON fallback"
        rows = _units_from_graph(graph) if graph is not None else data["units"]
        self._units = pd.DataFrame(rows).sort_values(["floor", "zone", "ac"]).reset_index(drop=True)

    @classmethod
    def load(cls, force_json=False):
        g = None if force_json else load_graph()
        return cls(g, None if g is not None else load_json())

    def units(self, analysed_only=True):
        """asset id, label, floor, zone, power column, zone temperature/RH columns, supply group."""
        u = self._units
        return (u[u["analysed"]] if analysed_only else u).reset_index(drop=True)

    def peer_groups(self):
        """{position (e.g. 'z1_AC4'): [power columns]}: the same equipment type in the same zone position, across floors."""
        u = self.units()
        return {pos: sorted(g["power_column"]) for pos, g in u.groupby("position")}

    def supply_groups(self):
        """{supply group id: [power columns]} for units fed by the same (assumed) chilled-water loop."""
        u = self.units()
        return {loop: sorted(g["power_column"]) for loop, g in u.groupby("supply_group")}

    def asset_for_column(self, column):
        row = self._units[self._units["power_column"].eq(column)]
        return row.iloc[0].to_dict() if len(row) else None

    def counts(self):
        if self.graph is not None:
            def n(cls):
                return len(list(self.graph.subjects(rdflib.RDF.type, rdflib.URIRef(f"https://brickschema.org/schema/Brick#{cls}"))))
            points = len(list(self.graph.subject_objects(rdflib.URIRef("https://brickschema.org/schema/Brick/ref#hasExternalReference"))))
            return {"floors": n("Floor"), "zones": n("HVAC_Zone"), "AC units": len(self.units()), "points": points}
        d = self.data
        return {"floors": 7, "zones": len(d["zones"]), "AC units": len(self.units()), "points": d["points"]}

    def table(self):
        u = self.units()
        t = u[["label", "floor", "zone", "power_column", "temp_column", "supply_group"]].copy()
        t["supply_group"] = t["supply_group"] + " (assumed)"
        return t.rename(columns={"label": "AC unit", "power_column": "Power point", "temp_column": "Temperature point",
                                 "supply_group": "Supply group", "floor": "Floor", "zone": "Zone"})


@functools.lru_cache(maxsize=1)
def get_model():
    """One shared model per process (parsing the graph takes ~0.5 s, and lookups happen per unit)."""
    return BrickModel.load()
