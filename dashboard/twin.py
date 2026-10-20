"""3D model of Chamchuri 5 for the dashboard's digital-twin view.

Geometry is traced from the CU-BEMS paper (Pipattanasomporn et al., Scientific Data 2020): Fig. 2 gives the
Floors 3-7 plan with Zones 1-5, Fig. 1 the 7-storey massing with a taller ground floor and rooftop PV.
It is a schematic to scale (about 52 x 33 m per floor, ~11,700 m2 in total), not a surveyed BIM model.
"""
import plotly.graph_objects as go

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
FOOTPRINT = [(425, 795, 185, 418)]
FLOOR_COUNT = 7
GROUND_HEIGHT, STOREY_HEIGHT, SLAB = 4.5, 3.8, 0.35
GAP = 0.25  # visual gap between neighbouring zones, metres
OVERHANG = -0.9  # slabs project past the facade like the balcony bands in Fig. 1 (negative gap = larger)


def floor_base(floor):
    return 0.0 if floor == 1 else GROUND_HEIGHT + (floor - 2) * STOREY_HEIGHT


def floor_height(floor):
    return GROUND_HEIGHT if floor == 1 else STOREY_HEIGHT


def _to_m(x0, x1, y0, y1, gap=GAP):
    """Pixel rectangle -> metres, with y flipped so the plan's top edge is the building's far side."""
    return ((x0 - PLAN_X0) * PX_TO_M + gap, (x1 - PLAN_X0) * PX_TO_M - gap,
            (PLAN_Y0 + PLAN_H - y1) * PX_TO_M + gap, (PLAN_Y0 + PLAN_H - y0) * PX_TO_M - gap)


def _boxes(rects, z0, z1, gap=GAP):
    """Vertices and triangles for a set of boxes, merged into one mesh."""
    xs, ys, zs, i, j, k = [], [], [], [], [], []
    faces = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    for rect in rects:
        x0, x1, y0, y1 = _to_m(*rect, gap=gap)
        base = len(xs)
        for z in (z0, z1):
            xs += [x0, x1, x1, x0]
            ys += [y0, y0, y1, y1]
            zs += [z, z, z, z]
        for a, b, c in faces:
            i.append(base + a)
            j.append(base + b)
            k.append(base + c)
    return dict(x=xs, y=ys, z=zs, i=i, j=j, k=k)


def zone_centre(zone, floor):
    """A point inside the zone's largest rectangle, at mid-storey height (used for labels)."""
    x0, x1, y0, y1 = _to_m(*max(ZONE_RECTS[zone], key=lambda r: (r[1] - r[0]) * (r[3] - r[2])))
    return (x0 + x1) / 2, (y0 + y1) / 2, floor_base(floor) + floor_height(floor) / 2


def building_figure(zone_styles, analysed_floors, colorbar=None, height=640):
    """zone_styles: {(floor, zone): {"color": hex, "hover": html}} for zones to highlight.
    Other zones on analysed floors and every zone on other floors are drawn as neutral context."""
    fig = go.Figure()
    for floor in range(1, FLOOR_COUNT + 1):
        z0 = floor_base(floor)
        z1 = z0 + floor_height(floor) - SLAB
        # floor slab
        fig.add_trace(go.Mesh3d(**_boxes(FOOTPRINT, z0 - SLAB, z0, gap=OVERHANG), color="#b9b3a8", opacity=1,
                                flatshading=True, hoverinfo="skip", showlegend=False))
        if floor not in analysed_floors:
            fig.add_trace(go.Mesh3d(**_boxes(FOOTPRINT, z0, z1, gap=0.4), color="#cfc8bd", opacity=0.18,
                                    flatshading=True, hovertemplate=f"Floor {floor}<br>not in this analysis<extra></extra>",
                                    showlegend=False))
            continue
        for zone, rects in ZONE_RECTS.items():
            style = zone_styles.get((floor, zone))
            if style is None:  # zone 3 (stairs, no AC) or no data
                fig.add_trace(go.Mesh3d(**_boxes(rects, z0, z1), color="#d9d4cc", opacity=0.35, flatshading=True,
                                        hovertemplate=f"Floor {floor} · Zone {zone}<br>"
                                                      f"{'Stairs and corridor, no AC' if zone == 3 else 'No data'}<extra></extra>",
                                        showlegend=False))
                continue
            fig.add_trace(go.Mesh3d(**_boxes(rects, z0, z1), color=style["color"], opacity=1, flatshading=True,
                                    hovertemplate=style["hover"] + "<extra></extra>", showlegend=False,
                                    lighting=dict(ambient=0.75, diffuse=0.5, specular=0.05)))
    # roof slab and PV array (Fig. 1)
    roof = floor_base(FLOOR_COUNT) + floor_height(FLOOR_COUNT)
    fig.add_trace(go.Mesh3d(**_boxes(FOOTPRINT, roof - SLAB, roof, gap=OVERHANG), color="#b9b3a8", flatshading=True,
                            hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Mesh3d(**_boxes([(470, 750, 200, 400)], roof, roof + 0.25, gap=0), color="#4a5d78",
                            flatshading=True, hovertemplate="Rooftop solar PV<extra></extra>", showlegend=False))
    # floor labels just outside the near long facade, at the left end
    fig.add_trace(go.Scatter3d(x=[-3.5] * len(analysed_floors), y=[-3.5] * len(analysed_floors),
                               z=[floor_base(f) + floor_height(f) / 2 for f in analysed_floors], mode="text",
                               text=[f"<b>Floor {f}</b>" for f in analysed_floors], textposition="middle left",
                               textfont=dict(size=13, color="#8a8984"), hoverinfo="skip", showlegend=False))
    if colorbar:  # invisible marker trace that only carries the colour scale legend
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                   marker=dict(colorscale=colorbar["colorscale"], cmin=colorbar["cmin"],
                                               cmax=colorbar["cmax"], color=[colorbar["cmin"]], showscale=True,
                                               colorbar=dict(title=colorbar["title"], thickness=12, len=0.6))))
    axis = dict(visible=False, showbackground=False)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data",
                                 camera=dict(eye=dict(x=-1.0, y=-1.3, z=0.6), center=dict(x=0, y=0, z=-0.12))))
    return fig
