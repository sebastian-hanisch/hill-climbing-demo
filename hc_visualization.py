"""Plotly-Abbildungen der Hill-Climbing-Demo: Karten mit Tour und Zug, Abstiegskurve, Verteilung der Nachbarn, Sweeps, Mehrfachstart.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import hc_constants as C

TOUR_COLOR = "#4c78a8"
OLD_COLOR = "#e45756"
NEW_COLOR = "#54a24b"
BOUND_COLOR = "#7f7f7f"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _line_trace(xy, edges, color, name, dash=None, width=2.5, showlegend=True):
    x, y = [], []
    for a, b in edges:
        x += [xy[a, 0], xy[b, 0], None]
        y += [xy[a, 1], xy[b, 1], None]
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip", showlegend=showlegend)


def _tour_edges_list(tour):
    t = np.asarray(tour)
    return list(zip(t.tolist(), np.roll(t, -1).tolist()))


def build_instance(xy):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=7, color=TOUR_COLOR, line=dict(width=1, color="white")), name="Stopps", hovertemplate="Stopp %{customdata}<extra></extra>", customdata=np.arange(1, len(xy))))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_tour(xy, tour, removed=(), added=(), title=None):
    """Tour; ein Zug wird hervorgehoben: entfernte Kanten rot gestrichelt, neue Kanten grün (vor dem Zug liegen die entfernten Kanten auf `tour`, danach die neuen)."""
    fig = go.Figure()
    hidden = {tuple(sorted(e)) for e in removed} | {tuple(sorted(e)) for e in added}
    keep = [e for e in _tour_edges_list(tour) if tuple(sorted(e)) not in hidden]
    fig.add_trace(_line_trace(xy, keep, TOUR_COLOR, "Tour"))
    if removed:
        fig.add_trace(_line_trace(xy, removed, OLD_COLOR, "entfernte Kanten", dash="dash", width=3.5))
    if added:
        fig.add_trace(_line_trace(xy, added, NEW_COLOR, "neue Kanten", width=3.5))
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=6, color="white", line=dict(width=1.5, color=TOUR_COLOR)), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.02, y=0.98))
    return _map_layout(fig)


def build_descent(lengths, current, bound):
    """Tourlänge nach jedem Zug; markiert den angezeigten Stand und die Schranke."""
    xs = list(range(len(lengths)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=lengths, mode="lines", line=dict(color=TOUR_COLOR, width=2), name="Tourlänge"))
    fig.add_trace(go.Scatter(x=[current], y=[lengths[current]], mode="markers", marker=dict(size=11, color=NEW_COLOR, line=dict(width=1, color="white")), name="angezeigter Stand"))
    fig.add_hline(y=bound, line=dict(color=BOUND_COLOR, dash="dot"), annotation_text="untere Schranke", annotation_position="bottom right")
    fig.update_xaxes(title_text="Zug")
    fig.update_yaxes(title_text="Länge (km)")
    return _base(fig, 300)


def build_neighbor_deltas(deltas):
    """Verteilung der Längenänderung aller Nachbarn der Startlösung: links von 0 verbessert der Zug."""
    deltas = np.asarray(deltas, dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=deltas[deltas >= 0], marker_color="#bab0ac", name="verschlechtern", nbinsx=40))
    fig.add_trace(go.Histogram(x=deltas[deltas < 0], marker_color=NEW_COLOR, name="verbessern", nbinsx=40))
    fig.update_layout(barmode="overlay", bargap=0.02)
    fig.update_xaxes(title_text="Längenänderung des Nachbarn (km)")
    fig.update_yaxes(title_text="Anzahl Nachbarn")
    return _base(fig, 280)


def build_sweep(rows, param_label, categorical=False, key_labels=None):
    """Abstand zur Schranke (Mittel, Streuung als Band) und Zahl der Züge über die Werte eines Reglers."""
    xs = [r["value"] for r in rows]
    if key_labels:
        xs = [key_labels.get(x, x) for x in xs]
    gap = np.array([r["gap"] for r in rows])
    sd = np.array([r["gap_sd"] for r in rows])
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Abstand zur Schranke (%)", "Züge"), horizontal_spacing=0.12)
    if categorical:
        fig.add_trace(go.Bar(x=xs, y=gap, error_y=dict(type="data", array=sd), marker_color=TOUR_COLOR, name="Abstand"), row=1, col=1)
        fig.add_trace(go.Bar(x=xs, y=[r["moves"] for r in rows], marker_color="#72b7b2", name="Züge"), row=1, col=2)
    else:
        fig.add_trace(go.Scatter(x=xs, y=gap + sd, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=np.maximum(gap - sd, 0), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(76,120,168,0.2)", showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=gap, mode="lines+markers", line=dict(color=TOUR_COLOR), name="Abstand"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[r["moves"] for r in rows], mode="lines+markers", line=dict(color="#72b7b2"), name="Züge"), row=1, col=2)
    fig.update_xaxes(title_text=param_label)
    fig.update_layout(showlegend=False)
    return _base(fig, 320)


def build_comparison(rows):
    """Abstand zur Schranke je Nachbarschaft, gruppiert nach Startlösung und Auswahlregel."""
    fig = go.Figure()
    colors = {("random", "first"): "#4c78a8", ("random", "best"): "#9ecae9", ("nearest", "first"): "#f58518", ("nearest", "best"): "#ffbf79"}
    for st in C.START_LABELS:
        for rule in C.RULE_LABELS:
            sel = [r for r in rows if r["start"] == st and r["rule"] == rule]
            fig.add_trace(go.Bar(x=[C.NEIGHBORHOOD_LABELS[r["neighborhood"]] for r in sel], y=[r["gap"] for r in sel], name=f"{C.START_LABELS[st]}, {C.RULE_LABELS[rule].lower()}", marker_color=colors[(st, rule)]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    return _base(fig, 330)


def build_multistart(report):
    """Histogramm der Endlängen vieler Abstiege und bester Stand nach k Starts."""
    gaps = report["gaps"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Abstand der lokalen Optima zur Schranke (%)", "Bester Abstand nach k Starts (%)"), horizontal_spacing=0.12)
    fig.add_trace(go.Histogram(x=gaps, nbinsx=25, marker_color=TOUR_COLOR, name="lokale Optima"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(1, len(gaps) + 1)), y=report["running_best"], mode="lines", line=dict(color=NEW_COLOR, width=2.5, shape="hv"), name="bester Stand"), row=1, col=2)
    fig.update_xaxes(title_text="Abstand (%)", row=1, col=1)
    fig.update_xaxes(title_text="Zahl der Starts", row=1, col=2)
    fig.update_yaxes(title_text="Anzahl", row=1, col=1)
    fig.update_layout(showlegend=False)
    return _base(fig, 320)


def build_scaling(blocks):
    """Abstand zur Schranke, Züge und bewertete Nachbarn über die Stoppzahl für die Verfahren in `blocks` ({label, rows})."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Abstand zur Schranke (%)", "Züge", "bewertete Nachbarn"), horizontal_spacing=0.08)
    colors = [TOUR_COLOR, "#f58518"]
    for k, blk in enumerate(blocks):
        xs = [r["value"] for r in blk["rows"]]
        for col, key in ((1, "gap"), (2, "moves"), (3, "evaluations")):
            fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in blk["rows"]], mode="lines+markers", line=dict(color=colors[k % 2]), name=blk["label"], showlegend=(col == 1), legendgroup=str(k)), row=1, col=col)
    fig.update_yaxes(type="log", row=1, col=3)
    fig.update_xaxes(title_text="Stopps")
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return _base(fig, 340)


def build_dlb(single, budget_rows):
    """Kandidatenlisten + Don't-Look-Bits gegen den vollen Rescan: links Bewertungen für einen Abstieg gleicher Güte
    (Balken, logarithmisch), rechts Abstand zur Schranke über das Budget (beide mit Neustarts)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Bewertungen für einen Abstieg (log.)", "Abstand zur Schranke über das Budget"), horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=["voller Rescan", "Kandidatenliste + DLB"], y=[single["full_evaluations"], single["dlb_evaluations"]], marker_color=[TOUR_COLOR, NEW_COLOR], showlegend=False), row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    xs = [r["value"] for r in budget_rows]
    fig.add_trace(go.Scatter(x=xs, y=[r["full_gap"] for r in budget_rows], mode="lines+markers", line=dict(color=TOUR_COLOR, width=2.5), name="voller Rescan"), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y=[r["dlb_gap"] for r in budget_rows], mode="lines+markers", line=dict(color=NEW_COLOR, width=2.5), name="Kandidatenliste + DLB"), row=1, col=2)
    fig.update_xaxes(type="log", row=1, col=2, title_text="Budget (Vorschläge)")
    fig.update_yaxes(title_text="Abstand zur Schranke (%)", row=1, col=2)
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return _base(fig, 340)
