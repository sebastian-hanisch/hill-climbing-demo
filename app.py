"""Hill Climbing - eine Tour, die im ersten lokalen Optimum stecken bleibt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Hill Climbing (lokale Suche) auf einer Lieferrunde -
und lässt stattdessen das Beispiel wachsen. Erstes Stück der Trajektorien-Metaheuristiken-Linie der "Konzepte"-Reihe: die einfachste Suche, an deren Schwäche (sie bleibt im ersten
lokalen Optimum stecken) die späteren Stücke ansetzen. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time
from dataclasses import replace

import numpy as np
import streamlit as st

import hc_algorithm as A
import hc_constants as C
from hc_evaluation import SWEEP_LABELS, Settings, analyse, comparison_table, dlb_budget_table, dlb_single_descent_table, local_optimum_hierarchy, multi_start_report, scaling_table, sweep, verdict
from hc_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    randomize_start_seed,
    sync_query_params,
)
from hc_visualization import build_comparison, build_descent, build_dlb, build_instance, build_multistart, build_neighbor_deltas, build_scaling, build_sweep, build_tour

st.set_page_config(page_title="Hill Climbing – Sebastian Hanisch", layout="wide")

MOVE_NAMES = {"swap": "Tausch", "2opt": "2-opt", "oropt": "Or-opt"}


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base, (10, 20, 40, 60, 100, 150) if param == "n" else None)


@st.cache_data(show_spinner=False)
def _comparison(base):
    return comparison_table(base)


@st.cache_data(show_spinner=False)
def _multi_start(settings):
    return multi_start_report(settings, C.MULTI_START_K)


@st.cache_data(show_spinner=False)
def _hierarchy(base):
    return local_optimum_hierarchy(base)


@st.cache_data(show_spinner=False)
def _scaling(base):
    return scaling_table(base)


@st.cache_data(show_spinner=False)
def _dlb_single(base):
    return dlb_single_descent_table(base)


@st.cache_data(show_spinner=False)
def _dlb_budget(base):
    return dlb_budget_table(base)


def _moves_text(kinds):
    return ", ".join(f"{v} × {MOVE_NAMES[k]}" for k, v in kinds.items()) or "keine"


st.title("🧗 Hill Climbing – eine Lieferrunde, die im ersten Optimum stecken bleibt")
st.markdown(
    """
Wie verbessert man eine Lieferrunde, ohne sie neu zu planen? Der einfachste Weg: **kleine Änderungen ausprobieren** – zwei Stopps vertauschen, ein Stück der Tour umdrehen (**2-opt**), ein Stück an anderer Stelle einfügen (**Or-opt**) – und jede behalten, die die Tour kürzer macht.
Wenn keine Änderung mehr hilft, ist die Tour ein **lokales Optimum**. Der Haken: es ist nicht das beste. Wie weit es davon entfernt liegt und wovon das abhängt – Startlösung, Nachbarschaft, Auswahlregel, Größe –, misst diese Demo,
gegen eine ehrliche **untere Schranke** der kürzesten Tour, mit Siegen und Niederlagen.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo – erstes Stück der Trajektorien-Metaheuristiken-Linie der \"Konzepte\"-Reihe – **ein** Verfahren an einem wachsenden Beispiel. "
    "Die Linie hat keinen Konvergenzpunkt: die nächsten Stücke (Simulated Annealing, Iterated Local Search mit VNS und ALNS, Tabu Search, GRASP) beheben jeweils **dieselbe** Schwäche auf einem anderen Weg. "
    "Vehikel ist ein einzelnes Fahrzeug, das ein Depot in der Mitte und n Kundenstopps in einem 100 × 100-km-Gebiet der Reihe nach anfährt (euklidische Entfernungen)."
)

with st.expander("So funktioniert Hill Climbing", expanded=True):
    st.markdown(
        """
1. **Startlösung.** Eine beliebige Rundtour: zufällig gemischt oder mit dem **Nächsten Nachbarn** gebaut (immer zum nächsten unbesuchten Stopp).
2. **Nachbarschaft.** Alle Touren, die durch **eine** kleine Änderung entstehen: **Tausch** zweier Stopps, **2-opt** (zwei Kanten löschen, das Stück dazwischen umdrehen – das entfernt Kreuzungen), **Or-opt** (ein Stück aus 1 bis 3 Stopps herausnehmen und an anderer Stelle einfügen, auch umgekehrt), oder 2-opt und Or-opt zusammen.
   Die Längenänderung jedes Nachbarn ergibt sich aus wenigen Kanten – die Tour muss nicht neu gemessen werden.
3. **Auswahlregel.** **Erste Verbesserung:** der erste Nachbar in fester Reihenfolge, der die Tour verkürzt, wird genommen. **Beste Verbesserung** (steilster Abstieg): alle Nachbarn bewerten, den mit der größten Verkürzung nehmen.
4. **Lokales Optimum.** Sobald kein Nachbar mehr kürzer ist, hält die Suche an. Sie kann nie eine Tour verlassen, aus der jeder Schritt bergauf führt – auch wenn eine viel kürzere Tour existiert.
5. **Bewertung.** Der Abstand zur **1-Baum-Schranke** (Held-Karp): eine untere Schranke der kürzesten Tour, die ohne Löser berechnet wird. Sie liegt bei gleichverteilten 60 Stopps im Mittel nur 0.5 % unter dem echten Optimum.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:4], preset_names[4:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider(
        "Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
        help="Anzahl der Kundenstopps (das Depot kommt dazu). Bei 2-opt mit zufälliger Startlösung und erster Verbesserung (Mittel über fünf feste Instanzen mit je drei Startlösungen) endet die Tour bei 10 / 20 / 40 / 60 / 100 / 150 / 200 Stopps "
             "1.7 % / 1.9 % / 7.7 % / 7.9 % / 9.1 % / 9.1 % / 9.5 % über der Schranke, nach 12 / 45 / 109 / 211 / 403 / 672 / 991 Zügen und 0.2 / 2.0 / 16 / 74 / 448 / 1 542 / 3 973 Tausend bewerteten Nachbarn. Mit Or-opt rechnet ein Lauf bei 200 Stopps einige Sekunden.",
    )
    cluster_share = st.slider(
        "Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
        help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet. Bei 0 / 25 / 50 / 75 / 100 % endet 2-opt (zufälliger Start, erste Verbesserung, 60 Stopps) 7.9 / 8.0 / 5.7 / 4.1 / 4.4 % über der Schranke. "
             "Bei 75 % ist der Nächste Nachbar mit erster Verbesserung schlechter als ein zufälliger Start (6.7 % gegen 4.1 %).",
    )
    neighborhood = st.selectbox(
        "Nachbarschaft", list(C.NEIGHBORHOOD_LABELS), key="neighborhood_select", format_func=lambda k: C.NEIGHBORHOOD_LABELS[k],
        help="Was als „kleine Änderung“ gilt. Bei zufälligem Start und erster Verbesserung (60 Stopps): Tausch 54.5 % über der Schranke (mit Nächstem Nachbarn 16.3 %), 2-opt 7.9 %, Or-opt 10.4 %, 2-opt + Or-opt 3.7 %. "
             "Tausch lässt im Mittel 13 Kreuzungen stehen, Or-opt 2.7, 2-opt und die Kombination keine. Anzahl der Nachbarn einer Tour bei 60 Stopps: Tausch 1 769, 2-opt 1 769, Or-opt 10 614, 2-opt + Or-opt 12 383.",
    )
    start = st.radio(
        "Startlösung", list(C.START_LABELS), key="start_radio", format_func=lambda k: C.START_LABELS[k], horizontal=True,
        help="Zufällig: gemischte Reihenfolge (im Mittel 420 % über der Schranke). Nächster Nachbar: immer zum nächsten Stopp (21 %). Bei 2-opt (60 Stopps) endet die Suche mit erster Verbesserung bei 7.9 % (zufällig) gegen 7.3 % (Nächster Nachbar) nach 211 gegen 17 Zügen, "
             "mit steilstem Abstieg bei 7.9 % gegen 4.0 %: eine gute Startlösung spart vor allem Züge.",
    )
    rule = st.radio(
        "Auswahlregel", list(C.RULE_LABELS), key="rule_radio", format_func=lambda k: C.RULE_LABELS[k], horizontal=True,
        help="Erste Verbesserung: erster kürzender Nachbar in fester Reihenfolge; Beste Verbesserung: der kürzeste unter allen. Bei 2-opt und zufälligem Start (60 Stopps) enden beide bei 7.9 %, aber mit 211 gegen 57 Zügen und 74 gegen 102 Tausend bewerteten Nachbarn; "
             "mit 2-opt + Or-opt bei 3.7 % gegen 3.8 %, 220 gegen 45 Zügen und 107 gegen 573 Tausend Bewertungen.",
    )
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für die Lage der Stopps.")
    if start == "random":
        start_seed = st.number_input(
            "Zufalls-Seed der Startlösung", *bounds("start_seed_input"), key="start_seed_input", step=1,
            help="Welche zufällige Reihenfolge die Suche als Startlösung bekommt. Sie entscheidet, wo die Suche stecken bleibt: bei 60 Stopps, 2-opt und erster Verbesserung streut der Abstand zur Schranke über die 15 Läufe "
                 "(fünf Instanzen × drei Startlösungen) von 0.9 % bis 14.1 % (Standardabweichung 4.0 Prozentpunkte).",
        )
        st.session_state["_kept_start_seed_input"] = start_seed
        st.button("🎲 Neue Startlösung würfeln", width="stretch", on_click=randomize_start_seed, help="Würfelt eine andere zufällige Startlösung für dieselbe Instanz.")
    else:
        start_seed = int(st.session_state.get("_kept_start_seed_input", C.DEFAULT_START_SEED))

sync_query_params({
    "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed), "neighborhood_select": neighborhood, "start_radio": start, "rule_radio": rule, "start_seed_input": int(start_seed),
})

settings = Settings(int(n_stops), int(cluster_share), int(seed), neighborhood, start, rule, int(start_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
    hist = a.descent
    steps = hist.steps
    n_moves = hist.n_moves
xy = a.inst.xy
lengths = [s.length for s in steps]
code = verdict(a)
data_key = settings

# --- Hill Climbing in Aktion ------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Hill Climbing in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Startlösung", 3: "3 · Nachbarschaft", 4: "4 · Abstieg", 5: "5 · Lokales Optimum"}
if "hc_step" not in st.session_state or st.session_state.get("hc_step_owner") != data_key:
    st.session_state["hc_step"] = 1
    st.session_state["hc_step_owner"] = data_key
    st.session_state.pop("hc_move", None)
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="hc_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

move = n_moves
play_moves = False
if step == 4 and n_moves > 0:
    mv_col, mvplay_col = st.columns([5, 2])
    with mv_col:
        move = st.slider("Zug", 0, n_moves, value=n_moves, key="hc_move", help="Wie viele verbessernde Züge schon ausgeführt sind (0 = Startlösung).")
    with mvplay_col:
        play_moves = st.button("▶️ Züge abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_moves == 0:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_moves, min(n_moves + 1, 40))})


def _move_edges(m):
    """Entfernte und neue Kanten des m-ten Zuges (m >= 1)."""
    return A.move_edges(steps[m - 1].tour, steps[m].move)


def _render(current_step, m):
    with view_slot.container():
        if current_step == 1:
            st.markdown(f"**{a.inst.n} Kundenstopps und das Depot (Stern)** – {a.inst.cluster_share} % der Stopps in Gruppen")
            st.plotly_chart(build_instance(xy), width="stretch", key="s1_map")
        elif current_step == 2:
            c1, c2 = st.columns(2)
            c1.markdown(f"**Startlösung: {C.START_LABELS[settings.start].lower()}** – Länge {a.start_length:,.0f} km".replace(",", "."))
            c1.plotly_chart(build_tour(xy, a.start_tour, title=None), width="stretch", key="s2_map")
            c2.markdown("**Was an der Startlösung auffällt**")
            c2.metric("Abstand zur Schranke", f"{a.start_gap:.0f} %", help="Länge der Startlösung gegenüber der unteren Schranke.")
            c2.metric("Kreuzungen", f"{a.crossings_start}", help="Paare von Tourkanten, die sich schneiden. Eine kürzeste Tour hat keine.")
        elif current_step == 3:
            c1, c2 = st.columns(2)
            deltas = A.neighbor_deltas(a.start_tour, a.D, settings.neighborhood)
            if n_moves == 0:
                c1.markdown("**Die Startlösung ist schon ein lokales Optimum** – kein Nachbar verkürzt sie.")
                c1.plotly_chart(build_tour(xy, a.start_tour), width="stretch", key="s3_map")
            else:
                removed, added = _move_edges(1)
                mv = steps[1].move
                c1.markdown(f"**Erster Zug: {MOVE_NAMES[mv[0]]}, {abs(mv[-1]):.1f} km kürzer** (rot gestrichelt: entfernt, grün: neu)")
                c1.plotly_chart(build_tour(xy, steps[0].tour, removed, added), width="stretch", key="s3_map")
            c2.markdown("**Längenänderung aller Nachbarn der Startlösung**")
            c2.plotly_chart(build_neighbor_deltas(deltas), width="stretch", key="s3_hist")
        elif current_step == 4:
            c1, c2 = st.columns(2)
            if m == 0:
                c1.markdown(f"**Vor dem ersten Zug** – Länge {steps[0].length:,.0f} km".replace(",", "."))
                c1.plotly_chart(build_tour(xy, steps[0].tour), width="stretch", key=f"s4_map_{m}")
            else:
                removed, added = _move_edges(m)
                mv = steps[m].move
                c1.markdown(f"**Zug {m} von {n_moves}: {MOVE_NAMES[mv[0]]}, {abs(mv[-1]):.1f} km kürzer** – Länge {steps[m].length:,.0f} km".replace(",", "."))
                c1.plotly_chart(build_tour(xy, steps[m].tour, removed, added), width="stretch", key=f"s4_map_{m}")
            c2.markdown("**Tourlänge über die Züge**")
            c2.plotly_chart(build_descent(lengths, m, a.bound), width="stretch", key=f"s4_curve_{m}")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(f"**Lokales Optimum nach {n_moves} Zügen** – Länge {a.length:,.0f} km".replace(",", "."))
            c1.plotly_chart(build_tour(xy, hist.tour), width="stretch", key="s5_map")
            c2.markdown("**Vorher und nachher**")
            c2.table({"": ["Länge (km)", "Abstand zur Schranke", "Kreuzungen"],
                      "Startlösung": [f"{a.start_length:.0f}", f"{a.start_gap:.1f} %", f"{a.crossings_start}"],
                      "lokales Optimum": [f"{a.length:.0f}", f"{a.gap:.1f} %", f"{a.crossings_end}"]})
            final_deltas = A.neighbor_deltas(hist.tour, a.D, settings.neighborhood)
            c2.caption(f"Kontrolle: von {len(final_deltas):,} Nachbarn der Endtour verkürzt keiner die Tour (bester: {final_deltas.min():+.2f} km).".replace(",", "."))


if auto_play:
    for s in STEP_LABELS:
        if s == 4:
            for f in _frames():
                _render(4, f)
                time.sleep(0.12)
            time.sleep(0.6)
        else:
            _render(s, n_moves)
            time.sleep(1.2)
    step = 5
elif play_moves:
    for f in _frames():                                                    # das letzte Bild ist der Endstand (Zug n_moves)
        _render(4, f)
        time.sleep(0.12)
else:
    _render(step, move)

if step == 1:
    st.caption(f"{a.inst.n} Stopps; die untere Schranke der kürzesten Rundtour liegt bei {a.bound:,.0f} km (1-Baum-Schranke, Held-Karp).".replace(",", "."))
elif step == 2:
    st.caption(f"Die Startlösung ist {a.start_gap:.0f} % länger als die Schranke und kreuzt sich {a.crossings_start}-mal. "
               "Ein zufälliges Mischen ist die schlechteste vernünftige Wahl; der Nächste Nachbar baut eine lange Tour mit wenigen langen Kanten am Schluss.")
elif step == 3:
    total = len(A.neighbor_deltas(a.start_tour, a.D, settings.neighborhood))
    improving = int((A.neighbor_deltas(a.start_tour, a.D, settings.neighborhood) < -1e-9).sum())
    st.caption(f"Die Nachbarschaft ({C.NEIGHBORHOOD_LABELS[settings.neighborhood]}) hat bei dieser Startlösung {total:,} Nachbarn, davon verkürzen {improving:,} die Tour. "
               f"{'Erste Verbesserung nimmt den ersten davon in fester Reihenfolge' if settings.rule == 'first' else 'Beste Verbesserung nimmt den kürzesten von allen'}.".replace(",", "."))
elif step == 4:
    st.caption(f"{n_moves} Züge ({_moves_text(hist.moves_by_kind())}); {hist.evaluations:,} bewertete Nachbarn insgesamt (nach jedem Zug beginnt die Suche von vorn).".replace(",", "."))
else:
    st.caption(f"Kein Nachbar verkürzt die Tour mehr – das lokale Optimum liegt {a.gap:.1f} % über der Schranke.")

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Suche gefunden hat")
st.caption(
    "**Abstand zur Schranke:** Länge der Endtour gegenüber einer unteren Schranke der kürzesten Rundtour (1-Baum, Held-Karp) in Prozent. Das wahre Optimum liegt bei gleichverteilten 60 Stopps im Mittel 0.5 % über der Schranke, bei gruppierten Stopps etwas mehr (1.1 %), "
    "der Abstand überschätzt die echte Lücke also um diesen Betrag. **Bewertete Nachbarn:** wie viele Nachbarn insgesamt auf ihre Längenänderung geprüft wurden – das Maß für den Aufwand, unabhängig vom Rechner."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Länge", f"{a.length:.0f} km", delta=f"Start {a.start_length:.0f} km", delta_color="off", help="Länge der Endtour; im Delta die der Startlösung.")
m2.metric("Abstand zur Schranke", f"{a.gap:.1f} %", delta=f"Start {a.start_gap:.0f} %", delta_color="off", help="Prozent über der unteren Schranke (1-Baum).")
m3.metric("Züge", f"{n_moves}", delta=_moves_text(hist.moves_by_kind()), delta_color="off", help="Zahl der verbessernden Züge bis zum lokalen Optimum, nach Art.")
m4.metric("Bewertete Nachbarn", f"{hist.evaluations:,}".replace(",", "."), delta=f"{a.seconds * 1000:.0f} ms", delta_color="off", help="Gesamtzahl der bewerteten Nachbarn; im Delta die Rechenzeit (rechnerabhängig).")

if code == "near_optimal":
    st.success(f"✅ Fast optimal: nur {a.gap:.1f} % über der unteren Schranke. Dass der Abstieg so weit kommt, hängt an dieser Instanz und dieser Startlösung – eine andere Startlösung endet in einem anderen Optimum (siehe Mehrfachstart unten).")
elif code == "stuck":
    st.warning(f"⚠️ Steckengeblieben: ein lokales Optimum {a.gap:.1f} % über der Schranke – kein Zug der Nachbarschaft ({C.NEIGHBORHOOD_LABELS[settings.neighborhood]}) verkürzt die Tour mehr, sie ist aber nicht die kürzeste. "
               "Eine andere Startlösung oder eine größere Nachbarschaft endet woanders.")
elif code == "crossings":
    st.warning(f"⚠️ Die Nachbarschaft ist zu klein: das lokale Optimum hat noch {a.crossings_end} Kreuzung{'en' if a.crossings_end != 1 else ''} und liegt {a.gap:.1f} % über der Schranke. Jede Kreuzung ließe sich mit einem 2-opt-Zug entfernen – "
               f"aber {C.NEIGHBORHOOD_LABELS[settings.neighborhood]} kennt diesen Zug nicht.")
else:
    st.warning("⚠️ Die Suche wurde vor dem lokalen Optimum abgebrochen.")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    st.table({"Kennzahl": ["Länge der Startlösung", "Länge der Endtour", "untere Schranke", "Abstand zur Schranke", "Kreuzungen (Start → Ende)", "Züge (nach Art)", "bewertete Nachbarn", "Rechenzeit"],
              "Wert": [f"{a.start_length:.1f} km", f"{a.length:.1f} km", f"{a.bound:.1f} km", f"{a.gap:.2f} %", f"{a.crossings_start} → {a.crossings_end}", f"{n_moves} ({_moves_text(hist.moves_by_kind())})",
                       f"{hist.evaluations:,}".replace(",", "."), f"{a.seconds * 1000:.0f} ms"]})
with d2:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Nachbarschaft", "Startlösung", "Auswahlregel", "Stopps (ohne Depot)", "Gruppenanteil"],
              "Einstellung": [C.NEIGHBORHOOD_LABELS[settings.neighborhood], C.START_LABELS[settings.start], C.RULE_LABELS[settings.rule], f"{a.inst.n}", f"{a.inst.cluster_share} %"]})
    st.caption("Die Bewertung aller Nachbarn ist in numpy vektorisiert; die Zahl der bewerteten Nachbarn zählt trotzdem jeden einzelnen (bei erster Verbesserung nur bis zum ersten kürzenden Nachbarn). Die Rechenzeit hängt vom Rechner ab, nur die Größenordnung zählt.")

st.markdown("---")

# --- Sweeps -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Größe, Gruppen, Nachbarschaft, Start und Regel ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
base_sweep = replace(settings, seed=0, start_seed=0)
if st.button("Sweep über 5 feste Instanzen berechnen (dauert etwa 10 bis 30 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen × 3 Startlösungen..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    categorical = sweep_param in ("neighborhood", "start", "rule")
    labels = {"neighborhood": C.NEIGHBORHOOD_LABELS, "start": C.START_LABELS, "rule": C.RULE_LABELS}.get(sweep_param)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], categorical=categorical, key_labels=labels), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band bzw. Balken) über 5 feste Instanzen (Seeds 100000–100004, getrennt vom Seed oben) mit je drei zufälligen Startlösungen (ein Lauf bei Nächstem Nachbarn); alle anderen Regler wie in der Seitenleiste. "
               "Die Streuung ist die Standardabweichung der Läufe, nicht des Mittels. Bei 200 Stopps dauert ein Lauf mit Or-opt einige Sekunden; der n-Sweep endet deshalb bei 150 Stopps (das Experiment „Skalierung“ geht bis 200).")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Nachbarschaft × Startlösung × Auswahlregel")
if st.button("Alle 16 Kombinationen vergleichen (dauert etwa 20 Sekunden)", key="comparison_start"):
    st.session_state["comparison_on"] = True
if st.session_state.get("comparison_on"):
    base_cmp = Settings(n=settings.n, cluster_share=settings.cluster_share)
    with st.spinner("Rechne 4 Nachbarschaften × 2 Startlösungen × 2 Regeln × 5 Instanzen..."):
        cmp_rows = _comparison(base_cmp)
    st.plotly_chart(build_comparison(cmp_rows), width="stretch", key="comparison_chart")
    st.table({"Nachbarschaft": [C.NEIGHBORHOOD_LABELS[r["neighborhood"]] for r in cmp_rows], "Startlösung": [C.START_LABELS[r["start"]] for r in cmp_rows], "Regel": [C.RULE_LABELS[r["rule"]] for r in cmp_rows],
              "Abstand (%)": [f"{r['gap']:.1f}" for r in cmp_rows], "Züge": [f"{r['moves']:.0f}" for r in cmp_rows], "bewertete Nachbarn": [f"{r['evaluations']:,.0f}".replace(",", ".") for r in cmp_rows],
              "Kreuzungen": [f"{r['crossings']:.1f}" for r in cmp_rows]})
    st.caption(f"Mittel über 5 feste Instanzen ({settings.n} Stopps, {settings.cluster_share} % in Gruppen), bei zufälligem Start je drei Startlösungen. Bei 60 gleichverteilten Stopps: die Kombination aus 2-opt und Or-opt ist in jeder Zeile besser als jede Nachbarschaft allein (3.4 bis 3.8 %); "
               "die Startlösung wirkt bei 2-opt nur mit steilstem Abstieg (4.0 % gegen 7.3 bis 7.9 %), und der Tausch braucht die gute Startlösung (16 % gegen 54 %).")

st.markdown("---")

st.subheader("🔬 Mehrfachstart: wie sehr entscheidet die Startlösung?")
if st.button("100 Abstiege aus zufälligen Startlösungen berechnen (dauert etwa 10 Sekunden, mit Or-opt länger)", key="multistart_start"):
    st.session_state["multistart_on"] = True
if st.session_state.get("multistart_on"):
    with st.spinner("Rechne 100 Abstiege..."):
        ms = _multi_start(settings)
    st.plotly_chart(build_multistart(ms), width="stretch", key="multistart_chart")
    g = ms["gaps"]
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Schlechtester / bester Abstieg", f"{g.max():.1f} % / {g.min():.1f} %", help="Abstand zur Schranke des schlechtesten und des besten der 100 lokalen Optima.")
    q2.metric("Verschiedene lokale Optima", f"{ms['distinct']} von {len(g)}", help="Wie viele der 100 Endtouren verschieden sind (gleiche Tour = gleiche Kantenmenge).")
    q3.metric("Höchstens 5 % über der Schranke", f"{(g <= 5).mean():.0%}", help="Anteil der Abstiege, die höchstens 5 % über der Schranke enden.")
    q4.metric("Gemeinsame Kanten mit der besten Tour", f"{ms['edge_share_mean']:.0%}", help="Mittlerer Anteil der Kanten eines lokalen Optimums, die auch in der besten der 100 Touren vorkommen.")
    st.caption("Alle Abstiege auf derselben Instanz, nur die Startlösung wechselt. Fast jeder Abstieg endet in einem **anderen** lokalen Optimum, und nur ein kleiner Teil kommt der besten Tour nahe – ein neuer Start ist ein neuer Versuch (der Vorgriff auf **GRASP** und **Iterated Local Search**). "
               "Die Optima sind sich aber ähnlich: sie teilen den größten Teil ihrer Kanten mit der besten gefundenen Tour, ein Hinweis, dass die guten Touren nahe beieinander liegen (deshalb lohnt es, von einem Optimum aus zu stören statt neu zu starten). "
               "Bei 60 gleichverteilten Stopps, 2-opt, zufälligem Start, erster Verbesserung (Mittel über 5 Instanzen): bester Abstieg von 1 / 5 / 20 / 100 Starts 7.9 / 4.0 / 1.8 / 1.2 %; 4 % der Abstiege enden höchstens 2 %, 25 % höchstens 5 % über der Schranke; im Mittel 99 von 100 Optima sind verschieden, 74 % der Kanten stimmen mit der besten Tour überein.")

st.markdown("---")

st.subheader("🔬 Ein 2-opt-Optimum ist kein Or-opt-Optimum")
if st.button("2-opt-Optima mit Or-opt weitersuchen (dauert etwa 20 Sekunden)", key="hierarchy_start"):
    st.session_state["hierarchy_on"] = True
if st.session_state.get("hierarchy_on"):
    base_h = Settings(n=settings.n, cluster_share=settings.cluster_share, rule=settings.rule)
    with st.spinner("Rechne 15 Abstiege mit 2-opt, dann mit 2-opt + Or-opt..."):
        hy = _hierarchy(base_h)
    h1, h2 = st.columns(2)
    h1.metric("2-opt-Optima, die weiter verbessert werden", f"{hy['share_not_optimal']:.0%}", help="Anteil der 2-opt-Optima, in denen ein Or-opt-Zug (oder ein weiterer 2-opt-Zug nach Or-opt) die Tour noch verkürzt.")
    h2.metric("Weitere Verkürzung", f"{hy['mean_improvement']:.1f} %", help="Mittlere Verkürzung der Tour in Prozent der Länge des 2-opt-Optimums.")
    st.caption(f"15 Abstiege ({settings.n} Stopps, {settings.cluster_share} % in Gruppen, Auswahlregel {C.RULE_LABELS[settings.rule].lower()}): erst 2-opt bis zum Optimum, dann von dort 2-opt + Or-opt. "
               "Ein lokales Optimum gehört zu einer Nachbarschaft – wer die Nachbarschaft wechselt, ist nicht mehr im Optimum. Das ist der Ansatz von **VNS** (systematischer Wechsel der Nachbarschaft, ein eigenes Stück der Linie). "
               "Bei 60 gleichverteilten Stopps: alle 15 2-opt-Optima lassen sich weiter verbessern, im Mittel um 3.8 %.")

st.markdown("---")

st.subheader("🔬 Skalierung: was kostet ein größeres Problem?")
if st.button("Stopps von 20 bis 200 durchfahren (dauert etwa 30 Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    base_sc = Settings(cluster_share=settings.cluster_share)
    with st.spinner("Rechne 6 Größen × 2 Verfahren × 5 Instanzen..."):
        sc = _scaling(base_sc)
    st.plotly_chart(build_scaling(sc), width="stretch", key="scaling_chart")
    st.caption(f"Mittel über 5 feste Instanzen ({settings.cluster_share} % in Gruppen), bei zufälligem Start je drei Startlösungen; die y-Achse rechts ist logarithmisch. Bei gleichverteilten Stopps: 2-opt mit zufälligem Start endet bei 20 / 40 / 60 / 100 / 150 / 200 Stopps 1.9 / 7.7 / 7.9 / 9.1 / 9.1 / 9.5 % über der Schranke "
               "(Züge 45 / 109 / 211 / 403 / 672 / 991, bewertete Nachbarn 2.0 / 15.9 / 73.9 / 448 / 1 542 / 3 973 Tausend); 2-opt + Or-opt mit Nächstem Nachbarn und steilstem Abstieg bei 0.05 / 1.1 / 3.4 / 4.4 / 5.3 / 5.1 % (Züge 3 / 6 / 9 / 18 / 27 / 31). "
               "Die Lücke wächst kaum mit der Größe, die Zahl der Züge etwa mit n und die der bewerteten Nachbarn etwa mit n³ – weil nach jedem Zug alle Nachbarn neu bewertet werden (kein Nachbarschaftslisten- und Don't-Look-Bit-Trick).")

st.markdown("---")

st.subheader("🔬 Kandidatenlisten + Don't-Look-Bits: was kostet die einfache Bewertung?")
if settings.neighborhood != "2opt":
    st.caption("Dieses Experiment ist nur für die Nachbarschaft 2-opt gemessen (Kandidatenlisten und Don't-Look-Bits für Or-opt und Tausch sind nicht Teil dieser Demo). "
               "Wählen Sie 2-opt in der Seitenleiste, um es zu berechnen.")
else:
    if st.button("Kandidatenliste + Don't-Look-Bits gegen vollen Rescan berechnen (dauert etwa 30 Sekunden)", key="dlb_start"):
        st.session_state["dlb_on"] = True
    if st.session_state.get("dlb_on"):
        base_dlb = Settings(cluster_share=settings.cluster_share)
        with st.spinner("Rechne einen Abstieg und 5 Budgets × 5 Instanzen × 3 Ketten mit Neustarts..."):
            dlb_single = _dlb_single(base_dlb)
            dlb_rows = _dlb_budget(base_dlb)
        st.plotly_chart(build_dlb(dlb_single, dlb_rows), width="stretch", key="dlb_chart")
        d1, d2 = st.columns(2)
        d1.metric("Bewertungen für ~7 % über der Schranke", f"{dlb_single['dlb_evaluations']:,.0f}".replace(",", "."), delta=f"voller Rescan {dlb_single['full_evaluations']:,.0f}".replace(",", "."), delta_color="off",
                  help="Ein Abstieg vom selben Start: Kandidatenliste (5 nächste Knoten) + Don't-Look-Bits gegen den vollen Rescan, gleiche Güte.")
        mid = dlb_rows[len(dlb_rows) // 2]
        d2.metric(f"Neustarts bei {mid['value']:,.0f} Vorschlägen".replace(",", "."), f"{mid['dlb_starts']:.0f}", delta=f"voller Rescan {mid['full_starts']:.1f}", delta_color="off",
                  help="Wie viele Abstiege bei gleichem Bewertungsbudget hineinpassen (der erste läuft immer zu Ende).")
        st.caption("Mittel über 5 feste Instanzen (60 Stopps, gleichverteilt oder wie oben eingestellt) mit je drei Ketten; die Kandidatenliste hat immer 5 nächste Knoten je Stopp, unabhängig von der Stoppzahl in der Seitenleiste. "
                   "Ein Abstieg erreicht dieselbe Güte (≈7 % über der Schranke) mit rund 650 statt 74 000 bewerteten Nachbarn – das Hundertfache weniger; bei gleichem Budget reicht das für weit mehr Neustarts: "
                   "25 / 100 / 200 / 500 Tausend / 1 Million Vorschläge geben 1.3 / 1.0 / 0.8 / 0.7 / 0.7 % über der Schranke (voller Rescan mit Neustarts: 7.9 / 7.9 / 4.9 / 2.9 / 2.5 %, da ein einzelner Abstieg schon 74 Tausend braucht). "
                   "Bei 60 Stopps landen dabei nur noch rund die Hälfte der Kandidatenlisten-Abstiege auf einem echten 2-opt-Optimum (gegen 100 % bei kleinen Instanzen) – die Kandidatenliste kostet Exaktheit, aber hier nicht Güte. "
                   "Vergleich zur [Simulated-Annealing-Demo](https://sebastianhanisch-simulated-annealing-demo.streamlit.app/): bei 200 Tausend Vorschlägen schlägt Hill Climbing mit Neustarts dort Simulated Annealing (0.8 % gegen 1.4 %), bei 1 Million liegen beide gleichauf (0.7 %) - Details dort.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur verbessernde Züge führen zum Ziel** | Die Suche bleibt im ersten lokalen Optimum stecken: 2-opt von zufälligen Startlösungen endet bei 60 Stopps im Mittel **7.9 %** über der Schranke (Streuung je Lauf von 0.9 bis 14.1 %). | **Simulated Annealing** (nimmt Verschlechterungen an), **Tabu Search** (Gedächtnis gegen Rückwege) |
| **Die Startlösung ist gleichgültig** | 100 Abstiege aus zufälligen Startlösungen: nur 4 % enden höchstens 2 %, 25 % höchstens 5 % über der Schranke; der beste von 1 / 5 / 20 / 100 Starts liegt bei 7.9 / 4.0 / 1.8 / 1.2 %. Neustarts helfen, kosten aber jeder einen ganzen Abstieg. | **GRASP** (randomisierte Konstruktion, viele Starts), **Iterated Local Search** (stört ein Optimum statt neu zu starten) |
| **Die Nachbarschaft ist groß genug** | Tausch allein bleibt bei **54.5 %** (16.3 % vom Nächsten Nachbarn) mit 13 Kreuzungen; Or-opt allein lässt 2.7 Kreuzungen stehen. Ein 2-opt-Optimum ist in 15 von 15 Fällen kein Optimum von 2-opt + Or-opt (weitere Verkürzung 3.8 %). | **VNS** (wechselt die Nachbarschaft systematisch), **ALNS** (lernt, welche Umbauten sich lohnen) |
| **Alle Nachbarn zu bewerten ist billig** | Die Zahl der bewerteten Nachbarn wächst etwa mit n³: 74 Tausend bei 60, **4.0 Millionen** bei 200 Stopps (Züge etwa 2- bis 5-mal n). Die Demo bewertet im Hauptteil nach jedem Zug alle Nachbarn neu. | Nachbarschaftslisten und Don't-Look-Bits (Experiment oben: bei 60 Stopps nur noch ~650 Bewertungen für dieselbe Güte) |
| **Die Schranke ist das Optimum** | Die 1-Baum-Schranke liegt bei gleichverteilten 60 Stopps im Mittel 0.5 % unter dem Optimum, bei gruppierten 1.1 % (in einem Einzelfall 3.7 %): der angezeigte Abstand überschätzt die echte Lücke. | Exakte Verfahren (CP-SAT, Branch-and-Cut; in den Tests der Demo als Kontrolle) |
"""
)
st.caption(
    "Die Nachbarn der Trajektorien-Metaheuristiken-Linie (noch nicht gebaut): Simulated Annealing, Iterated Local Search mit VNS und ALNS, Tabu Search und GRASP; mit dem genetischen Algorithmus der Populations-Linie ergäbe sich später ein Memetischer Algorithmus. "
    "Die Wurzel ist bewusst die einfachste Suche: nur bergab."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Kürzeste Rundtour über $N = n+1$ Knoten (Depot + Stopps) mit euklidischen Entfernungen $d_{ij}$: $\min_\pi \sum_{k} d_{\pi(k)\,\pi(k+1)}$ über alle Permutationen (zyklisch).

**Nachbarschaften und ihre Längenänderung.** Alle Deltas hängen nur von wenigen Kanten ab:
- **2-opt** (Kanten $(a,b)$ und $(c,d)$ der Tour werden durch $(a,c)$ und $(b,d)$ ersetzt, das Stück $b \dots c$ wird umgedreht): $\Delta = d_{ac} + d_{bd} - d_{ab} - d_{cd}$.
- **Or-opt** (Stück $s_0 \dots s_1$ mit Vorgänger $p$ und Nachfolger $q$ wird zwischen $u$ und $v$ eingefügt): $\Delta = \min\{d_{u s_0} + d_{s_1 v},\; d_{u s_1} + d_{s_0 v}\} - d_{uv} - (d_{p s_0} + d_{s_1 q} - d_{pq})$.
- **Tausch** von $x$ (Nachbarn $p,q$) und $y$ (Nachbarn $r,s$): $\Delta = d_{py} + d_{yq} + d_{rx} + d_{xs} - d_{px} - d_{xq} - d_{ry} - d_{ys}$.
Die Zahl der Nachbarn ist bei 2-opt und Tausch $N(N-3)/2$, bei Or-opt (Stücke der Länge 1 bis 3) etwa $3N^2$.

**Hill Climbing.** $\pi_{t+1} = \pi_t \circ m$ für einen Zug $m$ mit $\Delta(m) < 0$ (erste Verbesserung: der erste in fester Reihenfolge, beste Verbesserung: $\arg\min_m \Delta(m)$), bis kein Zug mit $\Delta < 0$ existiert: $\pi$ ist ein **lokales Optimum** der Nachbarschaft. Jeder Schritt verkürzt die Tour strikt, die Suche endet also nach endlich vielen Zügen.

**Untere Schranke (1-Baum, Held-Karp).** Ein 1-Baum ist ein Spannbaum über die Knoten $2..N$ plus die zwei billigsten Kanten des Knotens $1$; jede Rundtour ist ein 1-Baum. Mit Knotengewichten $\pi_i$ (Kosten $d_{ij} + \pi_i + \pi_j$) gilt für jede Tour
$L^\ast \ge w(\pi) = \text{Kosten des minimalen 1-Baums} - 2\sum_i \pi_i$. Das Subgradientenverfahren maximiert $w$: $\pi_i \leftarrow \pi_i + \lambda\,\frac{U - w(\pi)}{\lVert g\rVert^2}\,g_i$ mit $g_i = \deg_i - 2$ und $U$ der Länge einer guten Tour (300 Schritte, $\lambda$ wird halbiert, wenn 15 Schritte keine Verbesserung brachten).

**Kennzahl.** Abstand zur Schranke $= 100 \cdot (L - w) / w$. **Kreuzungen:** Zahl der Kantenpaare, die sich echt schneiden; eine kürzeste euklidische Tour hat keine, ein 2-opt-Optimum ebenfalls nicht.

**Grenzen.** (1) Nur Verbesserungen: die Suche kann ein lokales Optimum nie verlassen. (2) Das Ergebnis hängt von Start, Nachbarschaft und Auswahlregel ab. (3) Die Bewertung aller Nachbarn kostet je Zug $O(N^2)$ und insgesamt etwa $O(N^3)$. (4) Die Schranke ist nicht das Optimum.

Implementiert in `hc_algorithm.py` (Nachbarschaften, Abstieg, Kreuzungen, 1-Baum-Schranke, Mehrfachstart), `hc_scenario.py` (Instanzen), `hc_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
