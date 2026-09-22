"""Konstanten der Hill-Climbing-Demo: Szenario, Regler, Beschriftungen (Presets folgen nach den Messungen)."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0              # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0            # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_STARTS = 3                 # zufällige Startlösungen je Instanz in Sweeps und Vergleichstabellen
BOUND_ITERATIONS = 300

N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 200, 60, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_START_SEED = 0

NEIGHBORHOOD_LABELS = {"swap": "Tausch", "2opt": "2-opt", "oropt": "Or-opt", "2opt+oropt": "2-opt + Or-opt"}
START_LABELS = {"random": "Zufällig", "nearest": "Nächster Nachbar"}
RULE_LABELS = {"first": "Erste Verbesserung", "best": "Beste Verbesserung"}
DEFAULT_NEIGHBORHOOD = "2opt"
DEFAULT_START = "random"
DEFAULT_RULE = "first"

SCALING_N = (20, 40, 60, 100, 150, 200)
MULTI_START_K = 100
STEP_LABELS = ("1 Instanz", "2 Startlösung", "3 Nachbarschaft", "4 Abstieg", "5 Lokales Optimum")


def _preset(neighborhood="2opt", start="random", rule="first", n=DEFAULT_N, ballung=DEFAULT_BALLUNG):
    return {"n": n, "ballung": ballung, "seed": DEFAULT_SEED, "neighborhood": neighborhood, "start": start, "rule": rule, "start_seed": DEFAULT_START_SEED}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Nächster Nachbar als Start": _preset(start="nearest"),
    "Nächster Nachbar + steilster Abstieg": _preset(start="nearest", rule="best"),
    "Nur Tausch": _preset(neighborhood="swap"),
    "Nur Or-opt": _preset(neighborhood="oropt"),
    "2-opt + Or-opt": _preset(neighborhood="2opt+oropt"),
    "Große Instanz (200 Stopps)": _preset(n=200),
}
# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Startlösungen), 60 gleichverteilte Stopps, Abstand zur Schranke
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Stopps, zufällige Startlösung, 2-opt mit erster Verbesserung: im Mittel 7.9 % über der Schranke nach etwa 211 Zügen - ein lokales Optimum ohne Kreuzung, aber weit vom besten entfernt.",
    "Nächster Nachbar als Start": "Nächster Nachbar als Startlösung (21 % über der Schranke) spart Züge (17 statt 211), aber bei erster Verbesserung nicht Güte: 7.3 % statt 7.9 %.",
    "Nächster Nachbar + steilster Abstieg": "Mit steilstem Abstieg zahlt sich die gute Startlösung aus: 4.0 % über der Schranke nach nur 10 Zügen (zufälliger Start: 7.9 %).",
    "Nur Tausch": "Nur das Vertauschen zweier Stopps: von einer zufälligen Tour aus bleibt die Tour 54 % über der Schranke und hat im Mittel 13 Kreuzungen - die Nachbarschaft ist zu klein für das Problem.",
    "Nur Or-opt": "Nur Or-opt (Stücke aus 1-3 Stopps versetzen): 10.4 % über der Schranke, im Mittel 2.7 Kreuzungen bleiben - ein Optimum dieser Nachbarschaft ist keins der 2-opt-Nachbarschaft.",
    "2-opt + Or-opt": "Beide Nachbarschaften zusammen: 3.7 % über der Schranke (2-opt allein: 7.9 %) - mehr Nachbarn, tieferes Optimum, aber das Anderthalbfache an bewerteten Nachbarn (107 statt 74 Tausend).",
    "Große Instanz (200 Stopps)": "200 Stopps mit 2-opt: 9.5 % über der Schranke nach etwa 990 Zügen und 4.0 Millionen bewerteten Nachbarn - die Lücke wächst kaum, die Kosten wachsen etwa mit n³.",
}
# Urteile, die bei diesem Preset über verschiedene Instanzen und Startlösungen vorkommen (jedes Preset wird über 9 Instanzen x 2 Startlösungen gemessen)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"stuck", "near_optimal"},
    "Nächster Nachbar als Start": {"stuck"},
    "Nächster Nachbar + steilster Abstieg": {"stuck", "near_optimal"},
    "Nur Tausch": {"crossings"},
    "Nur Or-opt": {"crossings", "stuck"},
    "2-opt + Or-opt": {"stuck", "near_optimal"},
    "Große Instanz (200 Stopps)": {"stuck"},
}

# Kandidatenlisten + Don't-Look-Bits (hc_dlb.py): Budgetpunkte fuer den Mehrfachstart-Vergleich im Experiment
DLB_BUDGETS = (25000, 100000, 200000, 500000, 1000000)
DLB_CHAINS = 3
