"""Auswertung der Hill-Climbing-Demo: einzelne Läufe (Kennzahlen gegen die 1-Baum-Schranke), Sweeps über feste Instanzen, Vergleichstabellen und Mehrfachstart.

Der Abstand zur Schranke ist der Abstand zu einer *unteren* Schranke der kürzesten Tour; er überschätzt die wahre Lücke zum Optimum um die Schrankenlücke (gemessen: im Mittel unter 1 %)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import hc_algorithm as A
import hc_constants as C
import hc_dlb as DLB
import hc_scenario as S


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    seed: int = C.DEFAULT_SEED
    neighborhood: str = C.DEFAULT_NEIGHBORHOOD
    start: str = C.DEFAULT_START
    rule: str = C.DEFAULT_RULE
    start_seed: int = C.DEFAULT_START_SEED


@dataclass
class Analysis:
    settings: Settings
    inst: object
    D: np.ndarray
    start_tour: np.ndarray
    descent: object
    bound: float
    seconds: float
    crossings_start: int
    crossings_end: int

    @property
    def start_length(self):
        return self.descent.steps[0].length

    @property
    def length(self):
        return self.descent.length

    @property
    def gap(self):
        """Abstand der Endtour zur Schranke in Prozent der Schranke."""
        return 100.0 * (self.length - self.bound) / self.bound

    @property
    def start_gap(self):
        return 100.0 * (self.start_length - self.bound) / self.bound


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed):
    inst = S.generate(n, cluster_share, seed)
    return inst, A.dist_matrix(inst.xy)


@lru_cache(maxsize=256)
def reference_bound(n, cluster_share, seed):
    """1-Baum-Schranke der Instanz; Zielwert des Subgradientenverfahrens ist die Länge eines guten lokalen Optimums (Nächster Nachbar + 2-opt/Or-opt, steilster Abstieg)."""
    inst, D = instance(n, cluster_share, seed)
    ref = A.descend(D, A.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
    return A.held_karp_bound(D, ref.length, C.BOUND_ITERATIONS)


def make_start(settings, D):
    rng = np.random.default_rng(settings.start_seed)
    return A.start_tour(settings.start, D, rng)


def analyse(settings, keep_steps=True):
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)
    start = make_start(settings, D)
    t0 = time.perf_counter()
    descent = A.descend(D, start, settings.neighborhood, settings.rule, keep_steps=keep_steps)
    seconds = time.perf_counter() - t0
    return Analysis(settings, inst, D, start, descent, reference_bound(settings.n, settings.cluster_share, settings.seed), seconds,
                    A.count_crossings(inst.xy, start), A.count_crossings(inst.xy, descent.tour))


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------------

NEAR_GAP = 3.0                    # unter diesem Abstand zur Schranke gilt die Tour als fast optimal


def verdict(a):
    """Code: crossings (Optimum der Nachbarschaft, aber Kreuzungen und Lücke), stuck (lokales Optimum, Lücke >= NEAR_GAP), near_optimal, unfinished."""
    if a.descent.n_moves >= 100000:
        return "unfinished"
    if a.gap < NEAR_GAP:
        return "near_optimal"
    if a.crossings_end > 0:
        return "crossings"
    return "stuck"


# --- Sweeps und Vergleichstabellen -------------------------------------------------------------------------------------------------------


def _aggregate(runs):
    return {"gap": float(np.mean([r["gap"] for r in runs])), "gap_sd": float(np.std([r["gap"] for r in runs])), "gap_min": float(np.min([r["gap"] for r in runs])),
            "gap_max": float(np.max([r["gap"] for r in runs])), "moves": float(np.mean([r["moves"] for r in runs])), "evaluations": float(np.mean([r["evaluations"] for r in runs])),
            "seconds": float(np.mean([r["seconds"] for r in runs])), "crossings": float(np.mean([r["crossings"] for r in runs])),
            "start_gap": float(np.mean([r["start_gap"] for r in runs])), "n_runs": len(runs)}


def run_config(base, seeds=C.SWEEP_SEEDS, n_starts=C.SWEEP_STARTS, **changes):
    """Mittel über die festen Instanzen (und bei zufälligem Start `n_starts` Startlösungen) für die Einstellungen `base` mit `changes`."""
    s0 = replace(base, **changes)
    runs = []
    for seed in seeds:
        for k in range(n_starts if s0.start == "random" else 1):
            a = analyse(replace(s0, seed=seed, start_seed=k), keep_steps=False)
            runs.append({"gap": a.gap, "moves": a.descent.n_moves, "evaluations": a.descent.evaluations, "seconds": a.seconds, "crossings": a.crossings_end, "start_gap": a.start_gap})
    return _aggregate(runs)


SWEEP_VALUES = {"n": (10, 20, 40, 60, 100, 150, 200), "cluster_share": (0, 25, 50, 75, 100), "neighborhood": tuple(C.NEIGHBORHOOD_LABELS), "start": tuple(C.START_LABELS), "rule": tuple(C.RULE_LABELS)}
SWEEP_LABELS = {"n": "Stopps", "cluster_share": "Anteil in Gruppen (%)", "neighborhood": "Nachbarschaft", "start": "Startlösung", "rule": "Auswahlregel"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def comparison_table(base=Settings()):
    """Nachbarschaft x Startlösung x Auswahlregel bei den Einstellungen `base` (Stopps, Gruppenanteil)."""
    rows = []
    for nb in C.NEIGHBORHOOD_LABELS:
        for st in C.START_LABELS:
            for rule in C.RULE_LABELS:
                rows.append({"neighborhood": nb, "start": st, "rule": rule, **run_config(base, neighborhood=nb, start=st, rule=rule)})
    return rows


# --- Mehrfachstart ------------------------------------------------------------------------------------------------------------------------------


def multi_start_report(settings, k=100):
    """k Abstiege aus zufälligen Startlösungen derselben Instanz: Längen, Abstände zur Schranke, bester Zwischenstand, Kantenanteil an der besten Tour."""
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)
    bound = reference_bound(settings.n, settings.cluster_share, settings.seed)
    lengths, tours, moves, evals = A.multi_start(D, k, settings.start_seed, settings.neighborhood, settings.rule, "random")
    gaps = 100.0 * (lengths - bound) / bound
    best = int(np.argmin(lengths))
    shares = np.array([A.edge_share(t, tours[best]) for t in tours])
    return {"gaps": gaps, "best_gap": float(gaps[best]), "best_tour": tours[best], "running_best": np.minimum.accumulate(gaps), "edge_share_mean": float(np.mean(np.delete(shares, best))) if k > 1 else float("nan"),
            "distinct": len({tuple(t) for t in tours}), "moves": float(moves.mean()), "evaluations": float(evals.mean())}


def local_optimum_hierarchy(base=Settings(), seeds=C.SWEEP_SEEDS, n_starts=C.SWEEP_STARTS):
    """Anteil der 2-opt-Optima, die kein 2-opt+Or-opt-Optimum sind, und die mittlere weitere Verbesserung durch Or-opt (Prozent der Tourlänge)."""
    not_opt, improve = [], []
    for seed in seeds:
        inst, D = instance(base.n, base.cluster_share, seed)
        for k in range(n_starts):
            a = A.descend(D, A.start_tour("random", D, np.random.default_rng(k)), "2opt", base.rule, keep_steps=False)
            b = A.descend(D, a.tour, "2opt+oropt", base.rule, keep_steps=False)
            not_opt.append(b.n_moves > 0)
            improve.append(100.0 * (a.length - b.length) / a.length)
    return {"share_not_optimal": float(np.mean(not_opt)), "mean_improvement": float(np.mean(improve))}


SCALING_CONFIGS = (("2-opt, zufälliger Start, erste Verbesserung", dict(neighborhood="2opt", start="random", rule="first")),
                   ("2-opt + Or-opt, Nächster Nachbar, beste Verbesserung", dict(neighborhood="2opt+oropt", start="nearest", rule="best")))


def scaling_table(base=Settings()):
    """Züge, Bewertungen, Zeit und Abstand zur Schranke über die Stoppzahl, für zwei Verfahren."""
    return [{"label": label, "rows": sweep("n", replace(base, cluster_share=base.cluster_share, **kw), C.SCALING_N)} for label, kw in SCALING_CONFIGS]


# --- Kandidatenlisten + Don't-Look-Bits (hc_dlb.py) ---------------------------------------------------------------------------------------------


def full_restarts(D_mat, budget, seed, neighborhood="2opt", rule="first"):
    """Abstiege mit dem vollen Rescan aus zufälligen Startlösungen, bis die bewerteten Nachbarn das Budget erreichen
    (der erste läuft immer zu Ende, weitere mit dem Rest); wie hill_climbing_restarts in der Simulated-Annealing-Demo.
    Gibt (beste Länge, Zahl der Starts, verbrauchte Bewertungen) zurück."""
    rng = np.random.default_rng(seed)
    used, starts, best = 0, 0, None
    while used < budget or best is None:
        cap = None if best is None else budget - used
        r = A.descend(D_mat, A.random_tour(len(D_mat), rng), neighborhood, rule, keep_steps=False, max_evaluations=cap)
        used += r.evaluations
        starts += 1
        if best is None or r.length < best:
            best = r.length
    return best, starts, used


def dlb_restarts(D_mat, cand, budget, seed):
    """Dieselbe Regel wie full_restarts, mit dem Kandidatenlisten- + Don't-Look-Bit-Abstieg (hc_dlb.dlb_descend)."""
    rng = np.random.default_rng(seed)
    used, starts, best = 0, 0, None
    while used < budget or best is None:
        cap = None if best is None else budget - used
        r = DLB.dlb_descend(D_mat, A.random_tour(len(D_mat), rng), cand, seed=starts, max_evaluations=cap)
        used += r.evaluations
        starts += 1
        if best is None or r.length < best:
            best = r.length
    return best, starts, used


def dlb_single_descent_table(base=Settings(), seeds=C.SWEEP_SEEDS, n_starts=C.SWEEP_STARTS):
    """Ein Abstieg, voller Rescan gegen Kandidatenliste + Don't-Look-Bits: mittlere Bewertungen und Abstand zur Schranke
    (jeweils vom selben Start, damit die Güte direkt vergleichbar ist)."""
    full_gaps, full_evals, dlb_gaps, dlb_evals = [], [], [], []
    for seed in seeds:
        inst, D_mat = instance(base.n, base.cluster_share, seed)
        bound = reference_bound(base.n, base.cluster_share, seed)
        cand = DLB.build_candidate_lists(D_mat)
        for k in range(n_starts):
            t0 = A.random_tour(len(D_mat), np.random.default_rng(k))
            full = A.descend(D_mat, t0, base.neighborhood, base.rule, keep_steps=False)
            r = DLB.dlb_descend(D_mat, t0, cand, seed=k)
            full_gaps.append(100 * (full.length - bound) / bound)
            full_evals.append(full.evaluations)
            dlb_gaps.append(100 * (r.length - bound) / bound)
            dlb_evals.append(r.evaluations)
    return {"full_gap": float(np.mean(full_gaps)), "full_evaluations": float(np.mean(full_evals)),
            "dlb_gap": float(np.mean(dlb_gaps)), "dlb_evaluations": float(np.mean(dlb_evals))}


def dlb_budget_table(base=Settings(), budgets=C.DLB_BUDGETS, seeds=C.SWEEP_SEEDS, chains=C.DLB_CHAINS):
    """Abstand zur Schranke über das Budget: Neustarts mit vollem Rescan gegen Neustarts mit Kandidatenliste + Don't-Look-Bits."""
    rows = []
    for budget in budgets:
        full_gaps, full_starts, dlb_gaps, dlb_starts = [], [], [], []
        for seed in seeds:
            inst, D_mat = instance(base.n, base.cluster_share, seed)
            bound = reference_bound(base.n, base.cluster_share, seed)
            cand = DLB.build_candidate_lists(D_mat)
            for ch in range(chains):
                fb, fs, _ = full_restarts(D_mat, budget, ch, base.neighborhood, base.rule)
                db, ds, _ = dlb_restarts(D_mat, cand, budget, ch * 1000 + seed)
                full_gaps.append(100 * (fb - bound) / bound)
                full_starts.append(fs)
                dlb_gaps.append(100 * (db - bound) / bound)
                dlb_starts.append(ds)
        rows.append({"value": budget, "full_gap": float(np.mean(full_gaps)), "full_starts": float(np.mean(full_starts)),
                     "dlb_gap": float(np.mean(dlb_gaps)), "dlb_starts": float(np.mean(dlb_starts))})
    return rows
