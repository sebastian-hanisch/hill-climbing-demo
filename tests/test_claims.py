"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Instanzen (je drei zufällige Startlösungen) belegt.
Die Suche ist bis auf die Rundung der Entfernungen deterministisch; die Bänder sind trotzdem weit genug für Unterschiede zwischen numpy-Versionen und Plattformen (Ausreißer eines Laufs verschieben ein Mittel über 15 Läufe).
Positive UND negative Aussagen: wo Or-opt, der Nächste Nachbar oder steilster Abstieg nicht helfen, steht das hier ebenso als Test wie dort, wo sie helfen. Rechenzeiten sind nur als Größenordnung geprüft."""

import time
from functools import lru_cache

import numpy as np
import pytest

import hc_algorithm as A
import hc_constants as C
import hc_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


def rel(value, expected, frac, floor=2.0):
    assert abs(value - expected) <= max(floor, frac * expected), f"{value:.1f} statt {expected}"


# --- Seitenleiste ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,gap,moves,evals_k", [(10, 1.7, 12, 0.18), (20, 1.9, 45, 2.0), (40, 7.7, 109, 16), (60, 7.9, 211, 74), (100, 9.1, 403, 448), (150, 9.1, 672, 1542), (200, 9.5, 991, 3973)])
def test_stops_sweep(n, gap, moves, evals_k):
    r = cfg(n=n)
    near(r["gap"], gap, 0.9)
    rel(r["moves"], moves, 0.05, floor=2)
    rel(r["evaluations"] / 1000, evals_k, 0.10, floor=0.05)
    assert r["crossings"] == 0.0                                            # ein 2-opt-Optimum ist kreuzungsfrei


def test_moves_grow_about_linearly_and_evaluations_about_cubically():
    for n in (20, 40, 60, 100, 150, 200):
        assert 2.0 <= cfg(n=n)["moves"] / n <= 5.2
    ratio = cfg(n=200)["evaluations"] / cfg(n=60)["evaluations"]
    assert (200 / 60) ** 3 * 0.9 <= ratio <= (200 / 60) ** 3 * 2.0          # etwa n^3 und etwas darüber (gemessen: 54 gegen 37)
    assert cfg(n=200)["gap"] - cfg(n=100)["gap"] < 1.5                      # die Lücke wächst kaum


@pytest.mark.parametrize("share,gap", [(0, 7.9), (25, 8.0), (50, 5.7), (75, 4.1), (100, 4.4)])
def test_cluster_share_sweep(share, gap):
    near(cfg(cluster_share=share)["gap"], gap, 1.0)


def test_nearest_neighbor_with_first_improvement_is_worse_than_random_for_grouped_stops():
    nn, rnd = cfg(cluster_share=75, start="nearest")["gap"], cfg(cluster_share=75)["gap"]
    near(nn, 6.7, 1.0)
    assert nn > rnd + 1.5


def test_neighborhood_help_numbers():
    swap, opt2, oropt, both = (cfg(neighborhood=k) for k in ("swap", "2opt", "oropt", "2opt+oropt"))
    near(swap["gap"], 54.5, 4.0)
    near(opt2["gap"], 7.9, 0.9)
    near(oropt["gap"], 10.4, 1.5)
    near(both["gap"], 3.7, 0.9)
    near(cfg(neighborhood="swap", start="nearest")["gap"], 16.3, 2.0)
    near(swap["crossings"], 13.0, 2.0)
    near(oropt["crossings"], 2.7, 0.9)
    assert opt2["crossings"] == 0.0 and both["crossings"] == 0.0


def test_start_help_numbers():
    near(cfg()["start_gap"], 420, 20)
    near(cfg(start="nearest")["start_gap"], 21, 3)
    rnd, nn = cfg(), cfg(start="nearest")
    near(rnd["gap"], 7.9, 0.9)
    near(nn["gap"], 7.3, 1.0)
    rel(rnd["moves"], 211, 0.05)
    rel(nn["moves"], 17, 0.15, floor=2)
    rnd_b, nn_b = cfg(rule="best"), cfg(start="nearest", rule="best")
    near(rnd_b["gap"], 7.9, 0.9)
    near(nn_b["gap"], 4.0, 1.0)
    assert nn_b["gap"] < rnd_b["gap"] - 2.0 and abs(nn["gap"] - rnd["gap"]) < 2.0     # gute Startlösung: bei erster Verbesserung kaum Gewinn, bei steilstem Abstieg viel


def test_rule_help_numbers():
    first, best = cfg(), cfg(rule="best")
    rel(first["moves"], 211, 0.05)
    rel(best["moves"], 57, 0.10, floor=3)
    rel(first["evaluations"] / 1000, 74, 0.10)
    rel(best["evaluations"] / 1000, 102, 0.10)
    both_f, both_b = cfg(neighborhood="2opt+oropt"), cfg(neighborhood="2opt+oropt", rule="best")
    near(both_f["gap"], 3.7, 0.9)
    near(both_b["gap"], 3.8, 1.0)
    rel(both_f["moves"], 220, 0.05)
    rel(both_b["moves"], 45, 0.10, floor=3)
    rel(both_f["evaluations"] / 1000, 107, 0.10)
    rel(both_b["evaluations"] / 1000, 573, 0.10)
    assert best["moves"] < first["moves"] / 3 and both_b["evaluations"] > 4 * both_f["evaluations"]


def test_start_seed_spread_over_the_fifteen_runs():
    r = cfg()
    assert r["n_runs"] == 15
    near(r["gap_min"], 0.9, 0.6)
    near(r["gap_max"], 14.1, 1.2)
    near(r["gap_sd"], 4.0, 0.6)


def test_or_opt_alone_is_worse_than_two_opt_and_swap_needs_the_good_start():
    assert cfg(neighborhood="oropt")["gap"] > cfg()["gap"] + 1.5
    assert cfg(neighborhood="swap")["gap"] > 3 * cfg(neighborhood="swap", start="nearest")["gap"]


# --- Vergleichstabelle (16 Kombinationen, 60 gleichverteilte Stopps) ---------------------------------------------------------------------


def test_comparison_caption():
    for start in ("random", "nearest"):
        for rule in ("first", "best"):
            g = {k: cfg(neighborhood=k, start=start, rule=rule)["gap"] for k in C.NEIGHBORHOOD_LABELS}
            assert g["2opt+oropt"] < min(g["swap"], g["2opt"], g["oropt"]) and 3.0 <= g["2opt+oropt"] <= 4.3, (start, rule, g)
    near(cfg(start="nearest", rule="best")["gap"], 4.0, 1.0)
    assert 6.3 <= cfg(start="nearest")["gap"] <= 8.9 and 6.9 <= cfg()["gap"] <= 8.9


def test_preset_help_numbers():
    near(cfg(neighborhood="2opt+oropt")["gap"], 3.7, 0.9)
    rel(cfg(start="nearest", rule="best")["moves"], 10, 0.3, floor=3)
    ratio = cfg(neighborhood="2opt+oropt")["evaluations"] / cfg()["evaluations"]
    near(ratio, 1.45, 0.25)
    big = cfg(n=200)
    near(big["gap"], 9.5, 1.0)
    rel(big["moves"], 990, 0.05)
    rel(big["evaluations"] / 1e6, 4.0, 0.10, floor=0.2)


# --- Mehrfachstart, Hierarchie, Skalierung ------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _multi(kw_items=()):
    return [ev.multi_start_report(ev.Settings(seed=s, **dict(kw_items)), C.MULTI_START_K) for s in C.SWEEP_SEEDS]


def test_multi_start_numbers():
    res = _multi()
    for k, expected, tol in ((1, 7.9, 1.2), (5, 4.0, 1.0), (20, 1.8, 0.7), (100, 1.2, 0.6)):
        near(float(np.mean([m["running_best"][k - 1] for m in res])), expected, tol)
    g = np.concatenate([m["gaps"] for m in res])
    near(float((g <= 2).mean()), 0.04, 0.04)
    near(float((g <= 5).mean()), 0.25, 0.07)
    near(float(np.mean([m["distinct"] for m in res])), 99, 2)
    near(float(np.mean([m["edge_share_mean"] for m in res])), 0.74, 0.05)
    assert float(np.mean([m["running_best"][99] for m in res])) < float(np.mean([m["running_best"][0] for m in res])) / 4


def test_a_random_tour_shares_only_about_two_over_n_minus_one_of_its_edges():
    rng = np.random.default_rng(0)
    ref = A.random_tour(61, rng)
    share = float(np.mean([A.edge_share(A.random_tour(61, rng), ref) for _ in range(400)]))
    near(share, 2 / 60, 0.012)


def test_hierarchy_numbers():
    h = ev.local_optimum_hierarchy()
    assert h["share_not_optimal"] >= 0.93
    near(h["mean_improvement"], 3.8, 0.9)


@pytest.mark.parametrize("n,gap,moves", [(20, 0.05, 3.4), (40, 1.05, 5.8), (60, 3.4, 8.6), (100, 4.4, 18), (150, 5.3, 27), (200, 5.1, 31)])
def test_scaling_second_method(n, gap, moves):
    r = cfg(n=n, neighborhood="2opt+oropt", start="nearest", rule="best")
    near(r["gap"], gap, 0.6 if n <= 40 else 1.2)
    rel(r["moves"], moves, 0.25, floor=2)


def test_scaling_first_method_matches_the_caption():
    for n, gap in ((20, 1.9), (40, 7.7), (60, 7.9), (100, 9.1), (150, 9.1), (200, 9.5)):
        near(cfg(n=n)["gap"], gap, 0.9)
    assert cfg(n=200, neighborhood="2opt+oropt", start="nearest", rule="best")["gap"] < cfg(n=200)["gap"]


# --- Schranke gegen das echte Optimum (CP-SAT) --------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _optimum(n, share, seed):
    cp = pytest.importorskip("ortools.sat.python.cp_model")
    inst, D = ev.instance(n, share, seed)
    di = np.rint(D * 10000).astype(int)
    N = len(D)
    m = cp.CpModel()
    lits = {(i, j): m.NewBoolVar("") for i in range(N) for j in range(N) if i != j}
    m.AddCircuit([(i, j, l) for (i, j), l in lits.items()])
    m.Minimize(sum(int(di[i, j]) * l for (i, j), l in lits.items()))
    s = cp.CpSolver()
    s.parameters.max_time_in_seconds = 90
    s.parameters.num_workers = 4
    assert s.StatusName(s.Solve(m)) == "OPTIMAL"
    return s.ObjectiveValue() / 10000


def _bound_gaps(share):
    out = []
    for seed in C.SWEEP_SEEDS:
        opt = _optimum(60, share, seed)
        b = ev.reference_bound(60, share, seed)
        assert b <= opt + 0.01
        out.append(100 * (opt - b) / opt)
    return out


def test_bound_is_close_to_the_optimum_for_uniform_stops():
    g = _bound_gaps(0)
    near(float(np.mean(g)), 0.5, 0.3)
    assert max(g) < 1.0


def test_bound_is_looser_for_grouped_stops():
    g = _bound_gaps(100)
    near(float(np.mean(g)), 1.1, 0.7)
    assert 3.0 <= max(g) <= 4.5                                             # der Einzelfall aus dem Grenzen-Text


def test_local_optima_are_above_the_optimum_by_about_the_gap_to_the_bound_minus_the_bound_gap():
    for seed in C.SWEEP_SEEDS[:3]:
        opt = _optimum(60, 0, seed)
        a = ev.analyse(ev.Settings(seed=seed), keep_steps=False)
        assert a.length >= opt - 0.01
        true_gap = 100 * (a.length - opt) / opt
        assert 0 <= a.gap - true_gap <= 1.0


# --- Größenordnung der Rechenzeit ----------------------------------------------------------------------------------------------------------


def test_run_times_are_of_the_stated_order_of_magnitude():
    t0 = time.perf_counter()
    ev.analyse(ev.Settings(n=200), keep_steps=False)
    assert time.perf_counter() - t0 < 30
    assert cfg()["seconds"] < 1.0
    assert cfg(neighborhood="2opt+oropt")["seconds"] < 5.0
