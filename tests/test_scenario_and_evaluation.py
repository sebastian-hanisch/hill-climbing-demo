"""Szenario (Instanz, Gruppen) und Auswertung (Kennzahlen, Urteil, Sweeps, Mehrfachstart)."""

from dataclasses import replace

import numpy as np
import pytest

import hc_algorithm as A
import hc_constants as C
import hc_evaluation as ev
import hc_scenario as S


# --- Szenario ---------------------------------------------------------------------------------------------------------------------------------


def test_instance_shape_depot_and_area():
    inst = S.generate(60, 0, 3)
    assert inst.xy.shape == (61, 2) and inst.n == 60 and inst.n_nodes == 61
    assert inst.xy[0].tolist() == [50.0, 50.0]
    assert inst.xy.min() >= 0.0 and inst.xy.max() <= C.AREA


def test_instance_is_deterministic_and_seed_dependent():
    a, b, c = S.generate(40, 25, 5), S.generate(40, 25, 5), S.generate(40, 25, 6)
    assert np.array_equal(a.xy, b.xy) and not np.array_equal(a.xy, c.xy)


def test_grouped_stops_lie_closer_together_than_uniform_ones():
    def mean_nn(share):
        vals = []
        for seed in range(10):
            xy = S.generate(80, share, seed).xy[1:]
            d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            vals.append(d.min(axis=1).mean())
        return float(np.mean(vals))
    assert mean_nn(100) < 0.7 * mean_nn(0)
    inst = S.generate(80, 50, 1)
    assert inst.cluster_share == 50 and inst.n == 80


def test_exact_number_of_grouped_stops():
    for share in (0, 25, 50, 75, 100):
        assert len(S.generate(40, share, 2).xy) == 41


# --- Analyse ------------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings())
    assert a.length == pytest.approx(A.tour_length(a.descent.tour, a.D)) and a.start_length == pytest.approx(A.tour_length(a.start_tour, a.D))
    assert a.bound < a.length < a.start_length
    assert a.gap == pytest.approx(100 * (a.length - a.bound) / a.bound) and a.start_gap > a.gap > 0
    assert a.crossings_end == 0 and a.crossings_start > 0 and a.descent.n_moves == len(a.descent.steps) - 1


def test_analysis_is_deterministic_and_start_seed_matters_only_for_random_starts():
    s = ev.Settings(n=30)
    a, b = ev.analyse(s), ev.analyse(s)
    assert np.array_equal(a.descent.tour, b.descent.tour)
    assert not np.array_equal(ev.analyse(replace(s, start_seed=1)).start_tour, a.start_tour)
    n1, n2 = ev.analyse(replace(s, start="nearest", start_seed=1)), ev.analyse(replace(s, start="nearest", start_seed=2))
    assert np.array_equal(n1.start_tour, n2.start_tour)


def test_bound_is_below_every_found_tour():
    for seed in range(3):
        for nb in C.NEIGHBORHOOD_LABELS:
            a = ev.analyse(ev.Settings(n=30, seed=seed, neighborhood=nb), keep_steps=False)
            assert a.bound <= a.length + 1e-9


def test_reference_bound_is_cached_and_positive():
    assert ev.reference_bound(30, 0, 1) == ev.reference_bound(30, 0, 1) > 0


# --- Urteil -------------------------------------------------------------------------------------------------------------------------------------


def _fake(gap, crossings, moves=10):
    class D:
        n_moves = moves

    class F:
        descent = D()
        crossings_end = crossings
    f = F()
    f.gap = gap
    return f


def test_verdict_codes():
    assert ev.verdict(_fake(1.0, 0)) == "near_optimal" and ev.verdict(_fake(1.0, 3)) == "near_optimal"
    assert ev.verdict(_fake(6.0, 0)) == "stuck" and ev.verdict(_fake(6.0, 2)) == "crossings"
    assert ev.verdict(_fake(ev.NEAR_GAP, 0)) == "stuck" and ev.verdict(_fake(ev.NEAR_GAP - 0.01, 0)) == "near_optimal"
    assert ev.verdict(_fake(6.0, 0, moves=100000)) == "unfinished"


def test_verdict_of_real_runs():
    assert ev.verdict(ev.analyse(ev.Settings(neighborhood="swap"), keep_steps=False)) == "crossings"
    assert ev.verdict(ev.analyse(ev.Settings(n=10, neighborhood="2opt+oropt", start="nearest", rule="best", seed=1), keep_steps=False)) == "near_optimal"


# --- Sweeps und Tabellen --------------------------------------------------------------------------------------------------------------------


def test_run_config_counts_runs_and_aggregates():
    r = ev.run_config(ev.Settings(n=20))
    assert r["n_runs"] == len(C.SWEEP_SEEDS) * C.SWEEP_STARTS
    assert r["gap_min"] <= r["gap"] <= r["gap_max"] and r["gap_sd"] >= 0 and r["moves"] > 0 and r["evaluations"] > r["moves"]
    nn = ev.run_config(ev.Settings(n=20, start="nearest"))
    assert nn["n_runs"] == len(C.SWEEP_SEEDS)


def test_run_config_ignores_the_seed_of_the_base_settings():
    a = ev.run_config(ev.Settings(n=20, seed=1, start_seed=5))
    b = ev.run_config(ev.Settings(n=20, seed=999, start_seed=0))
    assert a == {**b, "seconds": a["seconds"]} or all(a[k] == b[k] for k in a if k != "seconds")


def test_sweep_values_and_labels():
    rows = ev.sweep("neighborhood", ev.Settings(n=15))
    assert [r["value"] for r in rows] == list(C.NEIGHBORHOOD_LABELS)
    assert rows[0]["gap"] > rows[1]["gap"]                                 # Tausch schlechter als 2-opt
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)


def test_comparison_table_has_sixteen_rows():
    rows = ev.comparison_table(ev.Settings(n=15))
    assert len(rows) == 16 and len({(r["neighborhood"], r["start"], r["rule"]) for r in rows}) == 16
    for start in C.START_LABELS:
        for rule in C.RULE_LABELS:
            grp = {r["neighborhood"]: r["gap"] for r in rows if r["start"] == start and r["rule"] == rule}
            assert grp["2opt+oropt"] <= min(grp["swap"], grp["2opt"], grp["oropt"]) + 1e-9


def test_multi_start_report():
    s = ev.Settings(n=25)
    m = ev.multi_start_report(s, 12)
    assert len(m["gaps"]) == 12 and m["best_gap"] == pytest.approx(m["gaps"].min()) and m["running_best"][-1] == pytest.approx(m["gaps"].min())
    assert all(a >= b - 1e-12 for a, b in zip(m["running_best"], m["running_best"][1:]))
    assert 1 <= m["distinct"] <= 12 and 0.0 <= m["edge_share_mean"] <= 1.0
    again = ev.multi_start_report(s, 12)
    assert np.array_equal(m["gaps"], again["gaps"])


def test_local_optimum_hierarchy_fields():
    h = ev.local_optimum_hierarchy(ev.Settings(n=20))
    assert 0.0 <= h["share_not_optimal"] <= 1.0 and h["mean_improvement"] >= 0.0


def test_scaling_table_structure(monkeypatch):
    monkeypatch.setattr(C, "SCALING_N", (10, 20))
    tab = ev.scaling_table()
    assert len(tab) == 2 and all([r["value"] for r in blk["rows"]] == [10, 20] for blk in tab)
    assert tab[0]["label"] != tab[1]["label"]


# --- Kandidatenlisten + Don't-Look-Bits --------------------------------------------------------------------------------------------------------


def test_full_restarts_and_dlb_restarts_use_at_least_one_full_descent_and_respect_the_budget():
    inst, D = ev.instance(40, 0, 100000)
    cand = ev.DLB.build_candidate_lists(D)
    single = A.descend(D, A.random_tour(len(D), np.random.default_rng(0)), "2opt", "first", keep_steps=False)
    _, starts, used = ev.full_restarts(D, 1000, 0)
    assert starts == 1 and used > 1000                             # Budget unter einem Abstieg: der erste läuft trotzdem zu Ende
    _, starts, used = ev.dlb_restarts(D, cand, 1000, 0)
    assert starts >= 1 and used >= 1000
    best_full, starts_full, used_full = ev.full_restarts(D, 5 * single.evaluations, 0)
    assert starts_full >= 3 and used_full <= 5 * single.evaluations + 2 * len(D) ** 2


def test_dlb_restarts_use_far_fewer_evaluations_per_start_than_full_restarts():
    inst, D = ev.instance(60, 0, 100000)
    cand = ev.DLB.build_candidate_lists(D)
    best_full, starts_full, used_full = ev.full_restarts(D, 200000, 0)
    best_dlb, starts_dlb, used_dlb = ev.dlb_restarts(D, cand, 200000, 0)
    assert starts_dlb > 20 * starts_full                            # gemessen: ~300 gegen ~3 Starts
    assert best_dlb <= best_full                                    # mehr, billigere Neustarts finden mindestens so gute Touren


def test_dlb_single_descent_table_fields():
    r = ev.dlb_single_descent_table(seeds=(100000, 100001), n_starts=2)
    assert r["dlb_evaluations"] < r["full_evaluations"] / 20         # gemessen: ~650 gegen ~74 000
    assert abs(r["dlb_gap"] - r["full_gap"]) < 3.0                   # vergleichbare Güte trotz weit weniger Bewertungen


def test_dlb_budget_table_structure_and_monotonicity():
    rows = ev.dlb_budget_table(budgets=(25000, 200000), seeds=(100000, 100001), chains=2)
    assert [r["value"] for r in rows] == [25000, 200000]
    assert rows[1]["dlb_starts"] > rows[0]["dlb_starts"] and rows[1]["full_starts"] >= rows[0]["full_starts"]
    assert rows[1]["dlb_gap"] <= rows[0]["dlb_gap"] + 1.0             # mehr Budget wird nicht schlechter (bis auf Rauschen)
