"""Hill Climbing: Längenänderung jedes Zuges gegen die neu gemessene Tourlänge, die Nachbarschaften gegen eine unabhängige Aufzählung, Abstieg (monoton, lokales Optimum), Kreuzungen,
Startlösungen und die 1-Baum-Schranke (gegen Brute-Force und CP-SAT)."""

import itertools

import numpy as np
import pytest

import hc_algorithm as A


def _instance(n_nodes, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n_nodes, 2)) * 100
    return xy, A.dist_matrix(xy)


# --- unabhängige Aufzählung der Nachbarschaften (explizite Listenoperationen) ---------------------------------------------------------------------


def _brute_neighbors(t, kind):
    """Alle Nachbartouren als Listen (mit Wiederholungen bei gleicher Tour erlaubt)."""
    t = list(t)
    n = len(t)
    out = []
    if kind in ("2opt", "2opt+oropt"):
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                out.append(t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:])
    if kind == "swap":
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                u = list(t)
                u[i], u[j] = u[j], u[i]
                out.append(u)
    if kind in ("oropt", "2opt+oropt"):
        for length in range(1, A.MAX_SEGMENT + 1):
            if n < length + 3:
                continue
            for i in range(n):
                seg = [t[(i + q) % n] for q in range(length)]
                rest = [t[(i + length + q) % n] for q in range(n - length)]     # beginnt nach dem Stück, endet davor
                for pos in range(len(rest) - 1):                                # Kante (rest[pos], rest[pos+1]); die schließende Kante (Ausgangsstelle) zählt nicht
                    for s in (seg, seg[::-1]) if length > 1 else (seg,):
                        out.append(rest[:pos + 1] + s + rest[pos + 1:])
    return out


def _brute_min_delta(t, D, kind):
    base = A.tour_length(t, D)
    lengths = [A.tour_length(np.array(u), D) for u in _brute_neighbors(t, kind)]
    return min(lengths) - base, len(_brute_neighbors(t, kind))


# --- Züge -------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_nodes", [7, 12])
def test_every_candidate_delta_equals_the_recomputed_length_change(n_nodes):
    xy, D = _instance(n_nodes, 1)
    t = A.random_tour(n_nodes, np.random.default_rng(2))
    base = A.tour_length(t, D)
    checked = 0
    for kind in ("swap", "2opt", "oropt"):
        for name, delta, ok, extra in A._candidates(t, D, kind):
            for i, j in zip(*np.where(ok)):
                move = (name, i, j, delta[i, j]) if name in ("2opt", "swap") else ("oropt", i, int(name[-1]), j, bool(extra[i, j]), delta[i, j])
                new = A.apply_move(t, move)
                assert sorted(new.tolist()) == list(range(n_nodes))
                assert A.tour_length(new, D) - base == pytest.approx(delta[i, j], abs=1e-9)
                checked += 1
    assert checked > 100


@pytest.mark.parametrize("kind", ["swap", "2opt", "oropt", "2opt+oropt"])
@pytest.mark.parametrize("n_nodes", [6, 9, 14])
def test_best_move_and_neighbor_count_match_the_independent_enumeration(kind, n_nodes):
    xy, D = _instance(n_nodes, 4)
    t = A.random_tour(n_nodes, np.random.default_rng(5))
    delta_all = A.neighbor_deltas(t, D, kind)
    brute_min, brute_count = _brute_min_delta(t, D, kind)
    assert delta_all.min() == pytest.approx(brute_min, abs=1e-9)
    move, evaluations = A.find_move(t, D, kind, "best")
    assert evaluations == len(delta_all)
    assert (move[-1] if move else 0.0) == pytest.approx(min(brute_min, 0.0), abs=1e-9)
    if kind in ("swap", "2opt"):
        assert len(delta_all) == brute_count == n_nodes * (n_nodes - 3) // 2
    if kind == "oropt":
        assert len(delta_all) == sum(n_nodes * (n_nodes - length - 1) for length in (1, 2, 3))                       # je Stück und Stelle die bessere Richtung
        assert brute_count == sum(n_nodes * (n_nodes - length - 1) * (1 if length == 1 else 2) for length in (1, 2, 3))   # die Aufzählung zählt beide Richtungen


def test_neighborhood_sizes_of_the_help_text():
    xy, D = _instance(61, 3)
    t = A.random_tour(61, np.random.default_rng(0))
    assert [len(A.neighbor_deltas(t, D, k)) for k in ("swap", "2opt", "oropt", "2opt+oropt")] == [1769, 1769, 10614, 12383]


def test_move_edges_lists_removed_and_added_edges():
    t = np.arange(8)
    move = ("2opt", 1, 4, 0.0)                                               # Kanten (1,2) und (4,5) werden durch (1,4) und (2,5) ersetzt
    removed, added = A.move_edges(t, move)
    assert removed == [(1, 2), (4, 5)] and added == [(1, 4), (2, 5)]
    assert A.apply_move(t, move).tolist() == [0, 1, 4, 3, 2, 5, 6, 7]


def test_or_opt_moves_a_segment_and_can_reverse_it():
    t = np.arange(8)
    # Stück (1, 2) wird in die Kante (5, 6) eingefügt, als Zyklus gelesen: 0 3 4 5 1 2 6 7
    assert A.tour_edges(A.apply_move(t, ("oropt", 1, 2, 5, False, 0.0))) == A.tour_edges(np.array([0, 3, 4, 5, 1, 2, 6, 7]))
    assert A.tour_edges(A.apply_move(t, ("oropt", 1, 2, 5, True, 0.0))) == A.tour_edges(np.array([0, 3, 4, 5, 2, 1, 6, 7]))
    wrapped = A.apply_move(t, ("oropt", 6, 3, 2, False, 0.0))                 # Stück 6, 7, 0 über das Array-Ende, wird in die Kante (2, 3) eingefügt
    assert sorted(wrapped.tolist()) == list(range(8)) and A.tour_edges(wrapped) >= {(6, 7), (0, 7), (2, 6), (0, 3), (1, 2)}


def test_first_improvement_takes_the_first_improving_neighbor_in_order():
    xy, D = _instance(20, 6)
    t = A.random_tour(20, np.random.default_rng(6))
    move, evaluations = A.find_move(t, D, "2opt", "first")
    d, ok = A._delta_2opt(t, D)
    flat = np.where(ok, d, np.inf).ravel()
    first = int(np.argmax(flat < -A.EPS))
    assert (move[1], move[2]) == divmod(first, 20) and evaluations == int(ok.ravel()[:first + 1].sum())
    best, _ = A.find_move(t, D, "2opt", "best")
    assert best[-1] <= move[-1] + 1e-12


# --- Abstieg ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["swap", "2opt", "oropt", "2opt+oropt"])
@pytest.mark.parametrize("rule", ["first", "best"])
def test_descent_is_monotone_ends_in_a_local_optimum_and_counts_its_moves(kind, rule):
    xy, D = _instance(13, 7)
    t0 = A.random_tour(13, np.random.default_rng(8))
    r = A.descend(D, t0, kind, rule)
    lengths = [s.length for s in r.steps]
    assert all(b < a - 1e-12 for a, b in zip(lengths, lengths[1:]))
    assert r.n_moves == len(r.steps) - 1 == sum(r.kinds.values()) and r.length == pytest.approx(A.tour_length(r.tour, D))
    assert r.length == pytest.approx(lengths[-1], abs=1e-6)
    brute_min, _ = _brute_min_delta(list(r.tour), D, kind)
    assert brute_min >= -1e-9                                              # kein Nachbar ist kürzer
    assert A.is_local_optimum(D, r.tour, kind)
    assert sorted(r.tour.tolist()) == list(range(13))


def test_descent_from_a_local_optimum_makes_no_move_and_is_deterministic():
    xy, D = _instance(15, 9)
    t0 = A.random_tour(15, np.random.default_rng(1))
    a = A.descend(D, t0, "2opt", "first")
    b = A.descend(D, t0, "2opt", "first")
    assert np.array_equal(a.tour, b.tour) and a.evaluations == b.evaluations and a.n_moves == b.n_moves
    again = A.descend(D, a.tour, "2opt", "first")
    assert again.n_moves == 0 and np.array_equal(again.tour, a.tour)
    fast = A.descend(D, t0, "2opt", "first", keep_steps=False)
    assert len(fast.steps) == 1 and fast.n_moves == a.n_moves and np.array_equal(fast.tour, a.tour)


def test_max_moves_stops_the_descent_early():
    xy, D = _instance(30, 2)
    r = A.descend(D, A.random_tour(30, np.random.default_rng(0)), "2opt", "first", max_moves=3)
    assert r.n_moves == 3


def test_max_evaluations_caps_the_evaluation_budget():
    xy, D = _instance(40, 2)
    start = A.random_tour(40, np.random.default_rng(1))
    full = A.descend(D, start, "2opt", "first", keep_steps=False)
    cut = A.descend(D, start, "2opt", "first", keep_steps=False, max_evaluations=full.evaluations // 3)
    assert cut.n_moves < full.n_moves and cut.evaluations >= full.evaluations // 3 and cut.length > full.length
    assert A.descend(D, start, "2opt", "first", keep_steps=False, max_evaluations=10 ** 9).n_moves == full.n_moves


def test_a_two_opt_local_optimum_has_no_crossing_and_or_opt_can_leave_some():
    crossings_swap = []
    for seed in range(6):
        xy, D = _instance(40, seed)
        t0 = A.random_tour(40, np.random.default_rng(seed))
        r = A.descend(D, t0, "2opt", "first", keep_steps=False)
        assert A.count_crossings(xy, r.tour) == 0
        assert A.count_crossings(xy, A.descend(D, t0, "2opt+oropt", "best", keep_steps=False).tour) == 0
        crossings_swap.append(A.count_crossings(xy, A.descend(D, t0, "swap", "first", keep_steps=False).tour))
    assert max(crossings_swap) > 0


def test_crossing_count_hand_instances():
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert A.count_crossings(xy, [0, 1, 2, 3]) == 1                          # Schleife 0-1 und 2-3 kreuzen sich, Kanten 1-2 und 3-0 nicht
    assert A.count_crossings(xy, [0, 2, 1, 3]) == 0                          # Rand des Quadrats
    star = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 3.0], [0.0, 2.0], [2.0, 2.0]])
    assert A.count_crossings(star, [0, 1, 3, 4, 2]) == A.count_crossings(star, [2, 4, 3, 1, 0])   # unabhängig von Startpunkt und Richtung


# --- Startlösungen und Hilfsfunktionen ------------------------------------------------------------------------------------------------------


def test_start_tours():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    D = A.dist_matrix(xy)
    assert A.nearest_neighbor_tour(D).tolist() == [0, 1, 3, 2, 4]
    assert A.start_tour("input", D).tolist() == [0, 1, 2, 3, 4]
    r = A.random_tour(30, np.random.default_rng(3))
    assert r[0] == 0 and sorted(r.tolist()) == list(range(30))
    assert not np.array_equal(A.random_tour(30, np.random.default_rng(1)), A.random_tour(30, np.random.default_rng(2)))
    with pytest.raises(ValueError):
        A.start_tour("best", D)


def test_canonical_edges_and_edge_share():
    t = np.array([3, 4, 0, 1, 2])
    assert A.canonical(t).tolist() in ([0, 1, 2, 3, 4], [0, 4, 3, 2, 1]) and A.canonical(t[::-1]).tolist() == A.canonical(t).tolist()
    assert A.edge_share(t, t[::-1]) == 1.0
    assert A.edge_share([0, 1, 2, 3], [0, 2, 1, 3]) == 0.5
    with pytest.raises(ValueError):
        A.descend(np.eye(4), np.arange(4), "3opt", "first")
    with pytest.raises(ValueError):
        A.descend(np.eye(4), np.arange(4), "2opt", "random")


def test_multi_start_returns_one_local_optimum_per_start():
    xy, D = _instance(20, 4)
    lengths, tours, moves, evals = A.multi_start(D, 6, 0, "2opt", "first")
    assert len(lengths) == len(tours) == len(moves) == len(evals) == 6
    for L, t in zip(lengths, tours):
        assert L == pytest.approx(A.tour_length(t, D)) and A.is_local_optimum(D, t, "2opt")
    l2 = A.multi_start(D, 6, 0, "2opt", "first")[0]
    assert np.array_equal(lengths, l2)


# --- Schranke -----------------------------------------------------------------------------------------------------------------------------------


def test_one_tree_bound_hand_instance_and_never_above_the_optimum():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    D = A.dist_matrix(square)
    assert A.held_karp_bound(D, 4.0) == pytest.approx(4.0, abs=1e-6)
    for seed in range(4):
        xy, D = _instance(8, seed)
        opt = min(A.tour_length([0, *p], D) for p in itertools.permutations(range(1, 8)))
        b = A.held_karp_bound(D, opt)
        assert b <= opt + 1e-9 and b >= 0.9 * opt


def test_bound_against_cp_sat_optimum():
    cp = pytest.importorskip("ortools.sat.python.cp_model")
    for seed in range(3):
        xy, D = _instance(25, seed)
        di = np.rint(D * 10000).astype(int)
        m = cp.CpModel()
        lits = {(i, j): m.NewBoolVar("") for i in range(25) for j in range(25) if i != j}
        m.AddCircuit([(i, j, l) for (i, j), l in lits.items()])
        m.Minimize(sum(int(di[i, j]) * l for (i, j), l in lits.items()))
        s = cp.CpSolver()
        s.parameters.max_time_in_seconds = 60
        assert s.StatusName(s.Solve(m)) == "OPTIMAL"
        opt = s.ObjectiveValue() / 10000
        ref = A.descend(D, A.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
        b = A.held_karp_bound(D, ref.length)
        assert b <= opt + 0.01 and (opt - b) / opt < 0.03
        assert ref.length >= opt - 0.01                                # CP-SAT rechnet mit auf 1e-4 gerundeten Entfernungen
