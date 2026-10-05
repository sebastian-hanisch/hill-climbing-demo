"""Unabhängiges Orakel für Hill Climbing: Schleifen-Neuimplementierung der Nachbarschaften und des Abstiegs (erste/beste Verbesserung) auf ganzzahligen
symmetrischen Entfernungsmatrizen (exakte Gleichstände), Schritt für Schritt und mit gleicher Bewertungszahl; Kreuzungszählung gegen die Definition; 1-Baum gegen networkx."""

import numpy as np
import pytest

import hc_algorithm as A


def _len(t, D):
    return sum(D[t[k]][t[(k + 1) % len(t)]] for k in range(len(t)))


def _key(t):
    t = list(t)
    k = t.index(0)
    t = t[k:] + t[:k]
    if len(t) > 2 and t[1] > t[-1]:
        t = [t[0]] + t[:0:-1]
    return tuple(t)


def _neighbors(t, D, nb):
    """(Art, Nachbartouren in der Reihenfolge der Demo; None = ungültige Zelle)."""
    n = len(t)
    out = []
    if nb in ("swap", "2opt", "2opt+oropt"):
        kinds = (["swap"] if nb == "swap" else ["2opt"])
        for kind in kinds:
            cells = []
            for i in range(n):
                for j in range(n):
                    if j >= i + 2 and not (i == 0 and j == n - 1):
                        if kind == "swap":
                            u = t[:]
                            u[i], u[j] = u[j], u[i]
                        else:
                            u = t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:]
                        cells.append(u)
            out.append((kind, cells))
    if nb in ("oropt", "2opt+oropt"):
        for length in (1, 2, 3):
            if n < length + 3:
                continue
            cells = []
            for i in range(n):
                seg = [t[(i + q) % n] for q in range(length)]
                rest = [t[(i + length + q) % n] for q in range(n - length)]
                for m in range(n):
                    if (m - (i - 1)) % n < length + 1:
                        cells.append(None)
                        continue
                    q = rest.index(t[m])
                    f, r = rest[:q + 1] + seg + rest[q + 1:], rest[:q + 1] + seg[::-1] + rest[q + 1:]
                    cells.append(r if _len(r, D) < _len(f, D) - A.EPS else f)
            out.append((f"oropt{length}", cells))
    return out


def _oracle_move(t, D, nb, rule):
    base, evals, best = _len(t, D), 0, None
    for _, cells in _neighbors(t, D, nb):
        vals = []
        for c in cells:
            if c is None:
                continue
            evals += 1
            d = _len(c, D) - base
            if rule == "first" and d < -A.EPS:
                return c, evals
            vals.append((d, c))
        if rule == "best" and vals:
            m = min(v for v, _ in vals)
            if m < -A.EPS and (best is None or m < best[0] - A.EPS):
                best = (m, next(c for v, c in vals if v == m))
    return (best[1] if best else None), evals


def _int_matrix(n, rng, hi):
    M = np.triu(rng.integers(0, hi + 1, size=(n, n)), 1)
    return (M + M.T).astype(float)


@pytest.mark.parametrize("nb", A.NEIGHBORHOODS)
@pytest.mark.parametrize("rule", A.RULES)
def test_descent_matches_a_loop_reimplementation_step_by_step(nb, rule):
    rng = np.random.default_rng(7)
    for _ in range(12):
        n = int(rng.integers(4, 10))
        D = _int_matrix(n, rng, int(rng.choice([2, 5, 20])))
        t = [0] + rng.permutation(np.arange(1, n)).tolist()
        r = A.descend(D, np.array(t), nb, rule)
        path, evals = [_key(t)], 0
        while True:
            c, e = _oracle_move(t, D, nb, rule)
            evals += e
            if c is None:
                break
            t = c
            path.append(_key(t))
        assert [_key(s.tour) for s in r.steps] == path
        assert r.evaluations == evals and r.n_moves == len(path) - 1


def test_crossings_match_the_definition_with_exact_arithmetic():
    rng = np.random.default_rng(3)

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    for _ in range(40):
        n = int(rng.integers(4, 12))
        xy = rng.integers(0, 6, size=(n, 2)).astype(float)
        t = rng.permutation(n)
        edges = [(t[k], t[(k + 1) % n]) for k in range(n)]
        count = 0
        for a in range(n):
            for b in range(a + 1, n):
                (p, q), (r, s) = edges[a], edges[b]
                if len({p, q, r, s}) == 4 and orient(xy[p], xy[q], xy[r]) * orient(xy[p], xy[q], xy[s]) < 0 and orient(xy[r], xy[s], xy[p]) * orient(xy[r], xy[s], xy[q]) < 0:
                    count += 1
        assert A.count_crossings(xy, t) == count


def test_one_tree_cost_matches_networkx_minimum_spanning_tree():
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(5)
    for _ in range(20):
        n = int(rng.integers(5, 10))
        D = A.dist_matrix(rng.random((n, 2)) * 100)
        pi = rng.random(n) * 10
        C = D + pi[:, None] + pi[None, :]
        G = nx.Graph()
        G.add_weighted_edges_from((i, j, C[i, j]) for i in range(1, n) for j in range(i + 1, n))
        ref = nx.minimum_spanning_tree(G).size(weight="weight") + sum(sorted(C[0, 1:])[:2])
        assert A._one_tree(C)[0] == pytest.approx(ref, abs=1e-9)
