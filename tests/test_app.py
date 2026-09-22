"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt, Randwerte, Abspielen ohne doppelte Schlüssel, ausgeblendeter Start-Seed, Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import hc_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(hc_step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if hc_step != 1:                                                       # die App setzt den Schritt beim ersten Lauf zurück: erst danach wählen
        at.select_slider(key="hc_step").set_value(hc_step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_measured_default():
    at = _run()
    _ok(at)
    assert _metric(at, "Abstand zur Schranke") == "6.8 %" and _metric(at, "Züge") == "202"
    assert any("Steckengeblieben" in w.value for w in at.warning)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["neighborhood_select"] == p["neighborhood"] and at.session_state["n_slider"] == p["n"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n", [10, 60])
def test_every_step_runs(step, n):
    at = _run(n_slider=n, hc_step=step)
    _ok(at)
    assert at.get("plotly_chart")
    assert at.session_state["hc_step"] == step


def test_step_four_with_the_move_slider_and_zero_moves():
    at = _run(hc_step=4)
    _ok(at)
    mv = next(s for s in at.slider if s.key == "hc_move")
    assert mv.value == mv.max
    mv.set_value(3).run()
    _ok(at)
    # eine Startlösung, die schon ein lokales Optimum ist: kein Zug, kein Zug-Regler
    at2 = _run(n_slider=10, neighborhood_select="2opt+oropt", start_radio="nearest", rule_radio="best", seed_input=1, hc_step=4)
    _ok(at2)
    if _metric(at2, "Züge") == "0":
        assert not [s for s in at2.slider if s.key == "hc_move"]


def test_step_three_and_five_with_no_move_do_not_crash():
    at = _run(n_slider=10, neighborhood_select="2opt+oropt", start_radio="nearest", rule_radio="best", seed_input=1, hc_step=3)
    _ok(at)
    at.select_slider(key="hc_step").set_value(5).run()
    _ok(at)


def test_play_runs_through_all_steps_without_duplicate_keys():
    at = _run(n_slider=20)
    next(b for b in at.button if b.label == "▶️ Abspielen").click().run()
    _ok(at)


def test_play_moves_in_step_four():
    at = _run(n_slider=20, hc_step=4)
    next(b for b in at.button if b.label == "▶️ Züge abspielen").click().run()
    _ok(at)


def test_start_seed_control_is_hidden_for_nearest_neighbor_and_kept():
    at = _run(start_seed_input=42)
    assert any(n.key == "start_seed_input" for n in at.number_input)
    at.radio(key="start_radio").set_value("nearest").run()
    _ok(at)
    assert not any(n.key == "start_seed_input" for n in at.number_input)
    at.radio(key="start_radio").set_value("random").run()
    _ok(at)
    assert next(n for n in at.number_input if n.key == "start_seed_input").value == 42


def test_dice_buttons_change_the_seeds():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old
    old_s = at.session_state["start_seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Startlösung würfeln").click().run()
    _ok(at)
    assert at.session_state["start_seed_input"] != old_s


@pytest.mark.parametrize("kw", [dict(n_slider=200), dict(n_slider=10, ballung_slider=100), dict(neighborhood_select="swap"), dict(neighborhood_select="oropt", rule_radio="best", n_slider=30)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["ballung"] = "40"
    at.query_params["nb"] = "nonsense"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["ballung_slider"] == 50 and at.session_state["neighborhood_select"] == C.DEFAULT_NEIGHBORHOOD


def test_sweeps_run_on_demand():
    at = _run(n_slider=10)
    at.selectbox(key="sweep_select").set_value("rule").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert any(k == "sweep_chart" for k in [e.proto.id for e in at.get("plotly_chart")] ) or at.get("plotly_chart")


@pytest.mark.parametrize("key", ["comparison_start", "multistart_start", "hierarchy_start"])
def test_experiments_run_on_demand(key):
    at = _run(n_slider=10)
    next(b for b in at.button if b.key == key).click().run()
    _ok(at)
    assert at.session_state[{"comparison_start": "comparison_on", "multistart_start": "multistart_on", "hierarchy_start": "hierarchy_on"}[key]]


def test_scaling_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "SCALING_N", (10, 20))
    at = _run(n_slider=10)
    next(b for b in at.button if b.key == "scaling_start").click().run()
    _ok(at)
    assert at.session_state["scaling_on"]


def test_dlb_experiment_runs_on_demand_and_is_gated_to_two_opt(monkeypatch):
    monkeypatch.setattr(C, "DLB_BUDGETS", (25000, 100000))
    monkeypatch.setattr(C, "DLB_CHAINS", 2)
    at = _run(n_slider=15)
    assert next(b for b in at.button if b.key == "dlb_start")
    next(b for b in at.button if b.key == "dlb_start").click().run()
    _ok(at)
    assert at.session_state["dlb_on"] and at.get("plotly_chart")
    at.selectbox(key="neighborhood_select").set_value("oropt").run()
    _ok(at)
    assert not any(b.key == "dlb_start" for b in at.button)
    assert any("nur für die Nachbarschaft 2-opt gemessen" in c.value for c in at.caption)


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Nur verbessernde Züge" in m.value for m in at.markdown)
