"""Presets: Vollständigkeit, gültige Werte, Urteile über mehrere Instanzen und Startlösungen (weite Bänder), Permalink-Grenzen."""

import pytest

import hc_constants as C
import hc_evaluation as ev
import hc_presets as P


def _settings(p, seed=None, start_seed=None):
    return ev.Settings(n=p["n"], cluster_share=p["ballung"], seed=p["seed"] if seed is None else seed, neighborhood=p["neighborhood"], start=p["start"], rule=p["rule"],
                       start_seed=p["start_seed"] if start_seed is None else start_seed)


def test_every_preset_has_help_bands_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS) and len(C.PRESETS) == 7
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0
        assert C.BALLUNG_MIN <= p["ballung"] <= C.BALLUNG_MAX and p["ballung"] % C.BALLUNG_STEP == 0
        assert p["neighborhood"] in C.NEIGHBORHOOD_LABELS and p["start"] in C.START_LABELS and p["rule"] in C.RULE_LABELS
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall (Voreinstellung)"]
    assert _settings(p) == ev.Settings()
    assert P.SETTING_SPECS["neighborhood_select"].default == C.DEFAULT_NEIGHBORHOOD and P.KEPT == {"start_seed_input": "_kept_start_seed_input"}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_verdicts_stay_in_their_bands_over_instances_and_starts(name):
    p = C.PRESETS[name]
    seeds = range(3) if p["n"] >= 150 else range(6)
    seen = set()
    for seed in seeds:
        for ss in (0, 1):
            seen.add(ev.verdict(ev.analyse(_settings(p, seed=seed, start_seed=ss), keep_steps=False)))
    assert seen <= C.PRESET_EXPECTED_BANDS[name], seen
    assert ev.verdict(ev.analyse(_settings(p), keep_steps=False)) in C.PRESET_EXPECTED_BANDS[name]


def test_preset_contrasts_at_the_default_instance():
    g = {name: ev.analyse(_settings(p), keep_steps=False).gap for name, p in C.PRESETS.items() if p["n"] == C.DEFAULT_N}
    assert g["Nur Tausch"] > 20 * 1.0 and g["Nur Tausch"] > 3 * g["Standardfall (Voreinstellung)"]
    assert g["2-opt + Or-opt"] < g["Standardfall (Voreinstellung)"]
    assert g["Nächster Nachbar + steilster Abstieg"] < g["Standardfall (Voreinstellung)"]
    assert g["Nur Or-opt"] > g["2-opt + Or-opt"]


def test_bounds_and_snapping_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert P.STEPS == {"n_slider": C.N_STEP, "ballung_slider": C.BALLUNG_STEP}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
