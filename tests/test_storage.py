import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gas_storage import SeasonalOU, StorageFacility, apply_policy, discount_factors, intrinsic, lsmc_value  # noqa: E402


def brute_force(fac, prices, disc):
    best = -np.inf
    for seq in itertools.product(fac.deltas, repeat=len(prices)):
        lvl, v, ok = fac.start_level, 0.0, True
        for n, d in enumerate(seq):
            lvl += d
            if not 0 <= lvl < fac.n_levels:
                ok = False
                break
            v += fac.cashflow(d, prices[n]) * disc[n]
        if ok and lvl == fac.end_level:
            best = max(best, v)
    return best


def test_intrinsic_matches_brute_force():
    fac = StorageFacility(capacity=100.0, n_levels=4, inj_steps=1, wdr_steps=2, rate=0.05)
    rng = np.random.default_rng(0)
    prices = rng.uniform(2.0, 5.0, size=7)
    disc = discount_factors(np.arange(7) / 252, 0.0, fac.rate)
    v, path = intrinsic(fac, prices, disc)
    assert v == pytest.approx(brute_force(fac, prices, disc), rel=1e-12)
    assert path[0] == fac.start_level and path[-1] == fac.end_level
    assert np.all(np.abs(np.diff(path)) <= max(fac.inj_steps, fac.wdr_steps))


def test_feasibility_respects_rates_and_terminal_condition():
    fac = StorageFacility(n_levels=11, inj_steps=1, wdr_steps=2)
    feas = fac.feasible(15)
    assert feas[0].sum() == 1 and feas[0, 0]
    assert feas[-1].sum() == 1 and feas[-1, 0]
    assert not feas[3, 4]          # cannot inject 4 levels in 3 days
    assert not feas[14, 3]         # cannot withdraw 3 levels in the last day


def test_ou_calibration_recovers_parameters():
    rng = np.random.default_rng(1)
    true = SeasonalOU(beta=np.array([1.2, 0.15, 0.05, 0.03, -0.02]), kappa=3.0, sigma=0.6)
    dates = np.cumsum(rng.choice([1, 1, 1, 1, 3], size=6000)) / 365.25        # irregular, weekend-like gaps
    x = true.simulate_x(0.0, 0.0, dates, n_paths=1, rng=rng, antithetic=False)[0]
    fitted = SeasonalOU.fit(dates, true.price_from_x(dates, x))
    assert fitted.kappa == pytest.approx(true.kappa, rel=0.35)
    assert fitted.sigma == pytest.approx(true.sigma, rel=0.05)
    assert fitted.beta[1] == pytest.approx(true.beta[1], abs=0.05)


def test_jump_model_recovers_parameters_and_beats_gaussian():
    rng = np.random.default_rng(4)
    true = SeasonalOU(beta=np.array([1.2, 0.1, 0.0, 0.0, 0.0]), kappa=3.0, sigma=0.5, lam=20.0, mu_j=0.0, sig_j=0.25)
    dates = np.cumsum(rng.choice([1, 1, 1, 1, 3], size=8000)) / 365.25
    x = true.simulate_x(0.0, 0.0, dates, n_paths=1, rng=rng, antithetic=False)[0]
    prices = true.price_from_x(dates, x)
    mrjd = SeasonalOU.fit(dates, prices, jumps=True)
    ou = SeasonalOU.fit(dates, prices, jumps=False)
    assert mrjd.sigma == pytest.approx(true.sigma, rel=0.1)
    assert mrjd.lam == pytest.approx(true.lam, rel=0.35)
    assert mrjd.sig_j == pytest.approx(true.sig_j, rel=0.2)
    assert mrjd.aic < ou.aic


def test_jump_expected_price_matches_monte_carlo():
    m = SeasonalOU(beta=np.array([1.1, 0.1, 0.0, 0.0, 0.0]), kappa=2.0, sigma=0.6, lam=18.0, mu_j=0.02, sig_j=0.25)
    t = np.arange(1, 253) / 252
    X = m.simulate_x(0.1, 0.0, t, 100_000, np.random.default_rng(5))
    mc = m.price_from_x(t, X).mean(axis=0)
    assert np.allclose(m.expected_price(0.1, 0.0, t)[[60, 250]], mc[[60, 250]], rtol=0.02)


def test_lsmc_collapses_to_intrinsic_without_uncertainty():
    model = SeasonalOU(beta=np.array([1.0, 0.2, 0.0, 0.0, 0.0]), kappa=2.0, sigma=1e-6)
    fac = StorageFacility(n_levels=11)
    t = np.arange(1, 121) / 365.25
    rng = np.random.default_rng(2)
    X = model.simulate_x(0.0, 0.0, t, 200, rng)
    S = model.price_from_x(t, X)
    disc = discount_factors(t, 0.0, fac.rate)
    v_int, _ = intrinsic(fac, model.expected_price(0.0, 0.0, t), disc)
    v_lsmc, _, _ = lsmc_value(fac, S, X, disc)
    assert v_lsmc == pytest.approx(v_int, rel=1e-4)


def test_out_of_sample_value_is_a_lower_bound():
    model = SeasonalOU(beta=np.array([1.0, 0.25, 0.0, 0.0, 0.0]), kappa=4.0, sigma=0.9)
    fac = StorageFacility(n_levels=11)
    t = np.arange(1, 181) / 365.25
    disc = discount_factors(t, 0.0, fac.rate)
    rng = np.random.default_rng(3)
    X = model.simulate_x(0.0, 0.0, t, 4000, rng)
    v_in, se_in, policy = lsmc_value(fac, model.price_from_x(t, X), X, disc)
    X2 = model.simulate_x(0.0, 0.0, t, 4000, rng)
    v_out, levels, _ = apply_policy(fac, policy, model.price_from_x(t, X2), X2, disc)
    v_int, _ = intrinsic(fac, model.expected_price(0.0, 0.0, t), disc)
    assert v_out.mean() <= v_in + 4 * se_in
    assert v_in > v_int                                   # optionality has positive value
    assert np.all(levels[:, -1] == fac.end_level)         # contract terms always honoured
    assert np.all((levels >= 0) & (levels < fac.n_levels))
