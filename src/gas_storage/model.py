"""Seasonal mean-reverting log-price models for gas spot prices.

    ln S(t) = f(t) + X(t),     f(t) = mu + sum_k [c_k cos(2 pi k t) + d_k sin(2 pi k t)]
    dX = -kappa X dt + sigma dW + J dN,     N ~ Poisson(lam),  J ~ N(mu_j, sig_j^2)

With lam = 0 this is the one-factor Schwartz (Ornstein-Uhlenbeck) model; with lam > 0 it is the
mean-reverting jump-diffusion (MRJD) used for spiky energy prices (Clewlow & Strickland, 2000).
Over a gap dt the diffusive part is exactly Gaussian with variance sigma^2 (1 - e^{-2 kappa dt}) / (2 kappa),
so weekends and holidays are handled exactly. Jumps are approximated as at most one per observation
interval, giving a two-component Gaussian-mixture transition density that is maximised directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def fourier_design(t: np.ndarray, n_harm: int) -> np.ndarray:
    cols = [np.ones_like(t)]
    for k in range(1, n_harm + 1):
        cols += [np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]
    return np.column_stack(cols)


def _normal_pdf(r: np.ndarray, mean, var) -> np.ndarray:
    return np.exp(-0.5 * (r - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)


def transition_loglik(x: np.ndarray, dt: np.ndarray, kappa: float, sigma: float,
                      lam: float = 0.0, mu_j: float = 0.0, sig_j: float = 0.0) -> float:
    phi = np.exp(-kappa * dt)
    var = sigma**2 * (1 - phi**2) / (2 * kappa)
    r = x[1:] - phi * x[:-1]
    if lam == 0.0:
        return float(np.sum(-0.5 * (np.log(2 * np.pi * var) + r**2 / var)))
    p = 1 - np.exp(-lam * dt)
    dens = (1 - p) * _normal_pdf(r, 0.0, var) + p * _normal_pdf(r, mu_j, var + sig_j**2)
    return float(np.sum(np.log(dens + 1e-300)))


@dataclass
class SeasonalOU:
    beta: np.ndarray      # [mu, c1, d1, c2, d2, ...]
    kappa: float          # mean-reversion speed, 1/year
    sigma: float          # diffusive volatility of X, 1/sqrt(year)
    lam: float = 0.0      # jump intensity, jumps/year
    mu_j: float = 0.0     # mean log jump size
    sig_j: float = 0.0    # std of log jump size
    n_harm: int = 2
    loglik: float = float("nan")

    @property
    def has_jumps(self) -> bool:
        return self.lam > 0

    @property
    def n_params(self) -> int:
        return len(self.beta) + (5 if self.has_jumps else 2)

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    @property
    def half_life_days(self) -> float:
        return np.log(2) / self.kappa * 365.25

    def seasonal(self, t: np.ndarray) -> np.ndarray:
        return fourier_design(np.asarray(t, float), self.n_harm) @ self.beta

    def deseasonalise(self, t: np.ndarray, log_price: np.ndarray) -> np.ndarray:
        return np.asarray(log_price) - self.seasonal(t)

    # ------------------------------------------------------------------ estimation
    @classmethod
    def fit(cls, t: np.ndarray, price: np.ndarray, n_harm: int = 2, jumps: bool = False) -> "SeasonalOU":
        """Two-step estimation: OLS for the seasonal curve, then exact maximum likelihood for X."""
        t, y = np.asarray(t, float), np.log(np.asarray(price, float))
        design = fourier_design(t, n_harm)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        x, dt = y - design @ beta, np.diff(t)
        phi0 = np.clip(np.polyfit(x[:-1], x[1:], 1)[0], 1e-4, 0.9999)
        k0 = -np.log(phi0) / np.median(dt)
        dx = np.diff(x)
        s_robust = 1.4826 * np.median(np.abs(dx - np.median(dx))) / np.sqrt(np.median(dt))
        if not jumps:
            def nll(p):
                return -transition_loglik(x, dt, np.exp(p[0]), np.exp(p[1]))
            res = minimize(nll, [np.log(k0), np.log(np.std(dx) / np.sqrt(np.median(dt)))], method="Nelder-Mead",
                           options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
            kappa, sigma = np.exp(res.x)
            return cls(beta, float(kappa), float(sigma), n_harm=n_harm, loglik=-float(res.fun))

        def nll_j(p):
            kappa, sigma, lam, sig_j = np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), np.exp(p[4])
            return -transition_loglik(x, dt, kappa, sigma, lam, p[3], sig_j)
        # robust diffusive vol as the start; jumps pick up the tails
        start = [np.log(k0), np.log(s_robust), np.log(10.0), 0.0, np.log(0.2)]
        res = minimize(nll_j, start, method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000, "maxfev": 20000})
        res = minimize(nll_j, res.x, method="BFGS")
        kappa, sigma, lam, sig_j = np.exp(res.x[[0, 1, 2, 4]])
        return cls(beta, float(kappa), float(sigma), float(lam), float(res.x[3]), float(sig_j),
                   n_harm=n_harm, loglik=-float(res.fun))

    # ------------------------------------------------------------------ moments
    def expected_price(self, x0: float, t0: float, t: np.ndarray) -> np.ndarray:
        """E[S(t) | X(t0) = x0] -- the model-implied 'forward curve' under the statistical measure.

        Diffusion: lognormal moment. Jumps: for compound-Poisson jumps decaying at rate kappa,
        E[exp(sum J_k e^{-kappa (t - s_k)})] = exp(lam * int_0^tau (M_J(e^{-kappa u}) - 1) du).
        """
        tau = np.asarray(t, float) - t0
        mean = x0 * np.exp(-self.kappa * tau)
        var = self.sigma**2 * (1 - np.exp(-2 * self.kappa * tau)) / (2 * self.kappa)
        log_m = self.seasonal(t) + mean + 0.5 * var
        if self.has_jumps:
            jump_term = np.empty_like(tau)
            for i, T in enumerate(tau):
                u = np.linspace(0.0, T, 400)
                a = np.exp(-self.kappa * u)
                jump_term[i] = self.lam * np.trapezoid(np.exp(a * self.mu_j + 0.5 * a**2 * self.sig_j**2) - 1, u)
            log_m = log_m + jump_term
        return np.exp(log_m)

    # ------------------------------------------------------------------ simulation
    def simulate_x(self, x0: float, t0: float, t: np.ndarray, n_paths: int,
                   rng: np.random.Generator, antithetic: bool = True) -> np.ndarray:
        """Exact OU simulation (plus at most one jump per step) on the grid t; returns (n_paths, len(t))."""
        t = np.asarray(t, float)
        dt = np.diff(np.concatenate([[t0], t]))
        phi = np.exp(-self.kappa * dt)
        sd = self.sigma * np.sqrt((1 - phi**2) / (2 * self.kappa))
        half = (n_paths + 1) // 2 if antithetic else n_paths
        z = rng.standard_normal((half, len(t)))
        if antithetic:
            z = np.vstack([z, -z])[:n_paths]
        if self.has_jumps:
            p_jump = 1 - np.exp(-self.lam * dt)
            jumps = (rng.random((n_paths, len(t))) < p_jump) * rng.normal(self.mu_j, self.sig_j, (n_paths, len(t)))
        else:
            jumps = np.zeros((n_paths, len(t)))
        x = np.empty((n_paths, len(t)))
        prev = np.full(n_paths, x0, dtype=float)
        for i in range(len(t)):
            prev = prev * phi[i] + sd[i] * z[:, i] + jumps[:, i]
            x[:, i] = prev
        return x

    def price_from_x(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.exp(self.seasonal(t) + x)
