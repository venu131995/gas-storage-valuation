"""Gas storage valuation: intrinsic dynamic programming and Least-Squares Monte Carlo (LSMC).

The facility is a discretised inventory grid. Each trading day the operator can inject up to
`inj_steps` grid levels, withdraw up to `wdr_steps` levels, or do nothing. Contract terms force
the facility to start and finish at given inventory levels (empty -> empty by default).

LSMC follows Boogert & de Jong (2008): a backward induction where, at each date, the value of
holding inventory level j until tomorrow is approximated by a regression of realised future
cash flows on polynomial functions of the state variable. Realised (not fitted) cash flows are
carried backwards, which keeps the in-sample estimate close to unbiased; an independent set of
paths run through the stored regression coefficients then gives a lower bound on the value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEG = -np.inf


@dataclass(frozen=True)
class StorageFacility:
    capacity: float = 1_000_000.0   # working gas, MMBtu
    n_levels: int = 61               # inventory grid points including empty and full
    inj_steps: int = 1               # max grid levels injected per day  (fill in 60 days)
    wdr_steps: int = 2               # max grid levels withdrawn per day (empty in 30 days)
    inj_cost: float = 0.02           # USD/MMBtu variable injection cost
    wdr_cost: float = 0.01           # USD/MMBtu variable withdrawal cost
    fuel_loss: float = 0.01          # fraction of injected gas consumed as compressor fuel
    rate: float = 0.04               # continuously-compounded discount rate
    start_level: int = 0
    end_level: int = 0

    @property
    def dv(self) -> float:
        return self.capacity / (self.n_levels - 1)

    @property
    def deltas(self) -> range:
        return range(-self.wdr_steps, self.inj_steps + 1)

    def cashflow(self, delta: int, price):
        """Undiscounted cash flow (USD) of moving `delta` grid levels at spot `price`."""
        vol = abs(delta) * self.dv
        if delta > 0:
            return -vol * (price * (1 + self.fuel_loss) + self.inj_cost)
        if delta < 0:
            return vol * (price - self.wdr_cost)
        return price * 0.0

    def feasible(self, n_steps: int) -> np.ndarray:
        """Boolean (n_steps+1, n_levels): levels reachable from the start AND able to reach the end."""
        n = np.arange(n_steps + 1)[:, None]
        lvl = np.arange(self.n_levels)[None, :]
        from_start = (lvl <= self.start_level + n * self.inj_steps) & (lvl >= self.start_level - n * self.wdr_steps)
        rem = n_steps - n
        to_end = (lvl - rem * self.wdr_steps <= self.end_level) & (lvl + rem * self.inj_steps >= self.end_level)
        return from_start & to_end


def discount_factors(t: np.ndarray, t0: float, rate: float) -> np.ndarray:
    return np.exp(-rate * (np.asarray(t, float) - t0))


# ----------------------------------------------------------------------------- intrinsic value

def intrinsic(fac: StorageFacility, prices: np.ndarray, disc: np.ndarray) -> tuple[float, np.ndarray]:
    """Optimal schedule against a single deterministic price path (a forward curve or a realised path).

    Returns the discounted value and the optimal inventory path (grid levels, length N+1).
    """
    prices = np.asarray(prices, float)
    n_steps, L = len(prices), fac.n_levels
    feas = fac.feasible(n_steps)
    V = np.where(feas[n_steps], 0.0, NEG)
    choice = np.zeros((n_steps, L), dtype=int)
    for n in range(n_steps - 1, -1, -1):
        Vn = np.full(L, NEG)
        for lvl in np.flatnonzero(feas[n]):
            for d in fac.deltas:
                j = lvl + d
                if 0 <= j < L and feas[n + 1, j]:
                    v = fac.cashflow(d, prices[n]) * disc[n] + V[j]
                    if v > Vn[lvl]:
                        Vn[lvl], choice[n, lvl] = v, d
        V = Vn
    path = [fac.start_level]
    for n in range(n_steps):
        path.append(path[-1] + choice[n, path[-1]])
    return float(V[fac.start_level]), np.array(path)


# ----------------------------------------------------------------------------- LSMC

def _basis(z: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(z, degree + 1, increasing=True)


@dataclass
class LSMCPolicy:
    coefs: np.ndarray        # (N, L, degree+1) continuation-value regressions per date and target level
    x_mean: np.ndarray       # (N,) standardisation of the state variable per date
    x_std: np.ndarray
    degree: int
    feas: np.ndarray

    def continuation(self, n: int, j: int, x: np.ndarray) -> np.ndarray:
        z = (x - self.x_mean[n]) / self.x_std[n]
        return _basis(z, self.degree) @ self.coefs[n, j]


def lsmc_value(fac: StorageFacility, S: np.ndarray, X: np.ndarray, disc: np.ndarray,
               degree: int = 3) -> tuple[float, float, LSMCPolicy]:
    """In-sample LSMC value (mean, standard error) and the fitted exercise policy."""
    M, N = S.shape
    L = fac.n_levels
    feas = fac.feasible(N)
    V = np.where(feas[N], 0.0, NEG)[None, :].repeat(M, axis=0)
    coefs = np.zeros((N, L, degree + 1))
    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-12
    for n in range(N - 1, -1, -1):
        B = _basis((X[:, n] - x_mean[n]) / x_std[n], degree)
        C = np.full((M, L), NEG)
        for j in np.flatnonzero(feas[n + 1]):
            coefs[n, j] = np.linalg.lstsq(B, V[:, j], rcond=None)[0]
            C[:, j] = B @ coefs[n, j]
        cf = {d: fac.cashflow(d, S[:, n]) * disc[n] for d in fac.deltas}
        Vn = np.full((M, L), NEG)
        for lvl in np.flatnonzero(feas[n]):
            best = np.full(M, NEG)
            realised = np.zeros(M)
            for d in fac.deltas:
                j = lvl + d
                if 0 <= j < L and feas[n + 1, j]:
                    score = cf[d] + C[:, j]
                    take = score > best
                    best = np.where(take, score, best)
                    realised = np.where(take, cf[d] + V[:, j], realised)
            Vn[:, lvl] = realised
        V = Vn
    v = V[:, fac.start_level]
    policy = LSMCPolicy(coefs=coefs, x_mean=x_mean, x_std=x_std, degree=degree, feas=feas)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(M)), policy


def apply_policy(fac: StorageFacility, policy: LSMCPolicy, S: np.ndarray, X: np.ndarray,
                 disc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the stored LSMC exercise rule forward on new paths (simulated or historical).

    Returns per-path discounted value, inventory paths (M, N+1) in grid levels, and daily
    discounted cash flows (M, N).
    """
    S, X = np.atleast_2d(S), np.atleast_2d(X)
    M, N = S.shape
    L = fac.n_levels
    lvl = np.full(M, fac.start_level)
    levels = np.empty((M, N + 1), dtype=int)
    levels[:, 0] = lvl
    flows = np.zeros((M, N))
    rows = np.arange(M)
    deltas = list(fac.deltas)
    for n in range(N):
        cf = np.array([fac.cashflow(d, S[:, n]) * disc[n] for d in deltas])          # (A, M)
        cont = np.full((L, M), NEG)
        for j in np.flatnonzero(policy.feas[n + 1]):
            cont[j] = policy.continuation(n, j, X[:, n])
        score = np.full((len(deltas), M), NEG)
        for a, d in enumerate(deltas):
            j = lvl + d
            ok = (j >= 0) & (j < L)
            jj = np.clip(j, 0, L - 1)
            score[a] = np.where(ok, cf[a] + cont[jj, rows], NEG)
        a_star = score.argmax(axis=0)
        flows[:, n] = cf[a_star, rows]
        lvl = lvl + np.array(deltas)[a_star]
        levels[:, n + 1] = lvl
    return flows.sum(axis=1), levels, flows
