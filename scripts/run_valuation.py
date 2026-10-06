"""End-to-end run: calibrate on Henry Hub, value a storage facility, backtest the policy, write figures."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gas_storage import SeasonalOU, StorageFacility, apply_policy, discount_factors, intrinsic, lsmc_value  # noqa: E402
from gas_storage.data import henry_hub, year_fraction  # noqa: E402

FIG, RES = ROOT / "figures", ROOT / "results"
CAL_START = "2010-01-01"             # post-shale-revolution regime
BACKTEST = ("2024-04-01", "2025-03-31")
N_PATHS, N_PATHS_SENS = 20_000, 6_000
SEED = 20261006

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def mm(x: float) -> float:
    return round(float(x) / 1e6, 4)


def model_summary(m: SeasonalOU) -> dict:
    d = {"kappa": round(m.kappa, 4), "sigma": round(m.sigma, 4), "half_life_days": round(m.half_life_days, 1),
         "loglik": round(m.loglik, 1), "aic": round(m.aic, 1)}
    if m.has_jumps:
        d.update(jumps_per_year=round(m.lam, 2), mean_log_jump=round(m.mu_j, 4), jump_std=round(m.sig_j, 4))
    return d


def value_facility(model: SeasonalOU, fac: StorageFacility, x0: float, t0: float, t: np.ndarray,
                   n_paths: int, rng: np.random.Generator) -> dict:
    disc = discount_factors(t, t0, fac.rate)
    fwd = model.expected_price(x0, t0, t)
    v_int, int_path = intrinsic(fac, fwd, disc)
    X = model.simulate_x(x0, t0, t, n_paths, rng)
    v_in, se_in, policy = lsmc_value(fac, model.price_from_x(t, X), X, disc)
    X2 = model.simulate_x(x0, t0, t, n_paths, rng)
    pv_out, levels, _ = apply_policy(fac, policy, model.price_from_x(t, X2), X2, disc)
    return dict(disc=disc, fwd=fwd, policy=policy, int_path=int_path, X_out=X2, levels_out=levels,
                intrinsic=v_int, lsmc_in=v_in, lsmc_in_se=se_in, lsmc_out=float(pv_out.mean()),
                lsmc_out_se=float(pv_out.std(ddof=1) / np.sqrt(n_paths)))


def value_table(r: dict) -> dict:
    return {"intrinsic": mm(r["intrinsic"]), "lsmc_in_sample": mm(r["lsmc_in"]), "lsmc_in_sample_se": mm(r["lsmc_in_se"]),
            "lsmc_out_of_sample": mm(r["lsmc_out"]), "lsmc_out_of_sample_se": mm(r["lsmc_out_se"]),
            "extrinsic_share": round(1 - r["intrinsic"] / r["lsmc_out"], 4)}


def schedule_flows(fac: StorageFacility, path: np.ndarray, prices: np.ndarray, disc: np.ndarray) -> np.ndarray:
    """Discounted daily cash flows of a fixed inventory schedule executed at the given prices."""
    return np.array([fac.cashflow(int(d), p) * df for d, p, df in zip(np.diff(path), prices, disc)])


def main() -> None:
    t_start = time.time()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    hh = henry_hub(ROOT / "data")
    fac = StorageFacility()
    out: dict = {"facility": asdict(fac), "data": {"series": "FRED DHHNGSP (Henry Hub spot, USD/MMBtu)",
                 "first": str(hh.index[0].date()), "last": str(hh.index[-1].date()), "n_obs": int(len(hh))}}

    # ------------------------------------------------------------------ 1. out-of-sample backtest year
    cal = hh[CAL_START:pd.Timestamp(BACKTEST[0]) - pd.Timedelta(days=1)]
    real = hh[BACKTEST[0]:BACKTEST[1]]
    t_cal, t_bt = np.asarray(year_fraction(cal.index)), np.asarray(year_fraction(real.index))
    out["backtest"] = {"calibration_window": [str(cal.index[0].date()), str(cal.index[-1].date())],
                       "storage_year": list(BACKTEST), "spot_range_usd": [float(real.min()), float(real.max())]}
    bt_runs = {}
    for name, jumps in [("ou", False), ("mrjd", True)]:
        m = SeasonalOU.fit(t_cal, cal.values, jumps=jumps)
        x0 = float(m.deseasonalise(t_cal[-1:], np.log(cal.values[-1:]))[0])
        r = value_facility(m, fac, x0, t_cal[-1], t_bt, N_PATHS, rng)
        x_real = m.deseasonalise(t_bt, np.log(real.values))
        pv_real, lev_real, fl_real = apply_policy(fac, r["policy"], real.values[None, :], x_real[None, :], r["disc"])
        r.update(model=m, pv_real=float(pv_real[0]), lev_real=lev_real[0], fl_real=fl_real[0])
        bt_runs[name] = r
    disc_bt = bt_runs["mrjd"]["disc"]
    v_pf, pf_path = intrinsic(fac, real.values, disc_bt)
    sched = bt_runs["mrjd"]["int_path"]
    sched_fl = schedule_flows(fac, sched, real.values, disc_bt)
    out["backtest"].update({
        "models": {k: model_summary(v["model"]) for k, v in bt_runs.items()},
        "exante_usd_mm": {k: value_table(v) for k, v in bt_runs.items()},
        "realised_usd_mm": {"lsmc_policy_mrjd": mm(bt_runs["mrjd"]["pv_real"]), "lsmc_policy_ou": mm(bt_runs["ou"]["pv_real"]),
                            "intrinsic_schedule": mm(sched_fl.sum()), "perfect_foresight": mm(v_pf)},
        "mrjd_policy_capture_of_perfect_foresight": round(bt_runs["mrjd"]["pv_real"] / v_pf, 4),
        "mrjd_policy_vs_intrinsic_schedule": round(bt_runs["mrjd"]["pv_real"] / sched_fl.sum(), 3),
    })

    # ------------------------------------------------------------------ 2. current valuation (next 12 months)
    cal_now = hh[CAL_START:]
    t_now = np.asarray(year_fraction(cal_now.index))
    fut = pd.bdate_range(cal_now.index[-1] + pd.Timedelta(days=1), periods=252)
    t_fut = np.asarray(year_fraction(fut))
    cur_runs = {}
    for name, jumps in [("ou", False), ("mrjd", True)]:
        m = SeasonalOU.fit(t_now, cal_now.values, jumps=jumps)
        x0 = float(m.deseasonalise(t_now[-1:], np.log(cal_now.values[-1:]))[0])
        cur_runs[name] = value_facility(m, fac, x0, t_now[-1], t_fut, N_PATHS, rng) | {"model": m, "x0": x0}
    m_now, cur, x0_now = cur_runs["mrjd"]["model"], cur_runs["mrjd"], cur_runs["mrjd"]["x0"]
    seas = m_now.seasonal(np.linspace(0, 1, 366, endpoint=False) + 26.0)
    out["current"] = {
        "valuation_date": str(cal_now.index[-1].date()), "last_spot": float(cal_now.values[-1]),
        "horizon": [str(fut[0].date()), str(fut[-1].date())],
        "models": {k: model_summary(v["model"]) for k, v in cur_runs.items()},
        "aic_improvement_mrjd_vs_ou": round(cur_runs["ou"]["model"].aic - m_now.aic, 1),
        "seasonal_peak_to_trough_pct": round(100 * (np.exp(seas.max() - seas.min()) - 1), 2),
        "usd_mm": {k: value_table(v) for k, v in cur_runs.items()},
        "usd_per_mmbtu_capacity_mrjd": round(cur["lsmc_out"] / fac.capacity, 4),
    }

    # ------------------------------------------------------------------ 3. sensitivities (MRJD)
    sens_vol = []
    for mult in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        r = value_facility(replace(m_now, sigma=m_now.sigma * mult, sig_j=m_now.sig_j * mult), fac,
                           x0_now, t_now[-1], t_fut, N_PATHS_SENS, rng)
        sens_vol.append({"vol_multiplier": mult, "intrinsic": mm(r["intrinsic"]), "lsmc_out": mm(r["lsmc_out"])})
    sens_rate = []
    for name, f in [("slow (fill 60d, empty 60d)", replace(fac, wdr_steps=1)),
                    ("base (fill 60d, empty 30d)", fac),
                    ("fast (fill 30d, empty 20d)", replace(fac, inj_steps=2, wdr_steps=3))]:
        r = value_facility(m_now, f, x0_now, t_now[-1], t_fut, N_PATHS_SENS, rng)
        sens_rate.append({"deliverability": name, "intrinsic": mm(r["intrinsic"]), "lsmc_out": mm(r["lsmc_out"])})
    out["sensitivity"] = {"volatility_mrjd": sens_vol, "deliverability_mrjd": sens_rate}
    out["runtime_seconds"] = round(time.time() - t_start, 1)
    (RES / "summary.json").write_text(json.dumps(out, indent=2))

    # ------------------------------------------------------------------ figures
    fit_t = t_now
    x_hist = m_now.deseasonalise(fit_t, np.log(cal_now.values))
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    ax[0].plot(cal_now.index, cal_now.values, lw=0.6, color="0.35", label="Henry Hub spot")
    ax[0].plot(cal_now.index, np.exp(m_now.seasonal(fit_t)), lw=1.4, color="tab:green", label="Fitted seasonal level f(t)")
    ax[0].set_yscale("log"); ax[0].set_ylabel("USD/MMBtu (log scale)"); ax[0].legend(loc="upper right")
    ax[0].set_title(f"MRJD fit since {CAL_START[:4]}: kappa = {m_now.kappa:.2f}/yr (half-life {m_now.half_life_days:.0f}d), "
                    f"sigma = {m_now.sigma:.2f}, {m_now.lam:.0f} jumps/yr (sd {m_now.sig_j:.2f})")
    ax[1].plot(cal_now.index, x_hist, lw=0.6, color="tab:blue")
    ax[1].axhline(0, color="k", lw=0.7); ax[1].set_ylabel("Deseasonalised X(t)")
    fig.tight_layout(); fig.savefig(FIG / "fig1_model_fit.png"); plt.close(fig)

    # one-business-day transition residuals vs the two fitted densities
    dt = np.diff(fit_t)
    one_day = np.isclose(dt, 1 / 365.25)
    m_ou = cur_runs["ou"]["model"]
    r_hist = (x_hist[1:] - np.exp(-m_now.kappa * dt) * x_hist[:-1])[one_day]
    grid = np.linspace(-0.6, 0.6, 600)
    d1 = 1 / 365.25
    v_ou = m_ou.sigma**2 * (1 - np.exp(-2 * m_ou.kappa * d1)) / (2 * m_ou.kappa)
    v_j = m_now.sigma**2 * (1 - np.exp(-2 * m_now.kappa * d1)) / (2 * m_now.kappa)
    p_j = 1 - np.exp(-m_now.lam * d1)
    pdf = lambda r, mu, v: np.exp(-0.5 * (r - mu) ** 2 / v) / np.sqrt(2 * np.pi * v)  # noqa: E731
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for a, logy in zip(ax, [False, True]):
        a.hist(r_hist, bins=200, range=(-0.6, 0.6), density=True, color="0.75", label="Henry Hub daily residuals")
        a.plot(grid, pdf(grid, 0, v_ou), color="tab:red", lw=1.3, label=f"Gaussian OU (AIC {m_ou.aic:,.0f})")
        a.plot(grid, (1 - p_j) * pdf(grid, 0, v_j) + p_j * pdf(grid, m_now.mu_j, v_j + m_now.sig_j**2),
               color="tab:blue", lw=1.3, label=f"Jump-diffusion (AIC {m_now.aic:,.0f})")
        if logy:
            a.set_yscale("log"); a.set_ylim(1e-3, None); a.set_title("Log scale: the tails")
        else:
            a.set_xlim(-0.3, 0.3); a.set_title("Body of the distribution"); a.legend(fontsize=7)
        a.set_xlabel("Daily change in X")
    fig.tight_layout(); fig.savefig(FIG / "fig0_jump_vs_gaussian.png"); plt.close(fig)

    S_out = m_now.price_from_x(t_fut, cur["X_out"])
    q = np.percentile(S_out, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(fut, q[0], q[4], color="tab:green", alpha=0.15, label="5-95%")
    ax.fill_between(fut, q[1], q[3], color="tab:green", alpha=0.3, label="25-75%")
    ax.plot(fut, cur["fwd"], color="tab:green", lw=1.6, label="Expected price (model forward curve)")
    ax.plot(cal_now.index[-120:], cal_now.values[-120:], color="0.3", lw=0.9, label="Realised spot")
    ax.set_ylabel("USD/MMBtu"); ax.legend(loc="upper left", ncol=2); ax.set_title("Simulated Henry Hub paths (MRJD) for the valuation horizon")
    fig.tight_layout(); fig.savefig(FIG / "fig2_price_simulation.png"); plt.close(fig)

    lev = cur["levels_out"] / (fac.n_levels - 1) * 100
    ql = np.percentile(lev, [10, 50, 90], axis=0)
    dates_lev = pd.DatetimeIndex([cal_now.index[-1]]).append(fut)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.fill_between(dates_lev, ql[0], ql[2], color="tab:blue", alpha=0.2, label="LSMC policy, 10-90% of paths")
    ax.plot(dates_lev, ql[1], color="tab:blue", lw=1.5, label="LSMC policy, median")
    ax.plot(dates_lev, cur["int_path"] / (fac.n_levels - 1) * 100, color="k", lw=1.2, ls="--", label="Intrinsic schedule")
    ax.set_ylabel("Inventory (% of capacity)"); ax.legend(loc="upper left"); ax.set_title("Optimal inventory under uncertainty vs. the intrinsic schedule")
    fig.tight_layout(); fig.savefig(FIG / "fig3_inventory_paths.png"); plt.close(fig)

    bt_dates = pd.DatetimeIndex([cal.index[-1]]).append(real.index)
    pf_fl = schedule_flows(fac, pf_path, real.values, disc_bt)
    fig, ax = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 1, 1]})
    ax[0].plot(real.index, real.values, color="0.3", lw=0.9, label="Realised Henry Hub")
    ax[0].plot(real.index, bt_runs["mrjd"]["fwd"], color="tab:green", ls="--", lw=1.2, label="Ex-ante model forward curve")
    ax[0].set_ylabel("USD/MMBtu"); ax[0].legend(loc="upper left")
    ax[0].set_title(f"Out-of-sample backtest, gas year {BACKTEST[0][:7]} to {BACKTEST[1][:7]} (calibrated only on prior data)")
    series = [(bt_runs["mrjd"]["lev_real"], bt_runs["mrjd"]["fl_real"], "LSMC policy (MRJD)", "-"),
              (bt_runs["ou"]["lev_real"], bt_runs["ou"]["fl_real"], "LSMC policy (Gaussian OU)", "-."),
              (sched, sched_fl, "Intrinsic schedule", "--"),
              (pf_path, pf_fl, "Perfect foresight (upper bound)", ":")]
    for path, fl, lab, st in series:
        ax[1].plot(bt_dates, np.asarray(path) / (fac.n_levels - 1) * 100, ls=st, lw=1.2, label=lab)
        ax[2].plot(real.index, np.cumsum(fl) / 1e6, ls=st, lw=1.2, label=f"{lab}: {fl.sum() / 1e6:.2f}m")
    ax[1].set_ylabel("Inventory (%)"); ax[1].legend(loc="upper left", fontsize=7)
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_ylabel("Cumulative PV (USD m)"); ax[2].legend(loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig4_backtest.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    mult = [s["vol_multiplier"] for s in sens_vol]
    ax.plot(mult, [s["lsmc_out"] for s in sens_vol], "o-", label="Total value (LSMC, out-of-sample)")
    ax.plot(mult, [s["intrinsic"] for s in sens_vol], "s--", label="Intrinsic value")
    ax.set_xlabel("Multiplier on calibrated volatilities (diffusion and jumps)"); ax.set_ylabel("USD m"); ax.legend()
    ax.set_title("Storage value is an option on volatility")
    fig.tight_layout(); fig.savefig(FIG / "fig5_vol_sensitivity.png"); plt.close(fig)

    pol, deltas = cur["policy"], list(fac.deltas)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for axx, target in zip(axs, ["07-15", "01-15"]):
        doy = pd.Timestamp(f"2001-{target}").dayofyear
        n = int(np.argmin(np.abs((np.asarray(fut.dayofyear) - doy + 182) % 365 - 182)))
        prices = np.linspace(1.0, 8.0, 200)
        x = np.log(prices) - m_now.seasonal(np.array([t_fut[n]]))[0]
        act = np.full((fac.n_levels, prices.size), np.nan)
        for lvl in np.flatnonzero(pol.feas[n]):
            best = np.full(prices.size, -np.inf)
            for d in deltas:
                j = lvl + d
                if 0 <= j < fac.n_levels and pol.feas[n + 1, j]:
                    sc = fac.cashflow(d, prices) * cur["disc"][n] + pol.continuation(n, j, x)
                    act[lvl] = np.where(sc > best, np.sign(d), act[lvl])
                    best = np.maximum(best, sc)
        im = axx.pcolormesh(prices, np.arange(fac.n_levels) / (fac.n_levels - 1) * 100, act, cmap="RdYlGn_r",
                            vmin=-1, vmax=1, shading="auto")
        axx.set_title(f"Decision rule on {fut[n].date()}"); axx.set_xlabel("Spot price (USD/MMBtu)")
    axs[0].set_ylabel("Inventory (% of capacity)")
    cb = fig.colorbar(im, ax=axs, ticks=[-1, 0, 1]); cb.ax.set_yticklabels(["withdraw", "hold", "inject"])
    fig.savefig(FIG / "fig6_decision_rule.png", bbox_inches="tight"); plt.close(fig)

    print(json.dumps({k: out[k] for k in ["backtest", "current", "sensitivity", "runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
