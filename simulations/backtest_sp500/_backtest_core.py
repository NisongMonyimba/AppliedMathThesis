"""
simulations/backtest_sp500/_backtest_core.py

Shared core logic for run_backtest.py and walk_forward_validation.py:
synthetic data generation, the four alpha signals (Chapter 8), and
mean-variance portfolio construction with a crowding discount.

============================================================================
IMPORTANT DATA DISCLAIMER -- READ BEFORE INTERPRETING ANY OUTPUT
============================================================================
This module generates SYNTHETIC market data. No real S&P 500 prices,
13F institutional holdings, or CFTC positioning data are used or
available in this environment (no network access to CRSP/Compustat/SEC
EDGAR). The synthetic generator constructs a panel of asset "ownership"
deviations that mean-revert toward a latent, common-noise-driven
equilibrium (re-using the same LQ-MFG structure as the rest of this
repository), and BY CONSTRUCTION embeds a genuine predictive
relationship between the ownership deviation and next-period returns.

This lets the backtest pipeline (signal construction -> portfolio
weights -> realised P&L -> Sharpe ratio) be tested end-to-end for
correctness ("does the machinery work and recover the signal it was
given?"), but the resulting Sharpe ratio, performance numbers, and any
comparison to the thesis's reported empirical results (Chapter 9,
Sharpe 1.91 on real 2005-2024 S&P 500 data) are NOT comparable and
NOT a substitute for the real empirical study. To run this against
real data, replace `generate_synthetic_panel()` with a loader that
reads actual CRSP returns and SEC 13F filings, keeping the rest of the
pipeline (signal construction, portfolio construction, evaluation)
unchanged.
============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BacktestConfig:
    n_assets: int = 50
    n_months: int = 240  # 20 years of monthly data, matching Ch.9's 2005-2024 window
    kappa: float = 0.4  # mean-reversion speed of ownership toward equilibrium
    sigma_idio: float = 0.15  # idiosyncratic monthly vol of ownership deviation
    sigma_common: float = 0.08  # common-noise monthly vol
    signal_strength: float = 0.02  # tuned so the synthetic Sharpe lands in a plausible
    # (not artificially inflated) range -- see module disclaimer; this is a sanity-check
    # signal strength, not calibrated to any real market property
    return_noise_std: float = 0.06  # monthly return idiosyncratic noise
    common_factor_loading_std: float = 0.15  # cross-sectional dispersion of factor betas
    risk_aversion: float = 2.0
    crowding_c0: float = 0.2
    transaction_cost_bps: float = 5.0
    turnover_cap: float = 0.20


def generate_synthetic_panel(config: BacktestConfig, seed: int) -> pd.DataFrame:
    """
    Generate a synthetic panel of (date, asset, ownership_deviation, return)
    with a built-in predictive relationship, as described in the module
    disclaimer above. Returns a long-format DataFrame.
    """
    rng = np.random.default_rng(seed)
    n, T = config.n_assets, config.n_months

    # Latent per-asset equilibrium ownership level (heterogeneous across assets).
    equilibrium = rng.normal(0.0, 0.3, size=n)
    # Per-asset factor loadings (beta to the single common factor).
    betas = rng.normal(1.0, config.common_factor_loading_std, size=n)

    ownership = rng.normal(0.0, 0.2, size=n)  # initial deviations from equilibrium
    records = []
    prev_spread = equilibrium - ownership  # spread known at the start of month 0

    for t in range(T):
        common_shock = rng.normal(0.0, config.sigma_common)
        idio_shock = rng.normal(0.0, config.sigma_idio, size=n)
        # Ownership deviation mean-reverts toward equilibrium, consistent
        # with the LQ-MFG structure used throughout this repository.
        ownership = ownership + config.kappa * (equilibrium - ownership) + idio_shock + common_shock

        spread = equilibrium - ownership  # spread known at the START of month t (for use in month t+1's trade)

        # Common market factor return for this month.
        factor_return = rng.normal(0.005, 0.04)
        # Built-in predictive relationship: the return realised OVER month t
        # responds to the spread known at the START of month t (prev_spread,
        # i.e. computed from the PREVIOUS month's ownership update), not the
        # spread computed using this same month's shocks -- this is the
        # correct lag structure for a tradable signal (avoids any
        # contemporaneous/look-ahead overlap between signal and the return
        # it is meant to predict).
        asset_return = (
            config.signal_strength * prev_spread
            + betas * factor_return
            + rng.normal(0.0, config.return_noise_std, size=n)
        )

        for i in range(n):
            records.append({
                "t": t,
                "asset": i,
                "ownership_deviation": ownership[i],
                "spread": prev_spread[i],  # the signal AVAILABLE for trading at month t
                "beta": betas[i],
                "factor_return": factor_return,
                "return": asset_return[i],  # the return realised using that signal
            })

        prev_spread = spread

    return pd.DataFrame.from_records(records)


def compute_crowding_measure(spread: FloatArray) -> float:
    """
    Crowding measure c_t = W2(mu_t, mu_t*), approximated via the closed-form
    Gaussian formula (Proposition 8.1) applied to the cross-sectional
    distribution of spreads: treat the "equilibrium" as spread=0 (mean
    zero, the same dispersion), so c_t reduces to the absolute value of the
    cross-sectional mean spread (the W2 distance between two Gaussians with
    equal variance and means m, 0 is exactly |m|).
    """
    return float(np.abs(np.mean(spread)))


def crowding_discount(c_t: float, c0: float) -> float:
    return c_t / (c_t + c0)


def construct_portfolio_weights(
    spread: FloatArray,
    beta: FloatArray,
    cov_diag: FloatArray,
    config: BacktestConfig,
) -> FloatArray:
    """
    Construct mean-variance portfolio weights from the spread signal,
    with a crowding discount applied. For tractability (and because a
    full cross-sectional covariance matrix is not the focus of this
    pipeline test), the covariance is treated as diagonal (idiosyncratic
    variances only); this is a simplification relative to Chapter 9's
    full Ledoit-Wolf shrinkage estimator.
    """
    standardised_spread = (spread - np.mean(spread)) / (np.std(spread) + 1e-8)
    expected_return = standardised_spread  # eta_1-scaled signal, eta_1=1 for simplicity
    raw_weights = expected_return / (config.risk_aversion * cov_diag)

    c_t = compute_crowding_measure(spread)
    discount = crowding_discount(c_t, config.crowding_c0)
    weights = discount * raw_weights

    # Dollar-neutral normalisation.
    weights = weights - np.mean(weights)
    gross = np.sum(np.abs(weights))
    if gross > 1e-8:
        weights = weights / gross  # normalise to unit gross exposure
    return weights, c_t


def apply_turnover_cap(new_weights: FloatArray, old_weights: FloatArray, cap: float) -> FloatArray:
    """Scale back trades if the implied one-way turnover exceeds the cap."""
    trade = new_weights - old_weights
    turnover = 0.5 * np.sum(np.abs(trade))
    if turnover > cap and turnover > 1e-8:
        scale = cap / turnover
        return old_weights + scale * trade
    return new_weights


def run_single_backtest(
    panel: pd.DataFrame,
    config: BacktestConfig,
    start_month: int,
    end_month: int,
) -> dict:
    """
    Run the portfolio construction + evaluation loop over months
    [start_month, end_month), returning realised monthly returns,
    turnover, and crowding measure series.
    """
    months = sorted(panel["t"].unique())
    months = [m for m in months if start_month <= m < end_month]

    monthly_returns = []
    crowding_series = []
    old_weights = None

    for t in months:
        month_data = panel[panel["t"] == t].sort_values("asset")
        spread = month_data["spread"].to_numpy()
        beta = month_data["beta"].to_numpy()
        realised_return = month_data["return"].to_numpy()
        cov_diag = np.full(len(spread), config.return_noise_std**2)

        new_weights, c_t = construct_portfolio_weights(spread, beta, cov_diag, config)
        if old_weights is None:
            old_weights = np.zeros_like(new_weights)
        capped_weights = apply_turnover_cap(new_weights, old_weights, config.turnover_cap)

        turnover = 0.5 * np.sum(np.abs(capped_weights - old_weights))
        cost = turnover * 2 * config.transaction_cost_bps / 10000.0  # round-trip cost

        portfolio_return = float(np.dot(capped_weights, realised_return)) - cost
        monthly_returns.append(portfolio_return)
        crowding_series.append(c_t)
        old_weights = capped_weights

    return {
        "months": months,
        "returns": np.array(monthly_returns),
        "crowding": np.array(crowding_series),
    }


def compute_performance_metrics(monthly_returns: FloatArray) -> dict:
    """Standard annualised performance metrics from a monthly return series."""
    mean_monthly = float(np.mean(monthly_returns))
    std_monthly = float(np.std(monthly_returns, ddof=1))
    annualised_return = mean_monthly * 12
    annualised_vol = std_monthly * np.sqrt(12)
    sharpe = annualised_return / annualised_vol if annualised_vol > 1e-12 else 0.0

    cumulative = np.cumprod(1 + monthly_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(np.min(drawdown))

    return {
        "annualised_return": annualised_return,
        "annualised_vol": annualised_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }
