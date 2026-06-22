"""
simulations/backtest_sp500/run_backtest.py

Runs a single train/test split backtest of the MFG-based alpha signals
and portfolio construction methodology described in Chapter 8/9, on
SYNTHETIC data (see the prominent disclaimer in _backtest_core.py --
no real S&P 500 / 13F data is used or available in this environment).

This script validates that the signal-construction -> portfolio
-construction -> evaluation pipeline is wired correctly end-to-end, by
checking that it recovers the predictive relationship deliberately
embedded in the synthetic data generator. It is NOT a replication of
the thesis's real empirical results (Sharpe 1.91 on 2005-2024 S&P 500
data); see walk_forward_validation.py for the more rigorous
walk-forward harness, and the module disclaimer for how to substitute
real data.

Usage:
    python run_backtest.py
    python run_backtest.py --n-assets 100 --n-months 360 --seed 7
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _backtest_core import (
    BacktestConfig,
    compute_performance_metrics,
    generate_synthetic_panel,
    run_single_backtest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-assets", type=int, default=50)
    parser.add_argument("--n-months", type=int, default=240)
    parser.add_argument("--train-frac", type=float, default=0.5, help="Fraction of months used for the (unused, illustrative) train split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("SYNTHETIC DATA DISCLAIMER: this backtest uses synthetic, not real,")
    print("market data. See _backtest_core.py module docstring for details.")
    print("=" * 70)

    config = BacktestConfig(n_assets=args.n_assets, n_months=args.n_months)
    print(f"\n==> Generating synthetic panel ({args.n_assets} assets x {args.n_months} months, seed={args.seed})...")
    panel = generate_synthetic_panel(config, seed=args.seed)

    train_end = int(args.n_months * args.train_frac)
    print(f"==> Train period: months [0, {train_end}) -- illustrative only, not used for fitting "
          f"(the synthetic signal here requires no estimation, unlike the real GMM calibration of Chapter 9)")
    print(f"==> Test (evaluation) period: months [{train_end}, {args.n_months})")

    result = run_single_backtest(panel, config, start_month=train_end, end_month=args.n_months)
    metrics = compute_performance_metrics(result["returns"])

    print(f"\n{'Metric':<25} {'Value':>12}")
    print("-" * 38)
    print(f"{'Annualised Return':<25} {metrics['annualised_return']*100:>11.2f}%")
    print(f"{'Annualised Volatility':<25} {metrics['annualised_vol']*100:>11.2f}%")
    print(f"{'Sharpe Ratio':<25} {metrics['sharpe_ratio']:>12.3f}")
    print(f"{'Maximum Drawdown':<25} {metrics['max_drawdown']*100:>11.2f}%")
    print(f"{'Mean Crowding Measure':<25} {result['crowding'].mean():>12.4f}")

    # Sanity check: a long-short equal-weight benchmark using the SAME
    # signal but without crowding discount / mean-variance weighting,
    # to confirm the optimisation step adds value over the raw signal.
    print("\n==> Sanity check: comparing against naive equal-weight signal sort...")
    naive_returns = []
    for t in result["months"]:
        month_data = panel[panel["t"] == t].sort_values("asset")
        spread = month_data["spread"].to_numpy()
        ret = month_data["return"].to_numpy()
        naive_weights = np.sign(spread) / len(spread)
        naive_weights = naive_weights - np.mean(naive_weights)
        gross = np.sum(np.abs(naive_weights))
        if gross > 1e-8:
            naive_weights = naive_weights / gross
        naive_returns.append(float(np.dot(naive_weights, ret)))
    naive_metrics = compute_performance_metrics(np.array(naive_returns))
    print(f"    Naive sign-sort Sharpe: {naive_metrics['sharpe_ratio']:.3f}")
    print(f"    MFG-weighted Sharpe:    {metrics['sharpe_ratio']:.3f}")

    output_path = (
        Path(args.output) if args.output
        else Path(__file__).resolve().parent / "backtest_results.csv"
    )
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "return", "crowding"])
        for t, r, c in zip(result["months"], result["returns"], result["crowding"]):
            writer.writerow([t, r, c])

    print(f"\n==> Monthly returns written to: {output_path}")


if __name__ == "__main__":
    main()
