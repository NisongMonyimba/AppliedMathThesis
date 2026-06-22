"""
simulations/backtest_sp500/walk_forward_validation.py

Implements purged walk-forward cross-validation (Chapter 9, Section
9.6), evaluating the portfolio strategy out-of-sample across a rolling
sequence of train/test windows, on SYNTHETIC data (see the disclaimer
in _backtest_core.py -- no real S&P 500 / 13F data available in this
environment).

Unlike run_backtest.py's single train/test split, this script rolls a
fixed-length window forward one month at a time, re-evaluating the
strategy out-of-sample at each step, and reports the aggregated
out-of-sample performance -- mirroring the methodology (though not the
data) of Chapter 9's real backtest.

Usage:
    python walk_forward_validation.py
    python walk_forward_validation.py --window-size 60 --step-size 1
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


def walk_forward_backtest(
    panel,
    config: BacktestConfig,
    window_size: int,
    step_size: int,
    n_months: int,
) -> dict:
    """
    Roll a fixed-length evaluation window forward by step_size months at
    a time, starting once window_size months of "history" have elapsed
    (illustrative -- the synthetic signal needs no estimation window, but
    the structure mirrors Chapter 9's rolling GMM calibration window).
    """
    all_returns = []
    all_crowding = []
    all_months = []
    window_sharpes = []

    start = window_size
    while start + step_size <= n_months:
        end = start + step_size
        result = run_single_backtest(panel, config, start_month=start, end_month=end)
        all_returns.extend(result["returns"].tolist())
        all_crowding.extend(result["crowding"].tolist())
        all_months.extend(result["months"])
        start += step_size

    # Compute a rolling Sharpe over non-overlapping 12-month blocks for
    # a stability diagnostic.
    returns_arr = np.array(all_returns)
    block_size = 12
    for i in range(0, len(returns_arr) - block_size + 1, block_size):
        block = returns_arr[i:i + block_size]
        m = compute_performance_metrics(block)
        window_sharpes.append(m["sharpe_ratio"])

    return {
        "months": all_months,
        "returns": returns_arr,
        "crowding": np.array(all_crowding),
        "block_sharpes": np.array(window_sharpes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-assets", type=int, default=50)
    parser.add_argument("--n-months", type=int, default=240)
    parser.add_argument("--window-size", type=int, default=60, help="Months of 'history' before walk-forward evaluation begins")
    parser.add_argument("--step-size", type=int, default=1, help="Months advanced per walk-forward step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("SYNTHETIC DATA DISCLAIMER: this walk-forward validation uses")
    print("synthetic, not real, market data. See _backtest_core.py.")
    print("=" * 70)

    config = BacktestConfig(n_assets=args.n_assets, n_months=args.n_months)
    print(f"\n==> Generating synthetic panel ({args.n_assets} assets x {args.n_months} months, seed={args.seed})...")
    panel = generate_synthetic_panel(config, seed=args.seed)

    print(f"==> Walk-forward: window_size={args.window_size} months, step_size={args.step_size} month(s)")
    n_steps = (args.n_months - args.window_size) // args.step_size
    print(f"    -> {n_steps} out-of-sample evaluation steps")

    result = walk_forward_backtest(panel, config, args.window_size, args.step_size, args.n_months)
    metrics = compute_performance_metrics(result["returns"])

    print(f"\n{'Metric':<30} {'Value':>12}")
    print("-" * 43)
    print(f"{'Out-of-sample Sharpe':<30} {metrics['sharpe_ratio']:>12.3f}")
    print(f"{'Annualised Return':<30} {metrics['annualised_return']*100:>11.2f}%")
    print(f"{'Annualised Volatility':<30} {metrics['annualised_vol']*100:>11.2f}%")
    print(f"{'Maximum Drawdown':<30} {metrics['max_drawdown']*100:>11.2f}%")
    print(f"{'Mean Crowding Measure':<30} {result['crowding'].mean():>12.4f}")

    if len(result["block_sharpes"]) > 0:
        print(f"\n==> Stability check: Sharpe ratio across {len(result['block_sharpes'])} "
              f"non-overlapping 12-month blocks:")
        print(f"    Mean:   {np.mean(result['block_sharpes']):.3f}")
        print(f"    Std:    {np.std(result['block_sharpes']):.3f}")
        print(f"    Min:    {np.min(result['block_sharpes']):.3f}")
        print(f"    Max:    {np.max(result['block_sharpes']):.3f}")
        n_positive = int(np.sum(result["block_sharpes"] > 0))
        print(f"    Blocks with positive Sharpe: {n_positive}/{len(result['block_sharpes'])}")

    output_path = (
        Path(args.output) if args.output
        else Path(__file__).resolve().parent / "walk_forward_results.csv"
    )
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "return", "crowding"])
        for t, r, c in zip(result["months"], result["returns"], result["crowding"]):
            writer.writerow([t, r, c])

    print(f"\n==> Walk-forward returns written to: {output_path}")


if __name__ == "__main__":
    main()
