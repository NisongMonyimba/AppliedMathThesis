"""
simulations/sobol_sensitivity/run_sobol_analysis.py

Sobol global sensitivity analysis of the LQ-MFG equilibrium variance
Var(X_T) with respect to the four model parameters (kappa, lambda,
sigma, sigma0), using the Riccati system (G.5) shared with the other
simulation scripts in this repository.

Quantity of interest (QoI):
    Var(X_T) = Var(Y_T) + Var(xbar_T)
where Var(Y_t) is the idiosyncratic cross-sectional dispersion and
Var(xbar_t) is the variance of the population mean, both obtained by
solving the moment ODEs derived in run_convergence_simulation.py's
module docstring.

Implementation note: SALib is not used here (not available in this
environment); first-order and total-order Sobol indices are instead
computed directly via the standard Saltelli (1999) / Jansen (1999)
Monte Carlo estimators, which require only numpy/scipy. This mirrors
the Sobol sensitivity methodology already used in the ProbOS project
(where Ea_SEI was identified as the dominant factor); here we instead
characterise which LQ-MFG parameter dominates equilibrium dispersion.

Usage:
    python run_sobol_analysis.py
    python run_sobol_analysis.py --n-samples 4096
"""

from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.stats import qmc

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

FloatArray = NDArray[np.float64]

# Parameter names and their [low, high] ranges for the sensitivity study.
# NOTE: the lower bound on lambda is chosen to stay within the region where
# the Riccati system (G.5) is well-posed on [0,T] -- for lambda too small
# relative to kappa and Q, the Riccati solution exhibits a genuine
# finite-time blow-up (a real mathematical phenomenon for backward Riccati
# equations, not a numerical artifact; verified directly by inspecting A_t
# for overflow across a parameter grid before finalising these bounds).
PARAM_NAMES = ["kappa", "lambda", "sigma", "sigma0"]
PARAM_BOUNDS = {
    "kappa": (0.3, 1.0),
    "lambda": (1.1, 2.0),
    "sigma": (0.1, 0.5),
    "sigma0": (0.05, 0.4),
}


@dataclass(frozen=True)
class FixedParams:
    """Parameters held fixed during the sensitivity study."""

    T: float = 1.0
    Q: float = 1.0
    P: float = 1.0
    x0_var: float = 0.5


def solve_riccati(kappa: float, lam: float, Q: float, P: float, T: float, n_grid: int = 400) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Solve the scalar Riccati system (G.5) backward from t=T to t=0."""

    def rhs(tau: float, y: FloatArray) -> FloatArray:
        A, B = y
        dA_dtau = -(2 * kappa * A - A**2 / lam + 2 * Q)
        dB_dtau = -(kappa * B + (A * B) / lam - kappa * A)  # CORRECTED sign
        return np.array([dA_dtau, dB_dtau])

    tau_grid = np.linspace(0.0, T, n_grid)
    sol = solve_ivp(
        rhs, t_span=(0.0, T), y0=np.array([2 * P, -2 * P]),
        t_eval=tau_grid, method="LSODA", rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Riccati ODE solve failed: {sol.message}")
    t_grid = T - tau_grid[::-1]
    A_t = sol.y[0][::-1]
    B_t = sol.y[1][::-1]
    return t_grid, A_t, B_t


def quantity_of_interest(kappa: float, lam: float, sigma: float, sigma0: float, fixed: FixedParams) -> float:
    """
    Compute Var(X_T) = Var(Y_T) + Var(xbar_T) for given LQ-MFG parameters,
    using the moment ODEs:
        d/dt Var(xbar_t) = -2*(A_t+B_t)/lambda * Var(xbar_t) + sigma0^2
        d/dt Var(Y_t)    = -2*(kappa + A_t/lambda) * Var(Y_t) + sigma^2

    Returns NaN (rather than raising) for parameter combinations that cause
    numerical stiffness in either ODE solve, so a handful of difficult
    samples do not crash the entire sensitivity study; these are dropped
    from the Sobol index estimators in compute_sobol_indices().
    """
    try:
        t_grid, A_t, B_t = solve_riccati(kappa, lam, fixed.Q, fixed.P, fixed.T)
    except RuntimeError:
        return float("nan")

    def rhs(t: float, y: FloatArray) -> FloatArray:
        var_xbar, var_Y = y
        A = np.interp(t, t_grid, A_t)
        B = np.interp(t, t_grid, B_t)
        kappa_t = kappa + A / lam
        rate_xbar = (A + B) / lam
        dvar_xbar = -2 * rate_xbar * var_xbar + sigma0**2
        dvar_Y = -2 * kappa_t * var_Y + sigma**2
        return np.array([dvar_xbar, dvar_Y])

    sol = solve_ivp(
        rhs, t_span=(0.0, fixed.T), y0=np.array([0.0, fixed.x0_var]),
        t_eval=[fixed.T], method="RK45", rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        # Some parameter combinations near the boundary of stability can
        # cause stiffness; treat as a (large) penalty value rather than
        # crashing the whole sensitivity study.
        return float("nan")
    var_xbar_T, var_Y_T = sol.y[:, -1]
    return float(var_xbar_T + var_Y_T)


def saltelli_sample(n_base: int, n_params: int, bounds: FloatArray, seed: int) -> tuple[FloatArray, FloatArray, list[FloatArray]]:
    """
    Generate the Saltelli (1999) sampling matrices A, B, and the
    n_params "A_B_i" matrices (A with column i replaced by B's column i),
    using Sobol low-discrepancy sequences via scipy.stats.qmc for the
    base samples (this is the standard, variance-efficient choice, and
    avoids requiring SALib).
    """
    sampler = qmc.Sobol(d=2 * n_params, scramble=True, seed=seed)
    n_pow2 = int(2 ** np.ceil(np.log2(n_base)))
    unit_samples = sampler.random(n_pow2)
    scaled = qmc.scale(unit_samples, np.concatenate([bounds[:, 0]] * 2), np.concatenate([bounds[:, 1]] * 2))

    A = scaled[:, :n_params]
    B = scaled[:, n_params:]

    A_B_list = []
    for i in range(n_params):
        A_B_i = A.copy()
        A_B_i[:, i] = B[:, i]
        A_B_list.append(A_B_i)

    return A, B, A_B_list


def compute_sobol_indices(
    qoi_fn: "callable[[FloatArray], FloatArray]",
    n_base: int,
    bounds: FloatArray,
    seed: int,
) -> tuple[FloatArray, FloatArray, int]:
    """
    Compute first-order (S_i) and total-order (S_Ti) Sobol indices using the
    Jansen (1999) estimators, which are more numerically stable than the
    original Sobol (1993) estimators for moderate sample sizes.
    """
    n_params = bounds.shape[0]
    A, B, A_B_list = saltelli_sample(n_base, n_params, bounds, seed)
    n = A.shape[0]

    f_A = qoi_fn(A)
    f_B = qoi_fn(B)
    f_AB = [qoi_fn(A_B_i) for A_B_i in A_B_list]

    # Drop any NaN rows (from failed ODE solves at extreme parameter combos)
    # consistently across all evaluated arrays.
    valid = np.isfinite(f_A) & np.isfinite(f_B)
    for f_AB_i in f_AB:
        valid &= np.isfinite(f_AB_i)
    n_valid = int(np.sum(valid))
    f_A, f_B = f_A[valid], f_B[valid]
    f_AB = [f_AB_i[valid] for f_AB_i in f_AB]

    var_total = np.var(np.concatenate([f_A, f_B]))

    S_first = np.zeros(n_params)
    S_total = np.zeros(n_params)
    for i in range(n_params):
        # Jansen (1999) first-order estimator
        S_first[i] = 1.0 - np.mean((f_B - f_AB[i]) ** 2) / (2 * var_total)
        # Jansen (1999) total-order estimator
        S_total[i] = np.mean((f_A - f_AB[i]) ** 2) / (2 * var_total)

    return S_first, S_total, n_valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=1024, help="Base Sobol sample size (rounded up to a power of 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    fixed = FixedParams()
    bounds = np.array([PARAM_BOUNDS[name] for name in PARAM_NAMES])

    def qoi_fn(param_matrix: FloatArray) -> FloatArray:
        results = np.empty(param_matrix.shape[0])
        for row_idx in range(param_matrix.shape[0]):
            kappa, lam, sigma, sigma0 = param_matrix[row_idx]
            results[row_idx] = quantity_of_interest(kappa, lam, sigma, sigma0, fixed)
        return results

    print(f"==> Running Sobol sensitivity analysis (base n={args.n_samples}, {len(PARAM_NAMES)} parameters)...")
    print(f"    Quantity of interest: Var(X_T), the equilibrium terminal variance")
    S_first, S_total, n_valid = compute_sobol_indices(qoi_fn, args.n_samples, bounds, args.seed)

    print(f"\n==> Valid evaluations: {n_valid} (after dropping any failed ODE solves)")
    print(f"\n{'Parameter':<10} {'S_first':>10} {'S_total':>10}")
    print("-" * 32)
    for name, s1, st in zip(PARAM_NAMES, S_first, S_total):
        print(f"{name:<10} {s1:>10.4f} {st:>10.4f}")

    dominant_idx = int(np.argmax(S_total))
    print(f"\n==> Dominant factor (highest S_total): {PARAM_NAMES[dominant_idx]} (S_total = {S_total[dominant_idx]:.4f})")
    interaction_strength = np.sum(S_total) - np.sum(S_first)
    print(f"==> Sum(S_total) - Sum(S_first) = {interaction_strength:.4f} (indicates parameter interaction strength)")

    output_path = (
        Path(args.output) if args.output
        else Path(__file__).resolve().parent / "sobol_indices_results.csv"
    )
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "S_first", "S_total"])
        for name, s1, st in zip(PARAM_NAMES, S_first, S_total):
            writer.writerow([name, s1, st])

    print(f"\n==> Results written to: {output_path}")


if __name__ == "__main__":
    main()
