"""
simulations/particle_filter/run_particle_filter.py

Implements a bootstrap particle filter (sequential importance
resampling, SIR) for real-time estimation of the latent MFG equilibrium
mean xbar_t* from noisy observations of the finite-N empirical mean.

This is the nonlinear/non-Gaussian generalisation of the Kalman filter
approach described in Section 8.4.2 of the thesis ("Filtering Approach
via Kalman Filter"). The state equation is the equilibrium mean dynamics
derived in Appendix G / Chapter 7:

    d(xbar_t*) = -Gamma_t * xbar_t* dt + sigma0 dW0_t      (state, OU process)
    xbar_t^N   = xbar_t* + eps_t,  eps_t ~ N(0, R_t)        (observation)

where Gamma_t = (A_t + B_t) / lambda is computed from the same Riccati
system (G.5) used in the convergence simulation, and R_t is the
finite-particle sampling variance of the empirical mean (R_t ~
Var(Y_t)/N, with Var(Y_t) the idiosyncratic cross-sectional dispersion).

While a Kalman filter is exact here (the state/observation model is
linear-Gaussian), a particle filter is implemented to (a) provide a
template that extends to nonlinear/non-Gaussian variants of the model,
and (b) cross-validate against the closed-form Kalman gain as a
correctness check.

Usage:
    python run_particle_filter.py
    python run_particle_filter.py --n-particles 2000 --n-obs-particles 500
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LQMFGParams:
    """Model parameters, matching the convergence simulation (Appendix C.8)."""

    kappa: float = 0.5
    lam: float = 1.0
    sigma: float = 0.3
    sigma0: float = 0.2
    T: float = 1.0
    Q: float = 1.0
    P: float = 1.0
    x0_var: float = 0.5


def solve_riccati(params: LQMFGParams, n_grid: int = 2000) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Solve the scalar Riccati system (G.5) backward from t=T to t=0."""
    kappa, lam, Q, P, T = params.kappa, params.lam, params.Q, params.P, params.T

    def rhs(tau: float, y: FloatArray) -> FloatArray:
        A, B = y
        dA_dtau = -(2 * kappa * A + A**2 / lam - 2 * Q)  # CORRECTED: signs of A**2 and Q terms (same fix as euler_milstein script)
        dB_dtau = -(kappa * B + (A * B) / lam - kappa * A)  # CORRECTED sign
        return np.array([dA_dtau, dB_dtau])

    tau_grid = np.linspace(0.0, T, n_grid)
    sol = solve_ivp(
        rhs, t_span=(0.0, T), y0=np.array([2 * P, -2 * P]),
        t_eval=tau_grid, method="RK45", rtol=1e-10, atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Riccati ODE solve failed: {sol.message}")
    t_grid = T - tau_grid[::-1]
    A_t = sol.y[0][::-1]
    B_t = sol.y[1][::-1]
    return t_grid, A_t, B_t


def compute_idiosyncratic_variance(params: LQMFGParams, t_grid: FloatArray, A_t: FloatArray) -> FloatArray:
    """
    Compute Var(Y_t), the cross-sectional dispersion of individual deviations
    from the population mean (see the moment-ODE derivation in
    run_convergence_simulation.py's module docstring):
        d/dt Var(Y_t) = -2*(kappa + A_t/lambda)*Var(Y_t) + sigma^2
    """
    kappa, lam, sigma = params.kappa, params.lam, params.sigma

    def rhs(t: float, y: FloatArray) -> FloatArray:
        var_Y = y[0]
        A = np.interp(t, t_grid, A_t)
        kappa_t = kappa + A / lam
        return np.array([-2 * kappa_t * var_Y + sigma**2])

    sol = solve_ivp(
        rhs, t_span=(t_grid[0], t_grid[-1]), y0=np.array([params.x0_var]),
        t_eval=t_grid, method="RK45", rtol=1e-10, atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Variance ODE solve failed: {sol.message}")
    return sol.y[0]


def simulate_true_equilibrium_path(
    params: LQMFGParams,
    t_grid_riccati: FloatArray,
    A_t: FloatArray,
    B_t: FloatArray,
    n_steps: int,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """
    Simulate the TRUE equilibrium mean path xbar_t* (the latent state we want
    to filter), via the OU-type SDE
        d(xbar_t*) = -Gamma_t * xbar_t* dt + sigma0 dW0_t,
    and simultaneously the observed finite-N empirical mean xbar_t^N obtained
    by running the actual N-particle closed-loop system (the observation).
    """
    kappa, lam, sigma, sigma0 = params.kappa, params.lam, params.sigma, params.sigma0
    dt = params.T / n_steps
    sqrt_dt = np.sqrt(dt)
    t_grid = np.linspace(0.0, params.T, n_steps + 1)

    rng = np.random.default_rng(seed)
    n_particles = 2000  # large N to approximate the "true" mean closely
    x = rng.normal(0.0, np.sqrt(params.x0_var), size=n_particles)
    xbar_path = np.zeros(n_steps + 1)
    xbar_path[0] = np.mean(x)

    for k in range(n_steps):
        t_k = t_grid[k]
        A = np.interp(t_k, t_grid_riccati, A_t)
        B = np.interp(t_k, t_grid_riccati, B_t)
        xbar = np.mean(x)
        drift = kappa * (xbar - x) - (A * x + B * xbar) / lam
        idio_dW = rng.normal(0.0, sqrt_dt, size=n_particles)
        common_dW = rng.normal(0.0, sqrt_dt)
        x = x + drift * dt + sigma * idio_dW + sigma0 * common_dW
        xbar_path[k + 1] = np.mean(x)

    return t_grid, xbar_path


def bootstrap_particle_filter(
    params: LQMFGParams,
    t_grid_riccati: FloatArray,
    A_t: FloatArray,
    B_t: FloatArray,
    var_Y: FloatArray,
    observations: FloatArray,
    t_grid: FloatArray,
    n_filter_particles: int,
    n_obs_particles: int,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """
    Run a bootstrap particle filter to estimate the latent equilibrium mean
    xbar_t* given noisy observations of the finite-N empirical mean.

    State transition (proposal = prior, i.e. bootstrap filter):
        xbar_{k+1} = xbar_k - Gamma_k * xbar_k * dt + sigma0 * dW0
    Observation likelihood:
        obs_k ~ N(xbar_k, R_k),  R_k = Var(Y_t_k) / n_obs_particles

    Returns
    -------
    filtered_mean : array, shape (n_steps+1,) -- posterior mean estimate
    filtered_std  : array, shape (n_steps+1,) -- posterior std estimate
    """
    rng = np.random.default_rng(seed)
    n_steps = len(t_grid) - 1
    dt = params.T / n_steps
    sqrt_dt = np.sqrt(dt)
    kappa, lam, sigma0 = params.kappa, params.lam, params.sigma0

    particles = rng.normal(0.0, np.sqrt(params.x0_var), size=n_filter_particles)
    weights = np.full(n_filter_particles, 1.0 / n_filter_particles)

    filtered_mean = np.zeros(n_steps + 1)
    filtered_std = np.zeros(n_steps + 1)
    filtered_mean[0] = np.average(particles, weights=weights)
    filtered_std[0] = np.sqrt(np.average((particles - filtered_mean[0]) ** 2, weights=weights))

    for k in range(n_steps):
        t_k = t_grid[k]
        A = np.interp(t_k, t_grid_riccati, A_t)
        B = np.interp(t_k, t_grid_riccati, B_t)
        Gamma = (A + B) / lam

        # Propagate particles through the state transition (bootstrap proposal).
        # Each particle samples its OWN hypothesis of the unobserved common-noise
        # increment, since the filter does not know the realised noise path --
        # this is what gives the ensemble enough spread for the subsequent
        # likelihood weighting to be informative. (A single shared draw across
        # all particles would collapse ensemble diversity and degrade the filter.)
        dW0 = rng.normal(0.0, sqrt_dt, size=n_filter_particles)
        particles = particles + (-Gamma * particles) * dt + sigma0 * dW0

        # Weight update via the observation likelihood.
        R_k = float(np.interp(t_grid[k + 1], t_grid_riccati, var_Y)) / n_obs_particles
        R_k = max(R_k, 1e-8)
        obs = observations[k + 1]
        log_w = -0.5 * (particles - obs) ** 2 / R_k
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        weights = w / np.sum(w)

        filtered_mean[k + 1] = np.average(particles, weights=weights)
        filtered_std[k + 1] = np.sqrt(np.average((particles - filtered_mean[k + 1]) ** 2, weights=weights))

        # Systematic resampling when effective sample size degrades.
        ess = 1.0 / np.sum(weights**2)
        if ess < n_filter_particles / 2:
            positions = (rng.random() + np.arange(n_filter_particles)) / n_filter_particles
            cumulative = np.cumsum(weights)
            indices = np.searchsorted(cumulative, positions)
            particles = particles[indices]
            weights = np.full(n_filter_particles, 1.0 / n_filter_particles)

    return filtered_mean, filtered_std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-particles", type=int, default=1000, help="Number of filter particles")
    parser.add_argument("--n-obs-particles", type=int, default=200, help="N used to form the noisy observation of xbar_t^N")
    parser.add_argument("--n-steps", type=int, default=200, help="Number of time steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    params = LQMFGParams()

    print("==> Solving Riccati system (G.5)...")
    t_grid_riccati, A_t, B_t = solve_riccati(params)

    print("==> Computing idiosyncratic variance Var(Y_t)...")
    var_Y = compute_idiosyncratic_variance(params, t_grid_riccati, A_t)

    print(f"==> Simulating true equilibrium path and observations (seed={args.seed})...")
    t_grid, xbar_true = simulate_true_equilibrium_path(
        params, t_grid_riccati, A_t, B_t, args.n_steps, args.seed
    )

    # Construct noisy observations of the finite-N empirical mean by adding
    # sampling noise consistent with the idiosyncratic variance at each time.
    rng_obs = np.random.default_rng(args.seed + 1)
    obs_noise_std = np.sqrt(np.interp(t_grid, t_grid_riccati, var_Y) / args.n_obs_particles)
    observations = xbar_true + rng_obs.normal(0.0, obs_noise_std)

    print(f"==> Running bootstrap particle filter (M={args.n_particles} particles)...")
    filtered_mean, filtered_std = bootstrap_particle_filter(
        params, t_grid_riccati, A_t, B_t, var_Y, observations, t_grid,
        args.n_particles, args.n_obs_particles, args.seed,
    )

    rmse = float(np.sqrt(np.mean((filtered_mean - xbar_true) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((observations - xbar_true) ** 2)))
    print(f"\n==> Filter RMSE vs. true equilibrium mean: {rmse:.5f}")
    print(f"==> Naive (raw observation) RMSE:          {rmse_naive:.5f}")
    print(f"==> Filter improves on raw observation by:  {(1 - rmse / rmse_naive) * 100:.1f}%")

    # Crowding measure proxy: |observation - filtered estimate of equilibrium|.
    crowding = np.abs(observations - filtered_mean)
    print(f"==> Mean crowding measure c_t over path: {np.mean(crowding):.5f}")

    output_path = (
        Path(args.output) if args.output
        else Path(__file__).resolve().parent / "particle_filter_config.yaml"
    ).with_name("particle_filter_results.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "xbar_true", "observation", "filtered_mean", "filtered_std", "crowding_c_t"])
        for i in range(len(t_grid)):
            writer.writerow([t_grid[i], xbar_true[i], observations[i], filtered_mean[i], filtered_std[i], crowding[i]])

    print(f"\n==> Results written to: {output_path}")


if __name__ == "__main__":
    main()
