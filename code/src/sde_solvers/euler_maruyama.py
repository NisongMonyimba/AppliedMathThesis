"""
Euler-Maruyama scheme for SDE simulation.

Implements:
    - EulerMaruyamaSDE: generic scalar/vector SDE solver
        dX_t = b(X_t, t) dt + sigma(X_t, t) dW_t
    - MeanFieldEulerMaruyama: McKean-Vlasov / MFG-specific subclass
        dX_t = b(X_t, mu_t, t) dt + sigma(X_t, mu_t, t) dW_t
      where mu_t is the empirical law of an N-particle system (propagation
      of chaos approximation), and b includes the common-noise coupling.

Supports strong-error convergence-rate estimation against a fine-grid
reference solution (Chapter 6 of the thesis).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
DriftFn = Callable[[FloatArray, float], FloatArray]
DiffusionFn = Callable[[FloatArray, float], FloatArray]
MeanFieldDriftFn = Callable[[FloatArray, FloatArray, float], FloatArray]
MeanFieldDiffusionFn = Callable[[FloatArray, FloatArray, float], FloatArray]


@dataclass
class EulerMaruyamaSDE:
    """
    Generic Euler-Maruyama solver for dX_t = b(X_t, t) dt + sigma(X_t, t) dW_t.

    Parameters
    ----------
    drift : Callable[[X, t], b(X, t)]
        Drift coefficient. Vectorized over the state dimension.
    diffusion : Callable[[X, t], sigma(X, t)]
        Diffusion coefficient. Vectorized over the state dimension.
    dim : int
        State-space dimension.
    seed : Optional[int]
        RNG seed for reproducibility.
    """

    drift: DriftFn
    diffusion: DiffusionFn
    dim: int = 1
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def simulate_path(
        self,
        x0: FloatArray,
        t0: float,
        t1: float,
        n_steps: int,
        n_paths: int = 1,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Simulate paths on a uniform grid over [t0, t1].

        Returns
        -------
        t_grid : array, shape (n_steps + 1,)
        paths  : array, shape (n_paths, n_steps + 1, dim)
        """
        dt = (t1 - t0) / n_steps
        sqrt_dt = np.sqrt(dt)
        t_grid = np.linspace(t0, t1, n_steps + 1)

        paths = np.zeros((n_paths, n_steps + 1, self.dim))
        paths[:, 0, :] = x0

        for k in range(n_steps):
            t_k = t_grid[k]
            x_k = paths[:, k, :]
            dW = self._rng.normal(0.0, sqrt_dt, size=(n_paths, self.dim))
            b = self.drift(x_k, t_k)
            sigma = self.diffusion(x_k, t_k)
            paths[:, k + 1, :] = x_k + b * dt + sigma * dW

        return t_grid.astype(np.float64), paths.astype(np.float64)

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """Reset the internal RNG, e.g. to reproduce a specific Brownian path."""
        self._rng = np.random.default_rng(seed if seed is not None else self.seed)


@dataclass
class MeanFieldEulerMaruyama:
    """
    Euler-Maruyama solver for the N-particle McKean-Vlasov / MFG system with
    common noise:

        dX_i_t = b(X_i_t, mu_t^N, t) dt + sigma(X_i_t, mu_t^N, t) dW_i_t
                 + (common noise term handled via shared dW0 increments)

    mu_t^N is the empirical measure of the N particles at time t, passed to
    drift/diffusion as the raw particle array (caller decides how to reduce
    it, e.g. via code/src/crowding_measure for a Wasserstein-based summary).

    Parameters
    ----------
    drift : Callable[[X, mu_particles, t], b]
        Drift coefficient, takes the particle's own state, the full particle
        array (representing mu_t^N), and time.
    diffusion : Callable[[X, mu_particles, t], sigma]
        Idiosyncratic diffusion coefficient (per-particle noise).
    common_noise_diffusion : Optional[Callable[[X, mu_particles, t], sigma_0]]
        Coefficient multiplying the shared common-noise Brownian increment
        dW0_t. If None, no common noise is applied (i.i.d. case).
    dim : int
        Per-particle state dimension.
    n_particles : int
        Number of particles N used to approximate the mean-field limit.
    seed : Optional[int]
    """

    drift: MeanFieldDriftFn
    diffusion: MeanFieldDiffusionFn
    common_noise_diffusion: Optional[MeanFieldDiffusionFn] = None
    dim: int = 1
    n_particles: int = 100
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def simulate(
        self,
        x0: FloatArray,
        t0: float,
        t1: float,
        n_steps: int,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Simulate the N-particle system on a uniform grid over [t0, t1].

        Parameters
        ----------
        x0 : array, shape (n_particles, dim)
            Initial particle configuration.

        Returns
        -------
        t_grid : array, shape (n_steps + 1,)
        paths  : array, shape (n_particles, n_steps + 1, dim)
        """
        dt = (t1 - t0) / n_steps
        sqrt_dt = np.sqrt(dt)
        t_grid = np.linspace(t0, t1, n_steps + 1)

        paths = np.zeros((self.n_particles, n_steps + 1, self.dim))
        paths[:, 0, :] = x0

        for k in range(n_steps):
            t_k = t_grid[k]
            x_k = paths[:, k, :]

            idio_dW = self._rng.normal(
                0.0, sqrt_dt, size=(self.n_particles, self.dim)
            )

            b = self.drift(x_k, x_k, t_k)
            sigma = self.diffusion(x_k, x_k, t_k)
            increment = b * dt + sigma * idio_dW

            if self.common_noise_diffusion is not None:
                dW0 = self._rng.normal(0.0, sqrt_dt, size=(1, self.dim))
                sigma0 = self.common_noise_diffusion(x_k, x_k, t_k)
                increment = increment + sigma0 * dW0

            paths[:, k + 1, :] = x_k + increment

        return t_grid.astype(np.float64), paths.astype(np.float64)

    def empirical_measure(self, paths: FloatArray, step: int) -> FloatArray:
        """Return the empirical measure (raw particle states) at a given step."""
        return paths[:, step, :]


def strong_error(
    reference_path: FloatArray,
    approx_path: FloatArray,
    reference_dt: float,
    approx_dt: float,
) -> float:
    """
    Compute the strong (L2) error between an approximate path and a
    fine-grid reference path, sub-sampling the reference to the coarser grid.

    Parameters
    ----------
    reference_path : array, shape (n_paths, n_ref_steps + 1, dim)
    approx_path : array, shape (n_paths, n_approx_steps + 1, dim)
    reference_dt, approx_dt : float
        Step sizes used to generate each path. approx_dt must be an integer
        multiple of reference_dt.

    Returns
    -------
    error : float
        E[ |X_T^ref - X_T^approx| ] estimated via Monte Carlo over n_paths.
    """
    ratio = round(approx_dt / reference_dt)
    if not np.isclose(ratio * reference_dt, approx_dt, rtol=1e-6):
        raise ValueError("approx_dt must be an integer multiple of reference_dt")

    subsampled_ref = reference_path[:, ::ratio, :]
    if subsampled_ref.shape[1] != approx_path.shape[1]:
        raise ValueError(
            f"Subsampled reference has {subsampled_ref.shape[1]} steps, "
            f"approx path has {approx_path.shape[1]} steps — grids don't align."
        )

    diff = subsampled_ref[:, -1, :] - approx_path[:, -1, :]
    return float(np.mean(np.linalg.norm(diff, axis=-1)))


def estimate_convergence_rate(
    step_sizes: FloatArray,
    errors: FloatArray,
) -> tuple[float, float]:
    """
    Estimate the strong convergence rate via log-log linear regression:
        log(error) ~ rate * log(dt) + intercept

    Parameters
    ----------
    step_sizes : array of dt values used (must be > 0)
    errors : array of corresponding strong errors (must be > 0)

    Returns
    -------
    rate : float
        Estimated convergence order (slope of the log-log fit).
    intercept : float
        log of the implied error constant (exp(intercept) ~ C in error ~ C * dt^rate).
    """
    if len(step_sizes) != len(errors):
        raise ValueError("step_sizes and errors must have the same length")
    if len(step_sizes) < 2:
        raise ValueError("need at least two (dt, error) pairs to fit a rate")

    log_dt = np.log(step_sizes)
    log_err = np.log(errors)
    rate, intercept = np.polyfit(log_dt, log_err, deg=1)
    return float(rate), float(intercept)
