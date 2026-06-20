"""
Milstein scheme for SDE simulation.

Implements:
    - MilsteinSDE: generic scalar/vector SDE solver with the Milstein
      correction term, giving strong order 1.0 (vs. 0.5 for Euler-Maruyama)
      when the diffusion coefficient's derivative is available.
    - MeanFieldMilstein: McKean-Vlasov / MFG-specific subclass.

Both reuse strong_error() and estimate_convergence_rate() from
euler_maruyama.py so Chapter 6's Euler-vs-Milstein comparison uses an
identical error metric and identical rate-fitting procedure.

Note on the Milstein correction:
    For diagonal/scalar noise, the correction term is:
        0.5 * sigma(X_t, t) * sigma_x(X_t, t) * (dW_t^2 - dt)
    where sigma_x is the partial derivative of sigma with respect to X.
    For the general multi-dimensional case with non-commuting noise, the
    Levy area terms are required; this implementation assumes commutative
    noise (diagonal or scalar diffusion), which holds for the thesis's
    1-D state / 1-D noise setting. This assumption is documented in
    Chapter 6 and should be revisited if extended to non-commutative
    multi-dimensional noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from euler_maruyama import (  # noqa: F401  (re-exported for convenience)
    DiffusionFn,
    DriftFn,
    FloatArray,
    MeanFieldDiffusionFn,
    MeanFieldDriftFn,
    estimate_convergence_rate,
    strong_error,
)


@dataclass
class MilsteinSDE:
    """
    Generic Milstein solver for dX_t = b(X_t, t) dt + sigma(X_t, t) dW_t,
    assuming commutative (diagonal/scalar) noise.

    Parameters
    ----------
    drift : Callable[[X, t], b(X, t)]
    diffusion : Callable[[X, t], sigma(X, t)]
    diffusion_derivative : Callable[[X, t], d(sigma)/dX]
        Analytic or numerically-differentiated derivative of the diffusion
        coefficient with respect to the state. Required for the Milstein
        correction term.
    dim : int
    seed : Optional[int]
    """

    drift: DriftFn
    diffusion: DiffusionFn
    diffusion_derivative: DiffusionFn
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
        Simulate paths on a uniform grid over [t0, t1] using the Milstein
        scheme.

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
            sigma_x = self.diffusion_derivative(x_k, t_k)

            milstein_correction = 0.5 * sigma * sigma_x * (dW**2 - dt)
            paths[:, k + 1, :] = x_k + b * dt + sigma * dW + milstein_correction

        return t_grid, paths

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """Reset the internal RNG, e.g. to reproduce a specific Brownian path."""
        self._rng = np.random.default_rng(seed if seed is not None else self.seed)


@dataclass
class MeanFieldMilstein:
    """
    Milstein solver for the N-particle McKean-Vlasov / MFG system with
    common noise, assuming commutative (diagonal/scalar) idiosyncratic and
    common-noise diffusion terms.

    Parameters mirror MeanFieldEulerMaruyama in euler_maruyama.py, with the
    addition of diffusion_derivative (and, if common noise is present,
    common_noise_diffusion_derivative) for the Milstein correction.
    """

    drift: MeanFieldDriftFn
    diffusion: MeanFieldDiffusionFn
    diffusion_derivative: MeanFieldDiffusionFn
    common_noise_diffusion: Optional[MeanFieldDiffusionFn] = None
    common_noise_diffusion_derivative: Optional[MeanFieldDiffusionFn] = None
    dim: int = 1
    n_particles: int = 100
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if (self.common_noise_diffusion is not None) != (
            self.common_noise_diffusion_derivative is not None
        ):
            raise ValueError(
                "common_noise_diffusion and common_noise_diffusion_derivative "
                "must both be provided, or both omitted."
            )

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
            sigma_x = self.diffusion_derivative(x_k, x_k, t_k)

            increment = (
                b * dt
                + sigma * idio_dW
                + 0.5 * sigma * sigma_x * (idio_dW**2 - dt)
            )

            if self.common_noise_diffusion is not None:
                dW0 = self._rng.normal(0.0, sqrt_dt, size=(1, self.dim))
                sigma0 = self.common_noise_diffusion(x_k, x_k, t_k)
                sigma0_x = self.common_noise_diffusion_derivative(x_k, x_k, t_k)  # type: ignore[misc]
                increment = increment + sigma0 * dW0 + 0.5 * sigma0 * sigma0_x * (
                    dW0**2 - dt
                )

            paths[:, k + 1, :] = x_k + increment

        return t_grid, paths

    def empirical_measure(self, paths: FloatArray, step: int) -> FloatArray:
        """Return the empirical measure (raw particle states) at a given step."""
        return paths[:, step, :]
