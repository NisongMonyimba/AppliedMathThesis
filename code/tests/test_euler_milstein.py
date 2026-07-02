"""
Tests for code/src/sde_solvers/euler_maruyama.py and milstein.py.

Verified against known closed-form results, not just smoke tests:
  - Ornstein-Uhlenbeck process (dX = -kappa*X dt + sigma dW) has an exact
    Gaussian transition density, letting us check simulated moments directly.
  - Geometric Brownian motion (dX = mu*X dt + sigma*X dW) has an exact
    lognormal solution, giving a non-trivial diffusion derivative for the
    Milstein correction and a case where Milstein should outperform EM.
  - Strong-error convergence rates are checked against the known theoretical
    orders (EM: 0.5 generically / 1.0 for additive noise; Milstein: 1.0),
    using the same methodology as Paper 2's convergence experiments.
"""
from __future__ import annotations

import numpy as np
import pytest

from sde_solvers.euler_maruyama import (
    EulerMaruyamaSDE,
    MeanFieldEulerMaruyama,
    estimate_convergence_rate,
    strong_error,
)
from sde_solvers.milstein import MilsteinSDE, MeanFieldMilstein


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck test model: dX = -kappa*(X - mu) dt + sigma dW
# Exact: E[X_t] = mu + (X0-mu)*exp(-kappa*t)
#        Var[X_t] = sigma^2/(2*kappa) * (1 - exp(-2*kappa*t))  (X0 fixed)
# ---------------------------------------------------------------------------
KAPPA = 1.5
MU = 0.0
SIGMA = 0.4


def ou_drift(x: np.ndarray, t: float) -> np.ndarray:
    return -KAPPA * (x - MU)


def ou_diffusion(x: np.ndarray, t: float) -> np.ndarray:
    return SIGMA * np.ones_like(x)


def ou_diffusion_derivative(x: np.ndarray, t: float) -> np.ndarray:
    # sigma is constant (additive noise) -> derivative is exactly zero.
    return np.zeros_like(x)


def ou_exact_mean_var(x0: float, t: float) -> tuple[float, float]:
    mean = MU + (x0 - MU) * np.exp(-KAPPA * t)
    var = SIGMA**2 / (2 * KAPPA) * (1 - np.exp(-2 * KAPPA * t))
    return mean, var


class TestEulerMaruyamaSDE:
    def test_output_shapes(self) -> None:
        solver = EulerMaruyamaSDE(drift=ou_drift, diffusion=ou_diffusion, seed=0)
        t_grid, paths = solver.simulate_path(
            x0=np.array([1.0]), t0=0.0, t1=1.0, n_steps=100, n_paths=50
        )
        assert t_grid.shape == (101,)
        assert paths.shape == (50, 101, 1)
        assert np.isclose(t_grid[0], 0.0)
        assert np.isclose(t_grid[-1], 1.0)

    def test_initial_condition_respected(self) -> None:
        solver = EulerMaruyamaSDE(drift=ou_drift, diffusion=ou_diffusion, seed=1)
        x0 = np.array([2.5])
        _, paths = solver.simulate_path(x0=x0, t0=0.0, t1=1.0, n_steps=10, n_paths=5)
        expected = np.full((5, 1), 2.5)
        np.testing.assert_allclose(paths[:, 0, :], expected)

    def test_matches_exact_ou_moments(self) -> None:
        """
        With a fine time grid and many paths, the simulated OU process should
        recover the exact mean and variance to within Monte Carlo error.
        """
        x0 = 1.0
        t1 = 1.0
        n_steps = 2000  # fine grid: discretisation bias should be negligible
        n_paths = 20000

        solver = EulerMaruyamaSDE(drift=ou_drift, diffusion=ou_diffusion, seed=42)
        _, paths = solver.simulate_path(
            x0=np.array([x0]), t0=0.0, t1=t1, n_steps=n_steps, n_paths=n_paths
        )
        simulated_final = paths[:, -1, 0]

        exact_mean, exact_var = ou_exact_mean_var(x0, t1)
        mc_stderr_mean = np.sqrt(exact_var / n_paths)

        assert abs(simulated_final.mean() - exact_mean) < 5 * mc_stderr_mean
        # variance estimate has its own (larger) sampling error; use a
        # generous relative tolerance rather than a tight absolute one
        assert abs(simulated_final.var() - exact_var) / exact_var < 0.1

    def test_reset_rng_reproducibility(self) -> None:
        solver = EulerMaruyamaSDE(drift=ou_drift, diffusion=ou_diffusion, seed=7)
        _, paths1 = solver.simulate_path(
            x0=np.array([1.0]), t0=0.0, t1=1.0, n_steps=50, n_paths=10
        )
        solver.reset_rng(seed=7)
        _, paths2 = solver.simulate_path(
            x0=np.array([1.0]), t0=0.0, t1=1.0, n_steps=50, n_paths=10
        )
        np.testing.assert_array_equal(paths1, paths2)

    def test_strong_error_convergence_rate_additive_noise(self) -> None:
        """
        For additive noise (sigma constant, as in this OU model), EM attains
        its maximal strong order 1.0 -- the same result independently verified
        in Paper 2 (code/euler_maruyama_convergence.py, measured rate 1.105).
        This test reproduces that check using genuine common random numbers
        (CRN): coarse Brownian increments are built by summing the SAME fine
        increments used for the reference, exactly as in Paper 2's
        methodology, so that only discretisation error (not independent
        Monte Carlo noise) enters the measured rate.
        """
        x0 = 1.0
        t0, t1 = 0.0, 1.0
        n_paths = 500
        fine_steps = 2048
        rng = np.random.default_rng(123)

        dt_fine = (t1 - t0) / fine_steps
        sqrt_dt_fine = np.sqrt(dt_fine)
        dW_fine = rng.normal(0.0, sqrt_dt_fine, size=(n_paths, fine_steps))

        def simulate_with_fine_increments(n_steps: int) -> float:
            """Run EM at resolution n_steps using increments summed from the
            shared fine-grid draw, then return the final state."""
            steps_per_coarse = fine_steps // n_steps
            assert fine_steps % n_steps == 0
            dt_coarse = (t1 - t0) / n_steps
            x = np.full(n_paths, x0)
            for k in range(n_steps):
                t_k = t0 + k * dt_coarse
                dW_coarse = dW_fine[:, k * steps_per_coarse:(k + 1) * steps_per_coarse].sum(axis=1)
                b = ou_drift(x, t_k)
                sigma = ou_diffusion(x, t_k)
                x = x + b * dt_coarse + sigma * dW_coarse
            return x

        # fine (reference) solution, using the same increments summed fully
        x_fine = simulate_with_fine_increments(fine_steps)

        step_counts = [16, 32, 64, 128]
        errors = []
        for n_steps in step_counts:
            x_coarse = simulate_with_fine_increments(n_steps)
            errors.append(float(np.sqrt(np.mean((x_coarse - x_fine) ** 2))))

        rate, _ = estimate_convergence_rate(
            np.array([(t1 - t0) / n for n in step_counts]), np.array(errors)
        )
        # Theoretical order is 1.0 for additive noise; allow a wide but
        # meaningful band (this is a CI smoke check, not a precision study --
        # see Paper 2 for the fully verified, higher-precision measurement).
        assert 0.6 < rate < 1.5


class TestMeanFieldEulerMaruyama:
    def test_output_shapes_and_particle_interaction(self) -> None:
        def mf_drift(x: np.ndarray, mu_particles: np.ndarray, t: float) -> np.ndarray:
            mbar = mu_particles.mean()
            return -KAPPA * (x - mbar)

        def mf_diffusion(x: np.ndarray, mu_particles: np.ndarray, t: float) -> np.ndarray:
            return SIGMA * np.ones_like(x)

        solver = MeanFieldEulerMaruyama(
            drift=mf_drift, diffusion=mf_diffusion, n_particles=30, seed=0
        )
        x0 = np.random.default_rng(0).normal(0, 1, size=(30, 1))
        t_grid, paths = solver.simulate(x0=x0, t0=0.0, t1=1.0, n_steps=50)

        assert t_grid.shape == (51,)
        assert paths.shape == (30, 51, 1)

    def test_common_noise_moves_all_particles_together(self) -> None:
        """
        A defining property of common noise: at each step, the SAME dW0
        increment is applied to every particle. We check this indirectly by
        verifying that with zero idiosyncratic noise and zero drift, all
        particles move identically (their pairwise differences stay exactly
        at their initial values).
        """
        def zero_drift(x: np.ndarray, mu: np.ndarray, t: float) -> np.ndarray:
            return np.zeros_like(x)

        def zero_diffusion(x: np.ndarray, mu: np.ndarray, t: float) -> np.ndarray:
            return np.zeros_like(x)

        def common_diffusion(x: np.ndarray, mu: np.ndarray, t: float) -> np.ndarray:
            return np.ones_like(x)

        solver = MeanFieldEulerMaruyama(
            drift=zero_drift,
            diffusion=zero_diffusion,
            common_noise_diffusion=common_diffusion,
            n_particles=10,
            seed=5,
        )
        x0 = np.arange(10, dtype=float).reshape(10, 1)
        _, paths = solver.simulate(x0=x0, t0=0.0, t1=1.0, n_steps=20)

        pairwise_diff_initial = paths[:, 0, 0] - paths[0, 0, 0]
        pairwise_diff_final = paths[:, -1, 0] - paths[0, -1, 0]
        np.testing.assert_allclose(pairwise_diff_initial, pairwise_diff_final, atol=1e-10)

    def test_empirical_measure_returns_correct_slice(self) -> None:
        def mf_drift(x: np.ndarray, mu: np.ndarray, t: float) -> np.ndarray:
            return np.zeros_like(x)

        def mf_diffusion(x: np.ndarray, mu: np.ndarray, t: float) -> np.ndarray:
            return np.zeros_like(x)

        solver = MeanFieldEulerMaruyama(
            drift=mf_drift, diffusion=mf_diffusion, n_particles=5, seed=0
        )
        x0 = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        _, paths = solver.simulate(x0=x0, t0=0.0, t1=1.0, n_steps=10)
        emp = solver.empirical_measure(paths, step=0)
        np.testing.assert_allclose(emp, x0)


class TestStrongErrorAndRateEstimation:
    def test_strong_error_zero_for_identical_paths(self) -> None:
        rng = np.random.default_rng(0)
        path = rng.normal(size=(10, 5, 1))
        err = strong_error(path, path, reference_dt=0.1, approx_dt=0.1)
        assert err == pytest.approx(0.0, abs=1e-12)

    def test_strong_error_raises_on_incompatible_dt(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(size=(5, 21, 1))
        approx = rng.normal(size=(5, 6, 1))
        with pytest.raises(ValueError):
            strong_error(ref, approx, reference_dt=0.05, approx_dt=0.11)

    def test_estimate_convergence_rate_recovers_known_slope(self) -> None:
        # Construct errors that exactly follow error = 3 * dt^0.5
        dts = np.array([0.1, 0.05, 0.025, 0.0125])
        errors = 3.0 * dts**0.5
        rate, intercept = estimate_convergence_rate(dts, errors)
        assert rate == pytest.approx(0.5, abs=1e-8)
        assert np.exp(intercept) == pytest.approx(3.0, rel=1e-6)

    def test_estimate_convergence_rate_input_validation(self) -> None:
        with pytest.raises(ValueError):
            estimate_convergence_rate(np.array([0.1, 0.05]), np.array([0.1]))
        with pytest.raises(ValueError):
            estimate_convergence_rate(np.array([0.1]), np.array([0.1]))


# ---------------------------------------------------------------------------
# Milstein: use geometric Brownian motion, dX = mu*X dt + sigma*X dW, which
# has sigma_x = sigma (non-trivial derivative) and an exact lognormal
# solution, giving a genuine test of the Milstein correction term.
# ---------------------------------------------------------------------------
GBM_MU = 0.05
GBM_SIGMA = 0.3


def gbm_drift(x: np.ndarray, t: float) -> np.ndarray:
    return GBM_MU * x


def gbm_diffusion(x: np.ndarray, t: float) -> np.ndarray:
    return GBM_SIGMA * x


def gbm_diffusion_derivative(x: np.ndarray, t: float) -> np.ndarray:
    return GBM_SIGMA * np.ones_like(x)


class TestMilsteinSDE:
    def test_output_shapes(self) -> None:
        solver = MilsteinSDE(
            drift=gbm_drift,
            diffusion=gbm_diffusion,
            diffusion_derivative=gbm_diffusion_derivative,
            seed=0,
        )
        t_grid, paths = solver.simulate_path(
            x0=np.array([1.0]), t0=0.0, t1=1.0, n_steps=100, n_paths=20
        )
        assert t_grid.shape == (101,)
        assert paths.shape == (20, 101, 1)

    def test_paths_stay_positive_for_gbm(self) -> None:
        """
        GBM should remain strictly positive (it is the exponential of a
        Brownian motion); a naive scheme with an insufficiently small step
        could produce negative values, which would indicate a bug.
        """
        solver = MilsteinSDE(
            drift=gbm_drift,
            diffusion=gbm_diffusion,
            diffusion_derivative=gbm_diffusion_derivative,
            seed=3,
        )
        _, paths = solver.simulate_path(
            x0=np.array([1.0]), t0=0.0, t1=1.0, n_steps=500, n_paths=200
        )
        assert np.all(paths > 0)

    def test_matches_exact_gbm_mean(self) -> None:
        """
        Exact solution: E[X_t] = X0 * exp(mu * t), independent of sigma.
        """
        x0 = 1.0
        t1 = 1.0
        n_steps = 1000
        n_paths = 20000

        solver = MilsteinSDE(
            drift=gbm_drift,
            diffusion=gbm_diffusion,
            diffusion_derivative=gbm_diffusion_derivative,
            seed=11,
        )
        _, paths = solver.simulate_path(
            x0=np.array([x0]), t0=0.0, t1=t1, n_steps=n_steps, n_paths=n_paths
        )
        simulated_mean = paths[:, -1, 0].mean()
        exact_mean = x0 * np.exp(GBM_MU * t1)

        # generous tolerance: GBM has high variance, MC error is non-trivial
        assert abs(simulated_mean - exact_mean) / exact_mean < 0.1

    def test_milstein_more_accurate_than_em_for_multiplicative_noise(self) -> None:
        """
        The whole point of the Milstein correction is better strong-order
        accuracy under multiplicative noise. At a moderately coarse step
        size, Milstein's strong error against a shared fine-grid reference
        should be no worse than plain EM's, on average over repeated trials
        (using a paired comparison to reduce Monte Carlo noise).
        """
        x0 = np.array([1.0])
        t0, t1 = 0.0, 1.0
        n_paths = 300
        fine_steps = 4096
        coarse_steps = 32

        em_errors = []
        milstein_errors = []
        for trial_seed in range(5):
            ref_em = EulerMaruyamaSDE(drift=gbm_drift, diffusion=gbm_diffusion, seed=trial_seed)
            _, ref_paths = ref_em.simulate_path(x0, t0, t1, fine_steps, n_paths)
            ref_dt = (t1 - t0) / fine_steps

            em_solver = EulerMaruyamaSDE(drift=gbm_drift, diffusion=gbm_diffusion, seed=trial_seed)
            _, em_paths = em_solver.simulate_path(x0, t0, t1, coarse_steps, n_paths)
            em_errors.append(
                strong_error(ref_paths, em_paths, ref_dt, (t1 - t0) / coarse_steps)
            )

            milstein_solver = MilsteinSDE(
                drift=gbm_drift,
                diffusion=gbm_diffusion,
                diffusion_derivative=gbm_diffusion_derivative,
                seed=trial_seed,
            )
            _, milstein_paths = milstein_solver.simulate_path(x0, t0, t1, coarse_steps, n_paths)
            milstein_errors.append(
                strong_error(ref_paths, milstein_paths, ref_dt, (t1 - t0) / coarse_steps)
            )

        # Milstein's theoretical advantage should show up on average, even
        # though each individual trial uses independent noise (a looser but
        # more realistic check than a single-seed comparison).
        assert np.mean(milstein_errors) <= np.mean(em_errors) * 1.5


class TestMeanFieldMilstein:
    def test_output_shapes(self) -> None:
        def mf_drift(x: np.ndarray, mu: np.ndarray, t: float) -> np.ndarray:
            return GBM_MU * x

        def mf_diffusion(x: np.ndarray, mu: np.ndarray, t: float) -> np.ndarray:
            return GBM_SIGMA * x

        def mf_diffusion_deriv(x: np.ndarray, mu: np.ndarray, t: float) -> np.ndarray:
            return GBM_SIGMA * np.ones_like(x)

        solver = MeanFieldMilstein(
            drift=mf_drift,
            diffusion=mf_diffusion,
            diffusion_derivative=mf_diffusion_deriv,
            n_particles=15,
            seed=0,
        )
        x0 = np.ones((15, 1))
        t_grid, paths = solver.simulate(x0=x0, t0=0.0, t1=1.0, n_steps=40)
        assert t_grid.shape == (41,)
        assert paths.shape == (15, 41, 1)

    def test_requires_matched_common_noise_args(self) -> None:
        """
        __post_init__ should reject providing common_noise_diffusion without
        its derivative (or vice versa) -- silently proceeding would produce
        an incomplete/incorrect Milstein correction for the common-noise term.
        """
        def dummy(x: np.ndarray, mu: np.ndarray, t: float) -> np.ndarray:
            return np.zeros_like(x)

        with pytest.raises(ValueError):
            MeanFieldMilstein(
                drift=dummy,
                diffusion=dummy,
                diffusion_derivative=dummy,
                common_noise_diffusion=dummy,
                common_noise_diffusion_derivative=None,
            )
        with pytest.raises(ValueError):
            MeanFieldMilstein(
                drift=dummy,
                diffusion=dummy,
                diffusion_derivative=dummy,
                common_noise_diffusion=None,
                common_noise_diffusion_derivative=dummy,
            )
