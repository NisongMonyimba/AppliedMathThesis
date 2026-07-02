"""
Strong-error convergence experiment: Euler-Maruyama for the mean-field-factored
LQ-MFG particle system, against the closed-form Riccati solution.
Uses common random numbers (CRN) across resolutions to isolate discretisation
error from Monte Carlo noise.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lq_mfg_common import KAPPA, LAM, C1, C2, SIGMA, SIGMA0, T, make_AB_interpolants

np.random.seed(42)

A_interp, B_interp = make_AB_interpolants()

def simulate_particle_system(N, n_steps, n_trials, dt_fine, fine_steps_per_step):
    """
    Simulate N-particle mean-field-factored LQ-MFG system with Euler-Maruyama
    at the given (coarse) step count, and compare against a reference solution
    obtained from a much finer discretisation using the SAME underlying
    Brownian increments (via Brownian bridge construction / summing fine
    increments), which is the correct way to do a CRN convergence study
    without needing the true continuous path.
    """
    dt_coarse = T / n_steps
    total_fine_steps = n_steps * fine_steps_per_step
    dt_f = T / total_fine_steps

    sqrt_dt_f = np.sqrt(dt_f)

    errors = np.zeros(n_trials)
    for trial in range(n_trials):
        rng = np.random.RandomState(1000 + trial)
        X0 = rng.normal(0, 1, size=N)  # mu_init = N(0,1)

        # fine idiosyncratic increments: (N, total_fine_steps)
        dW_fine = rng.normal(0, sqrt_dt_f, size=(N, total_fine_steps))
        dW0_fine = rng.normal(0, sqrt_dt_f, size=total_fine_steps)

        # === reference (fine) simulation ===
        X_fine = X0.copy()
        for k in range(total_fine_steps):
            t_k = k * dt_f
            mu_bar = X_fine.mean()
            A_t = A_interp(t_k); B_t = B_interp(t_k)
            alpha = -(A_t * X_fine + B_t * mu_bar) / LAM
            drift = KAPPA * (mu_bar - X_fine) + alpha
            X_fine = X_fine + drift * dt_f + SIGMA * dW_fine[:, k] + SIGMA0 * dW0_fine[k]
        X_fine_final = X_fine.copy()

        # === coarse simulation, using SUMMED fine increments (same noise realization) ===
        X_coarse = X0.copy()
        for j in range(n_steps):
            t_j = j * dt_coarse
            idx0 = j * fine_steps_per_step
            idx1 = (j+1) * fine_steps_per_step
            dW_coarse = dW_fine[:, idx0:idx1].sum(axis=1)
            dW0_coarse = dW0_fine[idx0:idx1].sum()
            mu_bar = X_coarse.mean()
            A_t = A_interp(t_j); B_t = B_interp(t_j)
            alpha = -(A_t * X_coarse + B_t * mu_bar) / LAM
            drift = KAPPA * (mu_bar - X_coarse) + alpha
            X_coarse = X_coarse + drift * dt_coarse + SIGMA * dW_coarse + SIGMA0 * dW0_coarse
        X_coarse_final = X_coarse.copy()

        # strong error: mean squared difference across particles
        errors[trial] = np.mean((X_coarse_final - X_fine_final)**2)

    return errors.mean(), errors.std() / np.sqrt(n_trials)

if __name__ == "__main__":
    N = 2000            # particle count, fixed (large enough that MC noise is small)
    n_trials = 50
    fine_steps_per_step = 32   # reference resolution multiplier

    step_counts = [8, 16, 32, 64]
    results = []
    print(f"{'n_steps':>8} {'dt':>10} {'mean_err':>14} {'stderr':>12}")
    for n_steps in step_counts:
        dt = T / n_steps
        mean_err, stderr = simulate_particle_system(N, n_steps, n_trials, dt, fine_steps_per_step)
        results.append((n_steps, dt, mean_err, stderr))
        print(f"{n_steps:>8} {dt:>10.6f} {mean_err:>14.8f} {stderr:>12.2e}")

    # estimate convergence rate via log-log regression
    dts = np.array([r[1] for r in results])
    errs = np.array([r[2] for r in results])
    log_dt = np.log(dts)
    log_err = np.log(errs)
    rate, intercept = np.polyfit(log_dt, log_err, 1)
    print(f"\nEstimated strong convergence rate: {rate:.4f}")

    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'em_convergence.csv'), 'w') as f:
        f.write("n_steps,dt,mean_sq_error,stderr\n")
        for n_steps, dt, err, se in results:
            f.write(f"{n_steps},{dt},{err},{se}\n")
    print("\nSaved to results/em_convergence.csv")
