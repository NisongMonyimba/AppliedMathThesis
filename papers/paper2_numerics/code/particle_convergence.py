"""
Particle convergence (propagation of chaos) experiment: empirical measure
convergence W2(mu_T^N, mu_T^{*,W0}) as N -> infinity, for the LQ-MFG benchmark,
where mu_T^{*,W0} is the exact CONDITIONAL Gaussian law given the specific
common-noise path W0 used in each trial (mean mbar_T^{W0} solves a linear SDE
driven by that same W0; variance v_T is deterministic and W0-independent).

CORRECTED (see derivation): the conditional variance ODE does NOT include a
sigma0^2 term -- an earlier draft of this experiment incorrectly included one,
found and fixed by cross-checking against direct large-N Monte Carlo variance,
which disagreed with the (buggy) ODE prediction by nearly a factor of 2.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lq_mfg_common import KAPPA, LAM, C1, C2, SIGMA, SIGMA0, T, make_AB_interpolants

np.random.seed(123)
A_interp, B_interp = make_AB_interpolants()

def exact_variance():
    """Conditional variance v_t solves a deterministic, W0-independent ODE:
       dv/dt = -2(kappa+A_t/lambda) v + sigma^2   (NO sigma0^2 term -- it
       cancels when centering by the stochastic conditional mean)."""
    from scipy.integrate import solve_ivp
    def rhs(t, y):
        v = y[0]
        A_t = float(A_interp(t))
        drift_coef = -(KAPPA + A_t/LAM)
        dv = 2*drift_coef*v + SIGMA**2
        return [dv]
    sol = solve_ivp(rhs, [0, T], [1.0], t_eval=[T], method='Radau', rtol=1e-12, atol=1e-14)
    return sol.y[0, -1]

def simulate_with_conditional_mean(N, n_steps, seed):
    """
    Simulate N particles AND the conditional mean process mbar_t^{W0} using the
    SAME W0 increments, so the reference measure matches the realized common
    noise path exactly (not the unconditional/averaged law).
    """
    rng = np.random.RandomState(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    X = rng.normal(0, 1, size=N)
    mbar_ref = 0.0   # conditional mean process, driven by the SAME dW0 draws
    for k in range(n_steps):
        t_k = k * dt
        mu_bar_emp = X.mean()
        A_t = float(A_interp(t_k)); B_t = float(B_interp(t_k))
        alpha = -(A_t * X + B_t * mu_bar_emp) / LAM
        drift = KAPPA * (mu_bar_emp - X) + alpha
        dW = rng.normal(0, sqrt_dt, size=N)
        dW0 = rng.normal(0, sqrt_dt)
        X = X + drift * dt + SIGMA * dW + SIGMA0 * dW0
        # conditional mean SDE: d(mbar) = -(A+B)/lam * mbar dt + sigma0 dW0
        mbar_ref = mbar_ref + (-(A_t+B_t)/LAM * mbar_ref) * dt + SIGMA0 * dW0
    return X, mbar_ref

def w2_gaussian_vs_empirical_1d(samples, m_star, v_star):
    n = len(samples)
    sorted_samples = np.sort(samples)
    quantile_levels = (np.arange(1, n+1) - 0.5) / n
    from scipy.stats import norm
    gaussian_quantiles = norm.ppf(quantile_levels, loc=m_star, scale=np.sqrt(v_star))
    return np.sqrt(np.mean((sorted_samples - gaussian_quantiles)**2))

if __name__ == "__main__":
    v_star = exact_variance()
    print(f"Exact conditional variance v_T = {v_star:.6f}")

    # sanity check against large-N direct Monte Carlo, using the CORRECT
    # per-trial conditional mean as the centering reference
    X_big, mbar_big = simulate_with_conditional_mean(N=50000, n_steps=512, seed=999)
    print(f"Sanity check (N=50000): sample var around empirical mean = {X_big.var():.6f}, "
          f"conditional-mean-SDE mbar_T = {mbar_big:.6f}, empirical mean = {X_big.mean():.6f}")
    print(f"  (empirical mean should be close to mbar_T for this realized W0 path)")

    n_steps_fine = 512
    N_values = [25, 50, 100, 200, 400, 800, 1600, 3200]
    n_reps = 60

    results = []
    print(f"\n{'N':>6} {'mean W2^2':>14} {'stderr':>12}")
    for N in N_values:
        w2_sq_vals = np.zeros(n_reps)
        for rep in range(n_reps):
            X_final, mbar_ref = simulate_with_conditional_mean(N, n_steps_fine, seed=2000+rep)
            w2 = w2_gaussian_vs_empirical_1d(X_final, mbar_ref, v_star)
            w2_sq_vals[rep] = w2**2
        mean_w2sq = w2_sq_vals.mean()
        stderr = w2_sq_vals.std() / np.sqrt(n_reps)
        results.append((N, mean_w2sq, stderr))
        print(f"{N:>6} {mean_w2sq:>14.6f} {stderr:>12.2e}")

    Ns = np.array([r[0] for r in results])
    w2sqs = np.array([r[1] for r in results])
    rate, intercept = np.polyfit(np.log(Ns), np.log(w2sqs), 1)
    print(f"\nEstimated rate of E[W2^2] vs N (theory predicts approx -1, i.e. O(1/N)): {rate:.4f}")

    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'particle_convergence.csv'), 'w') as f:
        f.write("N,mean_W2_sq,stderr\n")
        for N, w2sq, se in results:
            f.write(f"{N},{w2sq},{se}\n")
    print("Saved to results/particle_convergence.csv")
