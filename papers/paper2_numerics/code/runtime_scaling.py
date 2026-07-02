"""
Runtime scaling experiment: wall-clock time for the mean-field-factored
Euler-Maruyama particle scheme as a function of N, to verify O(N) complexity.
"""
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lq_mfg_common import KAPPA, LAM, C1, C2, SIGMA, SIGMA0, T, make_AB_interpolants

A_interp, B_interp = make_AB_interpolants()

def run_and_time(N, n_steps, seed):
    rng = np.random.RandomState(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    X = rng.normal(0, 1, size=N)
    t0 = time.perf_counter()
    for k in range(n_steps):
        t_k = k * dt
        mu_bar = X.mean()
        A_t = float(A_interp(t_k)); B_t = float(B_interp(t_k))
        alpha = -(A_t * X + B_t * mu_bar) / LAM
        drift = KAPPA * (mu_bar - X) + alpha
        dW = rng.normal(0, sqrt_dt, size=N)
        dW0 = rng.normal(0, sqrt_dt)
        X = X + drift * dt + SIGMA * dW + SIGMA0 * dW0
    t1 = time.perf_counter()
    return t1 - t0

if __name__ == "__main__":
    N_values = [1000, 10000, 100000, 1000000]
    n_steps = 64
    n_reps = 5

    results = []
    print(f"{'N':>10} {'wall_time(s)':>14} {'time/particle(us)':>20}")
    for N in N_values:
        times = [run_and_time(N, n_steps, seed=5000+r) for r in range(n_reps)]
        mean_time = np.median(times)  # median more robust to system jitter
        per_particle = mean_time / N * 1e6
        results.append((N, mean_time, per_particle))
        print(f"{N:>10} {mean_time:>14.4f} {per_particle:>20.3f}")

    Ns = np.array([r[0] for r in results])
    times_arr = np.array([r[1] for r in results])
    rate, intercept = np.polyfit(np.log(Ns), np.log(times_arr), 1)
    print(f"\nEstimated scaling exponent (expect ~1.0 for O(N)): {rate:.4f}")

    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'runtime_scaling.csv'), 'w') as f:
        f.write("N,wall_time_s,time_per_particle_us\n")
        for N, t, pp in results:
            f.write(f"{N},{t},{pp}\n")
    print("Saved to results/runtime_scaling.csv")
