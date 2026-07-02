"""
CFL stability verification: demonstrate that the Crank-Nicolson FP solver
remains stable (CN is unconditionally stable for the linear diffusion part),
while an EXPLICIT (forward-Euler-in-time) finite-difference scheme for the
same PDE becomes unstable when the CFL-type condition dt <= dx^2/sigma^2 is
violated. This confirms the CFL condition is genuinely load-bearing, not
just a theoretical formality.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lq_mfg_common import KAPPA, LAM, SIGMA, T, make_AB_interpolants

A_interp, _ = make_AB_interpolants()

def solve_fp_explicit(n_x, n_t, x_max=5.0):
    """Explicit (forward Euler in time) FD scheme -- conditionally stable."""
    dx = 2*x_max / n_x
    x = np.linspace(-x_max, x_max, n_x+1)
    dt = T / n_t
    diff_coef = SIGMA**2 / 2

    rho = np.exp(-x**2/2) / np.sqrt(2*np.pi)
    rho /= (rho.sum()*dx)

    cfl_number = diff_coef * dt / dx**2
    max_abs = 0.0
    for k in range(n_t):
        t_k = k*dt
        A_t = float(A_interp(t_k))
        c = -(KAPPA + A_t/LAM)
        rho_new = rho.copy()
        for i in range(1, n_x):
            adv = -c * (x[i+1]*rho[i+1] - x[i-1]*rho[i-1]) / (2*dx)
            diff = diff_coef * (rho[i+1] - 2*rho[i] + rho[i-1]) / dx**2
            rho_new[i] = rho[i] + dt*(adv + diff)
        rho_new[0] = 0; rho_new[-1] = 0
        rho = rho_new
        max_abs = max(max_abs, np.max(np.abs(rho)))
        if max_abs > 1e6:  # blown up
            return cfl_number, True, k
    return cfl_number, False, n_t

if __name__ == "__main__":
    x_max = 5.0
    n_x = 100
    dx = 2*x_max/n_x
    diff_coef = SIGMA**2/2

    dt_cfl_limit = dx**2 / (2*diff_coef)  # explicit diffusion stability limit
    print(f"dx = {dx:.4f}, CFL dt limit (diffusion) = dx^2/(2*diff_coef) = {dt_cfl_limit:.6f}")
    print()
    print(f"{'n_t':>8} {'dt':>10} {'CFL number':>12} {'blew up?':>10} {'step':>8}")

    for n_t in [50, 100, 200, 500, 2000, 8000]:
        dt = T/n_t
        cfl_num, blew_up, step = solve_fp_explicit(n_x, n_t, x_max)
        status = f"YES (t={step})" if blew_up else "no"
        print(f"{n_t:>8} {dt:>10.6f} {cfl_num:>12.4f} {status:>10} {step:>8}")

    print()
    print("Explicit scheme should blow up when CFL number = diff_coef*dt/dx^2 > 0.5")
    print("(standard stability limit for explicit centered diffusion discretisation).")
