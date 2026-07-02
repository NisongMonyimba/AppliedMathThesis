"""
Shared LQ-MFG model utilities for Paper 2 numerical experiments.
Baseline parameters match Paper 1 / thesis: kappa=0.5, lambda=1, c1=2, c2=5,
sigma=0.2, sigma0=0.3, T=1.
"""
import numpy as np
from scipy.integrate import solve_ivp

KAPPA = 0.5
LAM = 1.0
C1 = 2.0
C2 = 5.0
SIGMA = 0.2
SIGMA0 = 0.3
T = 1.0

def riccati_rhs(tau, y):
    """Backward-time (tau = T-t) Riccati system for A_t, B_t (corrected signs)."""
    A, B = y
    dA = -(2*KAPPA*A + A**2/LAM - C1)
    dB = -(KAPPA*B + 2*A*B/LAM + B**2/LAM - KAPPA*A)
    return [dA, dB]

def solve_riccati(n_points=2000):
    """Solve for A_t, B_t on a fine grid, return as callable interpolants."""
    tau_eval = np.linspace(0, T, n_points)
    sol = solve_ivp(riccati_rhs, [0, T], [C2, 0.0], t_eval=tau_eval,
                     method='Radau', rtol=1e-12, atol=1e-14)
    tau = sol.t
    A_tau = sol.y[0]
    B_tau = sol.y[1]
    t_fwd = T - tau
    order = np.argsort(t_fwd)
    return t_fwd[order], A_tau[order], B_tau[order]

def make_AB_interpolants():
    from scipy.interpolate import interp1d
    t_grid, A_grid, B_grid = solve_riccati()
    A_interp = interp1d(t_grid, A_grid, kind='cubic', fill_value='extrapolate')
    B_interp = interp1d(t_grid, B_grid, kind='cubic', fill_value='extrapolate')
    return A_interp, B_interp

if __name__ == "__main__":
    t, A, B = solve_riccati(11)
    for i in range(0, len(t), 1):
        print(f"t={t[i]:.2f}  A_t={A[i]:.6f}  B_t={B[i]:.6f}")
