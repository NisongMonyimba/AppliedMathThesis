"""
Finite-difference Fokker-Planck solver: spatial convergence experiment.
Solves the (deterministic-in-mean, since we condition on a fixed realized
mbar_t path from the closed-form/SDE solution) linear Fokker-Planck equation
for the LQ-MFG conditional density, using an implicit (Crank-Nicolson) scheme,
and checks convergence against the exact Gaussian conditional density.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lq_mfg_common import KAPPA, LAM, C1, C2, SIGMA, SIGMA0, T, make_AB_interpolants

A_interp, B_interp = make_AB_interpolants()

def exact_conditional_variance():
    from scipy.integrate import solve_ivp
    def rhs(t, y):
        v = y[0]
        A_t = float(A_interp(t))
        drift_coef = -(KAPPA + A_t/LAM)
        return [2*drift_coef*v + SIGMA**2]
    sol = solve_ivp(rhs, [0, T], [1.0], t_eval=np.linspace(0,T,2000),
                     method='Radau', rtol=1e-12, atol=1e-14)
    return sol.t, sol.y[0]

def solve_fp_crank_nicolson(n_x, n_t, x_max=6.0, mbar_path=None):
    """
    Solve the conditional Fokker-Planck equation
      d rho/dt = -d/dx[ drift(x,t) rho ] + (sigma^2/2) d^2rho/dx^2
    on x in [-x_max, x_max], where drift(x,t) = kappa(mbar_t - x) - (A_t x + B_t
    mbar_t)/lambda, using a fixed reference path mbar_t (here, mbar_t=0 for
    simplicity, i.e. we solve the CONDITIONAL problem centered at its own mean,
    which is exactly what the exact_conditional_variance() ODE describes).
    Crank-Nicolson (implicit, unconditionally stable) time-stepping.
    """
    dx = 2*x_max / n_x
    x = np.linspace(-x_max, x_max, n_x+1)
    dt = T / n_t

    # initial condition: N(0,1)
    rho = np.exp(-x**2/2) / np.sqrt(2*np.pi)
    rho /= (rho.sum()*dx)   # normalise on the grid

    for k in range(n_t):
        t_k = k*dt
        t_kp1 = (k+1)*dt
        A_tk = float(A_interp(t_k)); A_tkp1 = float(A_interp(t_kp1))
        # centered problem: mbar=0, drift(x,t) = -(kappa + A_t/lambda) x
        def drift_coef(A_t): return -(KAPPA + A_t/LAM)

        # Build the (linear, tridiagonal) Crank-Nicolson system for the FP PDE:
        # d rho/dt = -d/dx[c(t) x rho] + (sigma^2/2) rho''
        # Using centered differences for both advection and diffusion.
        n = n_x + 1
        diff_coef = SIGMA**2 / 2

        def build_operator(c):
            # L[rho]_i = -c * d/dx(x*rho)_i + diff_coef * rho''_i
            # advection term (conservative form, centered):
            #   d/dx(x rho)_i ~ (x_{i+1}rho_{i+1} - x_{i-1}rho_{i-1}) / (2dx)
            main = np.zeros(n); lower = np.zeros(n); upper = np.zeros(n)
            for i in range(1, n-1):
                adv_lower = -c * (-x[i-1]) / (2*dx)
                adv_upper = -c * ( x[i+1]) / (2*dx)
                diff_lower = diff_coef / dx**2
                diff_upper = diff_coef / dx**2
                diff_main = -2*diff_coef / dx**2
                lower[i] = adv_lower + diff_lower
                upper[i] = adv_upper + diff_upper
                main[i]  = diff_main
            return main, lower, upper

        c_k = drift_coef(A_tk)
        c_kp1 = drift_coef(A_tkp1)
        main_k, lower_k, upper_k = build_operator(c_k)
        main_kp1, lower_kp1, upper_kp1 = build_operator(c_kp1)

        # Crank-Nicolson: (I - dt/2 L^{k+1}) rho^{k+1} = (I + dt/2 L^k) rho^k
        # Build tridiagonal matrices
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve

        LHS_main = 1 - dt/2*main_kp1
        LHS_lower = -dt/2*lower_kp1
        LHS_upper = -dt/2*upper_kp1
        RHS_main = 1 + dt/2*main_k
        RHS_lower = dt/2*lower_k
        RHS_upper = dt/2*upper_k

        # zero-flux (Dirichlet, rho=0) boundary conditions at x=+-x_max
        LHS_main[0] = 1; LHS_main[-1] = 1
        LHS_lower[0] = 0; LHS_upper[0] = 0
        LHS_lower[-1] = 0; LHS_upper[-1] = 0

        A_mat = diags([LHS_lower[1:], LHS_main, LHS_upper[:-1]], [-1, 0, 1], format='csr')
        rho_shift_lower = np.roll(rho, 1); rho_shift_lower[0]=0
        rho_shift_upper = np.roll(rho, -1); rho_shift_upper[-1]=0
        rhs_vec = RHS_main*rho + RHS_lower*rho_shift_lower + RHS_upper*rho_shift_upper
        rhs_vec[0] = 0; rhs_vec[-1] = 0

        rho = spsolve(A_mat, rhs_vec)
        rho = np.maximum(rho, 0)  # guard against tiny negative numerical noise
        mass = rho.sum()*dx
        if mass > 1e-10:
            rho = rho / mass

    return x, rho

def exact_density(x, v_T):
    return np.exp(-x**2/(2*v_T)) / np.sqrt(2*np.pi*v_T)

if __name__ == "__main__":
    t_grid, v_grid = exact_conditional_variance()
    v_T = v_grid[-1]
    print(f"Exact conditional variance at T: v_T = {v_T:.6f}")

    n_t_fixed = 4000  # fine time resolution so spatial error dominates
    nx_values = [100, 200, 400, 800]
    x_max = 5.0

    results = []
    print(f"\n{'n_x':>6} {'dx':>10} {'L2 error':>14}")
    for n_x in nx_values:
        x, rho_num = solve_fp_crank_nicolson(n_x, n_t_fixed, x_max=x_max)
        rho_ex = exact_density(x, v_T)
        dx = 2*x_max/n_x
        l2_err = np.sqrt(np.sum((rho_num - rho_ex)**2) * dx)
        results.append((n_x, dx, l2_err))
        print(f"{n_x:>6} {dx:>10.4f} {l2_err:>14.8f}")

    dxs = np.array([r[1] for r in results])
    errs = np.array([r[2] for r in results])
    rate, intercept = np.polyfit(np.log(dxs), np.log(errs), 1)
    print(f"\nEstimated spatial convergence rate: {rate:.4f}")

    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'fd_convergence.csv'), 'w') as f:
        f.write("n_x,dx,L2_error\n")
        for n_x, dx, err in results:
            f.write(f"{n_x},{dx},{err}\n")
    print("Saved to results/fd_convergence.csv")
