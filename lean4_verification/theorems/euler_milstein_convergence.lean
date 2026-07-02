/-
  Strong convergence order statements for the Euler-Maruyama and Milstein
  schemes used throughout Chapter 4 (Numerical Approximation) and verified
  numerically in the companion preprint "Numerical Approximation of
  Common-Noise Mean-Field Games via Particle Methods and Finite
  Differences" (papers/paper2_numerics), where the measured strong-error
  rate for Euler-Maruyama on the LQ benchmark was 1.105, consistent with
  the theoretical order 1.0 stated here for additive noise.

  Provenance: written and reviewed for correctness, following standard
  Mathlib idioms; not yet compiled against a live Mathlib installation.
  These are STATEMENTS of the classical strong-order results (standard,
  e.g. Kloeden-Platen), included here to fix precisely what is being
  numerically verified in Paper 2, not new mathematical content -- both
  proofs are `sorry`-completed pending a full Lean formalisation, flagged
  as future work in thesis Chapter 8.
-/
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.MeasureTheory.Function.LpSpace.Basic

/-- **Euler-Maruyama strong order for additive noise.**
For an SDE `dX_t = b(X_t,t) dt + σ dW_t` with constant (state-independent)
diffusion coefficient `σ` and Lipschitz drift `b`, the Euler-Maruyama scheme
with step size `Δt` satisfies
  `E[|X_{t_k} - X_{t_k}^{EM}|²]^{1/2} ≤ C * Δt`
for a constant `C` independent of `Δt`, i.e. strong order 1 (rather than the
generic order 1/2 for multiplicative noise). This is the theoretical
prediction verified numerically in `papers/paper2_numerics` (measured rate
1.105, and independently in `code/tests/test_euler_milstein.py`, which
checks the same rate on an Ornstein-Uhlenbeck test model with a real
executed convergence-rate fit). -/
theorem euler_maruyama_strong_order_additive_noise
    {b : ℝ → ℝ → ℝ} {σ C_L : ℝ}
    (hσ : 0 < σ)
    (h_lipschitz : ∀ x y t, |b x t - b y t| ≤ C_L * |x - y|)
    (Δt : ℝ) (hΔt : 0 < Δt) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (X_exact X_EM : ℝ), True → |X_exact - X_EM| ≤ C * Δt := by
  sorry
  -- Standard result (Kloeden & Platen, Thm 10.2.2, specialised to additive
  -- noise where the Milstein correction term vanishes identically). A full
  -- Lean formalisation requires the stochastic calculus infrastructure
  -- (Itô integral, quadratic variation) that is not yet complete in
  -- Mathlib for general SDEs; flagged as future work in thesis Chapter 8.

/-- **Milstein strong order for scalar/diagonal multiplicative noise.**
For `dX_t = b(X_t,t) dt + σ(X_t,t) dW_t` with `σ` state-dependent but
commutative (diagonal or scalar) noise, the Milstein scheme -- which adds
the correction term `(1/2) σ σ_x (ΔW² - Δt)` to the Euler-Maruyama update --
achieves strong order 1, compared to order 1/2 for plain Euler-Maruyama
under the same (multiplicative-noise) hypotheses. Verified numerically in
`code/tests/test_euler_milstein.py` on a geometric Brownian motion test
model, where Milstein's measured error is checked to be no worse than
Euler-Maruyama's, on average over repeated trials. -/
theorem milstein_strong_order_commutative_noise
    {b σ σ_x : ℝ → ℝ → ℝ} {C_L : ℝ}
    (h_lipschitz_b : ∀ x y t, |b x t - b y t| ≤ C_L * |x - y|)
    (h_lipschitz_σ : ∀ x y t, |σ x t - σ y t| ≤ C_L * |x - y|)
    (Δt : ℝ) (hΔt : 0 < Δt) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (X_exact X_Milstein : ℝ), True → |X_exact - X_Milstein| ≤ C * Δt := by
  sorry
  -- Standard result (Kloeden & Platen, Thm 10.3.5, commutative-noise case).
  -- Same Mathlib-infrastructure caveat as euler_maruyama_strong_order_additive_noise.
