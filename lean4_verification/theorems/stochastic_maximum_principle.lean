/-
  Optimality of the linear feedback control in the canonical linear-quadratic
  (LQ) mean-field game benchmark, derived from the Stochastic Maximum
  Principle (Chapter 2). This is the closed-loop control α*(t,x,μ) =
  -(A_t x + B_t μ̄)/λ used as the reference benchmark throughout the thesis
  and both companion preprints.

  Provenance: written and reviewed for correctness, following standard
  Mathlib idioms; not yet compiled against a live Mathlib installation.
  See thesis Chapter 7 for the full provenance discussion.
-/
import Mathlib.Analysis.Calculus.Deriv.Linear
import Mathlib.LinearAlgebra.Matrix.DotProduct

/-- In the canonical LQ-MFG (Example 1.1, restated in Paper 1 as
Example "canonical LQ-MFG"), the optimal control maximising the Hamiltonian

  `H(t,x,μ̄,α) = -(A_t x + B_t μ̄) * α - (λ/2) * α²`

is `α*(t,x,μ̄) = -(A_t x + B_t μ̄) / λ`. -/
theorem lq_optimal_control_linear
    {κ λ c1 c2 : ℝ}
    (hκ : 0 < κ) (hλ : 0 < λ)
    (hc1 : 0 < c1) (hc2 : 0 < c2)
    {A B : ℝ → ℝ}
    (hA : ∀ t, HasDerivAt A (deriv A t) t)
    (hB : ∀ t, HasDerivAt B (deriv B t) t)
    (hA_riccati : ∀ t, deriv A t = 2 * κ * A t + (A t) ^ 2 / λ - c1)
    (hB_riccati : ∀ t, deriv B t = κ * B t + 2 * (A t) * (B t) / λ
        + (B t) ^ 2 / λ - κ * A t) :
    ∀ t x μ̄, IsMaxOn
      (fun α => -(A t * x + B t * μ̄) * α - λ / 2 * α ^ 2)
      Set.univ
      (-(A t * x + B t * μ̄) / λ) := by
  intro t x μ̄ α _
  simp only [IsMaxOn, IsMinOn, Set.mem_univ, forall_true_left]
  -- H(α) = -(A_t x + B_t μ̄) α - (λ/2) α²  is a downward parabola in α;
  -- its unique maximiser is where dH/dα = -(A_t x + B_t μ̄) - λα = 0, i.e.
  -- α* = -(A_t x + B_t μ̄)/λ. Completing the square gives the inequality
  -- directly, avoiding an explicit derivative computation.
  have key : -(A t * x + B t * μ̄) * (-(A t * x + B t * μ̄) / λ)
      - λ / 2 * (-(A t * x + B t * μ̄) / λ) ^ 2 ≥
      -(A t * x + B t * μ̄) * α - λ / 2 * α ^ 2 := by
    nlinarith [sq_nonneg (α + (A t * x + B t * μ̄) / λ), le_of_lt hλ]
  linarith
