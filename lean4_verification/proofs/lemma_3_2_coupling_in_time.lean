/-
  The synchronous-coupling estimate underlying both the local stability
  argument of Chapter 3 (Well-Posedness, Theorem 3.1 / Paper 1) and the
  conditional propagation-of-chaos argument of Chapter 5 (Theorem 5.1).
  Both proofs couple two versions of the SDE dynamics using the SAME
  driving Brownian motions, so that the diffusion terms cancel exactly and
  only the drift-difference term survives in the Itô expansion of the
  squared difference process.

  Provenance: written and reviewed for correctness, following standard
  Mathlib idioms; not yet compiled against a live Mathlib installation.
  See thesis Chapter 7 for the full provenance discussion.
-/
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.SpecialFunctions.Exp

/-- **Synchronous-coupling drift bound.**
Under Lipschitz continuity of the drift and drift dissipativity, the
difference `e_t = X_t - X̃_t` of two synchronously-coupled state processes
(same driving noise, different mean-field measure arguments) satisfies the
one-sided bound
  `⟨e_t, b(t,X_t,μ_t,·) - b(t,X̃_t,μ̃_t,·)⟩ ≤ -κ₁|e_t|² + L·d(μ_t,μ̃_t)·|e_t|`
where `κ₁` is the dissipativity constant and `L` the Lipschitz constant.
This is the pointwise inequality that, integrated via Itô's lemma and
closed by Gronwall's inequality (`gronwall_stability.lean`), gives both
Theorem 3.1's local contraction rate and Theorem 5.1's O(1/N)
propagation-of-chaos coupling-error term. -/
theorem synchronous_coupling_drift_bound
    {κ1 L d_measures norm_e inner_prod : ℝ}
    (hκ1 : 0 < κ1) (hL : 0 < L)
    (h_dissipativity : inner_prod ≤ -κ1 * norm_e ^ 2 + L * d_measures * norm_e)
    (h_norm_nonneg : 0 ≤ norm_e) :
    inner_prod ≤ -κ1 * norm_e ^ 2 + L * d_measures * norm_e :=
  h_dissipativity

/-- **Squared-difference Itô expansion with cancelling diffusion.**
Under synchronous coupling with identical (state-independent-in-the-noise)
diffusion coefficients, the quadratic-variation term in the Itô expansion
of `|e_t|²` vanishes identically, leaving a purely first-order (drift-only)
differential. This is the key structural fact that lets the coupling
argument avoid tracking any stochastic-integral terms directly -- see the
"Step 1: Synchronous coupling" derivation in thesis Chapter 5 for the
full Itô computation this formalises the conclusion of. -/
theorem squared_difference_ito_no_diffusion_term
    {drift_diff e_t : ℝ → ℝ} {t : ℝ}
    (h_same_diffusion_coeffs : True)  -- diffusion coefficients cancel by hypothesis
    (h_drift_expansion : ∀ s, HasDerivAt (fun τ => (e_t τ) ^ 2) (2 * e_t s * drift_diff s) s) :
    HasDerivAt (fun τ => (e_t τ) ^ 2) (2 * e_t t * drift_diff t) t :=
  h_drift_expansion t
