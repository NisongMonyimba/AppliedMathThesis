/-
  Gronwall's inequality, as used in the stability/contraction argument of
  Chapter 3 (Well-Posedness) and the local-contraction argument of the
  companion preprint "Quantitative Contraction Estimates for Common-Noise
  Mean-Field Games" (papers/paper1_wellposedness).

  Provenance: written and reviewed for correctness, following standard
  Mathlib idioms; not yet compiled against a live Mathlib installation.
  Run `lake build` from lean4_verification/ to check. See thesis Chapter 7
  for the full provenance discussion.
-/
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.ODE.Gronwall
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul

open Real

/-- **Differential Gronwall's inequality.**
If `y' ≤ C1 * y + C2` on `[0, ∞)` with `y(0) = 0`, `C1 > 0`, `C2 ≥ 0`, then
`y(t) ≤ (C2 / C1) * (exp(C1 * t) - 1)` for all `t ≥ 0`.

This is the estimate underlying the synchronous-coupling stability argument
of Theorem 3.1 (Wasserstein Stability) in the thesis and in the companion
well-posedness preprint. -/
theorem gronwall_estimate {y : ℝ → ℝ} {C1 C2 : ℝ}
    (hC1 : 0 < C1) (hC2 : 0 ≤ C2)
    (h_deriv : ∀ t ≥ 0, HasDerivAt y (deriv y t) t)
    (h_bound : ∀ t ≥ 0, deriv y t ≤ C1 * y t + C2)
    (hy0 : y 0 = 0) :
    ∀ t ≥ 0, y t ≤ C2 / C1 * (Real.exp (C1 * t) - 1) := by
  intro t ht
  -- Define z(t) = y(t) - (C2 / C1) * (exp(C1 * t) - 1).
  -- Show z' ≤ C1 * z, z(0) = 0, hence z ≤ 0 by the standard Gronwall lemma.
  have hC1_ne : C1 ≠ 0 := ne_of_gt hC1
  apply Gronwall.gronwall_nonneg_of_le hC1 _ _ _ ht
  · exact hy0
  · intro s hs
    calc deriv y s ≤ C1 * y s + C2 := h_bound s hs
    _ = C1 * (y s - C2 / C1 * (Real.exp (C1 * s) - 1))
        + C1 * (C2 / C1 * (Real.exp (C1 * s) - 1)) + C2 := by ring
    _ = C1 * (y s - C2 / C1 * (Real.exp (C1 * s) - 1))
        + C2 * Real.exp (C1 * s) := by field_simp; ring

/-- **Integral Gronwall's inequality.**
If `φ(t) ≤ a(t) + k * ∫₀ᵗ φ(s) ds` with `a` non-decreasing and `k ≥ 0`, then
`φ(t) ≤ a(t) * exp(k * t)`.

This is the *correct* closure for the self-referential inequality that arose
in the propagation-of-chaos argument of Chapter 5 / paper2_numerics. An
earlier draft of Appendix A (papers/paper1_wellposedness) closed a structurally
identical inequality using the ALGEBRAIC bound `φ ≤ a / (1 - k)`, which is
invalid for this integral form -- see the "Why a direct closure on [0,T]
fails" discussion in that paper's Appendix A for the full account, including
a numerical counterexample. This formal statement exists specifically to
pin down the correct closure and prevent that error from recurring. -/
theorem integral_gronwall_estimate {φ a : ℝ → ℝ} {k : ℝ}
    (hk : 0 ≤ k)
    (ha_mono : Monotone a)
    (h_bound : ∀ t ≥ 0, φ t ≤ a t + k * ∫ s in (0:ℝ)..t, φ s) :
    ∀ t ≥ 0, φ t ≤ a t * Real.exp (k * t) := by
  sorry
  -- Proof sketch: apply the standard integral form of Gronwall's lemma
  -- (available in Mathlib as a corollary of the ODE comparison theorem),
  -- using monotonicity of `a` to pull it outside the integral bound.
  -- Formalising this fully is flagged as future work in thesis Chapter 8.
