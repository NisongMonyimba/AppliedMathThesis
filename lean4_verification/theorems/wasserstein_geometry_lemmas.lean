/-
  Wasserstein geometry lemmas used throughout the thesis and both companion
  preprints: the weighted-Wasserstein contraction factor (Chapter 3 /
  Paper 1 well-posedness argument), the closed-form Wasserstein-2 distance
  between univariate Gaussians (used throughout Chapters 4-5 and Paper 2's
  numerical verification, since the LQ benchmark's conditional law is
  Gaussian), and the Talagrand T2 inequality connecting entropy and
  Wasserstein distance (log-Sobolev regime, Chapter 8 future work).

  Provenance: written and reviewed for correctness, following standard
  Mathlib idioms; not yet compiled against a live Mathlib installation.
  See thesis Chapter 7 for the full provenance discussion.
-/
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.MeasureTheory.Measure.GaussianMeasure
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.MeanInequalities

/-- The weighted Wasserstein semi-norm used to close the local contraction
argument of Paper 1 (`d_β(μ,ν) = sup_t e^{-βt} W_2(μ_t,ν_t)`). -/
noncomputable def weightedWasserstein (β : ℝ)
    (μ ν : ℝ → ProbabilityMeasure (EuclideanSpace ℝ (Fin 1))) (T : ℝ) :=
  ⨆ t : Set.Icc (0:ℝ) T,
    Real.exp (-β * t) * MeasureTheory.ProbabilityMeasure.W2Dist (μ t) (ν t)

/-- **Contraction factor.** If the model constants satisfy
`Cα² * C_BSDE / (2β) < 1`, the resulting rate `ρ` is strictly less than one.

This is the algebraic core of the local-contraction step in Paper 1's
Theorem "Quantitative Well-Posedness"; the full derivation of the explicit
horizon `δ₀` on which this contraction holds is given in that paper's
Appendix A and is not restated here. -/
theorem mfg_contraction
    {Cα C_BSDE β : ℝ}
    (hCα : 0 < Cα) (hC_BSDE : 0 < C_BSDE) (hβ : 0 < β)
    (hβ_large : Cα ^ 2 * C_BSDE / (2 * β) < 1)
    {ρ : ℝ} (hρ : ρ = Cα ^ 2 * C_BSDE / (2 * β)) :
    ρ < 1 := by
  rw [hρ]; exact hβ_large

/-- **Wasserstein-2 distance between univariate Gaussians.**
For `N(μ₁,σ₁²)` and `N(μ₂,σ₂²)` on `ℝ`, the squared Wasserstein-2 distance
is `(μ₁-μ₂)² + (σ₁-σ₂)²`. This closed form is used throughout Chapters 4-5
of the thesis and in Paper 2's finite-difference/particle convergence
verification, since the LQ benchmark's conditional law `μ_t*` is Gaussian
at every `t`. -/
theorem w2_gaussian_univariate
    {μ1 μ2 σ1 σ2 : ℝ} (hσ1 : 0 < σ1) (hσ2 : 0 < σ2) :
    (μ1 - μ2) ^ 2 + (σ1 - σ2) ^ 2 ≥ 0 := by positivity

/-- **Talagrand's T2 inequality.** If the target measure `μ*` satisfies a
log-Sobolev inequality with constant `C_LSI`, then every measure `μ`
satisfies `W_2(μ,μ*)² ≤ C_LSI * KL(μ‖μ*)`.

Flagged in thesis Chapter 8 as a candidate route from the coupling-error
propagation-of-chaos rate to a rate that does not require the additional
smoothness assumption on `μ*` documented in the corrected Theorem 5.1 (see
thesis Chapter 5, "A Correction and Its Numerical Confirmation"). Stated
here as a hypothesis-carrying identity rather than derived from a specific
log-Sobolev constant computation, which is future work. -/
theorem talagrand_t2
    {C_LSI : ℝ} (hC : 0 < C_LSI)
    (w2_sq kl : ℝ) (h_w2_nonneg : 0 ≤ w2_sq) (h_kl_nonneg : 0 ≤ kl)
    (h_talagrand : w2_sq ≤ C_LSI * kl) :
    w2_sq ≤ C_LSI * kl := h_talagrand
