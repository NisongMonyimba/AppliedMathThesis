import Lake
open Lake DSL

package «lean4_verification» where
  -- Formal verification companion to:
  --   N. Monyimba, "Quantitative Convergence Analysis of Stochastic Maximum
  --   Principles for Mean-Field Games with Common Noise" (MSc thesis, 2026)
  -- and the companion preprints in ../papers/.
  --
  -- Provenance note: these proofs were written and reviewed for mathematical
  -- correctness, following standard Mathlib idioms, but have not yet been
  -- compiled against a live Mathlib installation as of this commit. Run
  -- `lake update && lake build` to check; the lean4-check.yml CI workflow
  -- does this automatically on every push. See thesis Chapter 7 for the
  -- full provenance caveat and discussion of what each proof establishes.
  leanOptions := #[⟨`autoImplicit, false⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «Theorems» where
  srcDir := "theorems"

@[default_target]
lean_lib «Proofs» where
  srcDir := "proofs"
