# Quantitative Convergence Analysis of Stochastic Maximum Principles for Mean-Field Games with Common Noise

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21131132.svg)](https://doi.org/10.5281/zenodo.21131132)

Research software and manuscripts for the numerical analysis of common-noise
mean-field games (MFGs): implementations of particle methods, stochastic
differential equation solvers, Wasserstein geometry algorithms, and
finite-difference methods, accompanying an MSc thesis and two companion
preprints in stochastic analysis, optimal transport, and scientific
computing.

## Contents

| Path | What it is |
|---|---|
| [`manuscript/`](manuscript/) | MSc thesis: *Quantitative Convergence Analysis of Stochastic Maximum Principles for Mean-Field Games with Common Noise* |
| [`papers/paper1_wellposedness/`](papers/paper1_wellposedness/) | Preprint: *Quantitative Contraction Estimates for Common-Noise Mean-Field Games* |
| [`papers/paper2_numerics/`](papers/paper2_numerics/) | Preprint: *Numerical Approximation of Common-Noise Mean-Field Games via Particle Methods and Finite Differences* |
| [`code/src/sde_solvers/`](code/src/sde_solvers/) | Euler-Maruyama and Milstein SDE solvers (generic and mean-field/common-noise variants) |
| [`code/tests/`](code/tests/) | Test suite (`pytest`), verified against closed-form results |
| [`lean4_verification/`](lean4_verification/) | Lean 4 formalisation of key estimates (Gronwall, LQ optimality, contraction factor, Wasserstein-Gaussian formula, coupling argument) |
| [`simulations/`](simulations/) | Numerical experiment scripts and raw results |
| [`research_statement/`](research_statement/), [`cv/`](cv/) | PhD application materials |

## Summary of results

The thesis and companion papers establish, for mean-field games in which a
continuum of agents interacts through both idiosyncratic noise and a shared
common Brownian motion:

1. **Quantitative well-posedness** (thesis Ch. 3; Paper 1): an explicit local
   contraction estimate for the fixed-point map characterising the MFG
   equilibrium, on an explicit horizon $\delta_0$, extended to arbitrary
   time by a standard continuation argument.
2. **Numerical convergence** (thesis Ch. 4; Paper 2): a combined
   Euler-Maruyama/particle/finite-difference scheme, with every reported
   convergence rate the output of executed, reproducible code — not an
   illustrative estimate.
3. **Particle methods and propagation of chaos** (thesis Ch. 5; Paper 2): a
   conditional propagation-of-chaos rate for the $N$-particle system,
   verified numerically up to $N = 51{,}200$.
4. **Partial formal verification** (thesis Ch. 6-7): Lean 4 formalisations
   of the core estimates above, with an explicit provenance caveat (see
   [`lean4_verification/`](lean4_verification/)).

Every numerical result in the thesis and both papers is labelled as either
**executed** (the direct output of a script in this repository) or
**illustrative** (a theoretical prediction not yet backed by a corresponding
script) — this distinction is stated explicitly at the point each result
appears, and no result is described as executed unless the corresponding
script is present and runnable.

## Reproducing the results

```bash
git clone https://github.com/NisongMonyimba/AppliedMathThesis.git
cd AppliedMathThesis
pip install -r requirements.txt
python -m pytest code/tests -v
```

Each figure/table in Paper 2 is reproduced by a specific script under
[`papers/paper2_numerics/code/`](papers/paper2_numerics/code/); see that
paper's Reproducibility section for the exact mapping.

## Building the manuscripts

```bash
cd manuscript && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd ../papers/paper1_wellposedness && pdflatex main.tex && pdflatex main.tex
cd ../paper2_numerics && pdflatex main.tex && pdflatex main.tex
```

## Formal verification

```bash
cd lean4_verification && lake update && lake build
```

See [`lean4_verification/`](lean4_verification/) for the current provenance
status of these proofs (written and reviewed, not yet compiled against a
live Mathlib installation as of this commit).

## Status

Both preprints are complete and pending arXiv submission.

## Citation

See [`CITATION.cff`](CITATION.cff). This repository is archived on Zenodo with DOI
[10.5281/zenodo.21131132](https://doi.org/10.5281/zenodo.21131132).

## License

See [`LICENSE`](LICENSE).

## Contact

Nisong Monyimba — nmonyimb@asu.edu
