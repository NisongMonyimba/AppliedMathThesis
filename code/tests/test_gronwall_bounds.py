"""
Numerical verification of Gronwall's inequality in the two forms used
throughout the thesis and Paper 1:

  1. Differential Gronwall: if y'(t) <= C1*y(t) + C2*f(t) and y(0) = y0, then
     y(t) <= y0*exp(C1*t) + integral_0^t exp(C1*(t-s)) * C2*f(s) ds.

  2. Integral Gronwall: if phi(t) <= a(t) + k * integral_0^t phi(s) ds with
     a(t) non-decreasing, then phi(t) <= a(t) * exp(k*t).

This second form is exactly the inequality that Paper 1's Appendix A found
had been INCORRECTLY closed via an algebraic (1/(1-k)) argument in an
earlier draft -- these tests exist specifically to prevent that class of
error from recurring silently. See papers/paper1_wellposedness/main.tex,
Appendix A, "Why a direct closure on [0,T] fails" for the full account.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import odeint, quad


class TestDifferentialGronwall:
    def test_ode_solution_matches_closed_form_bound_with_equality(self) -> None:
        """
        For y'(t) = C1*y(t) + C2 (constant forcing), y(0)=y0, the exact
        solution is y(t) = (y0 + C2/C1)*exp(C1*t) - C2/C1. This should equal
        the Gronwall bound with equality, since the differential inequality
        is in fact an equality here.
        """
        C1, C2, y0 = 0.8, 1.3, 0.5

        def rhs(y: float, t: float) -> float:
            return C1 * y + C2

        t_eval = np.linspace(0, 2, 50)
        y_numeric = odeint(rhs, y0, t_eval).flatten()

        y_exact = (y0 + C2 / C1) * np.exp(C1 * t_eval) - C2 / C1
        np.testing.assert_allclose(y_numeric, y_exact, rtol=1e-6)

    def test_gronwall_bound_is_a_valid_upper_bound_under_perturbation(self) -> None:
        """
        Perturb the forcing to be strictly below its constant upper bound
        (y'(t) <= C1*y(t) + C2 with a time-varying, bounded-above forcing)
        and check the closed-form Gronwall bound still dominates the true
        (numerically integrated) solution pointwise.
        """
        C1, C2, y0 = 0.6, 1.0, 0.2

        def true_forcing(t: float) -> float:
            # bounded above by C2, sometimes strictly less
            return C2 * (0.5 + 0.5 * np.sin(3 * t) ** 2)

        def rhs(y: float, t: float) -> float:
            return C1 * y + true_forcing(t)

        t_eval = np.linspace(0, 3, 100)
        y_numeric = odeint(rhs, y0, t_eval).flatten()

        gronwall_bound = (y0 + C2 / C1) * np.exp(C1 * t_eval) - C2 / C1

        assert np.all(y_numeric <= gronwall_bound + 1e-8)


class TestIntegralGronwall:
    """
    These tests target exactly the inequality type mishandled in the
    earlier (incorrect) draft of Paper 1's Appendix A: phi(t) <= a(t) +
    k * integral_0^t phi(s) ds. The correct closure is phi(t) <= a(t)*exp(k*t);
    the (invalid) closure that was originally attempted was the algebraic
    phi(t) <= a(t)/(1-k), applicable only to a genuinely different
    (non-integral) inequality.
    """

    def test_equality_case_matches_exp_kt_bound_exactly(self) -> None:
        """
        For a(t) = a0 constant, the equation phi(t) = a0 + k*integral phi(s)ds
        has exact solution phi(t) = a0*exp(k*t) (differentiate both sides:
        phi'(t) = k*phi(t), phi(0) = a0).
        """
        a0, k = 1.0, 0.5
        t_eval = np.linspace(0, 4, 50)
        phi_exact = a0 * np.exp(k * t_eval)

        # Verify by direct substitution into the integral equation
        for i, t in enumerate(t_eval):
            integral_val, _ = quad(lambda s: a0 * np.exp(k * s), 0, t)
            rhs = a0 + k * integral_val
            assert rhs == pytest.approx(phi_exact[i], rel=1e-6)

    def test_incorrect_algebraic_closure_is_not_a_valid_bound(self) -> None:
        """
        This is the regression test for the specific bug found in Paper 1:
        directly checks that the (invalid) algebraic closure phi <= a/(1-k)
        is NOT a valid upper bound for the true solution phi(t) = a0*exp(k*t)
        once t is large enough -- confirming, numerically, why that closure
        technique was wrong and must not be reintroduced.
        """
        a0, k = 1.0, 0.5
        incorrect_bound = a0 / (1 - k)  # = 2.0, constant in t

        t_eval = np.linspace(0, 3, 100)
        true_phi = a0 * np.exp(k * t_eval)

        # The true solution MUST exceed the incorrect constant bound for
        # large enough t -- if this ever fails, something about the
        # underlying exponential-vs-algebraic relationship has changed
        # and the surrounding derivation should be re-examined.
        assert np.any(true_phi > incorrect_bound), (
            "Expected the true integral-Gronwall solution to eventually "
            "exceed the incorrect algebraic bound a/(1-k); if this "
            "assertion fails, re-verify the Appendix A bug account."
        )

        crossover_idx = np.argmax(true_phi > incorrect_bound)
        crossover_t = t_eval[crossover_idx]
        expected_crossover = np.log(1 - k) / (-k)  # solve a0*exp(k*t) = a0/(1-k)
        assert crossover_t == pytest.approx(expected_crossover, abs=0.15)

    def test_correct_exp_bound_dominates_numerically_integrated_inequality(self) -> None:
        """
        For a genuinely time-varying (non-decreasing) a(t), verify phi(t) <=
        a(t)*exp(k*t) holds for a numerically-integrated phi satisfying the
        integral inequality with equality (the worst case, tightest to the
        bound).
        """
        k = 0.4

        def a(t: float) -> float:
            return 1.0 + 0.3 * t  # non-decreasing, as required

        # phi(t) = a(t) + k*integral_0^t phi(s) ds, differentiate:
        # phi'(t) = a'(t) + k*phi(t), phi(0) = a(0)
        def rhs(phi: float, t: float) -> float:
            a_prime = 0.3
            return a_prime + k * phi

        t_eval = np.linspace(0, 5, 100)
        phi_numeric = odeint(rhs, a(0.0), t_eval).flatten()

        bound = np.array([a(t) for t in t_eval]) * np.exp(k * t_eval)
        assert np.all(phi_numeric <= bound + 1e-6)


class TestGronwallParameterSensitivity:
    def test_larger_k_gives_faster_growth(self) -> None:
        """Sanity check: the Gronwall exponential bound should grow strictly
        faster for larger k, at any fixed t > 0."""
        a0 = 1.0
        t = 2.0
        k_small, k_large = 0.2, 0.8
        bound_small = a0 * np.exp(k_small * t)
        bound_large = a0 * np.exp(k_large * t)
        assert bound_large > bound_small

    def test_zero_k_gives_constant_bound(self) -> None:
        """With k=0, the integral Gronwall bound degenerates to phi(t) <= a(t)
        (no self-referential growth), consistent with the k -> 0 limit."""
        a0 = 3.0
        t_eval = np.linspace(0, 5, 20)
        bound = a0 * np.exp(0.0 * t_eval)
        np.testing.assert_allclose(bound, a0)
