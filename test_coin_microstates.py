"""
test_coin_microstates.py - automatic checks for the project (pytest)
====================================================================

Run all checks from the project folder:

    pytest

(or, in VS Code: Testing panel (flask icon) -> Configure Python Tests ->
pytest -> root folder, then press "Run Tests").

Each test compares a formula or algorithm of the project with an
independent EXACT calculation (Python integers or fractions), so a passing
run shows the numbers in the report are correct, not just self-consistent.
The tests write no files and open no windows.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

import coin_common as cc
from part2 import team_parameters


# ---------------------------------------------------------------------------
# Multiplicities
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [0, 1, 7, 20, 56, 100])
def test_omega_row_sums_to_2_power_N(N):
    row = cc.omega_row_exact(N)
    assert len(row) == N + 1
    assert sum(row) == 2 ** N                      # binomial theorem, exact


def test_omega_row_matches_table_1_and_part_1a():
    assert cc.omega_row_exact(5) == [1, 5, 10, 10, 5, 1]          # Table 1 of the handout
    assert cc.omega_row_exact(7) == [1, 7, 21, 35, 35, 21, 7, 1]


def test_float_pascal_is_exact_up_to_N56():
    # Below 2^53 every double-precision value is an exact integer.
    for N in (10, 30, 56):
        assert [int(v) for v in cc.pascal_row_float(N)] == cc.omega_row_exact(N)


def test_float_pascal_overflows_after_N1029():
    assert np.all(np.isfinite(cc.pascal_row_float(1029)))
    assert not np.all(np.isfinite(cc.pascal_row_float(1030)))


# ---------------------------------------------------------------------------
# Part 1b enumeration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 2, 5, 13, 20, 23])
def test_enumeration_reproduces_binomial_coefficients(N):
    counts, elapsed = cc.enumerate_macrostates(N)
    assert counts.tolist() == cc.omega_row_exact(N)
    assert elapsed >= 0


def test_enumeration_lookup_table_fallback(monkeypatch):
    # Simulate NumPy < 2.0 (no np.bitwise_count) to test the fallback path.
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    counts, _ = cc.enumerate_macrostates(18)
    assert counts.tolist() == cc.omega_row_exact(18)


def test_enumeration_with_several_chunks():
    # Small chunks force the multi-chunk code path.
    counts, _ = cc.enumerate_macrostates(16, chunk_bits=10)
    assert counts.tolist() == cc.omega_row_exact(16)


# ---------------------------------------------------------------------------
# Logarithmic formulas (Parts 1e and 2k)
# ---------------------------------------------------------------------------
def test_ln_binom_matches_exact_logs():
    N = 1000
    ks = [0, 1, 250, 500, 999, 1000]
    exact = [math.log(math.comb(N, k)) for k in ks]
    assert np.allclose(cc.ln_binom(N, ks), exact, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("N,k", [(200, 8), (200, 30), (10 ** 6, 20)])
def test_ln_binom_small_k_matches_exact(N, k):
    assert cc.ln_binom_small_k(N, k) == pytest.approx(math.log(math.comb(N, k)), rel=1e-13)


@pytest.mark.parametrize("N,d", [(200, 10), (200, 25), (10 ** 6, 25)])
def test_ln_ratio_central_matches_exact(N, d):
    c = N // 2
    exact = math.log(Fraction(math.comb(N, c + d), math.comb(N, c)))
    assert cc.ln_ratio_central(N, d) == pytest.approx(exact, rel=1e-13)


def test_ln_p_central_stirling_accuracy():
    # Stirling series error is O(N^-3) relative: ~1e-9 at N = 200, less at 2000.
    for N, tol in ((200, 1e-8), (2000, 1e-11)):
        exact = math.log(Fraction(math.comb(N, N // 2), 2 ** N))
        assert cc.ln_p_central(N) == pytest.approx(exact, rel=tol)


def test_large_N_leading_orders():
    # For N = 1e23 the exact formulas must reduce to their leading orders.
    NL = 1e23
    assert cc.ln_ratio_central(NL, 10) == pytest.approx(-2 * 10 ** 2 / NL, rel=1e-6)
    assert math.exp(cc.ln_p_central(NL)) == pytest.approx(math.sqrt(2 / (math.pi * NL)), rel=1e-12)


# ---------------------------------------------------------------------------
# Part 2 team parameters and results for this team (N = 104)
# ---------------------------------------------------------------------------
def test_team_parameters_for_this_team():
    values, hm, m, n = team_parameters(["91", "01", "16", "29"], "round")
    assert values == [91, 1, 16, 29]
    assert hm == pytest.approx(float(4 / (Fraction(1, 91) + 1 + Fraction(1, 16) + Fraction(1, 29))))
    assert (m, n) == (4, 104)


def test_team_parameters_rounding_modes():
    assert team_parameters([91, 1, 16, 29], "floor")[2:] == (3, 103)
    assert team_parameters([91, 1, 16, 29], "ceil")[2:] == (4, 104)


def test_round_half_up():
    # Harmonic mean of [1, 2, 4, 8, 8] is 5 / (1 + 1/2 + 1/4 + 1/8 + 1/8)
    # = 5 / 2 = 2.5 EXACTLY (all reciprocals are exact in binary).
    # Rounding half up gives 3, but Python's built-in round(2.5) = 2.
    # The program must round halves up.
    values, hm, m, n = team_parameters([1, 2, 4, 8, 8], "round")
    assert hm == 2.5
    assert round(hm) == 2          # documents the Python behaviour we avoid
    assert m == 3


@pytest.mark.parametrize("bad", [["00", "12"], ["123"], ["ab"], []])
def test_team_parameters_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        team_parameters(bad, "round")


def test_part2_probabilities_N104_exact():
    N, two_n = 104, 2 ** 104
    om = cc.omega_row_exact(N)
    # Independent exact values with fractions:
    p30 = Fraction(math.comb(N, 30), two_n)
    p52 = Fraction(math.comb(N, 52), two_n)
    window10 = Fraction(sum(math.comb(N, k) for k in range(42, 63)), two_n)
    assert om[30] / two_n == pytest.approx(float(p30), rel=1e-15)
    assert om[52] / two_n == pytest.approx(float(p52), rel=1e-15)
    assert sum(om[42:63]) / two_n == pytest.approx(float(window10), rel=1e-15)
    # Most probable macrostate is N/2 = 52 for even N.
    assert max(range(N + 1), key=lambda k: om[k]) == 52
