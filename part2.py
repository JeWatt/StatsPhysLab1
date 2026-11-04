"""
part2.py - Part 2 of the coin microstate assignment
===================================================

PROBABILITIES, ENTROPY AND THERMODYNAMICS FOR N = 100 + M COINS

    a)-d) probabilities of single macrostates and of windows around N/2
    e)-g) entropy differences, S = kB ln(Omega)
    h)-j) waiting time for 8 heads, likelihood ratio, Second Law
    k)    the same questions for N ~ 1e23 (N -> infinity)

M is the harmonic mean of the last two digits of each team member's NIA
(rounded to an integer), so N = 100 + M < 200. For this team
(91, 01, 16, 29): harmonic mean = 3.610 -> M = 4 -> N = 104.

HOW TO RUN
    No input is needed. The team's NIA digits and all other settings are in
    settings.py. Then either
      - VS Code: open this file and press the Run button, or
      - terminal: python part2.py
    run_all.py runs part1.py and this program one after the other.

OUTPUT (folder "results" next to this file)
    part2_log.txt                     full console log (for the report)
    part2_probabilities_entropy.png   P_k and S(k) for the team's N
    part2_results_N<N>.csv            summary table with units and uncertainties
"""

from __future__ import annotations

import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import settings
from coin_common import (AGE_UNIVERSE_S, KB, SECONDS_PER_YEAR, banner,
                         get_pyplot, ln_binom_small_k, ln_p_central,
                         ln_ratio_central, omega_row_exact, save_fig,
                         start_logging, stop_logging)

# Relative error of one IEEE double rounding, 2^-53. Every probability and
# ratio below is an exact integer ratio rounded ONCE to a float, so this is
# its numerical uncertainty.
FLOAT_ROUNDING = 2.0 ** -53


# ===========================================================================
# TEAM PARAMETER M AND N
# ---------------------------------------------------------------------------
# Harmonic mean H = n / sum(1/x_i). statistics.harmonic_mean() from the
# standard library would also work, but it silently returns 0 when any
# value is 0; computing it by hand lets us stop with a clear message.
#
# The NIA digits come from settings.py as text ("01") or numbers (1); both
# are accepted and checked to be two-digit values 1..99.
#
# Note on this team's values (91, 01, 16, 29): the harmonic mean is pulled
# strongly towards the SMALLEST value, so the "01" dominates and H = 3.610
# (the arithmetic mean would be 34.25). This is a property of the harmonic
# mean, and it is how the assignment defines M.
#
# ROUNDING - a Python trap worth knowing:
#   Python's built-in round() uses "banker's rounding" (round half to
#   even): round(41.5) = 42 but round(42.5) = 42. The usual convention,
#   used here, rounds halves UP (42.5 -> 43), so "round" is implemented
#   as floor(x + 0.5) (x > 0).
# ===========================================================================
def team_parameters(nia_last2, rounding: str) -> tuple[list[int], float, int, int]:
    """Return (values, harmonic mean, M, N) from the NIA endings.

    Raises ValueError with an explanation if the input cannot be used.
    """
    values = []
    for v in nia_last2:
        text = str(v).strip()
        if not text.isdigit() or len(text) > 2:
            raise ValueError(f"NIA ending {v!r} is not a one- or two-digit number (check settings.py).")
        values.append(int(text))
    if not values:
        raise ValueError("NIA_LAST2 in settings.py is empty.")
    if any(v == 0 for v in values):
        raise ValueError("A two-digit ending of 00 makes the harmonic mean undefined; "
                         "agree a convention with the instructor.")
    hm = len(values) / sum(1 / v for v in values)
    if rounding == "round":
        m = math.floor(hm + 0.5)
    elif rounding == "floor":
        m = math.floor(hm)
    elif rounding == "ceil":
        m = math.ceil(hm)
    else:
        raise ValueError(f"M_ROUNDING must be 'round', 'floor' or 'ceil', not {rounding!r}.")
    n = 100 + m
    if n >= 200:
        raise ValueError(f"N = {n} must be below 200 (check the NIA values).")
    return values, hm, m, n


# ===========================================================================
# PART 2a-j - exact calculation for N < 200
# ---------------------------------------------------------------------------
# CHOSEN LOGIC
#   For N < 200 the whole row Omega(N, 0..N) is computed ONCE as exact
#   Python integers, and every requested quantity is a lookup or a sum:
#       P_k = Omega(N, k) / 2^N         (int / int: correctly rounded float)
#       Delta P = (sum of Omega over the window) / 2^N   (sum is exact)
#   Doing the sums on exact integers BEFORE dividing means the only error in
#   any printed probability is one final rounding, < 1.1e-16 relative.
#   (Building the row in floats instead gives relative errors up to
#   ~N*eps; Part 1c prints the measured value.)
#   ALTERNATIVE: scipy.stats.binom(N, 0.5).pmf(k) and .cdf(k); convenient,
#   but the cdf difference loses accuracy when Delta P is close to 1.
#
# ODD N (not the case for this team, N = 104, but handled for safety)
#   If N is odd, N/2 is not an integer; floor(N/2) and ceil(N/2) are then
#   equally probable since Omega(N, k) = Omega(N, N-k). The program uses
#   kc = floor(N/2) as "N/2". Windows "N/2 +- 10" are taken as
#   ceil(N/2 - 10) ... floor(N/2 + 10), keeping them symmetric (20 values
#   for odd N, 21 for even N).
# ===========================================================================
def part2_exact(nia_last2, rounding: str, out_dir: Path | None, plt) -> dict:
    banner("PART 2: team parameters")
    values, hm, M, N = team_parameters(nia_last2, rounding)
    print(f"NIA endings: {', '.join(f'{v:02d}' for v in values)}")
    print(f"Harmonic mean = {len(values)} / ({' + '.join(f'1/{v}' for v in values)}) = {hm:.4f}")
    print(f"M = {M} ({rounding}) -> N = 100 + M = {N}")

    om = omega_row_exact(N)                      # om[k] = Omega(N, k), exact
    two_n = 2 ** N                               # exact integer
    P = [o / two_n for o in om]                  # P_k as floats; sum = 1
    ln_om = [math.log(o) for o in om]            # S(k)/kB = ln Omega, natural log
    kc = N // 2                                  # "N/2" (floor for odd N)
    if N % 2:
        print(f'N is odd: N/2 is not an integer. "N/2 heads" is taken as k = {kc} '
              f"(P({kc}) = P({kc + 1})).")
    print(f"Numerical uncertainty of every probability and ratio below: relative < {FLOAT_ROUNDING:.1e}")
    print("(exact integer arithmetic, one final rounding to a double).")

    # --- 2a-d: probabilities -----------------------------------------------
    #  a) single unlikely macrostate (k = 30, far in the tail for N ~ 100)
    #  b) the most probable macrostate; ~ sqrt(2/(pi N)) for large N
    #  c) window of +-10 HEADS around N/2
    #  d) window of +-20 TAILS around N/2. Since Omega(N, k) = Omega(N, N-k),
    #     "t tails" is the same macrostate as "N - t heads"; the tails are
    #     converted explicitly (hk = N - t) to make the symmetry visible.
    banner("PART 2a-d: probabilities")
    a = P[30]
    b = P[kc]
    k1, k2 = math.ceil(N / 2 - 10), math.floor(N / 2 + 10)
    c = sum(om[k1:k2 + 1]) / two_n
    t1, t2 = math.ceil(N / 2 - 20), math.floor(N / 2 + 20)     # range given in TAILS
    hk = [N - t for t in range(t1, t2 + 1)]                     # corresponding heads
    d = sum(om[h] for h in hk) / two_n
    print(f"a) P(k = 30)                         = {a:.6e}")
    print(f"b) P(k = N/2 = {kc})                   = {b:.6e}")
    print(f"c) sum P_k, k = {k1}..{k2} heads         = {c:.6f}")
    print(f"d) sum P_k, {t1}..{t2} tails (={min(hk)}..{max(hk)} heads) = {d:.6f}")
    print("   (heads/tails symmetry: Omega(N,k) = Omega(N,N-k))")
    sigma = math.sqrt(N) / 2
    print(f"   For scale: sigma = sqrt(N)/2 = {sigma:.3f} heads, so +-10 = {10 / sigma:.2f} sigma "
          f"and +-20 = {20 / sigma:.2f} sigma.")

    # --- 2e-g: entropy changes ---------------------------------------------
    #  Boltzmann: S = kB ln(Omega)  =>  Delta S = S(k1) - S(k2)
    #                                         = kB ln[Omega(k1)/Omega(k2)].
    #  The ratio Omega(k1)/Omega(k2) is formed by exact integer division and
    #  only then passed to log, so no precision is lost.
    #  Sign convention of the handout: Delta S = S(k1) - S(k2), so e) and f)
    #  are NEGATIVE and g) is POSITIVE. Units: kB and J/K.
    #  "log" in the handout's S = kB log Omega is the NATURAL logarithm
    #  (the physics convention); log10 would give values 2.303x smaller.
    banner("PART 2e-g: entropy changes  dS = S(k1) - S(k2) = kB ln[Omega(k1)/Omega(k2)]")
    pairs = [(kc + 25, kc),     # e) high disorder vs maximum disorder
             (kc + 10, kc),     # f) moderate disorder vs maximum disorder
             (kc, 20)]          # g) maximum disorder vs high order
    tags = ["e", "f", "g"]
    ds_kb = []
    for tag, (q1, q2) in zip(tags, pairs):
        ds = math.log(om[q1] / om[q2])
        ds_kb.append(ds)
        print(f"{tag}) k1 = {q1:3d}, k2 = {q2:3d} : dS = {ds:+9.4f} kB = {KB * ds:+.4e} J/K   "
              f"[Omega ratio = {om[q1] / om[q2]:.4e}]")

    # --- 2h: waiting time for exactly 8 heads ------------------------------
    #  Interpretation: "a tossing rate of N coins per second" = the whole
    #  system is re-tossed once per second (one macrostate sample per s).
    #  Trials until first success are GEOMETRIC with p = P_8:
    #      mean <tau> = 1/p seconds,  std. dev. = sqrt(1 - p)/p  (~ <tau>)
    #  The std. dev. is the statistical uncertainty of the waiting time.
    #  ALTERNATIVES: "8 heads or fewer" (sum P_0..P_8, also printed) or
    #  "one coin per second" (multiply <tau> by N).
    banner("PART 2h-j: thermodynamic implications")
    p8 = P[8]
    tau = two_n / om[8]                          # exact division, s
    sd_tau = math.sqrt(1 - p8) / p8
    print(f"h) P(exactly 8 heads) = {p8:.4e}")
    print(f"   mean waiting time  = {tau:.4e} s (std. dev. {sd_tau:.4e} s) = "
          f"{tau / SECONDS_PER_YEAR:.4e} yr = {tau / AGE_UNIVERSE_S:.4e} x age of Universe")
    print(f'   (for "8 heads or fewer": {two_n / sum(om[:9]):.4e} s)')

    # --- 2i: likelihood ratio ----------------------------------------------
    #  The handout text ("M?+-10") is garbled; it is read as N/2 +- 10 heads,
    #  the same window as question 2c. The 2^N cancels in the ratio:
    #      P(N/2) / P(k) = Omega(N/2) / Omega(k)   (exact integer division)
    #  By symmetry both sides give the same ratio for even N.
    #  (The literal reading "M +- 10" would be 4 +- 10 for this team, and
    #  -6 heads does not exist, so it is not used.)
    r_lo = om[kc] / om[kc - 10]
    r_hi = om[kc] / om[kc + 10]
    print(f"i) P(N/2)/P(N/2 - 10 = {kc - 10}) = {r_lo:.4f},  "
          f"P(N/2)/P(N/2 + 10 = {kc + 10}) = {r_hi:.4f}")
    print(f"   (Gaussian estimate exp(2*10^2/N) = {math.exp(200 / N):.4f})")

    # --- 2j: Second Law ----------------------------------------------------
    #  P(k1)/P(k2) = Omega(k1)/Omega(k2) = exp(Delta S / kB). Lower-entropy
    #  macrostates are not forbidden, only exponentially improbable; for
    #  N ~ 100 such fluctuations are observable, so nothing here contradicts
    #  the Second Law, which becomes certainty only for macroscopic N (2k).
    print("j) No result contradicts the Second Law. Low-entropy macrostates are not forbidden,\n"
          "   only improbable: P(k1)/P(k2) = exp[dS/kB]. For this N:")
    for tag, ds in zip(tags, ds_kb):
        print(f"   {tag}) probability ratio exp(dS/kB) = {math.exp(ds):.4e}")
    print("   The Second Law is a statistical statement that becomes certain only as N -> inf (Part 2k).")

    # --- Figure ------------------------------------------------------------
    #  Left: P_k on a log axis (otherwise k = 8 and k = 30 would sit at 0),
    #  the +-10 window in blue and the special macrostates in red.
    #  Right: S/kB = ln Omega with the three pairs of e)-g).
    k = np.arange(N + 1)
    Pa = np.array(P)
    lna = np.array(ln_om)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.semilogy(k, Pa, "k.-")
    ax1.semilogy(np.arange(k1, k2 + 1), Pa[k1:k2 + 1], "b.", ms=10)
    for kk in (30, 8, kc):
        ax1.semilogy(kk, Pa[kk], "ro", mfc="r")
    ax1.annotate("  k = 30", (30, Pa[30]))
    ax1.annotate("  k = 8", (8, Pa[8]))
    ax1.set_xlim(0, N)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xlabel("k (heads)")
    ax1.set_ylabel(r"$P_k$  (dimensionless)")
    ax1.set_title(rf"N = {N} (blue: N/2 $\pm$ 10)")
    ax2.plot(k, lna, "k-")
    flat = [q for pair in pairs for q in pair]
    ax2.plot(flat, lna[flat], "ro", mfc="r")
    ax2.set_xlim(0, N)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("k (heads)")
    ax2.set_ylabel(r"$S / k_B = \ln \Omega$  (dimensionless)")
    ax2.set_title("Entropy of each macrostate")
    fig.tight_layout()
    save_fig(fig, out_dir, "part2_probabilities_entropy")

    return dict(M=M, N=N, kc=kc, a=a, b=b, c=c, d=d, ds_kb=ds_kb,
                tau=tau, sd_tau=sd_tau, r_lo=r_lo, r_hi=r_hi)


# ===========================================================================
# PART 2k - the same questions for N ~ 1e23 (N -> infinity)
# ---------------------------------------------------------------------------
# WHY NEW FORMULAS ARE NEEDED
#   math.comb(10**23, 10**23 // 2) would have ~3e22 digits: impossible.
#   Floats overflow (Part 1d), and lgamma(1e23) ~ 5e24 has an absolute error
#   ~1e9, so differences of lgamma values (ln of ratios ~ 1e-20) are noise.
#   Every quantity is therefore rewritten so that only small, exactly
#   representable numbers are subtracted (see coin_common.py):
#     (1) ln P(N/2) = -1/2 ln(pi N/2) - 1/(4N)                 (Stirling)
#     (2) ln[Omega(N/2+d)/Omega(N/2)] = sum log1p((1-2j)/(N/2+j))  (exact)
#     (3) ln Omega(N, k) = sum ln(N-j) - ln k!,  k << N             (exact)
#   They are VALIDATED at N = 200 against exact integers first; the
#   relative errors printed are the uncertainty of the method.
#
# ALTERNATIVES
#   - Gaussian approximation everywhere: good near the peak, WRONG in the
#     far tail (k = 8, 20, 30), where formula (3) is needed.
#   - Large-deviation form ln Omega ~ N H(x) - 1/2 ln(2 pi N x(1-x)),
#     H(x) = -x ln x - (1-x) ln(1-x)   (used in j).
#   - mpmath (arbitrary-precision floats): mpmath.loggamma with ~40 digits
#     evaluates the gamma-function expressions at N = 1e23 directly; a good
#     independent cross-check (conda install mpmath).
# ===========================================================================
def part2k(n_large: float) -> None:
    banner("PART 2k: validation of large-N formulas at N = 200")
    nv = 200
    cv = nv // 2
    ov = omega_row_exact(nv)
    checks = [(math.log(ov[cv] / 2 ** nv), ln_p_central(nv)),          # (1) ln P(N/2)
              (math.log(ov[cv + 25] / ov[cv]), ln_ratio_central(nv, 25)),  # (2) ratio d = 25
              (math.log(ov[30]), ln_binom_small_k(nv, 30))]            # (3) ln Omega(N,30)
    for exact, formula in checks:
        print(f"  exact {exact:+.10f}   formula {formula:+.10f}   "
              f"rel.err {abs(exact - formula) / abs(exact):.1e}")

    banner(f"PART 2k: results for N = {n_large:.0e}")
    NL = n_large
    ln2 = math.log(2)
    ln10 = math.log(10)
    ln_pc = ln_p_central(NL)                       # ln P(N/2)

    def lr(d: int) -> float:                       # ln[P(N/2+d)/P(N/2)]
        return ln_ratio_central(NL, d)

    # a) P(30): the N ln2 term dominates, P ~ 10^(-3e22).
    print(f"a) log10 P(k=30)   = {(ln_binom_small_k(NL, 30) - NL * ln2) / ln10:.6e}   (P ~ 10^-(3e22): never)")
    # b) P(N/2) -> sqrt(2/(pi N)) -> 0: even the most probable macrostate is
    #    itself improbable; probability is spread over ~sigma states.
    print(f"b) P(k = N/2)      = {math.exp(ln_pc):.4e}   (~ sqrt(2/(pi N)) -> 0)")
    # c, d) window sums = P(N/2) x sum of ratios (each 1 - O(1e-20)), i.e.
    #    21 P(N/2) and 41 P(N/2). Fixed-width windows are tiny compared with
    #    sigma ~ 1.6e11; windows of a few sigma hold essentially everything.
    sum10 = sum(math.exp(lr(d)) for d in range(0, 11)) + sum(math.exp(lr(d)) for d in range(1, 11))
    sum20 = sum(math.exp(lr(d)) for d in range(0, 21)) + sum(math.exp(lr(d)) for d in range(1, 21))
    print(f"c) dP(N/2 +- 10)   = {math.exp(ln_pc) * sum10:.4e}   "
          f"(= 21 P(N/2): window << sigma = {math.sqrt(NL) / 2:.2e})")
    print(f"d) dP(N/2 +- 20)   = {math.exp(ln_pc) * sum20:.4e}   (= 41 P(N/2))")
    print(f"   but P(|k-N/2| < 3 sigma) = {math.erf(3 / math.sqrt(2)):.6f} -> 1: "
          'a fixed +-10 window is no longer "macroscopic".')
    # e, f) Delta S ~ -2 d^2/N kB -> 0: N/2+25 and N/2+10 are the SAME
    #    macroscopic state as N/2.  g) Delta S ~ N ln2 kB ~ 0.96 J/K.
    #    Note: ln_oc = ln_pc + N ln2 adds ~ -27 to ~ 7e22; the -27 is below
    #    float resolution there and is lost (< 1e-15 relative effect).
    ds_e, ds_f = lr(25), lr(10)
    ln_oc = ln_pc + NL * ln2                       # ln Omega(N, N/2)
    ds_g = ln_oc - ln_binom_small_k(NL, 20)
    print(f"e) dS = {ds_e:.4e} kB = {KB * ds_e:.4e} J/K   (N/2+25 indistinguishable from N/2)")
    print(f"f) dS = {ds_f:.4e} kB = {KB * ds_f:.4e} J/K")
    print(f"g) dS = {ds_g:.4e} kB = {KB * ds_g:.4f} J/K   (macroscopic, ~ N kB ln2)")
    # h) <tau> = 2^N / Omega(N, 8) s, printed as log10 (~10^(3e22) s).
    log10_tau = (NL * ln2 - ln_binom_small_k(NL, 8)) / ln10
    print(f"h) waiting time for 8 heads ~ 10^({log10_tau:.4e}) s  "
          f"vs age of Universe ~ 10^{math.log10(AGE_UNIVERSE_S):.1f} s")
    # i) ratio -> 1, printed as ratio - 1 with expm1 (1 + 2e-21 == 1.0).
    print(f"i) P(N/2)/P(N/2+-10) - 1 = {math.expm1(-lr(10)):.4e}   (ratio -> 1)")
    # j) A macroscopically different state (51 % heads) has
    #    Delta S/kB ~ N [H(0.51) - ln2] ~ -2e19: relative probability
    #    exp(-2e19), i.e. zero for every practical purpose.
    xdev = 0.51
    hx = -xdev * math.log(xdev) - (1 - xdev) * math.log(1 - xdev)    # binary entropy (nats)
    ds_mac = NL * (hx - ln2)
    print(f"j) A macroscopically different state (k = 0.51 N) has dS ~ {ds_mac:.3e} kB = "
          f"{KB * ds_mac:.3e} J/K,")
    print(f"   i.e. relative probability exp({ds_mac:.2e}): the Second Law becomes effectively exact.")


def write_summary(res: dict, out_dir: Path) -> None:
    """Summary table with units and uncertainties for the report.

    The CSV opens directly in Excel. The "uncertainty" column states the
    numerical uncertainty (float rounding) and, for the waiting time, the
    statistical spread of the geometric distribution.
    ALTERNATIVE: pandas.DataFrame(rows).to_csv(...) if pandas is installed.
    """
    kc = res["kc"]
    rel = f"relative < {FLOAT_ROUNDING:.1e} (rounding)"
    rel_log = "absolute < 1e-15 (rounding)"
    rows = [("2a", "P(k=30)", res["a"], "-", rel),
            ("2b", f"P(k=N/2={kc})", res["b"], "-", rel),
            ("2c", "dP(N/2+-10 heads)", res["c"], "-", rel),
            ("2d", "dP(N/2+-20 tails)", res["d"], "-", rel),
            ("2e", "dS(N/2+25 vs N/2)", KB * res["ds_kb"][0], "J/K", rel_log + " in units of kB"),
            ("2f", "dS(N/2+10 vs N/2)", KB * res["ds_kb"][1], "J/K", rel_log + " in units of kB"),
            ("2g", "dS(N/2 vs 20)", KB * res["ds_kb"][2], "J/K", rel_log + " in units of kB"),
            ("2h", "mean wait for 8 heads", res["tau"], "s",
             f"std. dev. {res['sd_tau']:.4e} s (geometric distribution)"),
            ("2i", f"P(N/2)/P(N/2-10={kc - 10})", res["r_lo"], "-", rel),
            ("2i", f"P(N/2)/P(N/2+10={kc + 10})", res["r_hi"], "-", rel)]
    with open(out_dir / f"part2_results_N{res['N']}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "quantity", "value", "units", "uncertainty"])
        for item, q, v, u, unc in rows:
            w.writerow([item, q, f"{v:.10e}", u, unc])


# ===========================================================================
# MAIN
# ---------------------------------------------------------------------------
# main() reads everything from settings.py; nothing is typed in.
# `show` lets run_all.py postpone opening the figure windows until both
# parts have finished.
# ===========================================================================
def main(show: bool | None = None) -> None:
    if show is None:
        show = settings.SHOW_FIGURES
    out_dir = settings.OUTPUT_DIR if settings.SAVE_OUTPUTS else None
    plt = get_pyplot(show or settings.SHOW_FIGURES)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        start_logging(out_dir / "part2_log.txt")
    try:
        print(f"Coin microstate analysis, Part 2  -  {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")
        print(f"Settings: NIA_LAST2 = {settings.NIA_LAST2}, M_ROUNDING = {settings.M_ROUNDING!r}, "
              f"LARGE_N = {settings.LARGE_N:g}")
        res = part2_exact(settings.NIA_LAST2, settings.M_ROUNDING, out_dir, plt)
        part2k(settings.LARGE_N)
        if out_dir is not None:
            write_summary(res, out_dir)
            print(f"\nOutputs saved in folder \"{out_dir}\".")
    finally:
        stop_logging()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
