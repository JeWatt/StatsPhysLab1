"""
part1.py - Part 1 of the coin microstate assignment
===================================================

MICROSTATES AND MACROSTATES OF N UNBIASED COINS (two-state system)

    a) list every microstate for N = 7 (like Table 1 of the handout)
    b) largest N that can be fully enumerated (time and memory limits)
    c) Omega(N, k) for N = 20, 50, 80, 100 in a single plot
    d) largest N for which the distribution can be computed / plotted
    e) behaviour of the distribution as N -> 1e23 (thermodynamic limit)

HOW TO RUN
    No input is needed. All settings are in settings.py (benchmark on/off,
    output folder, show figures). Then either
      - VS Code: open this file and press the Run button, or
      - terminal: python part1.py
    run_all.py runs this program and part2.py one after the other.

OUTPUT (folder "results" next to this file)
    part1_log.txt                 full console log (for the report)
    part1a_N7_microstates.txt     complete N = 7 listing (appendix)
    part1b_timing.png             enumeration time vs N (if RUN_BENCHMARK)
    part1c_distributions.png      the "single plot" of part 1c
    part1c_Omega_table.csv        exact Omega(N, k) table, N = 20..100
    part1e_sharpening.png         normalised peaks, N = 1e2..1e6
"""

from __future__ import annotations

import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import settings
from coin_common import (available_ram_bytes, banner, enumerate_macrostates,
                         get_pyplot, hbytes, ln_binom, omega_row_exact,
                         pascal_row_float, save_fig, start_logging,
                         stop_logging, wrap_list)


# ===========================================================================
# PART 1a - Full enumeration for N = 7
# ---------------------------------------------------------------------------
# CHOSEN LOGIC
#   There is a one-to-one map between the 2^N microstates and the integers
#   0 ... 2^N - 1: write the integer in binary with N digits and read each
#   bit as one coin (1 = head, 0 = tail). format(x, "07b") gives the binary
#   string; replacing '1' -> 'H' and '0' -> 'T' gives the microstate. The
#   number of '1's is the macrostate k. Grouping by k reproduces Table 1.
#   Within each macrostate the microstates appear in increasing order of x.
#
# ALTERNATIVES
#   - itertools.product("HT", repeat=7): yields the 128 microstates as
#     tuples directly - the most "Pythonic" one-liner.
#   - itertools.combinations(range(7), k): lists the head POSITIONS for each
#     microstate with k heads, so the table is built already grouped by k
#     and Omega(7, k) = C(7, k) is visible in the code itself.
#   - Recursion: list(N) = [s + 'H' for s in list(N-1)] + [s + 'T' ...],
#     mirroring Omega(N, k) = Omega(N-1, k-1) + Omega(N-1, k).
#   RECOMMENDATION: the binary-integer version makes the "microstate =
#   integer" idea explicit, which Part 1b reuses.
# ===========================================================================
def part1a(out_dir: Path | None) -> None:
    banner("PART 1a: enumeration of all microstates for N = 7")
    N = 7
    micro = [format(x, f"0{N}b").replace("1", "H").replace("0", "T") for x in range(2 ** N)]
    n_heads = [m.count("H") for m in micro]

    omega7 = [0] * (N + 1)                    # omega7[k] = Omega(7, k)
    lines_out = []                            # full listing for the appendix file
    print(f"{'macrostate':<22} {'Omega':<6}  microstates")
    for k in range(N, -1, -1):                # descending k, as in Table 1
        group = [m for m, h in zip(micro, n_heads) if h == k]
        omega7[k] = len(group)                # multiplicity = how many
        label = f"{k} {'head' if k == 1 else 'heads'}, {N - k} {'tail' if N - k == 1 else 'tails'}"
        lines = wrap_list(group, 8)           # console: 8 microstates per line
        print(f"{label:<22} {omega7[k]:<6d}  {lines[0]}")
        for extra in lines[1:]:
            print(f"{'':<22} {'':<6}  {extra}")
        lines_out.append(f"{label}  (Omega = {omega7[k]})\n  {', '.join(group)}\n")

    if out_dir is not None:
        (out_dir / "part1a_N7_microstates.txt").write_text("\n".join(lines_out), encoding="utf-8")

    # Two independent checks: the multiplicities add up to 2^N, and equal the
    # binomial coefficients (row 7 of Tartaglia's triangle).
    print(f"Total microstates = {sum(omega7)} (2^N = {2 ** N}).  "
          f"Matches binomial coefficients: {omega7 == omega_row_exact(N)}")


# ===========================================================================
# PART 1b - Largest N for full enumeration (time and memory limits)
# ---------------------------------------------------------------------------
# CHOSEN LOGIC ("streaming" enumeration, see coin_common.enumerate_macrostates)
#   Every microstate x = 0 ... 2^N - 1 is generated, decoded (heads =
#   number of 1-bits = popcount) and added to the tally of its macrostate.
#   Chunks of 2^22 microstates are processed with NumPy and discarded, so
#   memory is constant: the limit is TIME, which doubles per extra coin.
#   N is increased until doubling the last run time would exceed the budget.
#   The benchmark is switched on/off with RUN_BENCHMARK in settings.py.
#
# WHY NOT THE STRING LIST OF 1a?
#   A Python string list costs ~(49 + N) bytes per microstate plus an 8-byte
#   pointer: ~2.5 GB at N = 25 and ~80 GB at N = 30. Storing the list is
#   limited by MEMORY long before time; the table printed afterwards
#   quantifies this (2 bytes / 1 byte / 1 bit per coin).
#
# ALTERNATIVES (to push N higher)
#   - Numba (@njit(parallel=True) with prange): compiled loop on all cores,
#     typically 10-50x faster than NumPy here -> +3 to +5 in N.
#   - CuPy on an NVIDIA GPU: same NumPy-style code on the GPU.
#   - multiprocessing.Pool over chunks: ~(number of cores)x, standard library.
#   - Gray-code order: consecutive microstates differ by one coin, so k
#     changes by +-1 per step. Only pays off in compiled code (Numba/C).
#   - "Split" trick: popcount(x) = popcount(low bits) + popcount(chunk
#     index), so every chunk is a shifted copy of one histogram. Very fast,
#     but it no longer decodes each microstate (arguably not enumeration).
#
# UNCERTAINTY OF THE TIMINGS
#   time.perf_counter measures wall-clock time; background load gives a few
#   percent scatter. ALTERNATIVE: repeat each N 3 times and report the mean
#   and standard deviation (or use the timeit module for short runs).
# ===========================================================================
def part1b(run_benchmark: bool, budget: float, start_n: int,
           out_dir: Path | None, plt) -> None:
    banner("PART 1b: maximum N for full enumeration")
    if run_benchmark:
        engine = "np.bitwise_count (hardware POPCNT)" if hasattr(np, "bitwise_count") \
            else "16-bit look-up table (NumPy < 2.0)"
        print(f"Streaming enumeration with NumPy, popcount via {engine}.")
        print(f"Budget {budget:.0f} s per run.")
        print(f"{'N':>4} {'2^N':>16} {'time (s)':>12} {'microstates/s':>16} {'check':>8}")
        bench_n, bench_t = [], []
        N = start_n
        while True:
            counts, t = enumerate_macrostates(N)
            # Correctness check: brute force must reproduce C(N, k) exactly.
            ok = counts.tolist() == omega_row_exact(N)
            bench_n.append(N)
            bench_t.append(t)
            print(f"{N:>4d} {2 ** N:>16d} {t:>12.3f} {2 ** N / t:>16.4e} {'OK' if ok else 'FAIL':>8}")
            # Stop rule: the next N costs ~2x; with a 5 % margin for scatter,
            # stop if 2.1 x this run would exceed the budget.
            if 2.1 * t > budget:
                break
            N += 1
        n_max = bench_n[-1]

        # Scaling law: least-squares fit log2(t) = a*N + b. A slope a ~ 1
        # confirms t ~ 2^N. Runs shorter than 0.05 s are dominated by
        # overheads and left out of the fit.
        bn, bt = np.array(bench_n), np.array(bench_t)
        use = bt > 0.05
        if use.sum() < 2:
            use[:] = True
        a, b = np.polyfit(bn[use], np.log2(bt[use]), 1)
        print(f"\nLargest N fully enumerated within {budget:.0f} s: N = {n_max} (t = {bt[-1]:.1f} s)")
        print(f"Fit: t ~ 2^({a:.3f}*N {b:+.2f}) s  (slope 1 => cost doubles per extra coin)")
        for dn in (1, 2, 3):
            print(f"  predicted time for N = {n_max + dn} : {2 ** (a * (n_max + dn) + b):.0f} s")
        print(f"  predicted time for N = 64 : {2 ** (a * 64 + b) / 3.156e7:.1e} years")

        fig, ax = plt.subplots()
        ax.semilogy(bn, bt, "o-", lw=1.2, label="measured")     # log y: exponential -> straight line
        nn = np.arange(bn[0], bn[-1] + 4)
        ax.semilogy(nn, 2.0 ** (a * nn + b), "--", label="fit $2^{aN+b}$")
        ax.axhline(budget, color="r", ls=":", label=f"time budget ({budget:.0f} s)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("N (coins)")
        ax.set_ylabel("wall-clock time (s)")
        ax.set_title("Full enumeration time vs N")
        ax.legend(loc="upper left")
        save_fig(fig, out_dir, "part1b_timing")
    else:
        print("Benchmark skipped (set RUN_BENCHMARK = True in settings.py to include it).")

    # --- Memory needed to STORE the list of all microstates ----------------
    # Storage formats, from most to least wasteful:
    #   2 bytes per coin (UTF-16 characters)            : 2 N * 2^N bytes
    #   1 byte per coin (UTF-8 / NumPy 'S' array)       : N * 2^N bytes
    #   bit-packed (store the integer itself)           : N * 2^N / 8 bytes
    # Even bit-packed, each extra coin roughly doubles the memory.
    ram = available_ram_bytes()
    print("\nMemory needed to store all 2^N microstates:")
    print(f"{'N':>4} {'2 B/coin':>14} {'1 B/coin':>14} {'1 bit/coin':>14}")
    for N in (20, 24, 28, 30, 32, 34, 36, 40):
        print(f"{N:>4d} {hbytes(2 * N * 2 ** N):>14} {hbytes(N * 2 ** N):>14} {hbytes(N * 2 ** N / 8):>14}")
    if not math.isnan(ram):
        print(f"Detected RAM: {hbytes(ram)}")

        def largest(bytes_per_coin: float) -> int:
            return max(n for n in range(1, 61) if bytes_per_coin * n * 2 ** n <= ram)

        print(f"Largest storable N:  2 B/coin {largest(2)} | 1 B/coin {largest(1)} "
              f"| bit-packed {largest(1 / 8)}")
    else:
        print("RAM size could not be detected; compare the table with your machine.")


# ===========================================================================
# PART 1c - Omega(N, k) for N = 20, 50, 80, 100
# ---------------------------------------------------------------------------
# CHOSEN LOGIC
#   Enumeration is impossible here (2^100 ~ 1.3e30 states). Python's exact
#   integers make Omega(N, k) = math.comb(N, k) EXACT for every N, so the
#   table written to CSV contains the full integers (all 30 digits for
#   N = 100). For comparison the same triangle is also built in double-
#   precision floats (float64) and its largest relative error is printed:
#   this is the rounding error a floating-point calculation would carry.
#
# PLOT CHOICE: one panel, logarithmic Omega axis
#   Omega spans 1.8e5 (N = 20) to 1.0e29 (N = 100). On a LINEAR axis like
#   Figure 2 of the handout, the N = 20, 50 and 80 curves would be squashed
#   flat against zero by the N = 100 peak. A logarithmic y-axis keeps all
#   four curves visible in a single plot and shows the ~2^N growth directly.
#   On a log axis the peaks look like rounded arches rather than bells,
#   because ln Omega is close to a parabola near k = N/2.
#   ALTERNATIVES: plot P_k = Omega/2^N on a linear axis (four bells of
#   comparable height), or Omega/Omega_max against k/N (shows the relative
#   width shrinking like 1/sqrt(N), done in Part 1e).
# ===========================================================================
def part1c(out_dir: Path | None, plt) -> None:
    banner("PART 1c: macrostate distributions")
    ns = [20, 50, 80, 100]
    table = {}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for N in ns:
        om = omega_row_exact(N)                      # exact integers
        k = np.arange(N + 1)
        # Check: sum_k Omega(N, k) = 2^N (binomial theorem). Exact here.
        k_max = max(range(N + 1), key=lambda i: om[i])
        flt = pascal_row_float(N)                    # float64 version
        rel_err = max(abs(int(f) - o) / o for f, o in zip(flt, om))
        print(f"N = {N:3d} : sum Omega = 2^N exactly: {sum(om) == 2 ** N}, "
              f"max Omega = {float(om[k_max]):.4e} at k = {k_max}, "
              f"double-precision Pascal rel. error = {rel_err:.1e}")
        table[N] = om
        ax.semilogy(k, np.array(om, dtype=float), "o-", ms=3, lw=1, label=f"N = {N}")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("k (number of heads)")
    ax.set_ylabel(r"$\Omega(N,k)$  (dimensionless, exact integers)")
    ax.set_title(r"Macrostate distribution $\Omega(N,k)$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, out_dir, "part1c_distributions")

    # CSV table (exact integers; empty cell where k > N).
    # ALTERNATIVE: pandas.DataFrame(...).to_csv(...) if pandas is installed.
    if out_dir is not None:
        with open(out_dir / "part1c_Omega_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["k"] + [f"Omega_N{N}" for N in ns])
            for k in range(max(ns) + 1):
                w.writerow([k] + [table[N][k] if k <= N else "" for N in ns])


# ===========================================================================
# PART 1d - Maximum N for the distribution (floating-point limits)
# ---------------------------------------------------------------------------
# CHOSEN LOGIC
#   In DOUBLE PRECISION (float64, the default in NumPy) the limiting
#   factor is the range: the largest float is ~1.8e308. The program finds
#   the limits by direct test:
#     (1) build float rows of Tartaglia's triangle until a value is inf;
#     (2) find the largest N with 2^N finite (2^1024 = inf), where
#         P_k = Omega / 2^N breaks even if Omega itself is still finite.
#   Expected: Omega overflows above N = 1029, 2^N above N = 1023.
#
# PYTHON-SPECIFIC POINT (worth one sentence in the report)
#   With Python's exact integers there is NO overflow: math.comb(10**5, k)
#   works (it just gets slower, ~N digits per number). The real limits
#   then become time and memory, and converting the result to float for
#   plotting (float(Omega) raises OverflowError above ~1.8e308). The
#   program demonstrates this with N = 5000 using exact integer division.
#
# HOW TO GO BEYOND
#   - Logarithms: ln Omega via math.lgamma never overflows; its absolute
#     error ~ eps * N ln N ruins differences near N ~ 1e15.
#   - Scaled recursion P(N,k) = [P(N-1,k-1) + P(N-1,k)] / 2: no overflow,
#     tails underflow to 0 (harmless for plotting).
#   - Gaussian (de Moivre-Laplace) approximation for any N.
#   - Plotting: a screen resolves only ~1e3-1e4 distinct k, so for large N
#     plot k/N or (k - N/2)/sigma instead of every point.
# ===========================================================================
def part1d() -> None:
    banner("PART 1d: maximum N for calculating/plotting the distribution")
    row = np.ones(1)
    n = 0
    with np.errstate(over="ignore", invalid="ignore"):
        while np.all(np.isfinite(row)):
            n += 1
            row = np.concatenate((row, [0.0])) + np.concatenate(([0.0], row))
        n_max_omega = n - 1                                     # last all-finite row
        powers = np.power(2.0, np.arange(1, 1101))              # 2^1024 -> inf
    n_max_2n = int(np.nonzero(np.isfinite(powers))[0][-1]) + 1
    print(f"largest double (sys.float_info.max)   = {sys.float_info.max:.4e}")
    print(f"Largest N with all Omega(N,k) finite = {n_max_omega} "
          f"(central Omega = {pascal_row_float(n_max_omega).max():.4e})")
    print(f"Largest N with 2^N finite            = {n_max_2n}  -> P_k = Omega/2^N fails from N = {n_max_2n + 1}")
    # Exact-integer route: no overflow at all (int / int is correctly rounded).
    N = 5000
    p_centre = math.comb(N, N // 2) / 2 ** N
    print(f"Python exact integers: N = {N}, P(N/2) = C(N,N/2)/2^N = {p_centre:.6e} "
          f"(C(N,N/2) has {len(str(math.comb(N, N // 2)))} digits) - no overflow,")
    print("  only time/memory limits; float(Omega) itself overflows above ~1.8e308.")
    print("Work-around for floats: logarithms, ln Omega = lgamma(N+1)-lgamma(k+1)-lgamma(N-k+1)")
    print("  then the limit is memory for the k-vector (8 B per k): 16 GB -> N ~ 2e9,")
    print("  and lgamma absolute error ~eps*N*lnN, which ruins differences near N ~ 1e15.")
    print("  Plot resolution: only ~1e3-1e4 distinct k fit across a screen.")


# ===========================================================================
# PART 1e - Towards the thermodynamic limit (N ~ 1e23)
# ---------------------------------------------------------------------------
# PHYSICS
#   Binomial distribution with p = 1/2: mean <k> = N/2, standard deviation
#   sigma = sqrt(N)/2, RELATIVE width sigma/<k> = 1/sqrt(N). For N = 1e23,
#   sigma/<k> ~ 3e-12: the fraction of heads is 1/2 to ~12 digits, i.e. the
#   macrostate is effectively deterministic.
#
# CHOSEN LOGIC
#   (1) Plot P_k / P_max against k/N for N = 1e2 ... 1e6 using lgamma.
#   (2) Print <k>, sigma, sigma/<k> up to N = 1e23.
#   (3) P(|k/N - 1/2| > 1e-10) = erfc(z) with z ~ 45, which underflows to 0
#       in math.erfc; the asymptotic form erfc(z) ~ exp(-z^2)/(z sqrt(pi))
#       is evaluated in log10 form instead.
#   ALTERNATIVE: plot P_k * sigma against (k - N/2)/sigma; all curves then
#   collapse onto the same standard Gaussian (shows universality).
# ===========================================================================
def part1e(n_large: float, out_dir: Path | None, plt) -> None:
    banner("PART 1e: extrapolation to N ~ 1e23")
    ns = [1e2, 1e3, 1e4, 1e5, 1e6]
    fig, ax = plt.subplots()
    for N in ns:
        # Window 0.4 <= k/N <= 0.6 with at most 4001 points (more points
        # than pixels add nothing, see 1d).
        k = np.round(np.linspace(0.4 * N, 0.6 * N, int(min(N * 0.2 + 1, 4001))))
        ln_p = ln_binom(N, k) - N * math.log(2)            # ln P_k, never overflows
        ax.plot(k / N, np.exp(ln_p - ln_p.max()), "-", lw=1.2,
                label=f"N = $10^{{{int(round(math.log10(N)))}}}$")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("k/N (fraction of heads, dimensionless)")
    ax.set_ylabel(r"$P_k / P_{max}$  (dimensionless)")
    ax.set_title("Distribution sharpens as N grows")
    ax.legend()
    save_fig(fig, out_dir, "part1e_sharpening")

    print(f"{'N':>10} {'<k>=N/2':>14} {'sigma=sqrt(N)/2':>16} {'sigma/<k>':>14}")
    for N in ns + [n_large]:
        print(f"{N:>10.0e} {N / 2:>14.4e} {math.sqrt(N) / 2:>16.4e} {1 / math.sqrt(N):>14.4e}")
    x = 1e-10                                         # deviation of k/N from 1/2
    z = x * 2 * math.sqrt(n_large) / math.sqrt(2)     # x / (sqrt(2) sigma/N), sigma/N = 1/(2 sqrt N)
    log10_tail = (-z ** 2 - math.log(z * math.sqrt(math.pi))) / math.log(10)
    print(f"N = {n_large:.0e}: P(|k/N - 1/2| > {x:.0e}) = erfc({z:.1f}) ~ 10^({log10_tail:.0f})")
    print("=> distribution is a delta-like peak at k/N = 1/2: deterministic macroscopic behaviour.")


# ===========================================================================
# MAIN
# ---------------------------------------------------------------------------
# main() reads everything from settings.py; nothing is typed in.
# `show` lets run_all.py postpone opening the figure windows until both
# parts have finished (otherwise Part 2 would wait for you to close them).
# ===========================================================================
def main(show: bool | None = None) -> None:
    if show is None:
        show = settings.SHOW_FIGURES
    out_dir = settings.OUTPUT_DIR if settings.SAVE_OUTPUTS else None
    plt = get_pyplot(show or settings.SHOW_FIGURES)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        start_logging(out_dir / "part1_log.txt")
    try:
        print(f"Coin microstate analysis, Part 1  -  {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")
        print(f"Settings: RUN_BENCHMARK = {settings.RUN_BENCHMARK}, "
              f"TIME_BUDGET_S = {settings.TIME_BUDGET_S:g}, LARGE_N = {settings.LARGE_N:g}")
        print("Tabulated Omega values are exact integers (Python int).")
        part1a(out_dir)
        part1b(settings.RUN_BENCHMARK, settings.TIME_BUDGET_S, settings.BENCH_START_N, out_dir, plt)
        part1c(out_dir, plt)
        part1d()
        part1e(settings.LARGE_N, out_dir, plt)
        if out_dir is not None:
            print(f"\nOutputs saved in folder \"{out_dir}\".")
    finally:
        stop_logging()
    if show:
        plt.show()


if __name__ == "__main__":
    # This guard runs main() only when the file is executed directly
    # ("python part1.py" or the Run button), not when run_all.py imports it.
    main()
