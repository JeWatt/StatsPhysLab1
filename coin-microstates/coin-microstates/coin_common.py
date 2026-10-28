"""
coin_common.py - shared helpers for part1.py and part2.py
=========================================================

This module is NOT meant to be run on its own; it is imported by the two
programs of the project:

    part1.py   Part 1 of the assignment (enumeration, distributions, limits)
    part2.py   Part 2 of the assignment (probabilities, entropy, N -> 1e23)

Keeping the shared code here (instead of copying it into both programs)
means every formula exists exactly once, so a fix in one place fixes both
programs. It corresponds to the "LOCAL FUNCTIONS" block at the end of the
MATLAB script CoinMicrostates.m.

Contents
--------
Physical constants ........ KB, SECONDS_PER_YEAR, AGE_UNIVERSE_S
Multiplicities ............ omega_row_exact, pascal_row_float
Logarithmic formulas ...... ln_binom, ln_binom_small_k, ln_p_central,
                            ln_ratio_central
Part 1b enumeration ....... enumerate_macrostates
Reporting utilities ....... start_logging, stop_logging, banner, hbytes,
                            wrap_list, save_fig, available_ram_bytes

Python vs MATLAB: one important difference
------------------------------------------
Python integers have UNLIMITED precision. math.comb(100, 50) returns the
exact 29-digit integer 100891344545564193334812497256, whereas MATLAB
doubles are exact only below 2^53 ~ 9.0e15. This project therefore uses
EXACT integers for every Omega(N, k) that is tabulated (Parts 1a-1c, 2a-2j)
and converts to float only at the very end (division by 2^N, logarithms,
plotting). Python's int/int division is correctly rounded, so each printed
probability carries a relative error of at most 2^-53 ~ 1.1e-16.

The floating-point (double precision) Pascal triangle used by the MATLAB
version is still provided (pascal_row_float) because Part 1d is precisely
about the limits of double precision, and Part 1c uses it to MEASURE the
rounding error that MATLAB would have.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
KB = 1.380649e-23                 # Boltzmann constant, J/K (exact since SI 2019)
SECONDS_PER_YEAR = 3.156e7        # s (Julian year, 365.25 d, rounded)
AGE_UNIVERSE_S = 13.8e9 * SECONDS_PER_YEAR   # s, 13.8 Gyr, for scale only


# ===========================================================================
# MULTIPLICITIES Omega(N, k)
# ===========================================================================
def omega_row_exact(N: int) -> list[int]:
    """Exact multiplicities Omega(N, k) = C(N, k) for k = 0..N.

    Uses math.comb, which returns an exact Python integer of any size.
    Element k of the returned list is Omega(N, k) (Python lists start at 0,
    so, unlike MATLAB, no "+1" index offset is needed).

    ALTERNATIVES
      - Build Tartaglia's triangle with integers:
            row = [1]
            for _ in range(N): row = [a + b for a, b in zip(row + [0], [0] + row)]
        Same exact result, O(N^2) operations; mirrors the handout formula.
      - scipy.special.comb(N, k, exact=True): same exact integers, needs
        SciPy.
      - scipy.stats.binom.pmf(k, N, 0.5): gives P_k directly as floats.
    """
    return [math.comb(N, k) for k in range(N + 1)]


def pascal_row_float(N: int) -> np.ndarray:
    """Row N of Tartaglia's triangle computed in IEEE double precision.

    This reproduces the MATLAB function pascalRow exactly:
        Omega(n, k) = Omega(n-1, k-1) + Omega(n-1, k)
    written as the previous row padded with a zero on the right plus the
    previous row padded with a zero on the left.

    Accuracy: exact while every value is < 2^53 (N <= 56); above that the
    relative error is bounded by ~N * eps / 2; values overflow to inf for
    N > 1029. Used in Part 1c (to measure the error against the exact
    integers) and Part 1d (to find the overflow limit).
    """
    row = np.ones(1)
    with np.errstate(over="ignore", invalid="ignore"):   # inf is expected in 1d
        for _ in range(N):
            row = np.concatenate((row, [0.0])) + np.concatenate(([0.0], row))
    return row


# ===========================================================================
# LOGARITHMIC FORMULAS (needed when Omega no longer fits in a float)
# ===========================================================================
def ln_binom(N: float, k) -> np.ndarray:
    """ln Omega(N, k) via the log-gamma function, ln n! = lgamma(n + 1).

    Never overflows and works for any N up to ~1e15. Above that the
    ABSOLUTE error of lgamma (~ eps * N ln N) destroys differences between
    two values, which is why Part 2k uses the special formulas below.

    `k` may be a scalar or an array. math.lgamma is applied element-wise.
    ALTERNATIVE: scipy.special.gammaln is vectorised and faster for very
    large arrays (only ~4000 values are needed here, so math.lgamma is fine
    and avoids a SciPy dependency).
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    lg = np.vectorize(math.lgamma, otypes=[float])
    return math.lgamma(N + 1) - lg(k + 1) - lg(N - k + 1)


def ln_binom_small_k(N: float, k: int) -> float:
    """ln Omega(N, k) for k << N, accurate even for N ~ 1e23.

    Omega(N, k) = N (N-1) ... (N-k+1) / k!
      => ln Omega = sum_{j=0}^{k-1} ln(N - j) - ln k!
    ln(N - j) is evaluated as ln N + log1p(-j/N): forming N - j directly
    would round back to N in double precision when N ~ 1e23.
    math.fsum adds the terms with exact rounding (no accumulated error).
    """
    return math.fsum(math.log(N) + math.log1p(-j / N) for j in range(k)) - math.lgamma(k + 1)


def ln_p_central(N: float) -> float:
    """ln[ Omega(N, N/2) / 2^N ] from the Stirling series (N even, large).

    Omega(N, N/2) / 2^N = sqrt(2 / (pi N)) * (1 - 1/(4N) + O(N^-3)),
    so ln P(N/2) = -1/2 ln(pi N / 2) - 1/(4N). Relative error ~1e-9 at
    N = 200 (checked in part2.py) and utterly negligible at N = 1e23.
    """
    return -0.5 * math.log(math.pi * N / 2) - 1 / (4 * N)


def ln_ratio_central(N: float, d: int) -> float:
    """ln[ Omega(N, N/2 + d) / Omega(N, N/2) ] for N even, d >= 0.

    Exact telescoping product of consecutive ratios, with c = N/2:
        Omega(N, c+j) / Omega(N, c+j-1) = (c-j+1)/(c+j) = 1 + (1-2j)/(c+j)
    so the log of the ratio is a sum of log1p terms. The numerator (1 - 2j)
    is a small exact integer, and log1p keeps full relative precision even
    for arguments ~1e-22. By symmetry the same value holds for -d.
    Leading order: -2 d^2 / N (the Gaussian).
    """
    return math.fsum(math.log1p((1 - 2 * j) / (N / 2 + j)) for j in range(1, d + 1))


# ===========================================================================
# PART 1b: STREAMING ENUMERATION OF ALL 2^N MICROSTATES
# ===========================================================================
_LUT16 = None   # popcount look-up table, built on first use (fallback only)


def _popcount(x: np.ndarray, N: int) -> np.ndarray:
    """Number of 1-bits (heads) of every uint64 microstate in x.

    Preferred: np.bitwise_count (NumPy >= 2.0), a ufunc that uses the CPU's
    hardware POPCNT instruction - one machine instruction per microstate.
    Fallback for NumPy 1.x: the same 65,536-entry look-up table as the
    MATLAB version, applied to each 16-bit slice of x.
    """
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    global _LUT16
    if _LUT16 is None:
        v = np.arange(65536, dtype=np.uint32)
        _LUT16 = np.zeros(65536, dtype=np.uint8)
        for b in range(16):
            _LUT16 += ((v >> b) & 1).astype(np.uint8)
    h = np.zeros(x.shape, dtype=np.uint8)
    for s in range(0, N, 16):
        h += _LUT16[((x >> np.uint64(s)) & np.uint64(0xFFFF)).astype(np.intp)]
    return h


def enumerate_macrostates(N: int, chunk_bits: int = 22) -> tuple[np.ndarray, float]:
    """Visit all 2^N microstates, count heads in each, tally macrostates.

    Microstate <-> integer x in [0, 2^N): bit b of x is coin b (1 = head).
    The integers are produced in chunks of 2^chunk_bits (default ~4.2
    million, 32 MB as uint64), decoded with a popcount and histogrammed
    with np.bincount, then discarded. Memory stays constant (~100 MB) for
    any N, so the limit is TIME: the work doubles with every extra coin.

    Returns (counts, elapsed_seconds), where counts[k] = Omega(N, k) as
    int64 (exact up to 2^63, far beyond any reachable N).

    ALTERNATIVES (not implemented, see README "Going faster")
      - Numba @njit(parallel=True): compiled loop on all CPU cores.
      - CuPy: the same code on an NVIDIA GPU.
      - multiprocessing.Pool: split chunks across CPU cores.
    Timing uses time.perf_counter, the highest-resolution wall clock.
    """
    chunk_bits = min(N, chunk_bits)
    chunk = 1 << chunk_bits
    n_chunks = 1 << (N - chunk_bits)
    counts = np.zeros(N + 1, dtype=np.int64)
    t0 = time.perf_counter()
    for c in range(n_chunks):
        start = c * chunk
        x = np.arange(start, start + chunk, dtype=np.uint64)   # this chunk's microstates
        h = _popcount(x, N)                                      # heads per microstate
        counts += np.bincount(h, minlength=N + 1)[: N + 1]       # tally macrostates
    return counts, time.perf_counter() - t0


# ===========================================================================
# REPORTING UTILITIES
# ===========================================================================
class _Tee:
    """Write everything to several streams (console + log file).

    Python equivalent of MATLAB's `diary`: after start_logging(), every
    print() goes both to the screen and to the log file.
    ALTERNATIVE: the `logging` module (levels, timestamps), or simply run
    `python part1.py > log.txt` and read the file afterwards.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


_log_file = None


def start_logging(path: Path) -> None:
    """Start copying console output to `path` (overwrites an old log)."""
    global _log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_file)


def stop_logging() -> None:
    """Stop copying console output and close the log file."""
    global _log_file
    sys.stdout = sys.__stdout__
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def banner(txt: str) -> None:
    """Print an underlined section heading."""
    print(f"\n{txt}\n{'-' * len(txt)}")


def hbytes(b: float) -> str:
    """Human-readable byte count (binary prefixes, 1 KB = 1024 B)."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while b >= 1024 and i < len(units) - 1:
        b /= 1024
        i += 1
    return f"{b:.1f} {units[i]}"


def wrap_list(items: list[str], per_line: int) -> list[str]:
    """Join strings into lines of `per_line` items each."""
    return [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]


def save_fig(fig, out_dir: Path | None, name: str) -> None:
    """Save a Matplotlib figure as a 200-dpi PNG in out_dir (if given).

    bbox_inches='tight' crops white margins, like MATLAB's exportgraphics.
    ALTERNATIVE: save as .pdf or .svg (vector graphics) for a LaTeX report:
        fig.savefig(out_dir / f"{name}.pdf")
    """
    if out_dir is None:
        return
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")


def available_ram_bytes() -> float:
    """Total physical RAM in bytes, or NaN if it cannot be determined.

    Uses only the standard library: GlobalMemoryStatusEx on Windows,
    sysconf on Linux and macOS.
    ALTERNATIVE: `psutil.virtual_memory().total` (one line, but an extra
    package to install).
    """
    try:
        if platform.system() == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(stat.ullTotalPhys)
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return float("nan")
