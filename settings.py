"""
settings.py - every setting of the project, in one place
========================================================

This is the ONLY file you need to edit. The programs never ask for input:
change a value here, save, then run part1.py, part2.py or run_all.py
(in VS Code: open the file and press the Run button).

The values below are read by part1.py, part2.py and run_all.py when they
start.
"""

from pathlib import Path

# ===========================================================================
# TEAM DATA (Part 2)
# ---------------------------------------------------------------------------
# Last two digits of each team member's NIA (academic identifier), one entry
# per member.
#
# They are written as TEXT in quotes ("01", not 01) on purpose: in Python a
# number with a leading zero, such as 01, is a SyntaxError. As text, "01" is
# kept exactly as printed on the ID card and converted to the number 1 when
# the harmonic mean is calculated.
# ===========================================================================
NIA_LAST2 = ["91", "01", "16", "29"]

# How to turn the harmonic mean into the integer M (N = 100 + M):
#   "round" - nearest integer, halves rounded UP (2.5 -> 3)
#             (Python's own round() would round 42.5 down to 42)
#   "floor" - always round down
#   "ceil"  - always round up
# For the NIA values above: harmonic mean = 3.610 -> M = 4 -> N = 104.
M_ROUNDING = "round"

# ===========================================================================
# PART 1b BENCHMARK
# ---------------------------------------------------------------------------
# The benchmark enumerates all 2^N microstates for N = BENCH_START_N, +1, +2,
# ... until one run would exceed TIME_BUDGET_S. Total running time is about
# 2 x TIME_BUDGET_S (~10 minutes with the 5-minute budget of the assignment),
# so it is OFF by default. Set RUN_BENCHMARK = True for the final report run.
# ===========================================================================
RUN_BENCHMARK = False
TIME_BUDGET_S = 300.0        # s per single enumeration (assignment: 5 min)
BENCH_START_N = 20           # below ~20 a run is too short to time reliably

# ===========================================================================
# OUTPUT
# ---------------------------------------------------------------------------
# Results (logs, figures, CSV tables) are written to the "results" folder
# NEXT TO THESE FILES, no matter which folder the terminal is in.
# ===========================================================================
SAVE_OUTPUTS = True
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# True  -> also open the figures in windows at the end of the run
#          (the program then waits until you close them)
# False -> figures are only saved as PNG files
SHOW_FIGURES = False

# ===========================================================================
# PHYSICS
# ---------------------------------------------------------------------------
# (Physical constants such as the Boltzmann constant are in coin_common.py,
# because they are not settings.)
# ===========================================================================
LARGE_N = 1e23               # "macroscopic" N for Parts 1e and 2k
