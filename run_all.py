"""
run_all.py - run the whole assignment in one go
===============================================

Runs part1.py and then part2.py with the settings in settings.py. Nothing
is typed in.

HOW TO RUN
    - VS Code: open this file and press the Run button, or
    - terminal: python run_all.py

Each part writes its own log (results/part1_log.txt, results/part2_log.txt)
plus its figures and tables. If SHOW_FIGURES = True in settings.py, the
figure windows of BOTH parts open together at the very end, so Part 2 does
not wait for you to close the Part 1 figures.

With RUN_BENCHMARK = True the whole run takes ~2 x TIME_BUDGET_S
(~10 minutes); otherwise it finishes in a few seconds.
"""

import time

import part1
import part2
import settings
from coin_common import get_pyplot


def main() -> None:
    t0 = time.perf_counter()
    part1.main(show=False)          # figures are kept open but not shown yet
    part2.main(show=False)
    elapsed = time.perf_counter() - t0
    print(f"\nAll parts finished in {elapsed:.1f} s.")
    if settings.SAVE_OUTPUTS:
        print(f"Results are in: {settings.OUTPUT_DIR}")
    if settings.SHOW_FIGURES:
        get_pyplot(True).show()     # open every figure from both parts


if __name__ == "__main__":
    main()
