# Coin Microstates — Statistical Physics Lab 1

Microstates, macrostates, probabilities and entropy of **N unbiased coins**, the simplest two-state system in statistical physics, computed for the practice assignment *"Probabilities, Entropy, and Thermodynamics"*.

The programs need **no input while running**. Every setting is in `settings.py`.

| File | Role |
|------|------|
| `settings.py` | **The only file to edit**: team NIA digits, benchmark on/off, output folder, show figures |
| `run_all.py` | Runs Part 1 and Part 2 in one go |
| `part1.py` | Part 1 a–e: lists all N = 7 microstates, benchmarks full enumeration, plots Ω(N,k) for N = 20–100, finds the floating-point limits, and shows the distribution sharpening as N → 10²³ |
| `part2.py` | Part 2 a–k: probabilities, entropy changes, the waiting time for 8 heads, the likelihood ratio and the Second Law for N = 100 + M, then the same questions for N ~ 10²³ |
| `coin_common.py` | Shared formulas and helpers imported by the programs (not run on its own) |
| `test_coin_microstates.py` | Automatic checks of the maths, run with `pytest` |

## Our Parameters

- NIA endings: **91, 01, 16, 29**
- Harmonic mean = 4 / (1/91 + 1/1 + 1/16 + 1/29) = **3.610**
- Rounded to M = **4**, so **N = 104**

The harmonic mean is pulled strongly toward the smallest value, so the 01 dominates; the arithmetic mean would be 34.25. This follows directly from the assignment's definition of M.

## Setup (once per computer)

```powershell
conda env create -f environment.yml
conda activate coin-microstates
```

**Windows PowerShell / VS Code:** if `conda` is "not recognized", or you see "running scripts is disabled on this system", run the following once and then open a new terminal:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

If `conda` still isn't recognised, run this once and then restart VS Code:

```powershell
& "C:\Anaconda3\Scripts\conda.exe" init powershell
```

## Running

**In VS Code:**

1. Press Ctrl+Shift+P, choose *Python: Select Interpreter*, and pick `coin-microstates`. If it isn't listed, choose *Enter interpreter path…* and paste `C:\Anaconda3\envs\coin-microstates\python.exe`.
2. Open `run_all.py` (or `part1.py` / `part2.py`) and press the **▶ Run** button.

**In a terminal**, with the environment active:

```powershell
python run_all.py        # both parts
python part1.py          # Part 1 only
python part2.py          # Part 2 only
```

To change anything (team digits, the 5-minute benchmark, showing figures), edit `settings.py`, save, and run again.

**Part 1b benchmark:** it is **off** by default so runs take a few seconds. For the final report run, set `RUN_BENCHMARK = True` in `settings.py`. The run then takes about 10 minutes, because each enumeration has a 5-minute budget and the benchmark stops after the run that comes closest to it.

## Output

Everything is written to the `results/` folder next to the programs:

| File | Content |
|------|---------|
| `part1_log.txt`, `part2_log.txt` | Full console output, to quote in the report |
| `part1a_N7_microstates.txt` | Complete N = 7 listing (appendix) |
| `part1b_timing.png` | Enumeration time vs N (when `RUN_BENCHMARK = True`) |
| `part1c_distributions.png` | Ω(N,k) for N = 20, 50, 80, 100 in a single plot (log axis) |
| `part1c_Omega_table.csv` | Exact Ω(N,k) integers for the same N |
| `part1e_sharpening.png` | Normalised peaks for N = 10² … 10⁶ |
| `part2_probabilities_entropy.png` | P_k and S(k) for N = 104 |
| `part2_results_N104.csv` | Part 2 summary table with units and uncertainties |

## Checking the maths

```powershell
pytest
```

This runs `test_coin_microstates.py` (34 checks). Each check compares a formula or algorithm with an independent exact calculation: binomial coefficients, the enumeration engine, the large-N formulas, M and N for this team, and the Part 2 probabilities for N = 104. In VS Code you can also use the Testing panel (flask icon) → *Configure Python Tests* → **pytest** → root folder.

## Conventions and interpretations

- **M:** the harmonic mean of the two-digit NIA endings, rounded half up, so 2.5 becomes 3 (Python's built-in `round()` would round 2.5 down to 2). `M_ROUNDING` in `settings.py` can switch to `floor` or `ceil`.
- **Odd N:** "N/2" means ⌊N/2⌋, with symmetric ±10 and ±20 windows. This doesn't apply to N = 104, but it's handled.
- **2h:** the whole set of N coins is re-tossed once per second. The mean waiting time is 1/P₈, and its standard deviation (geometric distribution) is reported as the uncertainty.
- **2i:** the garbled "M ± 10" in the handout is read as **N/2 ± 10** heads, the same window as question 2c. The literal reading, 4 ± 10, would include −6 heads, which doesn't exist.
- **Entropy:** S = k_B ln Ω (natural log), with k_B = 1.380649 × 10⁻²³ J/K (exact since the 2019 SI redefinition).
- **Uncertainties:** every tabulated Ω is an exact integer, and every probability or ratio is an exact integer ratio rounded once to a double. The numerical uncertainty is therefore relative < 1.1 × 10⁻¹⁶. The large-N formulas are validated against exact values at N = 200, and the logs print their errors.

## Numerical notes

- **Exact integers:** Python integers have unlimited precision, so every Ω(N,k) is exact (`math.comb`). For comparison, Part 1c also prints the rounding error the same calculation would have in double-precision floats.
- **Part 1d** shows that exact integers never overflow (it computes N = 5000 directly). With exact integers the limits become time and memory, plus the conversion to `float` for plotting.
- **Part 1b** uses NumPy's hardware popcount (`np.bitwise_count`, NumPy ≥ 2.0).

## Uploading to GitHub

`.gitignore` excludes `results/`, Python caches and the pytest cache, so only the source files are uploaded. If you want the figures and logs on GitHub, delete the `results/` line from `.gitignore`.

To upload, use VS Code's Source Control panel (Ctrl+Shift+G) or GitHub Desktop. Commit the changes, then push.

## AI Disclaimer

As directed from our instructor, we worked to use AI to format, comment, and clean up our program in order to make a nice github repo. We also used AI to help with logic for our program and not that originally we planned on using MATLAB but were directed to use python for our script. AI was very helpful in converting our MATLAB script to python.
