# Coin Microstates

Microstates, macrostates, probabilities and entropy of **N unbiased coins**, the simplest two-state system in statistical physics. This is a Python port of the MATLAB script `CoinMicrostates.m`, written for a practice assignment in statistical thermodynamics.

| Program    | Assignment questions | What it does |
|------------|----------------------|--------------|
| `part1.py` | Part 1 a–e | Lists all N = 7 microstates, benchmarks full enumeration, plots Ω(N,k) for N = 20–100, finds the floating-point limits, and shows the distribution sharpening as N → 10²³ |
| `part2.py` | Part 2 a–k | Computes probabilities, entropy changes, the waiting time for 8 heads, likelihood ratios and the Second Law for N = 100 + M, then repeats the questions for N ~ 10²³ |
| `coin_common.py` | — | Shared helpers imported by both programs (formulas, enumeration, logging, figures). Not run on its own. |

## Setup (conda)

You only need to do this once per computer.

```bash
cd coin-microstates
conda env create -f environment.yml
conda activate coin-microstates
```

If you use pip instead of conda, follow the instructions at the top of `requirements.txt`.

## Running

Run these with the environment active, from inside the project folder:

```bash
python part1.py                          # Part 1 without the long benchmark (a few seconds)
python part1.py --benchmark              # adds the Part 1b benchmark (~10 min, 5-min budget per run)
python part1.py --benchmark --budget 60  # quicker benchmark for testing

python part2.py --nia 47 83 26           # your team's last two NIA digits, one per member
python part2.py --nia 47 83 26 --rounding floor
```

Other options: `--outdir <folder>` changes where results go, `--no-save` writes nothing, and `--show` opens the figures in windows. Run any program with `--help` to list every option.

You can also open the files in **Spyder** (included with Anaconda) or **VS Code** and press Run. To pass options there, use Spyder's *Run > Configuration per file > Command line options*, or edit `DEFAULT_NIA` near the top of `part2.py`.

## Output

Everything is written to `results/`:

| File | Content |
|------|---------|
| `part1_log.txt`, `part2_log.txt` | Full console output, to quote in the report |
| `part1a_N7_microstates.txt` | Complete N = 7 listing (appendix) |
| `part1b_timing.png` | Enumeration time vs N (with `--benchmark`) |
| `part1c_distributions.png`, `part1c_Omega_table.csv` | Ω(N,k) for N = 20, 50, 80, 100 (exact integers in the CSV) |
| `part1e_sharpening.png` | Normalised peaks for N = 10² … 10⁶ |
| `part2_probabilities_entropy.png` | P_k and S(k) for your N |
| `part2_results_N<N>.csv` | Part 2 summary table with units |

## Conventions and interpretations

These are the same as in the MATLAB version. Confirm them with the instructor:

- **M** is the harmonic mean of each member's two-digit NIA ending, rounded half away from zero (MATLAB's convention). Choose `floor` or `ceil` with `--rounding`.
- **Odd N:** "N/2" means ⌊N/2⌋. The ±10 and ±20 windows are kept symmetric.
- **2h:** the whole set of N coins is re-tossed once per second. The mean waiting time is 1/P₈, and its standard deviation is reported as the uncertainty.
- **2i:** the handout text is garbled, so both readings, N/2 ± 10 and M ± 10, are printed.
- **Entropy** is S = k_B ln Ω (natural log), with k_B = 1.380649 × 10⁻²³ J/K (exact since the 2019 SI redefinition).

## Differences from the MATLAB version

- **Exact integers.** Python integers have unlimited precision, so every tabulated Ω(N,k) is exact (`math.comb`), and probabilities carry only one final rounding (< 1.1 × 10⁻¹⁶ relative). Part 1c also prints the measured error of MATLAB's double-precision Pascal triangle, for comparison.
- **Part 1d** also shows that exact integers never overflow (it computes N = 5000 directly). With exact integers the limits become time and memory, plus the conversion to `float` for plotting.
- **Rounding:** Python's built-in `round()` rounds half to even (`round(42.5) == 42`). `part2.py` uses `floor(x + 0.5)` so that M matches MATLAB.
- **Settings** are command-line options rather than variables edited inside the script.
- **Part 1b** uses NumPy's hardware popcount (`np.bitwise_count`, NumPy ≥ 2.0). It is roughly twice as fast as the MATLAB look-up table in single-core tests.

## Going faster (Part 1b)

Each doubling of speed adds one coin to the maximum N. These options are not implemented, but they fit the code structure (`enumerate_macrostates` in `coin_common.py`):

- **Numba** (`conda install numba`): an `@njit(parallel=True)` loop uses every CPU core, typically 10–50× faster than NumPy.
- **CuPy** on an NVIDIA GPU: the same NumPy-style code runs on the GPU, about 100× faster or more on an RTX-class card.
- **multiprocessing** (standard library): split the chunks across CPU cores.

## Uploading to GitHub

The `.gitignore` already excludes `results/` and Python caches, so only the source files are uploaded. If you want the figures on GitHub, remove the `results/` line from `.gitignore`.

There are three ways to upload:

- **Website (no tools needed).** Create a new empty repository on github.com, then use *Add file > Upload files* and drag in the contents of this folder. Hidden files such as `.gitignore` may need to be selected explicitly.
- **GitHub Desktop.** Use *File > Add local repository*, choose this folder, create the repository, then click *Publish repository*.
- **Command line:**

  ```bash
  git init
  git add .
  git commit -m "Coin microstates: Python port of Parts 1 and 2"
  git branch -M main
  git remote add origin https://github.com/<your-user>/coin-microstates.git
  git push -u origin main
  ```

To keep the MATLAB version in the same repository, add `CoinMicrostates.m` in a `matlab/` subfolder.

**License:** none is included. Add one (for example MIT) before making the repository public if you want others to be able to reuse the code.
