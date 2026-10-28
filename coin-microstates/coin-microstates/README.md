# Coin Microstates

Microstates, macrostates, probabilities and entropy of **N unbiased coins**, the simplest two-state system in statistical physics.

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


If running with anaconda and having issues with VSC, try the following in anaconda terminal:
conda info --base
where.exe conda
& "installLocation" init powershell

conda init powershell
to link to PS

If you then see “running scripts is disabled on this system”, run Set-ExecutionPolicy -Scope CurrentUser RemoteSigned in VS Code’s terminal and restart again

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
