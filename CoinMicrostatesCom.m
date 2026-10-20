%% CoinMicrostates.m
% =========================================================================
%  MICROSTATES, MACROSTATES, PROBABILITIES AND ENTROPY OF N UNBIASED COINS
% =========================================================================
%  James Watt 100615729,
%  
%  A set of N unbiased coins is the simplest "two-state" system in
%  statistical physics (spin-1/2 paramagnet, two-level atoms, lattice gas
%  with occupied/empty sites ...). Outline of Script below:
%
%     Part 1  a) list every microstate for N = 7 (like Table 1)
%             b) largest N that can be fully enumerated (time + memory)
%             c) Omega(N,k) for N = 20, 50, 80, 100 in one plot
%             d) largest N for which the distribution can be computed
%             e) behaviour of the distribution as N -> 1e23
%     Part 2  a)-d) probabilities of single macrostates and windows
%             e)-g) entropy differences  S = kB ln(Omega)
%             h)-j) waiting time, likelihood ratio, Second Law
%             k)    the same questions for N ~ 1e23 (N -> infinity)
%
%  HOW TO USE
%   1. Edit NIA_last2 in the USER SETTINGS block (one entry per member).
%   2. Run the whole file (F5). Part 1b is a deliberate ~10 min benchmark;
%      set runBenchmark = false while testing everything else.
%   3. Results appear in the Command Window and are saved in <outDir>:
%        results_log.txt                  full console log (for the report)
%        part1a_N7_microstates.txt        complete N = 7 listing (appendix)
%        part1b_timing.png                enumeration time vs N
%        part1c_distributions.png         the "single plot" of part 1c
%        part1c_Omega_table.csv           Omega(N,k) table, N = 20..100
%        part1e_sharpening.png            normalised peaks, N = 1e2..1e6
%        part2_probabilities_entropy.png  P_k and S(k) for your team's N
%        part2_results_N<N>.csv           Part 2 summary table with units
%
%
%  NUMERICAL CONVENTIONS (state these in the report)
%   * Omega(N,k) is a pure number (dimensionless); P_k too.
%   * Entropy uses the natural logarithm: S = kB*ln(Omega), reported both
%     in units of kB and in J/K with kB = 1.380649e-23 J/K (exact by
%     definition since the 2019 SI, so no uncertainty).
%   * Omega(N,k) is obtained from Tartaglia's (Pascal's) triangle in IEEE
%     double precision. Every Omega < 2^53 ~ 9.0e15 is EXACT (this holds
%     for all k when N <= 56). Above that each addition can round by at
%     most eps/2 ~ 1.1e-16 relative, so after N rows the relative error is
%     bounded by ~N*eps/2 (< 2.3e-14 for N <= 200). These are the
%     "quantified uncertainties": the inputs are exact integers, so the 
%     only uncertainty is floating-point rounding.
%   * For N ~ 1e23 nothing can be enumerated or tabulated. Part 1e and 2k
%     use exact log-ratio products and Stirling/asymptotic formulas, which
%     are first validated against exact values at N = 200.
% =========================================================================

clear; close all; clc; %gottta do it in matlab

%% ======================== USER SETTINGS =================================
% NIA_last2 : the LAST TWO DIGITS of each team member's NIA, read as a
%             two-digit number (e.g. NIA ...47 -> 47). Interpretation used:
%             M = harmonic mean of these numbers, rounded to an integer.
%             This interpretation is consistent with the requirement
%             N = 100 + M < 200, because a two-digit number is < 100.
%   ALTERNATIVE interpretations:
%     - harmonic mean of the two individual DIGITS of each NIA, then
%       averaged over members (gives M < 10, so N ~ 100-109);
%     - floor() or ceil() instead of round() for a non-integer mean.
NIA_last2    = [29 01 91 29];   % JW, IAB, MGG, TS <- one entry team member

% timeBudget : wall-clock limit (s) for a single full enumeration in 1b.
%              The assignment's "performance incentive" is 5 minutes.
timeBudget   = 300;

% runBenchmark : the 1b benchmark keeps doubling the work until one run
%                approaches timeBudget, so the total cost is ~2*timeBudget
%                (~10 min). Set false to skip it while testing the rest.
runBenchmark = false;

% benchStartN : first N timed in 1b. Below ~20 a run takes milliseconds,
%               which tic/toc cannot time reliably, so starting lower only
%               adds noise to the scaling fit.
benchStartN  = 20;

saveOutputs  = true;          % write figures/tables/log to outDir
outDir       = 'results';     % created if it does not exist
kB           = 1.380649e-23;  % Boltzmann constant, J/K (exact, SI 2019)
N_large      = 1e23;          % "macroscopic" N used in Part 1e and Part 2k
% =========================================================================

% --- Logging -------------------------------------------------------------
% diary copies everything printed in the Command Window into a text file,
% so the numbers quoted in the report can be traced back to one run.
% The old log is deleted first because diary APPENDS by default.
% ALTERNATIVE: write results with fprintf(fid, ...) to a dedicated file,
% or use "publish('CoinMicrostates.m','pdf')" to produce a PDF that
% interleaves code, output and figures (handy for the appendix).
if saveOutputs
    if ~exist(outDir,'dir'), mkdir(outDir); end
    logFile = fullfile(outDir,'results_log.txt');
    if exist(logFile,'file'), delete(logFile); end
    diary(logFile);
end
fprintf('Coin microstate analysis  -  %s\n', datestr(now));
fprintf('Relative numerical uncertainty of tabulated Omega values: < N*eps\n\n');

%% ========================================================================
%  PART 1a  -  Full enumeration for N = 7
%  ------------------------------------------------------------------------
%  CHOSEN LOGIC
%   There is a one-to-one map between the 2^N microstates and the integers
%   0 ... 2^N-1: write the integer in binary with N digits and read each
%   bit as one coin (1 = head, 0 = tail). dec2bin does this for all 128
%   integers at once and returns a 128 x 7 char matrix. Counting the '1's
%   in each row gives the macrostate k (number of heads) of that row.
%   Grouping rows by k reproduces the structure of Table 1.
%
%  ALTERNATIVES
%   - ff2n(N) (Statistics and ML Toolbox) returns the same 2^N x N matrix
%     as numeric 0/1 values.
%   - Generate each macrostate directly: nchoosek(1:N, k) lists the head
%     POSITIONS for every microstate with k heads. This builds the table
%     already grouped by k, and makes Omega(N,k) = nchoosek(N,k) visible
%     in the code itself.
%   - Recursive construction: the list for N is the list for N-1 with 'H'
%     appended plus the list for N-1 with 'T' appended (mirrors the
%     Pascal recursion Omega(N,k) = Omega(N-1,k-1) + Omega(N-1,k)).
%   RECOMMENDATION: dec2bin is the shortest and makes the "microstate =
%   binary integer" idea explicit, which is reused in Part 1b.
%  ========================================================================
banner('PART 1a: enumeration of all microstates for N = 7');
N    = 7;
bits = dec2bin(0:2^N-1, N);            % 128 x 7 char of '0'/'1'
micro = repmat('T', size(bits));       % start with all tails ...
micro(bits == '1') = 'H';              % ... and turn every '1' into a head
nHeads = sum(bits == '1', 2);          % k for every microstate (row)

Omega7 = zeros(N+1,1);                 % Omega7(k+1) = Omega(7,k)
% The complete listing goes to a text file for the report appendix; the
% console shows the same table wrapped to 8 microstates per line.
if saveOutputs, fid = fopen(fullfile(outDir,'part1a_N7_microstates.txt'),'w'); else, fid = -1; end
fprintf('%-22s %-6s  microstates\n','macrostate','Omega');
for k = N:-1:0                         % descending k, as in Table 1
    list = cellstr(micro(nHeads == k, :));   % all microstates with k heads
    Omega7(k+1) = numel(list);               % multiplicity = how many
    label = sprintf('%d heads, %d tails', k, N-k);
    lines = wrapList(list, 8);
    fprintf('%-22s %-6d  %s\n', label, Omega7(k+1), lines{1});
    for L = 2:numel(lines), fprintf('%-22s %-6s  %s\n','','',lines{L}); end
    if fid > 0
        fprintf(fid,'%s  (Omega = %d)\n  %s\n\n', label, Omega7(k+1), strjoin(list,', '));
    end
end
if fid > 0, fclose(fid); end
% Two independent checks: the multiplicities must add up to 2^N, and must
% equal the binomial coefficients (row 7 of Pascal's(Tartaglia's) triangle)
fprintf('Total microstates = %d (2^N = %d).  Matches binomial coefficients: %d\n', ...
    sum(Omega7), 2^N, isequal(Omega7, pascalRow(N)));

%% ========================================================================
%  PART 1b  -  Largest N for full enumeration (time and memory limits)
%  ------------------------------------------------------------------------
%  CHOSEN LOGIC ("streaming" enumeration)
%   Every one of the 2^N microstates is generated explicitly as the
%   integer x = 0 ... 2^N-1, decoded (number of heads = number of 1-bits,
%   the "popcount" of x) and added to the tally of its macrostate. The
%   microstates are produced in chunks of 2^22 (~4.2 million) and
%   discarded once counted, so the memory use is CONSTANT (~100 MB) no
%   matter how large N is.
%   => The limit is TIME. The work is proportional to 2^N (times a small
%      factor ceil(N/16)(bin size) for the popcount), so each extra coin 
%      doubles the run time. The loop increases N until doubling the last 
%      time would exceed timeBudget; that last N is the answer to 1b.
%      
%
%  WHY NOT dec2bin AS IN 1a?
%   dec2bin stores the whole 2^N x N char matrix. MATLAB char uses 2 bytes
%   per element, so the list needs 2*N*2^N bytes: ~31 GB at N = 29. Storing
%   the list is therefore limited by MEMORY, and fails long before time
%   becomes an issue. The table printed after the benchmark quantifies
%   this second limit (char / logical / bit-packed storage). Basically gets
%   around having to store each macrostate directly in memory
%
%  ALTERNATIVES (to push N higher depending on lazyiness or not)
%   - parfor over chunks (Parallel Computing Toolbox): near-linear speed-up
%     with the number of cores, i.e. +log2(cores) in N (+3 for 8 cores).
%   - gpuArray (same toolbox): the same vectorised code runs on a GPU.
%   - A small C MEX file using the hardware popcount instruction
%     (__builtin_popcountll) reaches ~1e9-1e10 states/s: about +5 in N.
%   - Gray-code order: consecutive microstates differ by ONE coin, so k
%     changes by exactly +-1 per step (O(1) work per state). In MATLAB
%     the required scalar loop is slow; it only pays off in C/MEX.
%   - Use uint32 instead of uint64 when N <= 32 (half the memory traffic).
%  
%   gonna keep the honest and easier streamed version (shown here) for the
%   report, and note that parfor/MEX as the realistic ways to gain N.
%
%  UNCERTAINTY OF THE TIMINGS
%   tic/toc measures wall-clock time; background load typically gives a
%   few percent run-to-run scatter. ALTERNATIVE: repeat each N 3 times and
%   report mean +- standard deviation, or use timeit() for short runs.
%  ========================================================================
banner('PART 1b: maximum N for full enumeration');
if runBenchmark
    fprintf('Streaming enumeration, budget %.0f s per run.\n', timeBudget);
    fprintf('%4s %16s %12s %16s %8s\n','N','2^N','time (s)','microstates/s','check');
    benchN = []; benchT = [];
    N = benchStartN;
    while true
        [cnt, t] = enumerateMacrostates(N);
        % Correctness check: the tally of the brute-force enumeration must
        % reproduce the binomial coefficients exactly (all values < 2^53
        % here, so the comparison is exact).
        ok = isequal(cnt, pascalRow(N));
        benchN(end+1) = N; benchT(end+1) = t; %#ok<SAGROW>
        fprintf('%4d %16.0f %12.3f %16.4e %8s\n', N, 2^N, t, 2^N/t, tf(ok));
        % Stop rule: the next N costs ~2x, so if 2x the current time (with
        % a 5% margin for scatter) exceeds the budget, this N is the max.
        if 2.1*t > timeBudget, break; end
        N = N + 1;
    end
    Nmax_time = benchN(end);

    % Scaling law: fit log2(t) = a*N + b by least squares soln, a slope ~1
    % confirms t ~ 2^N, i.e. exponential complexity(think fib seq) Short
    % runs are dominated by overheads and are excluded from the fit.
    % ALTERNATIVE: fit t = C*N*2^N (include the popcount factor) or fit
    % only the last 4-5 points, which are the most reliable.
    use = benchT > 0.05; if nnz(use) < 2, use = true(size(benchT)); end
    pfit = polyfit(benchN(use), log2(benchT(use)), 1);
    fprintf('\nLargest N fully enumerated within %.0f s: N = %d (t = %.1f s)\n', ...
        timeBudget, Nmax_time, benchT(end));
    fprintf('Fit: t ~ 2^(%.3f*N %+.2f) s  (slope 1 => cost doubles per extra coin)\n', pfit(1), pfit(2));
    for dN = 1:3
        fprintf('  predicted time for N = %d : %.0f s\n', Nmax_time+dN, 2^polyval(pfit,Nmax_time+dN));
    end
    fprintf('  predicted time for N = 64 : %.1e years\n', 2^polyval(pfit,64)/3.156e7);

    figure('Name','Part 1b');
    semilogy(benchN, benchT, 'o-', 'LineWidth', 1.2); hold on;   % log y:expon->straight line
    nn = benchN(1):benchN(end)+3;
    semilogy(nn, 2.^polyval(pfit,nn), '--');
    yline(timeBudget, 'r:', '5 min budget');
    grid on; xlabel('N (coins)'); ylabel('wall-clock time (s)');
    title('Full enumeration time vs N'); legend('measured','fit 2^{aN+b}','Location','northwest');
    saveFig(gcf, outDir, 'part1b_timing', saveOutputs);
else
    fprintf('Benchmark skipped (runBenchmark = false).\n');
end

% --- Memory needed to STORE the list of all microstates ------------------
% Three possible storage formats, from most to least wasteful:
%   char    : 2 bytes per coin  (what dec2bin returns)      -> 2*N*2^N B
%   logical : 1 byte per coin   (e.g. bits=='1')             -> N*2^N B
%   packed  : 1 bit per coin    (store the integer itself)   -> N*2^N/8 B
% Even bit-packed, every extra coin roughly doubles the memory, so the
% storable N is only ~4-5 larger than with char. Checks system memory to 
% determine best way to store microstates data
ram = availableRAMbytes();
fprintf('\nMemory needed to store all 2^N microstates:\n');
fprintf('%4s %14s %14s %14s\n','N','char (2N B)','logical (N B)','packed (N/8 B)');
for N = [20 24 28 30 32 34 36 40]
    fprintf('%4d %14s %14s %14s\n', N, hbytes(2*N*2^N), hbytes(N*2^N), hbytes(N*2^N/8));
end
if ~isnan(ram)
    fprintf('Detected RAM: %s\n', hbytes(ram));
    % largest n whose list fits in RAM for a given bytes-per-coin value
    f = @(bpc) find(arrayfun(@(n) bpc*n*2^n <= ram, 1:60), 1, 'last');
    fprintf('Largest storable N:  char %d | logical %d | bit-packed %d\n', f(2), f(1), f(1/8));
else
    fprintf('RAM size could not be detected; compare the table with your machine.\n');
end

%% ========================================================================
%  PART 1c  -  Omega(N,k) for N = 20, 50, 80, 100
%  ------------------------------------------------------------------------
%  CHOSEN LOGIC
%   Enumeration is impossible here (2^100 ~ 1.3e30 states EW), so Omega(N,k)
%   is computed from Tartaglia's(Pascal's) triangle:
%        Omega(N,k) = Omega(N-1,k-1) + Omega(N-1,k),   Omega(0,0) = 1.
%   This uses only additions of positive numbers, which are the most
%   accurate floating-point operation (no cancellation, no overflow until
%   ~1e308)
%
%  ALTERNATIVES
%   - nchoosek(N,k): same values, but it issues "Result may not be exact"
%     warnings once Omega > 2^53 (already for N = 100), and must be called
%     once per k.
%   - exp(gammaln(N+1)-gammaln(k+1)-gammaln(N-k+1)): one vectorised line,
%     relative error ~1e-13; slightly worse than Pascal, but works for
%     huge N; used in Part 1e
%   - Exact integers: nchoosek(sym(N),k) (Symbolic Math Toolbox) or
%     java.math.BigInteger (built into MATLAB's JVM, no toolbox). Exact
%     but slow; only needed if we want all 30 digits.
%   - binopdf(k,N,0.5) (Statistics Toolbox) gives P_k directly.
%
%  PLOT CHOICE
%   Omega spans 1.8e5 (N = 20) to 1.0e29 (N = 100): on a linear axis like
%   Figure 2 only the N = 100 curve would be visible. The "single plot"
%   therefore has two panels: Omega on a LOG axis (all four curves, shows
%   the growth ~2^N) and P_k = Omega/2^N on a LINEAR axis (shape directly
%   comparable with Figure 2; the peak gets taller-narrower in absolute
%   k and its area stays 1).
%   ALTERNATIVE: plot Omega/Omega_max against k/N to show directly that
%   the relative width shrinks like 1/sqrt(N) (done in Part 1e).
%  ========================================================================
banner('PART 1c: macrostate distributions');
Ns = [20 50 80 100];
T  = array2table((0:max(Ns))', 'VariableNames', {'k'});   % CSV table for the appendix
fig = figure('Name','Part 1c','Position',[100 100 1100 420]);
for i = 1:numel(Ns)
    N  = Ns(i); k = (0:N)'; Om = pascalRow(N);
    % Check: sum_k Omega(N,k) = 2^N (binomial theorem with a = b = 1).
    % The relative difference printed is the numerical uncertainty.
    fprintf('N = %3d : sum Omega = %.6e, 2^N = %.6e, rel.diff = %.1e, max Omega = %.4e at k = %d\n', ...
        N, sum(Om), 2^N, abs(sum(Om)-2^N)/2^N, max(Om), k(Om==max(Om)));
    col = nan(max(Ns)+1,1); col(1:N+1) = Om;   % pad shorter rows with NaN
    T.(sprintf('Omega_N%d',N)) = col;%creates text name
    subplot(1,2,1); semilogy(k, Om, 'o-', 'MarkerSize', 3); hold on;
    subplot(1,2,2); plot(k, Om/2^N, 'o-', 'MarkerSize', 3); hold on;
end
subplot(1,2,1); grid on; xlabel('k (number of heads)'); ylabel('\Omega(N,k)  (dimensionless)');
title('Multiplicity \Omega(N,k)'); legend(compose('N = %d',Ns),'Location','south');
subplot(1,2,2); grid on; xlabel('k (number of heads)'); ylabel('P_k = \Omega/2^N  (dimensionless)');
title('Probability of each macrostate'); legend(compose('N = %d',Ns));
saveFig(fig, outDir, 'part1c_distributions', saveOutputs);
if saveOutputs, writetable(T, fullfile(outDir,'part1c_Omega_table.csv')); end

%% ========================================================================
%  PART 1d  -  Maximum N for the distribution (floating-point limits)
%  ------------------------------------------------------------------------
%  CHOSEN LOGIC
%   With Pascal's triangle the cost is only ~N^2/2 additions, so time is
%   never the problem. The limiting factor is the RANGE of IEEE double
%   precision: the largest exact number is realmax ~ 1.8e308.
%   script finds the limits by direct test:
%     (1) keep building rows until some Omega(N,k) becomes Inf;
%     (2) find the largest N for which 2^N is finite (2^1024 = Inf), which
%         is where P_k = Omega/2^N breaks even if Omega itself is finite.
%   Expected: the central Omega ~ 2^N*sqrt(2/(pi*N)) overflows at N = 1030,
%   and 2^N at N = 1024, so the practical limit is N = 1023 for P_k and
%   N = 1029 for Omega.
%
%  Potential Alternatives we can pursue...
%   - Work with logarithms: ln Omega via gammaln never overflows. The next
%     limit is MEMORY for the k-vector (8 bytes per k: 16 GB -> N ~ 2e9)
%     and the ACCURACY of gammaln: its absolute error grows like
%     eps*N*ln(N), so differences such as ln Omega(N,k1) - ln Omega(N,k2)
%     lose all digits around N ~ 1e15.
%   - Scale each row by 2 while building it (P_k recursion
%     P(N,k) = [P(N-1,k-1) + P(N-1,k)]/2): no overflow, but the tails
%     underflow to 0 below realmin ~ 2.2e-308 (harmless for plotting).
%   - Arbitrary precision: vpa/sym or java.math.BigInteger. Exact, but
%     cost grows quickly; fine up to N ~ 1e4-1e5.
%   - Gaussian (de Moivre-Laplace) approximation: valid for any N, error
%     ~1/N near the peak.
%   - Plotting: a screen can resolve only ~1e3-1e4 distinct k, so for
%     larger N one plots k/N or (k-N/2)/sigma instead of every point.
%  ========================================================================
banner('PART 1d: maximum N for calculating/plotting the distribution');
row = 1; n = 0;
while all(isfinite(row))
    n = n + 1; row = [row 0] + [0 row];   %#ok<AGROW>  (row n of the triangle)
end
Nmax_Omega = n - 1;                                       % last all-finite row
Nmax_2N    = find(isfinite(2.^(1:1100)), 1, 'last');      % 2^1024 = Inf
fprintf('realmax (double)                     = %.4e\n', realmax);
fprintf('Largest N with all Omega(N,k) finite = %d (central Omega = %.4e)\n', ...
    Nmax_Omega, max(pascalRow(Nmax_Omega)));
fprintf('Largest N with 2^N finite            = %d  -> P_k = Omega/2^N fails from N = %d\n', ...
    Nmax_2N, Nmax_2N+1);
fprintf('Work-around: logarithms. ln Omega = gammaln(N+1)-gammaln(k+1)-gammaln(N-k+1)\n');
fprintf('  then the limit is memory for the k-vector (8 B per k): 16 GB -> N ~ 2e9,\n');
fprintf('  and gammaln absolute error ~eps*N*lnN, which ruins differences near N ~ 1e15.\n');
fprintf('  Plot resolution: only ~1e3-1e4 distinct k fit across a screen.\n');

%% ========================================================================
%  PART 1e  -  Towards the thermodynamic limit (N ~ 1e23)
%  ------------------------------------------------------------------------
%  PHYSICS
%   The binomial distribution with p = 1/2 has
%      expected mean of <k> = N/2, standard deviation  sigma = sqrt(N)/2,
%   so the RELATIVE width sigma/<k> = 1/sqrt(N). Absolute fluctuations
%   grow (sqrt(N)) but relative fluctuations vanish. For N = 1e23,
%   sigma/<k> ~ 3e-12 and the fraction of heads is 1/2 to ~12 digits, i.e.
%   the macrostate is essentially deterministic. Near the peak the
%   distribution looks Gaussian (central limit theorem / de Moivre-
%   Laplace), and far from it the probability falls like exp[-N*(...)]
%   (large-deviation theory, see Part 2k-j).
%
%  CHOSEN LOGIC
%   (1) Plot P_k/P_max against k/N for N = 1e2 ... 1e6 using gammaln
%       (Part 1d work-around) to SHOW the sharpening.
%   (2) Print <k>, sigma and sigma/<k> up to N = 1e23.
%   (3) Quantify "deterministic": the probability that k/N deviates from
%       1/2 by more than 1e-10 is erfc(x/(sqrt(2)*sigma/N)). Its argument
%       (~45) is far too large for erfc() (underflows to 0), so the
%       asymptotic form erfc(z) ~ exp(-z^2)/(z*sqrt(pi)) is used in
%       log10 form.
%   ALTERNATIVE: plot P_k*sigma against (k-N/2)/sigma; all curves then
%   collapse onto the same standard Gaussian (shows universality).
%  ========================================================================
banner('PART 1e: extrapolation to N ~ 1e23');
NsE = [1e2 1e3 1e4 1e5 1e6];
fig = figure('Name','Part 1e');
for N = NsE
    % Only the window 0.4 <= k/N <= 0.6 is plotted, with at most 4001
    % points (more points than pixels would add nothing, see 1d).
    k  = round(linspace(0.4*N, 0.6*N, min(N*0.2+1, 4001)));
    lnP = lnBinom(N,k) - N*log(2);                 % ln P_k, never overflows
    plot(k/N, exp(lnP - max(lnP)), '-', 'LineWidth', 1.2); hold on;  % normalised to peak = 1
end
grid on; xlabel('k/N (fraction of heads)'); ylabel('P_k / P_{max}');
title('Distribution sharpens as N grows'); legend(compose('N = 10^{%d}', log10(NsE)));
saveFig(fig, outDir, 'part1e_sharpening', saveOutputs);

fprintf('%10s %14s %14s %14s\n','N','<k>=N/2','sigma=sqrt(N)/2','sigma/<k>');
for N = [NsE N_large]
    fprintf('%10.0e %14.4e %14.4e %14.4e\n', N, N/2, sqrt(N)/2, 1/sqrt(N));
end
x = 1e-10;                       % deviation of k/N from 1/2
z = x*2*sqrt(N_large)/sqrt(2);   % erfc argument: x / (sqrt(2)*sigma/N), sigma/N = 1/(2 sqrt(N))
log10tail = (-z^2 - log(z*sqrt(pi)))/log(10);   % log10 of erfc(z) ~ exp(-z^2)/(z sqrt(pi))
fprintf('N = %.0e: P(|k/N - 1/2| > %.0e) = erfc(%.1f) ~ 10^(%.0f)\n', N_large, x, z, log10tail);
fprintf('=> distribution is a delta-like peak at k/N = 1/2: deterministic macroscopic behaviour.\n');

%% ========================================================================
%  PART 2  -  Probabilities, entropy and thermodynamics, N = 100 + M
%  ------------------------------------------------------------------------
%  TEAM PARAMETER
%   Harmonic mean H = n / sum(1/x_i). It is computed by hand rather than
%   with harmmean() so no Statistics Toolbox is needed. A value of 0 makes
%   1/x infinite (H undefined), so it is trapped with a clear message.
%
%  CHOSEN LOGIC FOR PROBABILITIES
%   N < 200, so the full row Omega(N,0..N) is computed once with Pascal's
%   triangle and every requested quantity is a lookup or a sum over it:
%       P_k = Omega(N,k)/2^N,   Delta P = sum of P_k over the window.
%   Indexing: MATLAB arrays start at 1, so P_k is stored in P(k+1).
%   The anonymous function Pk(k) hides this +1 offset.
%   ALTERNATIVES: binopdf/binocdf (Statistics Toolbox):
%       P_k = binopdf(k,N,0.5);  Delta P = binocdf(k2,N,0.5)-binocdf(k1-1,N,0.5)
%   (careful: the cdf difference loses accuracy when Delta P ~ 1).
%
%  ODD N
%   If N is odd, N/2 is not an integer. Both floor(N/2) and ceil(N/2) are
%   then the (equally) most probable macrostates since
%   Omega(N,k) = Omega(N,N-k). The script uses kc = floor(N/2) as "N/2".
%   Windows "N/2 +- 10" are taken as ceil(N/2-10) ... floor(N/2+10), which
%   keeps them symmetric (20 values for odd N, 21 for even N).
%   ALTERNATIVE: report the result for both floor and ceil, or use the
%   window floor(N/2)-10 ... floor(N/2)+10 (21 values, slightly asymmetric).
%  ========================================================================
banner('PART 2: team parameters');
x = double(NIA_last2(:));
if any(x == 0)
    error('A two-digit ending of 00 makes the harmonic mean undefined; agree a convention with the instructor.');
end
Hm = numel(x) / sum(1./x);       % harmonic mean of the two-digit endings
M  = round(Hm);                  % N must be an integer (see USER SETTINGS for floor/ceil)
N  = 100 + M;
assert(N < 200, 'N must be below 200');
fprintf('NIA endings: %s\nHarmonic mean = %.4f -> M = %d -> N = %d\n', mat2str(x'), Hm, M, N);

Om  = pascalRow(N);  k = (0:N)';
P   = Om / 2^N;                  % P_k (stored at index k+1); sum(P) = 1
lnO = log(Om);                   % S(k)/kB = ln Omega(N,k), natural log
Pk  = @(kk) P(kk+1);             % P_k with k counted from 0
kc  = floor(N/2);                % "N/2" (see ODD N note above)
if mod(N,2)
    fprintf('N is odd: N/2 is not an integer. "N/2 heads" is taken as k = %d (P(%d) = P(%d)).\n', kc, kc, kc+1);
end

% --- 2a-d: probabilities -------------------------------------------------
%  a) single unlikely macrostate (k = 30, far in the tail for N ~ 140)
%  b) the most probable macrostate; ~ sqrt(2/(pi*N)) for large N
%  c) window of +-10 HEADS around N/2
%  d) window of +-20 TAILS around N/2. Because Omega(N,k) = Omega(N,N-k),
%     "T tails" is the same macrostate as "N-T heads", and the tails
%     window maps onto the identical heads window. The code converts the
%     tails explicitly (hk = N - tails) to make the symmetry visible.
banner('PART 2a-d: probabilities');
a  = Pk(30);
b  = Pk(kc);
k1 = ceil(N/2-10); k2 = floor(N/2+10);
c  = sum(P(k1+1:k2+1));
t1 = ceil(N/2-20); t2 = floor(N/2+20);         % range given in TAILS
hk = N - (t1:t2);                              % corresponding numbers of heads
d  = sum(P(hk+1));
fprintf('a) P(k = 30)                         = %.6e\n', a);
fprintf('b) P(k = N/2 = %d)                   = %.6e\n', kc, b);
fprintf('c) sum P_k, k = %d..%d heads         = %.6f\n', k1, k2, c);
fprintf('d) sum P_k, %d..%d tails (=%d..%d heads) = %.6f\n', t1, t2, min(hk), max(hk), d);
fprintf('   (heads/tails symmetry: Omega(N,k) = Omega(N,N-k))\n');
% Physical reading: compare c and d with the standard deviation
% sigma = sqrt(N)/2 (~6 for N ~ 140). +-10 is ~1.7 sigma (~92%), +-20 is
% ~3.4 sigma (>99.9%): almost every toss lands near N/2.

% --- 2e-g: entropy changes -----------------------------------------------
%  Boltzmann: S = kB ln(Omega)  =>  Delta S = S(k1) - S(k2)
%                                         = kB * ln[Omega(k1)/Omega(k2)].
%  Computing the ratio inside ONE log (lnO(k1) - lnO(k2)) avoids
%  overflow and keeps full relative precision.
%  Sign convention follows the handout: Delta S = S(k1) - S(k2), so e) and
%  f) are NEGATIVE (moving away from maximum disorder lowers S) and g) is
%  POSITIVE (maximum disorder minus a highly ordered state).
%  Results are given in units of kB (natural for statistical physics) and
%  in J/K (SI, as requested by the report format).
%  ALTERNATIVE: log10 would give S in "kB*log10" units, a factor ln(10)
%  smaller; the physical convention is the natural log.
banner('PART 2e-g: entropy changes  dS = S(k1) - S(k2) = kB ln[Omega(k1)/Omega(k2)]');
pairs = [kc+25 kc;     % e) high disorder  vs maximum disorder
         kc+10 kc;     % f) moderate disorder vs maximum disorder
         kc    20];    % g) maximum disorder vs high order
tags  = {'e','f','g'};
dS_kB = zeros(3,1);
for i = 1:3
    dS_kB(i) = lnO(pairs(i,1)+1) - lnO(pairs(i,2)+1);
    fprintf('%s) k1 = %3d, k2 = %3d : dS = %+9.4f kB = %+.4e J/K   [Omega ratio = %.4e]\n', ...
        tags{i}, pairs(i,1), pairs(i,2), dS_kB(i), kB*dS_kB(i), exp(dS_kB(i)));
end

% --- 2h: waiting time for exactly 8 heads ----------------------------------
%  Interpretation: "a tossing rate of N coins per second" = the whole
%  system is re-tossed once per second, i.e. one independent macrostate
%  sample per second. The number of trials until the first success is
%  GEOMETRICALLY distributed with success probability p = P_8:
%      mean waiting time  <tau> = 1/p   (in s, one trial per s)
%      standard deviation       = sqrt(1-p)/p  (~ <tau> when p << 1)
%  The std dev is quoted as the statistical uncertainty: the waiting
%  time is a random variable, and its spread is as large as its mean.
%  ALTERNATIVES: "8 heads or fewer" (sum of P_0 ... P_8, also printed),
%  or "one coin per second" (then divide the rate by N: <tau> * N).
banner('PART 2h-j: thermodynamic implications');
p8    = Pk(8);
tau   = 1/p8;                         % s, one toss of all N coins per second
sdTau = sqrt(1-p8)/p8;                % s, std. dev. of the geometric distribution
ageU  = 13.8e9*3.156e7;               % s, age of the Universe (13.8 Gyr) for scale
fprintf('h) P(exactly 8 heads) = %.4e\n', p8);
fprintf('   mean waiting time  = %.4e s (std. dev. %.4e s) = %.4e yr = %.4e x age of Universe\n', ...
    tau, sdTau, tau/3.156e7, tau/ageU);
fprintf('   (for "8 heads or fewer": %.4e s)\n', 1/sum(P(1:9)));

% --- 2i: likelihood ratio --------------------------------------------------
%   "M?+-10" is given, does this mean:
%    M +- 10 heads (literal reading of "M"): far in the tail, ratio
%        of many orders of magnitude
%  The ratio of probabilities equals the ratio of multiplicities, since
%  the common factor 2^N cancels: P(N/2)/P(k) = Omega(N/2)/Omega(k).

if M-10 >= 0 && M+10 <= N
    R_M10 = [b/Pk(M-10), b/Pk(M+10)];
    fprintf('   P(N/2)/P(M - 10 = %d) = %.4e,  P(N/2)/P(M + 10 = %d) = %.4e\n', ...
        M-10, R_M10(1), M+10, R_M10(2));
end

% --- 2j: Second Law ----------------------------------------------------------
%  Key relation: P(k1)/P(k2) = Omega(k1)/Omega(k2) = exp(Delta S/kB).
%  A transition to a lower-entropy macrostate (Delta S < 0) is not
%  forbidden, only exponentially improbable. For N ~ 140 such fluctuations
%  are observable (e.g. +10 heads happens ~1/4 as often as N/2), so
%  nothing calculated here "contradicts" the Second Law: the Second Law
%  is a statement about overwhelming probability, which becomes certain
%  only when Delta S/kB is huge, i.e. for macroscopic N (see 2k) or real
%  systems
fprintf(['j) No result contradicts the Second Law. Low-entropy macrostates are not forbidden,\n' ...
         '   only improbable: P(k1)/P(k2) = exp[dS/kB]. For this N:\n']);
for i = 1:3
    fprintf('   %s) probability ratio exp(dS/kB) = %.4e\n', tags{i}, exp(dS_kB(i)));
end
fprintf('   The Second Law is a statistical statement that becomes certain only as N -> inf (Part 2k).\n');

% --- Figure for Part 2 -----------------------------------------------------
%  Left: P_k on a log axis (otherwise k = 8 and k = 30 would sit at zero)
%  with the +-10 window and the special macrostates highlighted.
%  Right: S/kB = ln Omega, showing the entropy maximum at N/2 and the
%  three pairs used in e)-g).
fig = figure('Name','Part 2','Position',[100 100 1100 420]);
subplot(1,2,1);
plot(k, P, 'k.-'); hold on;
plot(k1:k2, P(k1+1:k2+1), 'b.', 'MarkerSize', 14);        % N/2 +- 10 window
plot([30 8 kc], P([30 8 kc]+1), 'ro', 'MarkerFaceColor', 'r');
text(30, P(31), '  k = 30'); text(8, P(9), '  k = 8');
set(gca, 'YScale', 'log'); grid on; xlim([0 N]);
xlabel('k (heads)'); ylabel('P_k  (dimensionless)'); title(sprintf('N = %d (blue: N/2 \\pm 10)', N));
subplot(1,2,2);
plot(k, lnO, 'k-'); hold on;
plot(pairs(:), lnO(pairs(:)+1), 'ro', 'MarkerFaceColor', 'r');
grid on; xlim([0 N]); xlabel('k (heads)'); ylabel('S / k_B = ln \Omega');
title('Entropy of each macrostate');
saveFig(fig, outDir, 'part2_probabilities_entropy', saveOutputs);

%% ========================================================================
%  PART 2k  -  The same questions for N ~ 1e23 (N -> infinity)
%  ------------------------------------------------------------------------
%  WHY NEW FORMULAS ARE NEEDED
%   - Omega and 2^N overflow (Part 1d), so everything is done in logs.
%   - gammaln(1e23) ~ 5e24 carries an absolute error ~ eps*5e24 ~ 1e9,
%     so DIFFERENCES of gammaln values (ln of ratios ~ 1e-20) are ignorable
%     noise. Also N/2 + 25 cannot even be stored exactly: the spacing of
%     doubles near 5e22 (eps(5e22)) is ~8e6.
%   The fix is to rewrite every quantity so that only small, exactly
%   representable numbers are subtracted:
%
%   (1) Central probability (Stirling series, N even):
%         ln P(N/2) = ln[Omega(N,N/2)/2^N] = -1/2 ln(pi*N/2) - 1/(4N) + O(N^-3)
%   (2) Ratio to the centre, EXACT product with c = N/2:
%         Omega(N,c+j)/Omega(N,c+j-1) = (c-j+1)/(c+j) = 1 + (1-2j)/(c+j)
%         => ln[Omega(c+d)/Omega(c)] = sum_{j=1..d} log1p((1-2j)/(c+j))
%       The numerator 1-2j is an exact small integer and log1p keeps full
%       relative precision for arguments ~1e-22. Leading order: -d^2/c
%       = -2 d^2/N (the Gaussian), but the product is exact.
%   (3) Small k (k << N), EXACT:
%         ln Omega(N,k) = sum_{j=0..k-1} ln(N-j) - ln k!,
%       with ln(N-j) written as ln N + log1p(-j/N) (N-j itself would round
%       to N in double precision).
%
%   These are VALIDATED at N = 200 against the exact Pascal values; the
%   relative errors printed are the uncertainty of the method.
%
%  ALTERNATIVES
%   - Gaussian approximation everywhere: P_k ~ sqrt(2/(pi N))
%     exp[-2(k-N/2)^2/N]. Simple, excellent near the peak, but WRONG in the
%     far tail (k = 8, 20, 30), where formula (3) must be used.
%   - Large-deviation (entropy) form for k = x*N:
%       ln Omega ~ N*H(x) - 1/2 ln(2 pi N x(1-x)),
%       H(x) = -x ln x - (1-x) ln(1-x)   (used in 2k-j).
%   - vpa/sym with ~40 significant digits can evaluate gammaln-type
%     expressions at N = 1e23 directly (Symbolic Math Toolbox), useful as
%     an independent cross-check.
%  ========================================================================
banner('PART 2k: validation of large-N formulas at N = 200');
Nv = 200; Ov = pascalRow(Nv); cv = Nv/2;
chk = [log(Ov(cv+1)/2^Nv),            lnPcentral(Nv);          % (1) ln P(N/2)
       log(Ov(cv+26)/Ov(cv+1)),       lnRatioCentral(Nv,25);   % (2) ratio d = 25
       log(Ov(31)),                   lnBinomSmallK(Nv,30)];   % (3) ln Omega(N,30)
for i = 1:3
    fprintf('  exact %+.10f   formula %+.10f   rel.err %.1e\n', chk(i,1), chk(i,2), ...
        abs(chk(i,1)-chk(i,2))/abs(chk(i,1)));
end

banner(sprintf('PART 2k: results for N = %.0e (M = %d keeps its meaning)', N_large, M));
NL  = N_large; ln2 = log(2);
lnPc = lnPcentral(NL);                 % ln P(N/2)
lr  = @(d) lnRatioCentral(NL, d);      % ln[P(N/2+d)/P(N/2)]

% a) P(30) = Omega(N,30)/2^N: the N ln2 term dominates, P ~ 10^(-3e22).
fprintf('a) log10 P(k=30)   = %.6e   (P ~ 10^-(3e22): never)\n', (lnBinomSmallK(NL,30) - NL*ln2)/log(10));
% b) P(N/2) -> sqrt(2/(pi N)) -> 0: even the MOST probable macrostate is
%    itself extremely improbable; probability is spread over ~sigma states.
fprintf('b) P(k = N/2)      = %.4e   (~ sqrt(2/(pi N)) -> 0)\n', exp(lnPc));
% c, d) Window sums = P(N/2) * sum of ratios; each ratio is 1 - O(1e-20),
%    so dP = 21 P(N/2) and 41 P(N/2). The windows are tiny compared with
%    sigma ~ 1.6e11: a window of fixed width is no longer the relevant
%    scale. The relevant statement is about windows of a few sigma (or a
%    fixed FRACTION of N), which contain essentially all the probability.
sum10 = sum(exp(arrayfun(lr, 0:10))) + sum(exp(arrayfun(lr, 1:10)));   % d = -10..10 (symmetric)
sum20 = sum(exp(arrayfun(lr, 0:20))) + sum(exp(arrayfun(lr, 1:20)));   % d = -20..20
fprintf('c) dP(N/2 +- 10)   = %.4e   (= 21 P(N/2): window << sigma = %.2e)\n', exp(lnPc)*sum10, sqrt(NL)/2);
fprintf('d) dP(N/2 +- 20)   = %.4e   (= 41 P(N/2))\n', exp(lnPc)*sum20);
fprintf('   but P(|k-N/2| < 3 sigma) = %.6f -> 1: a fixed +-10 window is no longer "macroscopic".\n', erf(3/sqrt(2)));
% e, f) Delta S ~ -2 d^2/N kB -> 0: N/2+25 and N/2+10 are, for N = 1e23,
%    the SAME macroscopic state as N/2 (the labels "high/moderate
%    disorder" lose their meaning).
% g)  Delta S = ln Omega(N/2) - ln Omega(20) ~ N ln2 kB ~ 0.96 J/K: a
%    genuinely macroscopic entropy difference.
%    Note: lnOc = lnPc + N ln2 adds ~ -27 to ~ 7e22; the -27 is below the
%    double resolution (~1e7 there) and is lost, which changes the result
%    by < 1e-15 relative (irrelevant at the printed precision).
dSe = lr(25); dSf = lr(10);
lnOc = lnPc + NL*ln2;                  % ln Omega(N, N/2)
dSg = lnOc - lnBinomSmallK(NL,20);
fprintf('e) dS = %.4e kB = %.4e J/K   (N/2+25 indistinguishable from N/2)\n', dSe, kB*dSe);
fprintf('f) dS = %.4e kB = %.4e J/K\n', dSf, kB*dSf);
fprintf('g) dS = %.4e kB = %.4f J/K   (macroscopic, ~ N kB ln2)\n', dSg, kB*dSg);
% h) <tau> = 1/P(8) = 2^N / Omega(N,8) seconds, printed as log10 because
%    the number itself (~10^(3e22)) cannot be stored in any float.
log10tau = (NL*ln2 - lnBinomSmallK(NL,8))/log(10);
fprintf('h) waiting time for 8 heads ~ 10^(%.4e) s  vs age of Universe ~ 10^%.1f s\n', log10tau, log10(ageU));
% i) Reading (1): ratio -> 1 (printed as ratio - 1 with expm1 because
%    1 + 2e-21 = 1 exactly in double). Reading (2): astronomically large.
fprintf('i) P(N/2)/P(N/2+-10) - 1 = %.4e   (ratio -> 1)\n', expm1(-lr(10)));
log10RM = (lnOc - lnBinomSmallK(NL, M+10))/log(10);
fprintf('   P(N/2)/P(M+10)  ~ 10^(%.4e)\n', log10RM);
% j) A state that differs MACROSCOPICALLY (here 51% instead of 50% heads)
%    has Delta S/kB ~ N*[H(0.51) - ln2] ~ -2e19: its relative probability
%    exp(-2e19) is zero for every practical purpose. This is why the
%    Second Law is never seen to fail in macroscopic systems.
xdev = 0.51; Hx = -xdev*log(xdev) - (1-xdev)*log(1-xdev);   % binary entropy (nats)
dSmac = NL*(Hx - ln2);             % leading order of [S(0.51N) - S(N/2)]/kB
fprintf('j) A macroscopically different state (k = 0.51 N) has dS ~ %.3e kB = %.3e J/K,\n', dSmac, kB*dSmac);
fprintf('   i.e. relative probability exp(%.2e): the Second Law becomes effectively exact.\n', dSmac);

% --- Summary table (with units) for the report -----------------------------
if saveOutputs
    S = table({'2a';'2b';'2c';'2d';'2e';'2f';'2g';'2h'}, ...
        {'P(k=30)';'P(k=N/2)';'dP(N/2+-10 heads)';'dP(N/2+-20 tails)'; ...
         'dS(N/2+25 vs N/2)';'dS(N/2+10 vs N/2)';'dS(N/2 vs 20)';'mean wait for 8 heads'}, ...
        [a;b;c;d;kB*dS_kB;tau], {'-';'-';'-';'-';'J/K';'J/K';'J/K';'s'}, ...
        'VariableNames', {'item','quantity','value','units'});
    writetable(S, fullfile(outDir, sprintf('part2_results_N%d.csv', N)));
    diary off;
    fprintf('\nOutputs saved in folder "%s".\n', outDir);
end

%% ======================== LOCAL FUNCTIONS ===============================

function row = pascalRow(N)
% PASCALROW  Row N of Tartaglia's triangle: Omega(N,k) for k = 0..N.
%   Builds the triangle row by row with Omega(n,k) = Omega(n-1,k-1) +
%   Omega(n-1,k), written as the sum of the previous row padded with a
%   zero on the right and on the left. Cost O(N^2), memory O(N).
%   Exact for N <= 56; relative error < ~N*eps/2 beyond; Inf for N > 1029.
%   ALTERNATIVE: multiplicative recurrence Omega(N,k) = Omega(N,k-1) *
%   (N-k+1)/k gives the row in O(N) operations, but each step mixes a
%   multiplication and a division (slightly more rounding). For N < 200
%   the difference in speed is irrelevant.
row = 1;
for n = 1:N
    row = [row 0] + [0 row]; %#ok<AGROW>
end
row = row(:);                  % column vector, row(k+1) = Omega(N,k)
end

function [counts, tElapsed] = enumerateMacrostates(N)
% ENUMERATEMACROSTATES  Visit all 2^N microstates and tally macrostates.
%   Microstate = integer x in 0..2^N-1, bit b of x = coin b (1 = head).
%   For each x the number of heads h = popcount(x) is computed, and
%   counts(h+1) is incremented. Returns the tally and the elapsed time.
%
%   Implementation details
%   * Chunks of 2^22 consecutive integers are processed with vectorised
%     operations (MATLAB loops over single elements are ~100x slow)
%     2^22 uint64 values = 32 MB; with temporaries the peak is ~100 MB.
%     ALTERNATIVE: a larger chunk (2^24) reduces loop overhead slightly
%     but needs ~4x the memory; 2^20 is safer on small laptops.
%   * popcount uses a 65,536-entry look-up table LUT(v+1) = number of
%     1-bits in v (0 <= v < 2^16). Each 64-bit x is split into 16-bit
%     pieces with bitshift/bitand, so N <= 32 needs 2 look-ups per state.
%     ALTERNATIVE: an 8-bit table (256 entries, 4 look-ups for N <= 32)
%     or the bit-twiddling "SWAR" popcount; in MATLAB the 16-bit table
%     is usually fastest.
%   * The LUT is "persistent" so it is built only once per session, not
%     at every call of the benchmark loop.
%   * accumarray(h+1, 1) is a fast histogram of the head counts.
%     ALTERNATIVE: histcounts(h, -0.5:1:N+0.5) (similar speed).
%   * Valid for N <= 53: chunk start values are formed from doubles, and
%     2^53 is far beyond any reachable run time anyway.
persistent LUT
if isempty(LUT)                                   % popcount of 0..65535
    v = uint16(0:65535)'; LUT = zeros(65536,1,'uint8');
    for bb = 1:16, LUT = LUT + uint8(bitget(v, bb)); end
end
chunkBits = min(N, 24); chunk = 2^chunkBits; nChunks = 2^(N - chunkBits);
offs   = uint64(0:chunk-1)';                      % 0..chunk-1, reused by every chunk
mask   = uint64(65535);                           % lowest 16 bits
counts = zeros(N+1,1);
tStart = tic;
for c = 0:nChunks-1
    x = offs + uint64(c)*uint64(chunk);           % microstates c*chunk ... (c+1)*chunk-1
    h = zeros(chunk,1,'uint8');                   % heads per microstate (<= 255 OK)
    for s = 0:16:N-1                              % popcount, 16 bits at a time
        h = h + LUT(double(bitand(bitshift(x, -s), mask)) + 1);
    end
    counts = counts + accumarray(double(h) + 1, 1, [N+1 1]);   % tally macrostates
end
tElapsed = toc(tStart);
end

function y = lnBinom(N, k)
% LNBINOM  ln Omega(N,k) via the log-gamma function, ln n! = gammaln(n+1).
%   Never overflows; vectorised in k. Accurate to ~1e-13 relative for the
%   values used in Part 1e (N <= 1e6). Becomes useless for DIFFERENCES at
%   N >~ 1e15 (absolute error ~ eps*N*ln N), see Part 2k.
y = gammaln(N+1) - gammaln(k+1) - gammaln(N-k+1);
end

function y = lnBinomSmallK(N, k)
% LNBINOMSMALLK  ln Omega(N,k) for k << N, accurate even for N ~ 1e23.
%   Omega(N,k) = N(N-1)...(N-k+1)/k!  =>  ln Omega = sum ln(N-j) - ln k!
%   ln(N-j) = ln N + log1p(-j/N) avoids forming N-j (which rounds to N).
j = 0:k-1;
y = sum(log(N) + log1p(-j/N)) - gammaln(k+1);
end

function y = lnPcentral(N)
% LNPCENTRAL  ln[ Omega(N,N/2) / 2^N ] from the Stirling series (N even).
%   Omega(N,N/2)/2^N = sqrt(2/(pi N)) * (1 - 1/(4N) + O(N^-3)).
%   Relative error ~1e-9 already at N = 200 (printed in the validation)
%   and ~1e-70 at N = 1e23.
y = -0.5*log(pi*N/2) - 1/(4*N);
end

function y = lnRatioCentral(N, d)
% LNRATIOCENTRAL  ln[ Omega(N,N/2+d) / Omega(N,N/2) ] for N even, d >= 0.
%   Exact telescoping product of consecutive ratios (see Part 2k notes);
%   by symmetry the same value holds for -d. Leading order -2 d^2/N.
if d == 0, y = 0; return; end
j = 1:d;
y = sum(log1p((1 - 2*j) ./ (N/2 + j)));
end

function b = availableRAMbytes()
% AVAILABLERAMBYTES  Total physical RAM in bytes (NaN if unknown).
%   Tries, in order: MATLAB's memory() (Windows only), /proc/meminfo
%   (Linux), sysctl (macOS). Used only to report the storage limit in 1b.
b = NaN;
try, [~, s] = memory; b = s.PhysicalMemory.Total; return; catch, end            %#ok<CTCH> Windows
try
    [st, out] = system('grep MemTotal /proc/meminfo');                            % Linux
    if st == 0, b = sscanf(out, 'MemTotal: %f') * 1024; return; end
catch, end
try
    [st, out] = system('sysctl -n hw.memsize');                                   % macOS
    if st == 0, b = str2double(strtrim(out)); return; end
catch, end
end

function s = hbytes(b)
% HBYTES  Human-readable byte count (binary prefixes, 1 KB = 1024 B).
u = {'B','KB','MB','GB','TB','PB'}; i = 1;
while b >= 1024 && i < numel(u), b = b/1024; i = i + 1; end
s = sprintf('%.1f %s', b, u{i});
end

function lines = wrapList(list, perLine)
% WRAPLIST  Join a cell array of strings into lines of perLine items.
n = numel(list); lines = {};
for i = 1:perLine:n
    lines{end+1} = strjoin(list(i:min(i+perLine-1, n)), ', '); %#ok<AGROW>
end
end

function s = tf(ok)
% TF  'OK'/'FAIL' label for the enumeration check.
if ok, s = 'OK'; else, s = 'FAIL'; end
end

function banner(txt)
% BANNER  Print an underlined section heading in the Command Window/log;
% I should have just made a livescript but whatever
fprintf('\n%s\n%s\n', txt, repmat('-', 1, numel(txt)));
end

function saveFig(fig, outDir, name, doSave)
% SAVEFIG  Save a figure as a 200-dpi PNG.
%   exportgraphics crops white margins; older releases fall back
%   to print. ALTERNATIVE: save as .pdf or .eps (vector graphics) for a
%   LaTeX report: exportgraphics(fig, 'name.pdf', 'ContentType', 'vector').
if ~doSave, return; end
f = fullfile(outDir, [name '.png']);
try, exportgraphics(fig, f, 'Resolution', 200);   % R2020a+
catch, print(fig, f, '-dpng', '-r200'); end
end
