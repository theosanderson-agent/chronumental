"""Whether the data can support dating at all, checked before fitting.

chronumental will currently return a root in the year -1598 without comment.
The three failures behind numbers like that are all visible in O(n) from the
tree and the tip dates alone, before a single step is taken -- so the tool can
say what is wrong instead of quietly answering anyway.

What the checks are, and what each is evidence of:

Clock signal. If root-to-tip divergence does not increase with sampling date,
there is no molecular clock to read and every date is arbitrary. A negative
slope is already a hard error. A slope that is positive but barely correlated
is the same failure in weaker form, and it is the single best predictor of a
bad root among the datasets checked (Spearman -0.57 of the correlation
against root error, p = 0.002, n = 27).

Saturation. Above roughly 0.3 substitutions per site, sites are hit more than
once often enough that branch lengths understate divergence, and they do so
increasingly with depth -- so the clock is not linear and no per-branch
correction fixes it, because saturation accumulates along a path rather than
within a branch. Lassa's three segments sit at 1.56 to 2.13, where the
Jukes-Cantor correction is undefined.

What is deliberately not checked. "Few mutations per branch" was the obvious
third test and the data refutes it: ebola/ebov-2013 has a median of zero
mutations per branch and 52% of its branches empty, and its root lands within
half a year, because those empty branches end at dated tips that pin them.
The count that does mean something -- whether a node has any path of
mutation-carrying branches to a dated tip -- is per-node, not per-dataset, and
chronumental.uncertainty already reports it that way. At dataset level it runs
the wrong way entirely (Spearman -0.43 against root error): trees full of
empty branches are densely sampled recent outbreaks, which are the easy ones.

What this actually catches, on the 29 Nextstrain builds in chron_analysis.
Seven are flagged. The three worst roots are all among them -- lassa/s at 42.6
years out, WNV at 30.7, lassa/gpc at 29.9 -- as are the two datasets that
return no usable answer at all, norovirus and seasonal-cov/oc43. lassa/l is
flagged with a root only 1.4 years out but the worst internal disagreement in
the set, so flagging it is right for the other reason. dengue/all/genome is a
false positive: 0.361 substitutions per site, just over the saturation line,
and it dates correctly.

Two failures are missed, and they matter more than the false positive.
measles/N450/global is 15.6 years out and hmpv/all/genome 5.2, with none of
these statistics out of the ordinary. A clean report means nothing was
detected, not that the answer is right.

The correlation threshold separates cleanly on this sample: below 0.2 sit
WNV, all three Lassa segments, norovirus and seasonal-cov/oc43, while chikv at
0.39 and measles/genome at 0.31 are the nearest datasets that date correctly.
But it is 29 datasets, so read a warning as a prompt to look rather than a
verdict.
"""
import numpy as np

# Substitutions per site above which multiple hits stop being negligible.
SATURATION_PER_SITE = 0.3
# Correlation of root-to-tip divergence with date below which the clock signal
# is too weak to trust. See the module docstring for how this was chosen.
WEAK_CLOCK_CORRELATION = 0.2


def clock_signal(root_to_tip, dates_in_days):
    """Slope per year, and the correlation, of divergence against date."""
    years = np.asarray(dates_in_days, dtype=np.float64) / 365.25
    divergence = np.asarray(root_to_tip, dtype=np.float64)
    if len(years) < 3 or np.ptp(years) == 0 or np.ptp(divergence) == 0:
        return float("nan"), float("nan")
    slope = np.polyfit(years, divergence, 1)[0]
    correlation = np.corrcoef(years, divergence)[0, 1]
    return float(slope), float(correlation)


def report(root_to_tip, dates_in_days, genome_length=None):
    """Print what the data says about its own dateability; return the warnings.

    `root_to_tip` is each dated tip's divergence from the root in whatever
    units the tree carries; `genome_length` converts those to substitutions
    per site, and where it is unknown the saturation check is skipped and
    said to be skipped rather than guessed at.
    """
    warnings = []
    slope, correlation = clock_signal(root_to_tip, dates_in_days)
    if np.isnan(slope):
        print("Clock signal: not measurable from these tips.")
    else:
        print(f"Clock signal: root-to-tip divergence rises {slope:.3g} per "
              f"year, correlation {correlation:.2f} with sampling date.")
        if slope <= 0:
            warnings.append(
                "Root-to-tip divergence does not increase with sampling "
                "date, so these tips carry no clock signal. Any dates "
                "returned are determined by the priors, not by the data.")
        elif correlation < WEAK_CLOCK_CORRELATION:
            warnings.append(
                f"Root-to-tip divergence is only weakly related to sampling "
                f"date (correlation {correlation:.2f}). On the datasets this "
                f"was calibrated against, that was the best single warning "
                f"of a badly placed root. Treat deep dates with suspicion, "
                f"and consider --clock_filter_iqd or an externally estimated "
                f"--clock.")

    divergence = np.asarray(root_to_tip, dtype=np.float64)
    if genome_length:
        per_site = float(np.median(divergence)) / float(genome_length)
        print(f"Divergence: median root-to-tip {per_site:.3g} substitutions "
              f"per site.")
        if per_site > SATURATION_PER_SITE:
            warnings.append(
                f"Median root-to-tip divergence is {per_site:.2f} "
                f"substitutions per site. Above about "
                f"{SATURATION_PER_SITE} sites are hit more than once often "
                f"enough that branch lengths understate divergence, and "
                f"increasingly so with depth, so the clock is not linear. "
                f"Deep nodes will be dated too recently and no per-branch "
                f"correction fixes this.")
    else:
        print("Divergence: saturation not checked -- pass "
              "--treat_mutation_units_as_normalised_to_genome_size to give "
              "the genome length these branch lengths are relative to.")

    for warning in warnings:
        print("\nWARNING: " + warning)
    if warnings:
        print("\nchronumental will fit anyway. Nothing above proves the "
              "result is wrong, and a clean report does not prove it is "
              "right -- these checks catch some failures, not all.\n")
    return warnings
