import os
import sys

GPU_REQUESTED = "--use_gpu" in sys.argv
if not GPU_REQUESTED:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import datetime
import math

import pandas as pd
import jax.numpy as jnp
import numpy as np
from . import helpers
from . import input_mod
import collections
import jax
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from . import models
from . import uncertainty
from . import diagnostics
from scipy import stats

try:
    from . import _version
    version = _version.version
except ImportError:
    version = "dev"

print(f"Chronumental {version}")

platform = jax.default_backend()
print(f"Platform: {platform}")

if GPU_REQUESTED and platform == "cpu":
    print("GPU requested but was not available")
    print("This probably reflects your CUDA/jaxlib installation")

import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description=
        'Convert a distance tree into time tree with distances in days.')
    parser.add_argument(
        '--tree',
        help=
        'an input newick tree, potentially gzipped, with branch lengths reflecting genetic distance in integer number of mutations',
        required=True)

    parser.add_argument(
        '--dates',
        help=
        'A metadata file with columns strain and date (in "2020-01-02" format, or less precisely, "2021-01", "2021")',
        required=True)

    parser.add_argument(
        '--dates_out',
        default=None,
        type=str,
        help="Output for date tsv (otherwise will use default)")

    parser.add_argument('--tree_out',
                        default=None,
                        type=str,
                        help="Output for tree (otherwise will use default)")

    parser.add_argument(
        "--treat_mutation_units_as_normalised_to_genome_size",
        default=None,
        type=int,
        help=
        "If your branch sizes, and mutation rate, are normalised to per-site values, then enter the genome size here."
    )

    parser.add_argument(
        '--clock',
        help=
        'Molecular clock rate. This should be in units of something per year, where the "something" is the units on the tree. If not given we will attempt to estimate this by RTT. By default this value is held fixed; pass --floating_clock_rate to treat it as a starting point instead.',
        default=None,
        type=float)

    parser.add_argument(
        '--clock_estimator',
        choices=('theil-sen', 'phylogenetic'),
        default='theil-sen',
        help=(
            "Estimator for the starting clock rate when --clock is omitted. "
            "'theil-sen' is a robust root-to-tip regression that treats tips "
            "as independent observations even where they share ancestry. "
            "'phylogenetic' instead uses Felsenstein's independent contrasts, "
            "which corrects for that shared ancestry. Contrasts are the more "
            "principled estimator, but they fit a lower rate, and since the "
            "root is the node furthest from any dated tip it is the most "
            "sensitive to that: on deep trees the lower rate can push the "
            "root implausibly far back. 'phylogenetic' does better on "
            "shallow, densely sampled trees; 'theil-sen' is the safer "
            "default across a range of tree depths."))

    parser.add_argument(
        '--phylogenetic_clock_variance_floor',
        default=5.0,
        type=float,
        help=(
            "Minimum per-branch mutation variance used by "
            "--clock_estimator phylogenetic. The default prevents branches "
            "with very few observed mutations from being treated as nearly "
            "noiseless by the Gaussian contrast approximation."))

    parser.add_argument(
        '--clock_filter_iqd',
        default=0.0,
        type=float,
        help=(
            "Discard the date of any tip whose root-to-tip divergence sits "
            "more than this many interquartile ranges from the regression "
            "line, before fitting. Those tips keep their place in the tree "
            "and simply become undated. 0, the default, keeps every date. "
            "4 is a reasonable value if you suspect swapped or mistyped "
            "metadata: on datasets with a tight clock it identifies "
            "genuinely wrong dates with high precision and recovers much of "
            "the damage they do, but it finds only the worst of them -- two "
            "sequences of similar divergence can swap dates and leave almost "
            "no trace in this statistic -- so it is a mitigation rather than "
            "a fix."))

    parser.add_argument(
        '--profile_clock_rate',
        default=0,
        type=int,
        help=(
            "Choose the clock rate by profiling instead of by root-to-tip "
            "regression: refit the durations at this many fixed rates and "
            "take the best. Costs one fit per grid point, and 7 is a "
            "reasonable value. On simulations with a known rate this halved "
            "the error in the rate, and it is the only route that yields a "
            "confidence interval on the rate, and hence on the root date. "
            "0, the default, is off."))

    parser.add_argument(
        '--profile_clock_range',
        default=1.5,
        type=float,
        help=(
            "How wide the --profile_clock_rate grid is, as a multiple of the "
            "starting estimate either way. The grid recentres once if the "
            "best rate lands on an edge."))

    parser.add_argument(
        '--robust_passes',
        default=1,
        type=int,
        help=(
            "How many times to fit. After each fit but the last, tips the "
            "fitted tree cannot place are dropped and the fit is repeated "
            "without them. 1, the default, fits once and drops nothing. 3 is "
            "a reasonable value if you suspect wrong dates: unlike "
            "--clock_filter_iqd, which screens against a straight line "
            "before fitting, this compares each tip to where the rest of the "
            "tree puts it, which is a much sharper question -- but it costs "
            "one full fit per pass."))

    parser.add_argument(
        '--robust_iqd',
        default=4.0,
        type=float,
        help=(
            "How far from the fitted tree a tip's date may sit, in "
            "interquartile ranges of the residuals, before --robust_passes "
            "discards it."))

    parser.add_argument(
        '--no_confidence_intervals',
        action='store_true',
        help=(
            "Skip the confidence intervals on node dates. They are computed "
            "by default: the curvature of the objective at the fitted point "
            "is tree-sparse when written in node dates, so the intervals "
            "cost one more pass over the tree and no refitting. See "
            "chronumental.uncertainty."))

    parser.add_argument(
        '--confidence_conditions_on_clock_rate',
        action='store_true',
        help=(
            "Report intervals that hold the clock rate at its fitted value "
            "rather than propagating the rate's own uncertainty. The root's "
            "date is roughly the oldest tip minus divergence over the rate, "
            "so conditioning makes deep intervals much too narrow; this is "
            "here for comparison rather than for use."))

    parser.add_argument(
        '--variance_dates',
        default=3.0,
        type=float,
        help=
        ("How uncertain the reported tip dates are taken to be. It multiplies "
         "each tip's date-precision category (1 for a full date, 30 for "
         "month-only, 365 for year-only) to give the standard deviation of "
         "the date likelihood, in days. The old default of 0.3 treated a full "
         "date as known to within about seven hours, which over-constrains "
         "the fit: the tip dates then pin each root-to-tip total so tightly "
         "that the mutation counts have almost no say in how that total is "
         "divided among the branches on the path.")
    )

    parser.add_argument(
        '--steps',
        default=20000,
        type=int,
        help=
        "Upper bound on the number of SVI steps. By default this is a ceiling, "
        "not a target: fitting stops earlier once the predicted dates stop "
        "changing by more than --convergence_tol_days, since extra steps beyond "
        "that point change the answer only in ways too small to matter. Pass "
        "--disable_early_stopping to always run exactly this many steps, e.g. "
        "for a reproducible step count."
    )

    parser.add_argument('--lr',
                        default=0.03,
                        type=float,
                        help="Adam learning rate")

    parser.add_argument(
        '--convergence_rate_tol',
        default=0.002,
        type=float,
        help=
        ("Relative change in the fitted clock rate below which it counts as "
         "settled, for early stopping. Checked alongside "
         "--convergence_tol_days because node dates can look stable while "
         "the rate is still moving, and the dates are more sensitive to the "
         "rate than a test on the dates alone can detect.")
    )

    parser.add_argument(
        '--convergence_tol_days',
        default=0.1,
        type=float,
        help=
        ("Stop once the mean absolute change in predicted node dates between "
         "convergence checks falls below this many days. The old default of 1 "
         "day stopped too early: the node dates had settled but the clock "
         "rate was still moving, and the rate is what the dates are most "
         "sensitive to. On one real dataset a 5%% error in the fitted rate "
         "cost 3 days of median disagreement with treetime, and tightening "
         "this from 1 to 0.1 took that disagreement from 11.3 days to 8.1. "
         "Simulated benchmarks improve too, from 14.9 to 14.1 days mean "
         "absolute error, for about 11%% more runtime.")
    )

    parser.add_argument(
        '--convergence_check_every',
        default=50,
        type=int,
        help="How often, in steps, to evaluate the early-stopping criterion.")

    parser.add_argument(
        '--convergence_patience',
        default=3,
        type=int,
        help=
        "Number of consecutive checks (each --convergence_check_every steps apart) "
        "that must be below --convergence_tol_days before stopping early.")

    parser.add_argument(
        '--disable_early_stopping',
        action='store_true',
        help=
        "Always run the full --steps, ignoring the convergence criterion. Use this "
        "if you need an exact, reproducible number of SVI steps.")

    parser.add_argument('--name_all_nodes',
                        action='store_true',
                        help="Should we name all nodes in the output tree?")

    parser.add_argument(
        '--expected_min_between_transmissions',
        default=3,
        type=int,
        help=
        "For forming the prior, an expected minimum time between transmissions in days"
    )

    parser.add_argument(
        '--only_use_full_dates',
        action='store_true',
        help="Only use full dates, given to the precision of a day")

    parser.add_argument(
        '--output_unit',
        type=str,
        help="Unit for the output branch lengths on the time tree.",
        choices=["days", "years"],
        default="days")

    parser.add_argument(
        '--multiply_date_precision',
        action='store_true',
        help=
        "Restore the old date-likelihood scale, --variance_dates multiplied "
        "by each tip's precision window, rather than the two combined in "
        "quadrature. Multiplying conflates the width of the reported interval "
        "with how far a reported date can be from the truth for other "
        "reasons, so raising one inflates the other: at the default it made a "
        "year-only date uncertain to within ten years.")

    parser.add_argument(
        '--variance_on_clock_rate',
        action='store_true',
        help=("Will cause the clock rate to be "
              "drawn from a random distribution with a learnt variance. "
              "Requires --floating_clock_rate, since a fixed rate "
              "has no variance to learn."))

    parser.add_argument(
        '--root_date_prior_scale_days',
        default=36500.0,
        type=float,
        help=("Scale of the Cauchy prior on the root date, in days before "
              "the oldest tip. Heavy-tailed, so this mainly sets where the "
              "prior stops being flat rather than how far back the root may "
              "go; the default of 100 years is uninformative for almost any "
              "tree."))

    parser.add_argument(
        '--clock_likelihood',
        choices=('poisson', 'gamma-poisson'),
        default='poisson',
        help=(
            "Mutation-count likelihood. The default Poisson is a strict "
            "clock. 'gamma-poisson' marginalizes independent Gamma-distributed "
            "branch rates, producing a negative-binomial likelihood with a "
            "single learned branch-rate coefficient of variation."))

    parser.add_argument(
        '--branch_rate_cv_init',
        default=0.3,
        type=float,
        help=(
            "Starting coefficient of variation for branch rates under "
            "--clock_likelihood gamma-poisson. This is optimized during the "
            "fit, not held fixed."))

    parser.add_argument(
        '--floating_clock_rate',
        action='store_true',
        help=(
            "Fit the clock rate as a free parameter instead of holding it at "
            "the estimate. This was the behaviour before the rate was fixed "
            "by default. It lets the fit recover from a poor starting "
            "estimate, at the cost of a rate biased low: fitting one free "
            "duration per branch alongside a single shared rate is the "
            "classic incidental-parameters problem, and the free fit landed "
            "below the reference rate on 18 of 24 real datasets."))

    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help=
        ("Will attempt to use the GPU. You will need a version of CUDA installed to suit Numpyro."
         ))

    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help=
        "This flag will trigger the use of Weights and Biases to log the fitting process. This must be installed with 'pip install wandb'"
    )

    parser.add_argument('--wandb_project_name',
                        default="chronumental",
                        type=str,
                        help="Wandb project name")

    parser.add_argument('--clipped_adam',
                        action='store_true',
                        help=("Will use the clipped version of Adam"))

    parser.add_argument(
        '--reference_node',
        default=None,
        type=str,
        help=
        "A reference node to use for computing dates. This should be early in the tree, and have a correct date. If not specified it will be picked as the oldest node, but often these can be metadata errors."
    )

    parser.add_argument(
        '--always_use_final_params',
        action='store_true',
        help=
        "Will force the model to always use the final parameters, rather than simply using those that gave the lowest loss"
    )

    return parser


def _make_convergence_check(path_sum):
    """Build the two jitted functions behind the early-stopping convergence
    check.

    `node_days` computes every node's predicted date on device, reusing the
    same pointer-jumping path sum the model uses for terminal dates.
    `mean_abs_change` reduces two such arrays' difference to a single scalar,
    also on device, so a check costs one scalar host sync rather than an
    O(nodes) transfer and a walk in Python.

    Because the path sum already produces every node's date, the check needs
    no structure of its own. It previously built a second sparse matrix over
    all nodes, which cost about a gigabyte at 100k tips and had to be
    subsampled to stay affordable.

    Kept as two functions, rather than one that also does the comparison,
    because there is no previous value to compare against on a check's first
    call.
    """

    @jax.jit
    def node_days(branch_times, root_date):
        return path_sum(branch_times) + root_date

    @jax.jit
    def mean_abs_change(current_node_days, previous_node_days):
        return jnp.mean(jnp.abs(current_node_days - previous_node_days))

    return node_days, mean_abs_change


def prepend_to_file_name(full_path, to_prepend):
    if "/" in full_path:
        path, file = full_path.rsplit('/', 1)
        return f"{path}/{to_prepend}_{file}"
    else:
        return f"{to_prepend}_{full_path}"


TerminalState = collections.namedtuple(
    "TerminalState", "names indices dates errors target_dates")


def _drop_terminals(terminals, keep):
    """The same tip state with a boolean mask applied."""
    names = [name for name, keeping in zip(terminals.names, keep) if keeping]
    kept = set(names)
    return TerminalState(
        names=names,
        indices=terminals.indices[keep],
        dates=terminals.dates[keep],
        errors=terminals.errors[keep],
        target_dates={name: value
                      for name, value in terminals.target_dates.items()
                      if name in kept})


def _fit_with_robustness_passes(args, run_fit, build_model, clock_rate,
                                terminals, branch_distances_array,
                                parent_indices, root_index, path_sum):
    """Fit, look at what the fit could not explain, drop it, and fit again.

    --clock_filter_iqd screens tips before fitting, against the root-to-tip
    regression. That statistic is noisy: it compares a tip only to a straight
    line through the whole dataset, so a wrong date on a tip whose divergence
    happens to suit the line leaves no trace, and it measured 3-76% recall
    across the contaminated simulations. Residuals against the *fitted tree*
    ask a sharper question -- how far is this tip from where its own place in
    the tree puts it -- and they only exist after a fit, which is why this is
    a loop rather than a filter.

    The residual has to be the tree's opinion of the tip, not the fit's. A
    tip's date likelihood has a standard deviation of about three days, so the
    fit believes the date it was given: hand it a date that belongs to another
    sequence and it drags that tip onto the wrong date, leaving a residual of
    well under a day. Measured on contaminated simulations, the fitted-minus-
    reported residual had an interquartile range of 0.7 days and found 17% of
    the swaps. So the comparison here is against where the tree alone puts the
    tip -- its parent's fitted date plus what its own branch's mutations say
    the gap should be -- which does not use the tip's reported date at all
    except to disagree with it.

    This is ordinary iteratively reweighted robust regression, with hard
    rejection rather than a weight function because a tip's date either
    belongs to that tip or belongs to some other sequence entirely; there is
    not much in between. Each pass drops what the previous fit could not
    explain and refits without it. One pass, the default, is the old
    behaviour exactly.

    What no screen can do is find an error smaller than the sampling window.
    Two tips exchanging dates within a two-year window are wrong by eight
    months on average, which is the same size as an ordinary tip's residual,
    and nothing separates them. The same swap in a twenty-year window is
    obvious. Expect this to help where the dates span years and not where
    they span months.

    Measured on simulated trees of 400 tips with 10% of dates swapped in
    pairs, over a twenty-year sampling window, as median and mean error on
    internal nodes in days:

        no screen              214.5   944.3     0 tips dropped
        --clock_filter_iqd 4    50.1   197.0    17.7
        --robust_passes 3      130.2   681.0    97.7
        both                    39.7   179.8    52.0

    Use it with --clock_filter_iqd, not instead of it. On its own it throws
    out a quarter of the tree to catch a tenth of it and lands worse than the
    cheap up-front filter; after that filter has taken the obvious cases it
    finds another 21% off the median error. The signal is weak -- the
    standardised residual has a median of 1.7 for a correct tip and 3.9 for a
    swapped one -- so it is a screen with poor precision that happens to be
    worth its false positives, not a detector.

    On uncontaminated data all four configurations agree to the day and this
    drops nothing at all, which is the property that matters most.
    """
    model = build_model(clock_rate, terminals)
    params, was_interrupted, _loss = run_fit(model)

    for pass_number in range(1, max(args.robust_passes, 1)):
        if was_interrupted:
            break
        indices = np.asarray(terminals.indices)
        if len(indices) < 20:
            print("Robustness pass: too few dated tips to judge outliers.")
            break
        residual, residual_sd = _tree_residuals(
            model, params, terminals, branch_distances_array, parent_indices,
            root_index, path_sum)
        # Standardised, not ranked against a global spread. Branches differ
        # enormously in how well the tree determines them -- a tip on a branch
        # with three mutations is placed to within months, one with sixty to
        # within days -- so a single interquartile range over all of them
        # flags the badly determined rather than the badly dated. Measured on
        # contaminated simulations, thresholding the raw residual threw out 74
        # of 400 tips to catch 40 wrong ones and made the answer worse.
        standardised = np.abs(residual) / residual_sd
        keep = ~(standardised > args.robust_iqd)
        dropped = int((~keep).sum())
        if dropped == 0:
            print(f"Robustness pass {pass_number}: every tip sits within "
                  f"{args.robust_iqd} standard errors of where the tree puts "
                  f"it; nothing more to drop.")
            break
        if dropped >= len(keep) - 20:
            print("Robustness pass: this would drop almost every tip, which "
                  "means the residuals are not outliers but the whole fit. "
                  "Keeping all dates.")
            break
        worst = float(np.max(np.abs(residual[~keep])))
        print(f"Robustness pass {pass_number}: dropping the dates of "
              f"{dropped} of {len(keep)} tips that sit more than "
              f"{args.robust_iqd} standard errors from where the tree puts "
              f"them (worst {worst:.0f} days out). Refitting without them.")
        terminals = _drop_terminals(terminals, keep)
        model = build_model(clock_rate, terminals)
        params, was_interrupted, _loss = run_fit(model)

    return params, was_interrupted, model


def _tree_residuals(model, params, terminals, branch_distances_array,
                    parent_indices, root_index, path_sum):
    """How far each tip's date is from where the tree puts it, in its own SDs.

    The tree's opinion of a tip is its parent's fitted date plus the duration
    this branch's mutations imply, and that opinion has a standard error of
    its own: the parent's, from the same curvature the confidence intervals
    use, and the branch's, which for k mutations is the duration over root k.
    Dividing by it is what separates a badly dated tip from a badly determined
    one.

    A branch with no mutations gets an infinite standard error and therefore a
    residual of zero, which is right: it has no opinion about how long it is,
    so it cannot disagree with anything.
    """
    branch_times = np.asarray(model.get_branch_times(params))
    node_dates = np.asarray(path_sum(model.get_branch_times(params))) + float(
        params['root_date_mu'])
    sd = uncertainty.node_date_intervals(
        parent_indices=np.asarray(parent_indices),
        root_index=root_index,
        mutations_per_branch=np.asarray(branch_distances_array),
        branch_times=branch_times,
        node_dates=node_dates,
        terminal_indices=np.asarray(terminals.indices),
        terminal_sigmas=np.asarray(model.get_date_sigmas()),
        clock_rate=float(model.get_mutation_rate(params)))[0]

    indices = np.asarray(terminals.indices)
    parents = np.asarray(parent_indices)[indices]
    rate = float(model.get_mutation_rate(params))
    mutations = np.asarray(branch_distances_array)[indices]
    implied = mutations * helpers.DAYS_PER_YEAR / max(rate, 1e-12)
    predicted = node_dates[parents] + implied

    with np.errstate(divide="ignore", invalid="ignore"):
        # Var(k) = k for a Poisson, so the implied duration k*Y/rate has a
        # standard error of itself over root k.
        duration_sd = np.where(mutations > 0, implied / np.sqrt(
            np.maximum(mutations, 1e-12)), np.inf)
    parent_sd = np.nan_to_num(sd[parents], nan=np.inf)
    tip_sd = np.asarray(model.get_date_sigmas())
    residual_sd = np.sqrt(parent_sd**2 + duration_sd**2 + tip_sd**2)
    return predicted - np.asarray(terminals.dates), residual_sd


def _profile_clock_rate(args, run_fit, build_model, initial_rate, terminals):
    """Choose the clock rate by profiling, and get an interval for it.

    Refitting the durations with the rate held at each of a series of values
    and taking the best is the profile likelihood. It is worth being exact
    about what that does and does not do.

    It does not remove the Neyman-Scott bias. The objective has one shared
    rate against a duration per branch, the shared rate comes out biased, and
    profiling cannot help with that because

        argmax_mu [ max_t L(mu, t) ]  =  the joint maximiser

    by construction. Profiling *is* the joint MLE, not a correction to it.
    Cox and Reid's adjusted profile likelihood is the textbook correction and
    was tried: it fails here, running to whichever end of the grid it is
    given, because the durations are nowhere near orthogonal to the rate --
    scaling the rate by a and every duration by 1/a leaves the count
    likelihood almost unchanged, so the adjustment term tracks that scaling
    rather than measuring anything. On simulations it made the bias worse,
    -27.7% against -9.9%.

    What it does do is beat what chronumental does today, for two reasons
    that have nothing to do with the bias. First, the default pins the rate at
    a root-to-tip regression, which is a worse estimator than the MLE: on
    simulations with a known rate of 8.00e-4, the default read 7.20e-4 (-9.9%)
    and the profile maximum 8.38e-4 (+4.8%). Second, the alternative of
    co-fitting the rate does not actually reach the joint optimum -- the
    co-fitted rate moved 0.7% in 34,000 further steps, and a fixed-rate fit
    reached a lower loss than the floating fit that has the extra freedom.

    And it is the only route to a rate interval that works. The closed-form
    one is a small difference of large numbers and comes out negative on real
    trees; see chronumental.uncertainty. The profile's curvature in log(rate)
    is clean, measured at a standard error near 0.035 on simulations. Because
    each grid point also has its own fitted root, the root's interval comes
    straight off the same grid rather than needing anything further.

    The cost is one fit per grid point, and the grid can widen once if the
    best rate lands on an edge.
    """
    points = max(args.profile_clock_rate, 3)
    span = math.log(max(args.profile_clock_range, 1.01))
    centre = math.log(initial_rate)
    grid, losses, roots = [], [], []

    for attempt in range(2):
        wanted = np.exp(np.linspace(centre - span, centre + span, points))
        for rate in wanted:
            if any(abs(math.log(rate / seen)) < 1e-9 for seen in grid):
                continue
            print(f"Profiling clock rate: fitting at {rate:.6g}")
            params, interrupted, loss = run_fit(build_model(rate, terminals))
            if interrupted or not np.isfinite(loss):
                continue
            grid.append(float(rate))
            losses.append(float(loss))
            roots.append(float(params['root_date_mu']))
        order = np.argsort(grid)
        grid = [grid[i] for i in order]
        losses = [losses[i] for i in order]
        roots = [roots[i] for i in order]
        if len(grid) < 3:
            print("Profiling the clock rate did not produce enough usable "
                  "fits; keeping the estimated rate.")
            return initial_rate, None
        best = int(np.argmin(losses))
        if 0 < best < len(grid) - 1 or attempt == 1:
            break
        # The best rate is on an edge, so the grid was in the wrong place.
        centre = math.log(grid[best])
        print(f"Profiling clock rate: best value was at the edge of the "
              f"grid; recentring on {grid[best]:.6g}")

    rate, profile = uncertainty.profile_interval(
        np.array(grid), np.array(losses), np.array(roots), initial_rate)
    if profile is None:
        print("The profiled likelihood has no maximum inside the grid, so "
              "there is nothing to read a rate or an interval off. Keeping "
              "the estimated rate. Widening --profile_clock_range may help.")
    return rate, profile


def _report_profile(profile, origin_date):
    """Print the profiled rate and root, with their intervals."""
    print("")
    print(f"Profiled clock rate {profile['rate']:.6g} "
          f"(95% {profile['rate_low']:.6g} to {profile['rate_high']:.6g}; "
          f"standard error {profile['standard_error_log_rate']:.4f} on the "
          f"log scale)")
    dates = _days_to_dates(origin_date, [profile["root_low"],
                                         profile["root"],
                                         profile["root_high"]])
    if all(date is not None for date in dates):
        print(f"Profiled root date {dates[1]:%Y-%m-%d} "
              f"(95% {dates[0]:%Y-%m-%d} to {dates[2]:%Y-%m-%d})")
    print("This interval covers the clock rate's uncertainty and nothing "
          "else: the tree, the topology and the tip dates are all taken as "
          "given.")
    print("")


def _days_to_dates(origin_date, days):
    """Calendar dates for day offsets, clamped to what datetime can hold.

    An unidentified node's bound can be arbitrarily far away, and a date
    thousands of years out of range should come back as the end of the
    representable range rather than crash the whole run at the last step.
    """
    # The origin can arrive as a pandas Timestamp or a plain datetime
    # depending on how the metadata parsed, and the two disagree about what
    # years they can represent, so bound against the plain calendar and let
    # anything the origin's own type still rejects come back empty.
    plain = origin_date.date() if hasattr(origin_date, "date") else origin_date
    low = (datetime.date.min - plain).days + 1
    high = (datetime.date.max - plain).days - 1
    out = []
    for day in days:
        if not np.isfinite(day):
            out.append(None)
            continue
        try:
            out.append(origin_date +
                       datetime.timedelta(days=float(min(max(day, low),
                                                         high))))
        except (OverflowError, ValueError):
            out.append(None)
    return out


def _confidence_columns(args, my_model, params, path_sum, parent_indices,
                        root_index, branch_distances_array, terminal_indices,
                        name_to_pos, names, origin_date):
    """The interval columns for the dates file, or none if they cannot be had.

    A failure here must not cost someone the fit they just waited for, so the
    dates are still written -- with a warning saying what was lost, rather
    than silently.
    """
    try:
        branch_times = np.asarray(my_model.get_branch_times(params))
        node_days = np.asarray(path_sum(jnp.asarray(branch_times))) + float(
            params['root_date_mu'])
        sd, identified, lower, upper, rate_included = (
            uncertainty.node_date_intervals(
                parent_indices=np.asarray(parent_indices),
                root_index=root_index,
                mutations_per_branch=np.asarray(branch_distances_array),
                branch_times=branch_times,
                node_dates=node_days,
                terminal_indices=np.asarray(terminal_indices),
                terminal_sigmas=np.asarray(my_model.get_date_sigmas()),
                clock_rate=float(my_model.get_mutation_rate(params)),
                include_rate_uncertainty=(
                    not args.confidence_conditions_on_clock_rate)))
    except Exception as exception:  # noqa: BLE001 - reported, not swallowed
        print(f"Could not compute confidence intervals ({exception}); "
              f"writing dates without them.")
        return {}

    if not rate_included and not args.confidence_conditions_on_clock_rate:
        print("These intervals are conditional on the clock rate: the "
              "closed-form correction for the rate's own uncertainty came "
              "out negative, which is what it does on real trees. Deep "
              "nodes are therefore narrower than they should be. See "
              "chronumental/uncertainty.py for why.")
    positions = np.array([name_to_pos[name] for name in names])
    unresolved = int((~identified).sum())
    if unresolved:
        print(f"{unresolved} of {len(identified)} nodes sit in parts of the "
              f"tree with no mutations to date them. Their dates are bounded "
              f"by their neighbours rather than estimated, and are marked in "
              f"the bounded_by_tree_order column.")
    columns = {
        "lower_95": _days_to_dates(origin_date, lower[positions]),
        "upper_95": _days_to_dates(origin_date, upper[positions]),
        "date_sd_days": sd[positions],
    }
    if unresolved:
        columns["bounded_by_tree_order"] = ~identified[positions]
    return columns


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.use_wandb:
        try:
            import wandb
        except ImportError:
            raise ValueError(
                "Wandb not installed. Please install it with `pip install wandb`"
            )
        wandb.init(project=args.wandb_project_name)
        wandb.config.update(args)

    # Whether the early-stopping convergence check will run at all, decided
    # up front because it also gates building the extra sparse matrix the
    # check needs (see below) -- no point paying for that on a run that will
    # never use it.
    check_convergence = (not args.disable_early_stopping
                         and args.convergence_tol_days > 0)

    if args.dates_out is None:
        args.dates_out = prepend_to_file_name(args.dates,
                                              "chronumental_dates") + ".tsv"

    if args.tree_out is None:
        args.tree_out = prepend_to_file_name(args.tree,
                                             "chronumental_timetree")

    metadata = input_mod.get_metadata(args.dates)

    print("Reading tree")
    tree = input_mod.read_tree(args.tree)

    print("Processing dates")
    input_mod.process_dates(metadata)

    full = input_mod.get_present_dates(
        metadata, only_use_full_dates=args.only_use_full_dates)
    lookup = dict(
        zip(full['strain'],
            zip(full['processed_date'], full['processed_date_error'])))

    # Get oldest date in full, and corresponding strain:
    if args.reference_node:
        reference_point, ref_point_distance = input_mod.get_specific(
            full, tree, args.reference_node)
    else:
        reference_point, ref_point_distance = input_mod.get_oldest(full, tree)

    if args.treat_mutation_units_as_normalised_to_genome_size:
        ref_point_distance = ref_point_distance * args.treat_mutation_units_as_normalised_to_genome_size

    print(
        f"Using {reference_point}, with date: {lookup[reference_point][0]} and distance from root {ref_point_distance} as an arbitrary reference point"
    )

    target_dates, target_errors = input_mod.get_target_dates(
        tree, lookup, reference_point)
    terminal_names = sorted(target_dates.keys())

    terminal_target_dates_array = jnp.asarray(
        [float(target_dates[x]) for x in terminal_names])

    terminal_target_errors_array = jnp.asarray(
        [float(target_errors[x]) for x in terminal_names])

    print(
        f"Found {len(terminal_names)} terminals with usable date metadata{' [full date mode is on]' if args.only_use_full_dates else ''}"
    )

    terminal_name_to_pos = {x: i for i, x in enumerate(terminal_names)}

    initial_branch_lengths, invented_labels = (
        input_mod.get_initial_branch_lengths_and_name_all_nodes(tree))
    names_init = sorted(initial_branch_lengths.keys())
    branch_distances_array = jnp.array(
        [initial_branch_lengths[x] for x in names_init])
    if args.treat_mutation_units_as_normalised_to_genome_size:
        branch_distances_array = branch_distances_array * args.treat_mutation_units_as_normalised_to_genome_size

    name_to_pos = {x: i for i, x in enumerate(names_init)}

    # Root-to-tip sums are computed by pointer jumping over a parent-index
    # array rather than a sparse matrix of (node, ancestor) pairs. The sparse
    # form cost memory proportional to the total path length over the tree --
    # 116 million entries and 2.6 GB on a 300k-tip tree, the largest single
    # allocation in a run. See helpers.make_path_sum.
    parent_indices, root_index, max_depth = input_mod.get_parent_indices(
        tree, name_to_pos)
    n_rounds = max(1, int(math.ceil(math.log2(max_depth + 1))))
    print(f"Tree depth {max_depth}; using {n_rounds} pointer-jumping rounds")
    path_sum = helpers.make_path_sum(jnp.asarray(parent_indices), n_rounds,
                                     root_index)
    terminal_indices = jnp.asarray(
        [name_to_pos[name] for name in terminal_names], dtype=jnp.int32)

    if args.clock_filter_iqd > 0:
        residuals = np.asarray(path_sum(branch_distances_array)[terminal_indices])
        days = np.asarray(terminal_target_dates_array)
        if len(days) >= 20:
            slope, intercept = np.polyfit(days, residuals, 1)
            offset = residuals - (slope * days + intercept)
            q1, q3 = np.percentile(offset, [25, 75])
            span = q3 - q1
            middle = np.median(offset)
            keep = np.abs(offset - middle) <= args.clock_filter_iqd * span
            dropped = int((~keep).sum())
            if dropped and dropped < len(days):
                print(f"Clock filter: dropping the dates of {dropped} of "
                      f"{len(days)} tips more than {args.clock_filter_iqd} "
                      f"interquartile ranges off the root-to-tip line")
                terminal_names = [n for n, k in zip(terminal_names, keep) if k]
                terminal_target_dates_array = terminal_target_dates_array[keep]
                terminal_target_errors_array = terminal_target_errors_array[keep]
                terminal_name_to_pos = {x: i for i, x in enumerate(terminal_names)}
                target_dates = {k: v for k, v in target_dates.items()
                                if k in terminal_name_to_pos}
                terminal_indices = jnp.asarray(
                    [name_to_pos[name] for name in terminal_names],
                    dtype=jnp.int32)
            elif dropped:
                print("Clock filter: would drop every tip; keeping all dates")

    # What the data says about whether it can be dated at all, before any
    # time is spent fitting it. See chronumental.diagnostics: these are O(n)
    # and they catch some failures, not all.
    diagnostics.report(
        np.asarray(path_sum(branch_distances_array)[terminal_indices]),
        np.asarray(terminal_target_dates_array),
        genome_length=args.treat_mutation_units_as_normalised_to_genome_size)

    if args.clock:
        print(f"Using clock rate {args.clock}")
        clock_rate = args.clock
        if args.treat_mutation_units_as_normalised_to_genome_size:
            clock_rate = clock_rate * args.treat_mutation_units_as_normalised_to_genome_size
        clock_candidates = [('provided', clock_rate)]
    else:
        root_to_tip = path_sum(branch_distances_array)[terminal_indices]

        print(
            "No clock rate specified, performing root-to-tip regression to estimate starting value"
        )
        # Root-to-tip regression to get a starting value for the clock rate.
        # This seeds the prior (Uniform(0, clock_rate * 1000)) and every
        # initial value in the guide, so a bad estimate here poisons the
        # whole fit. Ordinary least squares is not robust: under a relaxed
        # (non-strict) clock, root-to-tip divergence is noisy, and a
        # handful of tips with extreme dates or divergences get high
        # leverage and can drag the unweighted OLS slope far from the
        # truth. Theil-Sen (the median of all pairwise slopes) has a 29%
        # breakdown point and is far less sensitive to such outliers,
        # while agreeing closely with OLS when the strict-clock
        # assumption actually holds.
        x = np.asarray(terminal_target_dates_array)
        y = np.asarray(root_to_tip)
        # Theil-Sen is O(n^2) in the number of tips (it forms every
        # pairwise slope), which is fine for hundreds or a few thousand
        # tips but can exhaust memory on the much larger real-world trees
        # chronumental is sometimes run on. Cap the number of points fed
        # to it by subsampling; a few thousand tips is already far more
        # than needed to estimate one slope robustly.
        max_points_for_theilsen = 5000
        if x.shape[0] > max_points_for_theilsen:
            rng = np.random.default_rng(0)
            idx = rng.choice(x.shape[0],
                             size=max_points_for_theilsen,
                             replace=False)
            x_fit, y_fit = x[idx], y[idx]
        else:
            x_fit, y_fit = x, y
        slope_per_day, intercept, lo_slope, hi_slope = stats.theilslopes(
            y_fit, x_fit)
        theil_sen_rate = slope_per_day * helpers.DAYS_PER_YEAR
        if args.clock_estimator == 'phylogenetic':
            phylogenetic_rate = input_mod.estimate_clock_rate_phylogenetic(
                tree, name_to_pos, np.asarray(branch_distances_array),
                np.asarray(terminal_indices),
                np.asarray(terminal_target_dates_array),
                np.asarray(terminal_target_errors_array),
                variance_floor=args.phylogenetic_clock_variance_floor)

            clock_candidates = [('phylogenetic', phylogenetic_rate)]
        else:
            clock_candidates = [('theil-sen', theil_sen_rate)]

        for candidate_name, candidate_rate in clock_candidates:
            print(f"{candidate_name} clock regression: got rate of: "
                  f"{candidate_rate}")
        if any(rate <= 0 for _, rate in clock_candidates):
            raise ValueError(
                "ERROR: Root-to-tip regression predicted a negative mutation rate. If your dataset is correct you will need to manually specify an initial clock rate with --clock."
            )

    if (any(rate < 1 for _, rate in clock_candidates)
            and not args.treat_mutation_units_as_normalised_to_genome_size):
        raise ValueError(
            "Clock rate is less than 1 mutation per year. This probably means you need to specify a genome_size with --treat_mutation_units_as_normalised_to_genome_size size. If you are sure that you do not, set that parameter to 1.0."
        )

    # The clock rate is held at the estimate unless asked otherwise. Fitting
    # it jointly with one free duration per branch biases it low -- those
    # durations are incidental parameters that grow with the tree -- and on
    # real data the free fit landed below the reference rate on 18 of 24
    # datasets, costing 10% in median accuracy against published time trees.
    if args.variance_on_clock_rate and not args.floating_clock_rate:
        raise ValueError(
            "--variance_on_clock_rate needs a rate to put variance on, so it "
            "requires --floating_clock_rate.")
    fix_clock_rate = not args.floating_clock_rate

    def build_model(candidate_rate, terminals):
        model_configuration = {
            "clock_rate": candidate_rate,
            "variance_dates": args.variance_dates,
            "expected_min_between_transmissions": args.expected_min_between_transmissions,
            "quadrature_date_scale": not args.multiply_date_precision,
            "fix_clock_rate": fix_clock_rate,
            "variance_on_clock_rate": args.variance_on_clock_rate,
            "clock_likelihood": args.clock_likelihood,
            "branch_rate_cv_init": args.branch_rate_cv_init,
            "root_date_prior_scale": args.root_date_prior_scale_days,
        }

        branch_time_init, initial_root_date = (
            input_mod.estimate_initial_times_local(
                tree, name_to_pos, branch_distances_array,
                terminals.target_dates, candidate_rate))
        initial_branch_times_array = jnp.asarray(
            [branch_time_init[x] for x in names_init])

        return models.DeltaGuideWithStrictLearntClock(
            path_sum=path_sum,
            terminal_indices=terminals.indices,
            branch_distances_array=branch_distances_array,
            terminal_target_dates_array=terminals.dates,
            terminal_target_errors_array=terminals.errors,
            ref_point_distance=ref_point_distance,
            model_configuration=model_configuration,
            terminal_names=terminals.names,
            initial_branch_times_array=initial_branch_times_array,
            initial_root_date=initial_root_date)

    terminal_state = TerminalState(names=terminal_names,
                                   indices=terminal_indices,
                                   dates=terminal_target_dates_array,
                                   errors=terminal_target_errors_array,
                                   target_dates=target_dates)

    def run_fit(my_model):
        """Fit one model to convergence, and hand back its parameters.

        Pulled out of main so that a robustness pass can call it more than
        once: refitting after dropping outlying tips needs the identical
        fit, not a copy of it that can drift.
        """
        print("Performing SVI:")
        num_steps = args.steps
        optimiser = optim.ClippedAdam(
            args.lr) if args.clipped_adam else optim.Adam(args.lr)


        svi = SVI(my_model.model, my_model.guide, optimiser, Trace_ELBO())
        state = svi.init(jax.random.PRNGKey(0))

        was_interrupted = False

        # Early-stopping convergence check: has the fit's predicted output
        # (root date plus every node's cumulative branch time) actually stopped
        # moving? This is a better stopping signal than the loss, which can keep
        # crawling long after the dates a user would see have settled -- see
        # _make_convergence_check for how this is kept cheap on a huge tree.
        # (check_convergence itself was decided earlier, before it was needed to
        # decide whether to build the sparse matrix below.)
        convergence_check_every = max(args.convergence_check_every, 1)
        if check_convergence:
            convergence_node_days_fn, convergence_mean_abs_change_fn = (
                _make_convergence_check(path_sum))
        previous_node_days = None
        consecutive_converged = 0
        previous_rate = None
        checks_done = 0

        # Run the fitting steps in chunks under jax.lax.scan rather than one at a
        # time from Python. The dominant cost of the old loop was not the gradient
        # computation but the per-step round trip: every step dispatched from
        # Python, and reading `loss` to test it forced a blocking device-to-host
        # transfer. Copying the whole parameter set off the device whenever the
        # loss improved, which early in fitting is most steps, cost more again.
        #
        # Chunk boundaries are chosen so the logged step numbers are unchanged
        # (step 0, then every tenth, then the last), because that progress output
        # is something people watch. Best-loss parameters are tracked on device
        # with jnp.where instead of being copied to the host.
        #
        # One behaviour change: Ctrl-C is only noticed between chunks, so it now
        # stops within CHUNK_SIZE steps rather than after exactly one. A gradient
        # explosion is likewise reported once per chunk rather than once per step.
        CHUNK_SIZE = 10

        def scan_body(carry, _):
            state, lowest_loss, best_params = carry
            state, loss = svi.update(state)
            current_params = svi.get_params(state)
            # A NaN loss compares False here, so NaN steps never become "best".
            improved = loss < lowest_loss
            new_lowest_loss = jnp.where(improved, loss, lowest_loss)
            new_best_params = jax.tree_util.tree_map(
                lambda best, current: jnp.where(improved, current, best),
                best_params, current_params)
            return (state, new_lowest_loss, new_best_params), (loss,
                                                               jnp.isnan(loss))

        def run_chunk(state, lowest_loss, best_params, length):
            carry, (losses, nans) = jax.lax.scan(scan_body,
                                                 (state, lowest_loss, best_params),
                                                 xs=None,
                                                 length=length)
            return carry + (losses, nans)

        run_chunk = jax.jit(run_chunk, static_argnames=("length", ))

        lowest_loss = jnp.inf
        # Seeded with the initial parameters, so this is never unset even if every
        # step produces a NaN loss.
        best_params = svi.get_params(state)

        step = -1
        try:
            remaining = num_steps
            first_chunk = True
            while remaining > 0:
                # The first chunk is a single step so that step 0 is logged, as it
                # was before.
                this_chunk_size = 1 if first_chunk else min(CHUNK_SIZE, remaining)
                first_chunk = False

                state, lowest_loss, best_params, losses, nans = run_chunk(
                    state, lowest_loss, best_params, length=this_chunk_size)

                # The only host sync per chunk, against several per step before.
                losses = np.asarray(losses)
                nans = np.asarray(nans)
                step += this_chunk_size
                remaining -= this_chunk_size

                if nans.any():
                    print(
                        "There may have been a 'gradient explosion'. This run may not be successful (you can stop it with ctrl-C). Suggested troubleshooting steps: specify a low learning rate e.g. '--lr 0.005'."
                    )

                if check_convergence:
                    # Only check at chunk boundaries -- chunks are already the
                    # sync point, so this rides along rather than adding one.
                    # `step // convergence_check_every` increasing means a
                    # multiple of it has been crossed since the last chunk; using
                    # "crossed" rather than "step % check_every == 0" means this
                    # is correct even when convergence_check_every is not a
                    # multiple of the chunk size.
                    new_checks_done = step // convergence_check_every
                    if new_checks_done > checks_done:
                        checks_done = new_checks_done
                        current_params = svi.get_params(state)
                        current_node_days = convergence_node_days_fn(
                            my_model.get_branch_times(current_params),
                            current_params['root_date_mu'])
                        if previous_node_days is not None:
                            # The only place a convergence check touches the
                            # host: one scalar, regardless of tree size, because
                            # the per-node dates and their comparison both ran
                            # on device inside the jitted functions above.
                            mean_abs_change_days = float(
                                convergence_mean_abs_change_fn(
                                    current_node_days, previous_node_days))
                            # The dates settling is necessary but not
                            # sufficient. They can look stable while the clock
                            # rate is still descending, and the dates are far
                            # more sensitive to the rate than this test is: on
                            # one real dataset a 5% rate error moved the median
                            # disagreement with treetime by 3 days. So require
                            # the rate to have stopped moving too, as a relative
                            # change, since its magnitude varies with the units
                            # the tree is in.
                            current_rate = float(
                                my_model.get_mutation_rate(current_params))
                            if previous_rate is None or previous_rate == 0:
                                rate_settled = False
                            else:
                                rate_settled = (
                                    abs(current_rate - previous_rate) /
                                    abs(previous_rate) <
                                    args.convergence_rate_tol)
                            previous_rate = current_rate
                            if (mean_abs_change_days < args.convergence_tol_days
                                    and rate_settled):
                                consecutive_converged += 1
                            else:
                                consecutive_converged = 0
                            if consecutive_converged >= args.convergence_patience:
                                print(
                                    f"Converged: mean absolute change in "
                                    f"predicted node dates was "
                                    f"{mean_abs_change_days:.4f} days (< "
                                    f"--convergence_tol_days "
                                    f"{args.convergence_tol_days}) for "
                                    f"{consecutive_converged} consecutive checks "
                                    f"~{convergence_check_every} steps apart. "
                                    f"Stopping early at step {step} "
                                    f"(of {num_steps} requested).")
                                loss = losses[-1]
                                results = my_model.get_logging_results(
                                    current_params)
                                results['step'] = step
                                results['loss'] = loss
                                results.move_to_end('loss', last=False)
                                results.move_to_end('step', last=False)
                                print("\t".join([
                                    f"{name}:{value:.4f}"
                                    if "." in str(value) else f"{name}:{value}"
                                    for name, value in results.items()
                                ]))
                                break
                        previous_node_days = current_node_days
                if step % 10 == 0 or step == num_steps - 1:
                    loss = losses[-1]
                    results = my_model.get_logging_results(svi.get_params(state))
                    results['step'] = step
                    results['loss'] = loss
                    results.move_to_end('loss', last=False)
                    results.move_to_end('step', last=False)

                    result_string = "\t".join([
                        f"{name}:{value:.4f}"
                        if "." in str(value) else f"{name}:{value}"
                        for name, value in results.items()
                    ])
                    print(result_string)
                    if args.use_wandb:
                        wandb.log(results)
        except KeyboardInterrupt:
            print(f"Interrupting model fitting after {step} steps.")
            was_interrupted = True
        print("Fit completed. Extracting parameters.")

        # best_params is None only if every step produced a NaN loss, in which
        # case there is no "best" set of parameters to fall back to.
        if args.always_use_final_params or best_params is None:
            params = svi.get_params(state)
        else:
            params = best_params
        return params, was_interrupted, float(lowest_loss)

    fit_rate = clock_candidates[0][1]
    profile = None
    if args.profile_clock_rate > 1:
        fit_rate, profile = _profile_clock_rate(args, run_fit, build_model,
                                                fit_rate, terminal_state)

    params, was_interrupted, my_model = _fit_with_robustness_passes(
        args, run_fit, build_model, fit_rate, terminal_state,
        branch_distances_array, parent_indices, root_index, path_sum)

    if profile is not None:
        _report_profile(profile, lookup[reference_point][0])
    to_save = ""
    if was_interrupted:
        while to_save.strip().lower() not in ['y', 'n']:
            to_save = input("Do you want to save the results? [y/n]")
    else:
        to_save = "y"
    if to_save.strip().lower() == "y":
        # Reuse the tree already parsed rather than reading the file again.
        # Setup labelled every node; the ones it invented are dropped again
        # here unless --name_all_nodes asked for them, so the output tree is
        # the same either way. Parsing twice cost about three seconds and a
        # second copy of the tree at 300k tips, and more above that.
        tree2 = tree
        if not args.name_all_nodes:
            for node in helpers.preorder_traversal(tree2.root):
                if node.label in invented_labels:
                    node.label = None

        branch_length_lookup = dict(
            zip(names_init,
                my_model.get_branch_times(params).tolist()))

        total_lengths_in_time = {}

        total_lengths = dict()

        for i, node in enumerate(helpers.preorder_traversal(tree2.root)):

            if not node.label:
                node_name = helpers.get_unnnamed_node_label(i)
                if args.name_all_nodes:
                    node.label = node_name
            else:
                node_name = node.label.replace("'", "")
            node.edge_length = branch_length_lookup[node_name] / (
                helpers.DAYS_PER_YEAR
                if args.output_unit == "years" else 1)
            if not node.parent:
                # Written as zero for the same reason its cumulative time is
                # zero: nothing above the root, so nothing for a branch to
                # span. Leaving the fitted value here would make the output
                # tree and the output dates disagree, since walking the tree
                # from the root would accumulate a duration the dates omit.
                # The root spans no time: it has no parent for time to elapse
                # from. The fitted model agrees -- helpers.make_path_sum zeroes
                # the root's entry, so its branch time never enters any
                # root-to-tip sum the likelihood sees. Adding it here would
                # shift every node's date by a quantity the fit never used.
                node.edge_length = 0.0
                total_lengths[node] = 0.0
            else:
                total_lengths[node] = branch_length_lookup[
                    node_name] + total_lengths[node.parent]

            if node.label:
                total_lengths_in_time[node.label.replace(
                    "'", "")] = total_lengths[node]

        print("Writing tree to file")
        tree2.write_tree_newick(args.tree_out)
        print("")
        print(f"Wrote tree to {args.tree_out}")

        origin_date = lookup[reference_point][0]
        output_dates = {
            name:
            origin_date +
            datetime.timedelta(days=(x + params['root_date_mu'].tolist()))
            for name, x in total_lengths_in_time.items()
        }

        names, values = zip(*output_dates.items())
        columns = {"strain": list(names), "predicted_date": list(values)}

        if not args.no_confidence_intervals:
            columns.update(
                _confidence_columns(args, my_model, params, path_sum,
                                    parent_indices, root_index,
                                    branch_distances_array, terminal_indices,
                                    name_to_pos, names, origin_date))

        output_meta = pd.DataFrame(columns)

        output_meta.to_csv(args.dates_out, sep="\t", index=False)
        print(f"Wrote predicted dates to {args.dates_out}")


if __name__ == "__main__":
    main()
