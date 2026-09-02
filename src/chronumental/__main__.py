import os
import sys

GPU_REQUESTED = "--use_gpu" in sys.argv
if not GPU_REQUESTED:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import datetime

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
        'Molecular clock rate. This should be in units of something per year, where the "something" is the units on the tree. If not given we will attempt to estimate this by RTT. This is only used as a starting point, unless you supply --enforce_exact_clock.',
        default=None,
        type=float)

    parser.add_argument(
        '--variance_dates',
        default=0.3,
        type=float,
        help=
        "Scale factor for date distribution. Essentially a measure of how uncertain we think the measured dates are."
    )

    parser.add_argument(
        '--initial_tau',
        default=3.2,
        type=float,
        help=
        "Initial value for the tau parameter in the model. Only applies to Horseshoe."
    )

    parser.add_argument(
        '--hs_scale',
        default=86917549.587,
        type=float,
        help="hs scale parameter in the model. Only applies to Horseshoe.")

    parser.add_argument(
        '--steps',
        default=20000,
        type=int,
        help=
        "Number of steps to use for the SVI. Increasing this number will make runtime increase, but yield more accurate results."
    )

    parser.add_argument('--lr',
                        default=0.03,
                        type=float,
                        help="Adam learning rate")

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

    parser.add_argument('--model',
                        default="DeltaGuideWithStrictLearntClock",
                        type=str,
                        help="Model type to use")

    parser.add_argument(
        '--output_unit',
        type=str,
        help="Unit for the output branch lengths on the time tree.",
        choices=["days", "years"],
        default="days")

    parser.add_argument(
        '--variance_on_clock_rate',
        action='store_true',
        help=("Will cause the clock rate to be "
              "drawn from a random distribution with a learnt variance."))

    parser.add_argument(
        '--enforce_exact_clock',
        action='store_true',
        help=("Will cause the clock rate to be exactly"
              " fixed at the value specified in clock, rather than learnt"))

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


def prepend_to_file_name(full_path, to_prepend):
    if "/" in full_path:
        path, file = full_path.rsplit('/', 1)
        return f"{path}/{to_prepend}_{file}"
    else:
        return f"{to_prepend}_{full_path}"


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

    if args.enforce_exact_clock and args.clock is None:
        raise ValueError(
            "If you want to enforce the exact clock rate, you must specify it with --clock"
        )

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

    initial_branch_lengths = input_mod.get_initial_branch_lengths_and_name_all_nodes(
        tree)
    names_init = sorted(initial_branch_lengths.keys())
    branch_distances_array = jnp.array(
        [initial_branch_lengths[x] for x in names_init])
    if args.treat_mutation_units_as_normalised_to_genome_size:
        branch_distances_array = branch_distances_array * args.treat_mutation_units_as_normalised_to_genome_size

    name_to_pos = {x: i for i, x in enumerate(names_init)}

    rows, cols = input_mod.get_rows_and_cols_of_sparse_matrix(
        tree, terminal_name_to_pos, name_to_pos)

    rows = jnp.asarray(rows)
    print("Rows array created")
    cols = jnp.asarray(cols)
    print("Cols array created")

    if args.clock:
        print(f"Using clock rate {args.clock}")
        clock_rate = args.clock
        if args.treat_mutation_units_as_normalised_to_genome_size:
            clock_rate = clock_rate * args.treat_mutation_units_as_normalised_to_genome_size
    else:
        root_to_tip = helpers.do_branch_matmul(rows,
                                               cols,
                                               branch_distances_array,
                                               final_size=len(terminal_names))

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
        slope_per_year = slope_per_day * 365

        print(f"Root to tip regression: got rate of: {slope_per_year}")
        clock_rate = slope_per_year
        if clock_rate < 0:
            raise ValueError(
                "ERROR: Root-to-tip regression predicted a negative mutation rate. If your dataset is correct you will need to manually specify an initial clock rate with --clock."
            )

    if clock_rate < 1 and not args.treat_mutation_units_as_normalised_to_genome_size:
        raise ValueError(
            "Clock rate is less than 1 mutation per year. This probably means you need to specify a genome_size with --treat_mutation_units_as_normalised_to_genome_size size. If you are sure that you do not, set that parameter to 1.0."
        )

    model_configuration = {
        "clock_rate": clock_rate,
        "variance_dates": args.variance_dates,
        "expected_min_between_transmissions":
        args.expected_min_between_transmissions,
        "enforce_exact_clock": args.enforce_exact_clock,
        "variance_on_clock_rate": args.variance_on_clock_rate,
        "initial_tau": args.initial_tau,
        "hs_scale": args.hs_scale,
        "fixed_tau": True
    }

    my_model = models.models[args.model](
        rows=rows,
        cols=cols,
        branch_distances_array=branch_distances_array,
        terminal_target_dates_array=terminal_target_dates_array,
        terminal_target_errors_array=terminal_target_errors_array,
        ref_point_distance=ref_point_distance,
        model_configuration=model_configuration,
        terminal_names=terminal_names)

    print("Performing SVI:")
    optimiser = optim.ClippedAdam(
        args.lr) if args.clipped_adam else optim.Adam(args.lr)
    svi = SVI(my_model.model, my_model.guide, optimiser, Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(0))

    num_steps = args.steps
    was_interrupted = False

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
    to_save = ""
    if was_interrupted:
        while to_save.strip().lower() not in ['y', 'n']:
            to_save = input("Do you want to save the results? [y/n]")
    else:
        to_save = "y"
    if to_save.strip().lower() == "y":
        tree2 = input_mod.read_tree(args.tree)

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
                365 if args.output_unit == "years" else 1)
            if not node.parent:
                total_lengths[node] = branch_length_lookup[node_name]
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
        output_meta = pd.DataFrame({"strain": names, "predicted_date": values})

        output_meta.to_csv(args.dates_out, sep="\t", index=False)
        print(f"Wrote predicted dates to {args.dates_out}")


if __name__ == "__main__":
    main()
