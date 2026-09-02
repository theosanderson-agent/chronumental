from os import name
import pandas as pd
import numpy as np
import gzip
import datetime
from alive_progress import alive_it
import treeswift
import xopen
import lzma
from . import helpers
from datetime import datetime as dt


def read_tabular_file(tabular_file_name, **kwargs):
    # Handle gzipped files, and csv and tsv
    tabular_file = xopen.xopen(tabular_file_name, "r")
    print(f"Reading {tabular_file_name}")

    stripped_name = tabular_file_name.replace(".gz", "").replace(
        ".bz2", "").replace(".xz", "").replace(".lzma", "")
    if stripped_name.endswith(".csv"):
        return pd.read_csv(tabular_file,
                           dtype={
                               "strain": str,
                               "name": str,
                               "taxon": str
                           },
                           **kwargs)
    if stripped_name.endswith(".tsv"):
        return pd.read_csv(tabular_file,
                           sep="\t",
                           dtype={
                               "strain": str,
                               "name": str,
                               "taxon": str
                           },
                           **kwargs)
    raise Exception(
        f"Tabular file {tabular_file_name} was expected to end in tsv or csv")


def get_correct_column(columns, possible_values):
    for column in columns:
        if str(column).strip().lower() in possible_values:
            return column
    raise Exception(
        f"""Could not find a column with one of the following names: {possible_values}. Available were:
    {columns}""")


def fromYearFraction(yearFraction):
    #check type is float
    if not isinstance(yearFraction, float):
        raise ValueError("Not a float")
    if np.isnan(yearFraction):
        raise ValueError("Is NaN")
    year = int(yearFraction)
    if year == 0:
        raise ValueError(
            "The year zero does not exist in the Gregorian calendar")
    fraction = yearFraction - year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year + 1, month=1, day=1)
    date = startOfThisYear + (fraction * ((startOfNextYear) -
                                          (startOfThisYear)))
    return date


def get_metadata(metadata_file):
    # get just the top row of the file
    metadata = read_tabular_file(metadata_file, nrows=1)
    # get the column names
    metadata_columns = metadata.columns
    name_column = get_correct_column(
        metadata_columns, possible_values=["strain", "name", "taxon"])
    print(
        f"Using {name_column} as the name column. This must be the name of the taxa in the tree."
    )

    date_column = get_correct_column(metadata_columns,
                                     possible_values=["date"])
    fields = [date_column, name_column]
    print(f"Using {fields} as the fields to parse.")

    print("Reading metadata")
    metadata = read_tabular_file(metadata_file,
                                 low_memory=False,
                                 usecols=fields).rename(columns={
                                     name_column: 'strain',
                                     date_column: 'date'
                                 })

    for field in ['date']:
        if field not in metadata:
            raise Exception(f"Metadata has no {field} column")
    return metadata


def read_tree(tree_file):
    extension = tree_file.replace(".gz", "").replace(".bz2", "").split(".")[-1]
    if extension == "nex" or extension == "nexus":
        trees = treeswift.read_tree_nexus(tree_file)
        keys = list(trees.keys())
        print(f"Using tree {keys[0]} from Nexus file")
        tree = trees[keys[0]]
    elif extension == "nwk" or extension == "newick":
        tree = treeswift.read_tree_newick(tree_file)
    else:
        print("Assuming tree file is newick (change extension for nexus)")
        tree = treeswift.read_tree_newick(tree_file)
    #raise ValueError(tree)

    # The root has no parent, so any branch length on it does not represent
    # elapsed time between two nodes and there is nothing above it to date.
    # Some tools write one anyway: treetime's divergence trees carry a root
    # branch, and on the ebola example it was 0.001 substitutions per site,
    # over a year at that dataset's clock rate. Left in place it was added to
    # every node's root-to-tip sum and, worse, pushed the reported root date
    # that far later, which is the single largest disagreement with treetime
    # on that dataset. Zeroing it moved the median disagreement across
    # internal nodes from 18.5 days to 12.6 and the root from 183 days out to
    # 75. It also removes a spurious Poisson term asking the fit to explain
    # the stem's mutations with a duration nothing else constrains.
    if tree.root.edge_length:
        print(f"Ignoring the root's branch length of {tree.root.edge_length}: "
              f"a root has no parent, so it spans no time.")
        tree.root.edge_length = 0

    for node in tree.traverse_preorder():
        if node.label:
            node.label = node.label.replace("'", "")
    return tree


def get_datetime_and_error(x):

    try:
        return [datetime.datetime.strptime(x, '%Y-%m-%d'), 1]
    except TypeError:
        try:
            return [fromYearFraction(x), 1]
        except ValueError:
            print(
                f"Warning: could not parse date {x}, it will not feature in calculation."
            )
            return [None, None]
    except ValueError:
        try:
            return fromYearFraction(x)
        except ValueError:
            pass

        try:
            return [
                datetime.datetime.strptime(x, '%Y-%m') +
                datetime.timedelta(days=30 // 2), 30
            ]
        except ValueError:
            try:
                return [
                    datetime.datetime.strptime(x, '%Y') +
                    datetime.timedelta(days=365 // 2), 365
                ]
            except ValueError:
                if x != "" and x != "?":
                    print(
                        f"Warning: could not parse date {x}, it will not feature in calculation."
                    )
                return [None, None]


def process_dates(metadata):
    metadata['date_and_error'] = metadata['date'].apply(get_datetime_and_error)
    metadata['processed_date'] = metadata['date_and_error'].apply(
        lambda x: x[0])
    metadata['processed_date_error'] = metadata['date_and_error'].apply(
        lambda x: x[1])
    metadata.drop(columns=['date_and_error'], inplace=True)


def get_present_dates(metadata, only_use_full_dates):
    if only_use_full_dates:
        return metadata[(~metadata['processed_date'].isnull())
                        & (metadata['processed_date_error'] < 5)]
    else:
        return metadata[~metadata['processed_date'].isnull()]


def get_oldest(full, tree):
    leaf_to_node = tree.label_to_node(selection="leaves")
    filtered = full[full['strain'].isin(leaf_to_node.keys())]
    oldest_date = filtered['processed_date'].min()
    the_oldest = filtered[filtered['processed_date'] == oldest_date]

    try:
        reference_point = the_oldest['strain'].values[0]
    except IndexError:
        raise ValueError(
            "Could not find a reference point on the tree. This probably means that the names on your tree don't match the strain/name/taxon column of the dates file."
        )

    distance = tree.distance_between(tree.root, leaf_to_node[reference_point])
    return reference_point, distance


def get_specific(full, tree, name):
    leaf_to_node = tree.label_to_node(selection="leaves")
    reference_point = name
    distance = tree.distance_between(tree.root, leaf_to_node[reference_point])
    return reference_point, distance


def get_target_dates(tree, lookup, reference_point):
    """
    Returns a list of dictionary mapping names to integer dates being targeted.
    Dates are relative to the date of the reference point, which forms an arbitary origin.
    """
    terminal_targets = {}
    terminal_targets_error = {}
    for terminal in alive_it(tree.traverse_leaves(),
                             title="Creating target date array"):

        terminal.label = terminal.label.replace("'", "")
        if terminal.label in lookup:
            date = lookup[terminal.label][0]
            diff = (date - lookup[reference_point][0]).days
            terminal_targets[terminal.label] = diff
            terminal_targets_error[terminal.label] = lookup[terminal.label][1]
    return terminal_targets, terminal_targets_error


def get_initial_branch_lengths_and_name_all_nodes(tree):
    initial_branch_lengths = {}
    for i, node in alive_it(enumerate(helpers.preorder_traversal(tree.root)),
                            title="finding initial branch_lengths"):
        if not node.label:
            name = helpers.get_unnnamed_node_label(i)
            node.label = name
        if node.edge_length is None:
            node.edge_length = 0

        initial_branch_lengths[node.label] = node.edge_length
    return initial_branch_lengths


def get_rows_and_cols_of_sparse_matrix(tree, terminal_name_to_pos,
                                       name_to_pos):
    # Here we define row col coordinates for 1s in a sparse matrix of mostly 0s
    count = 0

    for leaf in alive_it(tree.traverse_leaves(),
                         title="Counting tree for sparse matrix creation"):
        if leaf.label in terminal_name_to_pos:
            cur_node = leaf
            count += 1
            while cur_node.parent is not None:
                count += 1
                cur_node = cur_node.parent

    rows = np.zeros(count, dtype=int)
    cols = np.zeros(count, dtype=int)

    location = 0
    for leaf in alive_it(tree.traverse_leaves(),
                         title="Populating sparse matrix rows, cols"):
        if leaf.label in terminal_name_to_pos:
            cur_node = leaf
            rows[location] = terminal_name_to_pos[leaf.label]
            cols[location] = name_to_pos[cur_node.label]
            location += 1
            while cur_node.parent is not None:
                rows[location] = terminal_name_to_pos[leaf.label]
                cols[location] = name_to_pos[cur_node.parent.label]
                location += 1
                cur_node = cur_node.parent
    return rows, cols


def estimate_initial_times(tree, name_to_pos, branch_distances_array,
                           target_dates, clock_rate, floor_days=0.01):
    """Estimate a per-branch initial time and an initial root date from the
    tree, the tip dates and the (starting) clock rate, without using any
    ground truth.

    The crude default (mutations / clock_rate, floored) ignores the tip
    dates entirely. This instead asks, for every node, what date its
    descendant tips imply if you walk back from each of them at the
    (starting) clock rate, and averages those implied dates over the tips
    below the node. That average is recentred locally per subtree, so parts
    of the tree whose mutations under- or over-count elapsed time relative
    to the global clock rate (rate variation, missing data, a long
    zero-length run) get corrected using the actual dates below them,
    rather than all being pinned to the single global regression line.

    Concretely: for a dated tip t, define
        base(t) = target_dates[t] - cumulative_mutation_time(t)
    where cumulative_mutation_time is the mutation-clock-implied elapsed
    time from the root. This is the root date that tip alone would imply.
    For any node n, base(t) - base_of_root_relationship stays constant in n,
    so the average of base(t) over the dated tips under n, plus n's own
    cumulative_mutation_time, is an estimate of n's date that uses only
    information downstream of n. Root-to-tip monotonicity is then enforced
    with a single top-down pass, and branch times are the differences.

    The per-subtree average is shrunk towards the whole-tree average in
    proportion to how many dated tips support it (see shrinkage_pseudocount
    below), so a clade with few or noisy dates is not overcorrected.

    Returns (branch_time_init, root_date_init): a dict from node label to an
    initial branch time in days (always >= floor_days), and a float initial
    root date (in the same day-relative-to-reference units as
    target_dates).
    """
    days_per_mutation_year = 365.0 / clock_rate

    own_mutation_time = {}
    cumulative_mutation_time = {}
    for node in tree.traverse_preorder():
        label = node.label
        pos = name_to_pos.get(label)
        my_time = float(branch_distances_array[pos]) * days_per_mutation_year if pos is not None else 0.0
        own_mutation_time[label] = my_time
        parent_time = cumulative_mutation_time[
            node.parent.label] if node.parent is not None else 0.0
        cumulative_mutation_time[label] = parent_time + my_time

    sum_base = {}
    count = {}
    for node in tree.traverse_postorder():
        label = node.label
        if node.is_leaf():
            if label in target_dates:
                sum_base[label] = target_dates[
                    label] - cumulative_mutation_time[label]
                count[label] = 1
            else:
                sum_base[label] = 0.0
                count[label] = 0
        else:
            total = 0.0
            n = 0
            for child in node.children:
                total += sum_base[child.label]
                n += count[child.label]
            sum_base[label] = total
            count[label] = n

    root_label = tree.root.label
    global_average_base = (sum_base[root_label] /
                           count[root_label]) if count[root_label] else 0.0

    # Shrink each subtree's local average towards the global average,
    # weighted by how many dated tips actually support it. A clade with few
    # dated descendants (or high per-branch rate variance under a relaxed
    # clock) gets a noisy local average; treating the global average as
    # `shrinkage_pseudocount` extra "tips" of evidence keeps such clades from
    # being yanked to an unreliable local estimate, while well-supported
    # clades (many dated tips) are barely pulled off their own average.
    shrinkage_pseudocount = 10.0

    date_estimate = {}
    for node in tree.traverse_preorder():
        label = node.label
        average_base = (sum_base[label] + shrinkage_pseudocount *
                        global_average_base) / (count[label] +
                                                shrinkage_pseudocount)
        date_estimate[label] = average_base + cumulative_mutation_time[label]

    # Enforce that dates never decrease from parent to child, which both
    # keeps branch times positive and stops the per-subtree recentring above
    # from producing a child estimated earlier than its own parent. The
    # per-node floor is never smaller than that branch's own mutation count
    # would imply under the starting clock rate: without this, a branch
    # carrying real mutations could be initialised at (near) zero time,
    # which asks for an enormous rate to explain those mutations and makes
    # the first SVI steps unstable.
    adjusted = {}
    branch_time_init = {}
    for node in tree.traverse_preorder():
        label = node.label
        node_floor = max(floor_days, own_mutation_time[label])
        if node.parent is None:
            adjusted[label] = date_estimate[label]
            branch_time_init[label] = node_floor
        else:
            parent_label = node.parent.label
            adjusted[label] = max(date_estimate[label],
                                  adjusted[parent_label] + node_floor)
            branch_time_init[label] = adjusted[label] - adjusted[parent_label]

    root_date_init = adjusted[root_label]
    return branch_time_init, root_date_init


def get_rows_and_cols_of_full_sparse_matrix(tree, name_to_pos,
                                           max_nodes=None, seed=0):
    """Like get_rows_and_cols_of_sparse_matrix, but rows range over nodes
    generally (internal nodes included), not just the terminals.

    `max_nodes` caps how many nodes are included, sampling a fixed random
    subset when the tree is bigger. The convergence check only needs the
    *mean* absolute change in predicted dates, and the mean over a few
    thousand randomly chosen nodes estimates that to well under the
    tolerance it is compared against. Building rows for every node instead
    costs memory proportional to the total path length over the whole tree,
    which on a 100k-tip tree measured at about a gigabyte -- a real cost on
    the very large trees this tool is for. The subset is drawn with a fixed
    seed so a run stays reproducible.

    Used only for the early-stopping convergence check: with this, every
    node's predicted date can be computed on device with the same sparse
    matmul the model already uses for terminals (helpers.do_branch_matmul),
    so a convergence check costs one small host sync (a single scalar)
    rather than transferring the whole branch-length array and walking the
    tree from Python. `tree` must already have every node labelled, e.g. by
    get_initial_branch_lengths_and_name_all_nodes.
    """
    nodes = list(helpers.preorder_traversal(tree.root))
    if max_nodes is not None and len(nodes) > max_nodes:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(nodes), size=max_nodes, replace=False)
        chosen.sort()
        nodes = [nodes[i] for i in chosen]

    count = 0
    for node in alive_it(
            nodes, title="Counting tree for convergence-check matrix"):
        cur_node = node
        count += 1
        while cur_node.parent is not None:
            count += 1
            cur_node = cur_node.parent

    rows = np.zeros(count, dtype=int)
    cols = np.zeros(count, dtype=int)

    location = 0
    for row, node in enumerate(
            alive_it(nodes, title="Populating convergence-check matrix")):
        # Rows are indices into the selected subset, not into the full node
        # ordering, so the matmul's output has one entry per selected node.
        cur_node = node
        rows[location] = row
        cols[location] = name_to_pos[cur_node.label]
        location += 1
        while cur_node.parent is not None:
            rows[location] = row
            cols[location] = name_to_pos[cur_node.parent.label]
            location += 1
            cur_node = cur_node.parent
    return rows, cols, len(nodes)
