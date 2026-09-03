"""Confidence intervals on the fitted node dates.

The fit is parameterised in branch durations, because there positivity is a
free coordinate transform and the optimiser needs nothing else. Curvature is
easier to read in node dates, because there the two halves of the objective
swap roles: the tip-date term becomes diagonal, and the Poisson term becomes
2x2 parent-child blocks, since t_j = d_j - d_parent(j) touches exactly one
pair. In durations the tip term is a squared sum over each tip's whole
root-to-tip path, so second derivatives couple every ancestor-descendant
pair -- the dense structure make_path_sum exists to avoid.

Nothing has to be refitted to move between them. The dates are the path sums
the fit already computes, and the curvature is written analytically, because
the log-posterior is a sum of simple terms:

    Poisson    sum_j  k_j log(mu t_j / Y) - mu t_j / Y
    tip dates  sum_i  -(d_i - target_i)^2 / (2 sigma_i^2)

The linear part of the Poisson has no curvature, so only k_j log t_j
survives, and the Hessian of the negative log-posterior is

    d2/dd_j^2                 k_j/t_j^2, plus k_c/t_c^2 for each child c
    d2/dd_j dd_parent(j)     -k_j/t_j^2
    tips                     +1/sigma_i^2 on the diagonal

which is tree-structured, with one off-diagonal entry per branch. Eliminating
a tree in post-order creates no fill-in, so both the factorisation and the
selected inverse that gives the marginal variances are O(nodes), computed
here as two sweeps over the tree's levels.

Two things this has to get right beyond writing the matrix down.

The clock rate. The date block above conditions on the fitted rate, but the
root date is essentially (oldest tip - divergence/rate), so its uncertainty
is dominated by the rate's. Conditioning on the rate makes exactly the
interval users care about most far too narrow. The rate enters as one extra
parameter coupled to everything, so the augmented Hessian is the tree block
A bordered by a dense vector b and a scalar c, and

    var_v = (A^-1)_vv + (A^-1 b)_v^2 / (c - b' A^-1 b)

which costs one extra tree solve. Note this is the uncertainty in the rate
*implied by the tree*, and it is included even when --clock pinned the rate
for the fit: pinning a value does not make it known exactly.

Zero-mutation branches. A branch with no mutations contributes no curvature
at all, so the weights k_j/t_j^2 disconnect the tree. Take the components of
the graph made only of branches that do carry mutations: a component holding
no dated tip has no curvature in the direction that shifts all of it at once,
and its members' dates are not estimated by this objective -- they are only
bracketed by the ordering constraints, which are inequalities the quadratic
approximation cannot see. Those nodes are reported with the bracket their
neighbours put them in and flagged, rather than given a symmetric interval
that would be wrong on the side with nowhere to go. On real trees this is not
a rare corner: half of ebola/ebov-2013's branches carry no mutations, and 378
of its 2648 nodes come back unidentified.

How well calibrated this is, measured rather than assumed. Coverage can only
be checked where the truth is known, so it was checked against the simulator
in chron_analysis, not against another tool's point estimates -- comparing to
another dater measures agreement with that dater and cannot tell an interval
that is right from one that is merely wide.

Two measurements, because they answer different questions. Against simulated
truth the 95% intervals contain the true internal date 99% of the time, which
is conservative; but the simulator draws node times from a coalescent that
this objective knows nothing about, so the truth comes from a tighter
distribution than the model assumes and some excess is expected from that
alone. The frequentist check has no prior in it: holding one true tree fixed
and redrawing only the mutations, the reported SD is 1.25 times the spread
the estimate actually shows across replicates (quartiles 0.98 and 1.60), with
a bias of 0.29 SD.

That residual 25% is the boundary. Splitting nodes by how many of their
incident branches carry no mutations, the ratio of reported to actual spread
runs 0.90, 1.16, 1.46, 4.05 across the quartets from none to most
(Spearman 0.54, p = 3e-12). Where the quadratic approximation's assumptions
hold the standard error is right; positivity truncates the posterior into a
cone and a Gaussian ignoring that truncation is always the wider of the two.
So these intervals are honest but conservative, and most conservative exactly
where the tree runs out of mutations.

Cost, at a million nodes: 1.3 s at depth 223, 4.7 s at depth 1017, 20.5 s at
depth 5025. The work is linear in the tree but the sweeps step one level at a
time from Python, so it is the tree's depth rather than its size that shows
up. Real trees here are shallow -- 34 for ebola, 69 for measles/genome.
"""
import numpy as np

from . import helpers

# Below this the curvature weight k/t^2 is treated as absent rather than
# small. Mutation counts are counts, so anything under this is a branch the
# input gave no evidence for.
_NO_MUTATIONS = 1e-9


def _depths(parent_indices, root_index):
    """Edges from the root to each node, by pointer jumping.

    The same trick as helpers.make_path_sum, over a value of one per branch,
    and for the same reason: a per-node loop is O(nodes * depth) and these
    trees are deep.
    """
    n = len(parent_indices)
    acc = np.ones(n, dtype=np.int64)
    acc[root_index] = 0
    ptr = np.asarray(parent_indices, dtype=np.int64).copy()
    # Doubling the ancestor pointer each round, so depth doubles per round.
    while True:
        acc = acc + acc[ptr]
        new_ptr = ptr[ptr]
        if np.array_equal(new_ptr, ptr):
            break
        ptr = new_ptr
    return acc


def _levels(depth):
    """Node indices grouped by depth, shallowest first."""
    order = np.argsort(depth, kind="stable")
    ordered_depth = depth[order]
    boundaries = np.searchsorted(ordered_depth,
                                 np.arange(ordered_depth[-1] + 2))
    return [order[boundaries[d]:boundaries[d + 1]]
            for d in range(len(boundaries) - 1)]


class _TreeFactor(object):
    """LDL' factorisation of a tree-structured matrix, and solves with it.

    The matrix is A = diag(diagonal) with -weight[v] in entries (v, parent[v])
    and (parent[v], v). Eliminating children before parents gives an L with
    exactly one off-diagonal entry per node, so nothing here is more than a
    gather and a segmented sum.
    """

    def __init__(self, parent, weight, diagonal, levels, root_index):
        self.parent = parent
        self.weight = weight
        self.levels = levels
        self.root_index = root_index
        n = len(parent)
        self.n = n

        pivot = diagonal.astype(np.float64).copy()
        # Post-order: deepest level first, each node folding into its parent.
        for level in reversed(levels[1:]):
            w = weight[level]
            contribution = w * w / pivot[level]
            # D_p = A_pp - sum over children of A_pc^2 / D_c.
            pivot -= np.bincount(parent[level],
                                 weights=contribution,
                                 minlength=n)
        self.pivot = pivot
        # L[parent[v], v], the single off-diagonal entry of column v.
        self.below = np.zeros(n)
        nonroot = np.ones(n, dtype=bool)
        nonroot[root_index] = False
        self.below[nonroot] = -weight[nonroot] / pivot[nonroot]

    def marginal_variances(self):
        """The diagonal of the inverse, by Takahashi's recursion.

        On the sparsity pattern of a tree this collapses to one term:
        var[v] = 1/pivot[v] + L[parent, v]^2 * var[parent], sweeping down.
        """
        var = np.empty(self.n)
        var[self.root_index] = 1.0 / self.pivot[self.root_index]
        for level in self.levels[1:]:
            var[level] = (1.0 / self.pivot[level] +
                          self.below[level]**2 * var[self.parent[level]])
        return var

    def solve(self, rhs):
        """A^-1 rhs, as forward substitution, a scale, and back substitution."""
        y = rhs.astype(np.float64).copy()
        for level in reversed(self.levels[1:]):
            y -= np.bincount(self.parent[level],
                             weights=self.below[level] * y[level],
                             minlength=self.n)
        z = y / self.pivot
        for level in self.levels[1:]:
            z[level] -= self.below[level] * z[self.parent[level]]
        return z


def _components(parent, has_weight, levels, root_index):
    """Label the connected components of the branches that carry curvature.

    A downward sweep suffices: a node joins its parent's component if the
    branch between them has a nonzero weight, and starts its own otherwise.
    """
    n = len(parent)
    label = np.arange(n)
    for level in levels[1:]:
        joined = level[has_weight[level]]
        label[joined] = label[parent[joined]]
    return label


def node_date_intervals(parent_indices,
                        root_index,
                        mutations_per_branch,
                        branch_times,
                        node_dates,
                        terminal_indices,
                        terminal_sigmas,
                        clock_rate,
                        include_rate_uncertainty=True):
    """Marginal standard deviations, in days, for every node's date.

    `branch_times` and `node_dates` are the fitted durations and the path sums
    of them, both in days; `mutations_per_branch` is the observed count on
    each branch, indexed like the fit; `clock_rate` is in mutations per year.

    Returns (sd, identified, lower, upper), where `sd` is the standard
    deviation for identified nodes and NaN elsewhere, `identified` is False
    for nodes whose date this objective does not estimate, and `lower`/`upper`
    are the 95% interval, replaced for unidentified nodes by the bracket the
    ordering constraints put them in.
    """
    parent = np.asarray(parent_indices, dtype=np.int64)
    n = len(parent)
    k = np.asarray(mutations_per_branch, dtype=np.float64).copy()
    t = np.asarray(branch_times, dtype=np.float64).copy()
    dates = np.asarray(node_dates, dtype=np.float64)

    # The root's own branch is in the Poisson likelihood but not in any date:
    # make_path_sum zeroes it, so it cannot carry curvature about a date.
    k[root_index] = 0.0
    t[root_index] = 0.0

    weight = np.zeros(n)
    usable = (k > _NO_MUTATIONS) & (t > 0)
    weight[usable] = k[usable] / t[usable]**2

    diagonal = np.zeros(n)
    diagonal += weight
    diagonal += np.bincount(parent, weights=weight, minlength=n)
    # The root's self-parent entry would otherwise count its own zero weight.
    diagonal[root_index] -= weight[root_index]

    terminal_indices = np.asarray(terminal_indices, dtype=np.int64)
    precision = 1.0 / np.asarray(terminal_sigmas, dtype=np.float64)**2
    diagonal += np.bincount(terminal_indices,
                            weights=precision,
                            minlength=n)

    depth = _depths(parent, root_index)
    levels = _levels(depth)

    component = _components(parent, usable, levels, root_index)
    dated = np.zeros(n, dtype=bool)
    dated[terminal_indices] = True
    components_with_a_tip = np.zeros(n, dtype=bool)
    components_with_a_tip[component[dated]] = True
    identified = components_with_a_tip[component]

    # An unidentified component is a block of its own -- every branch leaving
    # it has zero weight -- so anchoring it changes nothing else. Anchor it
    # loosely, purely so the factorisation is defined, and discard the number
    # afterwards in favour of the ordering bracket.
    span = float(np.ptp(dates)) if n > 1 else 1.0
    anchor = 1.0 / max(span, 1.0)**2
    diagonal[~identified] += anchor

    factor = _TreeFactor(parent, weight, diagonal, levels, root_index)
    variance = factor.marginal_variances()

    if include_rate_uncertainty:
        variance = _add_rate_uncertainty(factor, parent, t, k, clock_rate,
                                         identified, root_index, variance, n)

    sd = np.sqrt(np.maximum(variance, 0.0))
    lower = dates - 1.96 * sd
    upper = dates + 1.96 * sd
    if not identified.all():
        lower[~identified], upper[~identified] = _ordering_bracket(
            parent, dates, identified, levels, lower, upper)
        sd[~identified] = np.nan
    return sd, identified, lower, upper


def _add_rate_uncertainty(factor, parent, t, k, clock_rate, identified,
                          root_index, variance, n):
    """Widen each variance by the share of it the clock rate is responsible for.

    Writing the rate as theta = log(mu), the Poisson term contributes
    d2/dtheta2 = (mu/Y) sum_j t_j and d2/dtheta dd_v = (mu/Y) dS/dd_v, where
    S is the tree's total duration, so the cross term counts each node's
    children against its own branch. Blocking the augmented Hessian gives the
    correction below.

    Unidentified nodes are held out of the cross vector: their entry in it is
    real, but their block was anchored arbitrarily above, so including them
    would let that arbitrary choice leak into the rate's variance.
    """
    per_day = clock_rate / helpers.DAYS_PER_YEAR
    total_duration = float(t.sum())
    if total_duration <= 0 or per_day <= 0:
        return variance
    children = np.bincount(parent, minlength=n).astype(np.float64)
    children[root_index] -= 1.0  # its self-parent edge is not a child
    has_parent = np.ones(n, dtype=np.float64)
    has_parent[root_index] = 0.0
    # dS/dd_v = [v is not the root] - (number of v's children).
    cross = per_day * (has_parent - children)
    cross[~identified] = 0.0

    rate_curvature = per_day * total_duration
    u = factor.solve(cross)
    schur = rate_curvature - float(cross @ u)
    if not np.isfinite(schur) or schur <= 0:
        # The tree alone does not pin the rate; leave the conditional
        # variances rather than report a negative correction.
        return variance
    return variance + u**2 / schur


def _ordering_bracket(parent, dates, identified, levels, lower, upper):
    """Bounds for nodes the curvature does not reach, from the tree's order.

    Such a node still has to sit after everything above it and before
    everything below, so report the nearest identified ancestor's lower bound
    and the latest identified descendant's upper bound. That is an honest
    statement of what is known: an interval, but one the data narrows only
    through the neighbours.
    """
    n = len(parent)
    from_above = np.where(identified, lower, -np.inf)
    for level in levels[1:]:
        unknown = level[~identified[level]]
        from_above[unknown] = np.maximum(from_above[unknown],
                                         from_above[parent[unknown]])
    from_below = np.where(identified, upper, np.inf)
    for level in reversed(levels[1:]):
        unknown = level
        np.minimum.at(from_below, parent[unknown], from_below[unknown])
    unidentified = ~identified
    lo = from_above[unidentified]
    hi = from_below[unidentified]
    # A node with no identified descendant is bounded only from above.
    hi = np.where(np.isfinite(hi), hi, dates[unidentified])
    lo = np.where(np.isfinite(lo), lo, dates[unidentified])
    return lo, np.maximum(hi, lo)
