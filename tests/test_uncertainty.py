"""The interval maths, checked against dense linear algebra.

Everything in chronumental.uncertainty is a sparse shortcut for something
with an exact small-scale reference: the factorisation and its selected
inverse should agree with numpy.linalg.inv on the same matrix written out in
full, and the clock-rate correction should agree with inverting the bordered
matrix directly. On trees of a few dozen nodes both are cheap, so the
shortcuts can be held to machine precision rather than to a tolerance.

Run with `python -m pytest tests/`. Only numpy is needed.
"""
import numpy as np
import pytest

from chronumental import helpers, uncertainty

RATE = 25.0  # mutations per year over the whole genome


def random_tree(rng, n_leaves):
    """A random tree as a parent-index array, with the root pointing at itself."""
    parent = [0]
    leaves = {0}
    while len(leaves) < n_leaves:
        chosen = int(rng.choice(sorted(leaves)))
        leaves.discard(chosen)
        for _ in range(2):
            parent.append(chosen)
            leaves.add(len(parent) - 1)
    return np.array(parent), 0, np.array(sorted(leaves))


def simulate(rng, n_leaves):
    """A tree whose mutations are consistent with its durations.

    Independent noise would leave the point far from any mode, where the
    bordered matrix is indefinite and the rate correction correctly declines
    to say anything -- which would leave that branch of the code untested.
    """
    parent, root, leaves = random_tree(rng, n_leaves)
    n = len(parent)
    times = rng.exponential(120.0, n)
    times[root] = 0.0
    depth = uncertainty._depths(parent, root)
    levels = uncertainty._levels(depth)
    dates = np.zeros(n)
    for level in levels[1:]:
        dates[level] = dates[parent[level]] + times[level]
    mutations = rng.poisson(RATE * times / helpers.DAYS_PER_YEAR).astype(float)
    mutations[root] = 0.0
    sigmas = rng.uniform(1.0, 30.0, len(leaves))
    return parent, root, leaves, times, dates, mutations, sigmas


def build_dense(parent, root, leaves, times, dates, mutations, sigmas):
    """The same Hessian the module builds, written out as a full matrix."""
    n = len(parent)
    weight = np.zeros(n)
    usable = (mutations > 0) & (times > 0)
    weight[usable] = mutations[usable] / times[usable]**2
    diagonal = weight + np.bincount(parent, weights=weight, minlength=n)
    diagonal[root] -= weight[root]
    diagonal += np.bincount(leaves, weights=1.0 / sigmas**2, minlength=n)

    depth = uncertainty._depths(parent, root)
    levels = uncertainty._levels(depth)
    component = uncertainty._components(parent, usable, levels, root)
    dated = np.zeros(n, dtype=bool)
    dated[leaves] = True
    with_a_tip = np.zeros(n, dtype=bool)
    with_a_tip[component[dated]] = True
    identified = with_a_tip[component]
    diagonal[~identified] += 1.0 / max(float(np.ptp(dates)), 1.0)**2

    dense = np.diag(diagonal.copy())
    for node in range(n):
        if node == root:
            continue
        dense[node, parent[node]] -= weight[node]
        dense[parent[node], node] -= weight[node]
    return dense, weight, diagonal, identified, levels


@pytest.mark.parametrize("trial", range(6))
def test_marginal_variances_match_a_dense_inverse(trial):
    rng = np.random.default_rng(trial)
    parent, root, leaves, times, dates, mutations, sigmas = simulate(
        rng, int(rng.integers(8, 40)))
    dense, _, _, identified, _ = build_dense(parent, root, leaves, times,
                                             dates, mutations, sigmas)
    sd, got_identified, _, _, _ = uncertainty.node_date_intervals(
        parent, root, mutations, times, dates, leaves, sigmas,
        clock_rate=RATE, include_rate_uncertainty=False)
    expected = np.diag(np.linalg.inv(dense))
    assert np.array_equal(got_identified, identified)
    np.testing.assert_allclose(sd[identified]**2, expected[identified],
                               rtol=1e-9)


@pytest.mark.parametrize("trial", range(6))
def test_solves_match_a_dense_solve(trial):
    rng = np.random.default_rng(100 + trial)
    parent, root, leaves, times, dates, mutations, sigmas = simulate(
        rng, int(rng.integers(8, 40)))
    dense, weight, diagonal, _, levels = build_dense(parent, root, leaves,
                                                     times, dates, mutations,
                                                     sigmas)
    factor = uncertainty._TreeFactor(parent, weight, diagonal, levels, root)
    rhs = rng.normal(size=len(parent))
    np.testing.assert_allclose(factor.solve(rhs), np.linalg.solve(dense, rhs),
                               rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("trial", range(6))
def test_rate_correction_matches_the_bordered_inverse(trial):
    rng = np.random.default_rng(200 + trial)
    parent, root, leaves, times, dates, mutations, sigmas = simulate(
        rng, int(rng.integers(8, 40)))
    dense, _, _, identified, _ = build_dense(parent, root, leaves, times,
                                             dates, mutations, sigmas)
    n = len(parent)
    per_day = RATE / helpers.DAYS_PER_YEAR
    children = np.bincount(parent, minlength=n).astype(float)
    children[root] -= 1.0
    has_parent = np.ones(n)
    has_parent[root] = 0.0
    cross = per_day * (has_parent - children)
    cross[~identified] = 0.0

    bordered = np.zeros((n + 1, n + 1))
    bordered[:n, :n] = dense
    bordered[:n, n] = cross
    bordered[n, :n] = cross
    bordered[n, n] = per_day * times.sum()
    schur = bordered[n, n] - cross @ np.linalg.solve(dense, cross)

    sd, _, _, _, applied = uncertainty.node_date_intervals(
        parent, root, mutations, times, dates, leaves, sigmas,
        clock_rate=RATE, include_rate_uncertainty=True)
    assert applied == (schur > 0)
    if schur <= 0:
        # The tree does not pin the rate, so the correction is declined and
        # the conditional variances stand.
        expected = np.diag(np.linalg.inv(dense))
    else:
        expected = np.diag(np.linalg.inv(bordered))[:n]
    np.testing.assert_allclose(sd[identified]**2, expected[identified],
                               rtol=1e-7)


def test_rate_uncertainty_only_widens():
    """It adds u^2/schur, which cannot be negative, so no interval shrinks."""
    rng = np.random.default_rng(7)
    parent, root, leaves, times, dates, mutations, sigmas = simulate(rng, 60)
    conditioned, identified, _, _, _ = uncertainty.node_date_intervals(
        parent, root, mutations, times, dates, leaves, sigmas,
        clock_rate=RATE, include_rate_uncertainty=False)
    widened, _, _, _, _ = uncertainty.node_date_intervals(
        parent, root, mutations, times, dates, leaves, sigmas,
        clock_rate=RATE, include_rate_uncertainty=True)
    assert np.all(widened[identified] >= conditioned[identified] - 1e-12)


def test_an_internal_node_with_no_mutations_around_it_is_unidentified():
    """A node no mutation reaches is bounded by its neighbours, not estimated.

    A tip in the same position is still identified, because its own reported
    date is curvature; only nodes with nothing at all are set aside.
    """
    #          0                 branch mutations: 1 -> 0, 2 -> 0, 3 -> 5
    #        /   \               1 is internal and nothing dates it
    #       1     3 (tip)
    #       |
    #       2 (tip)
    parent = np.array([0, 0, 1, 0])
    mutations = np.array([0.0, 0.0, 0.0, 5.0])
    times = np.array([0.0, 40.0, 60.0, 100.0])
    dates = np.array([0.0, 40.0, 100.0, 100.0])
    leaves = np.array([2, 3])
    sigmas = np.array([3.0, 3.0])
    sd, identified, lower, upper, _ = uncertainty.node_date_intervals(
        parent, 0, mutations, times, dates, leaves, sigmas, clock_rate=RATE)
    assert identified[0] and identified[3]
    assert identified[2], "a dated tip is identified by its own date"
    assert not identified[1]
    assert np.isnan(sd[1])
    # It still has to sit after the root and before the tip below it.
    assert lower[0] <= lower[1] <= upper[1] <= upper[2]


def test_depths_and_levels_agree_with_a_direct_walk():
    rng = np.random.default_rng(3)
    parent, root, _ = random_tree(rng, 30)
    depth = uncertainty._depths(parent, root)
    for node in range(len(parent)):
        walked, current = 0, node
        while current != root:
            current = parent[current]
            walked += 1
        assert depth[node] == walked
    levels = uncertainty._levels(depth)
    assert sorted(np.concatenate(levels)) == list(range(len(parent)))
    for distance, level in enumerate(levels):
        assert np.all(depth[level] == distance)


def test_a_flat_clock_is_reported_as_having_no_signal():
    """Divergence that does not rise with date is the hard case."""
    from chronumental import diagnostics
    rng = np.random.default_rng(0)
    days = np.linspace(0, 3650, 200)
    divergence = rng.uniform(0, 100, 200)  # unrelated to the dates
    warnings = diagnostics.report(divergence, days, genome_length=10000)
    assert any("no clock signal" in w or "weakly related" in w
               for w in warnings)


def test_a_clean_clock_raises_nothing():
    from chronumental import diagnostics
    rng = np.random.default_rng(1)
    days = np.linspace(0, 3650, 200)
    divergence = 0.02 * days + rng.normal(0, 1.0, 200)
    assert diagnostics.report(divergence, days, genome_length=100000) == []


def test_saturation_is_reported_only_when_the_genome_length_is_known():
    from chronumental import diagnostics
    rng = np.random.default_rng(2)
    days = np.linspace(0, 3650, 200)
    divergence = 5.0 * days + rng.normal(0, 10.0, 200)  # ~1.8 subs/site at 10kb
    assert any("substitutions per site" in w
               for w in diagnostics.report(divergence, days,
                                           genome_length=10000))
    # Without a genome length the units are unknown, so it is skipped rather
    # than guessed at.
    assert diagnostics.report(divergence, days, genome_length=None) == []
