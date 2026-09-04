import functools
import jax
import jax.numpy as jnp


def get_unnnamed_node_label(i):
    name = f"NODE_{i:07d}"
    return name


def preorder_traversal(node):
    """Yield every node at or below `node`, parents before children.

    Written with an explicit stack rather than recursion. The recursive
    version overflowed Python's call stack on trees deep enough to matter:
    a simulated million-tip coalescent tree raised RecursionError before
    fitting could start, and the large SARS-CoV-2 trees this tool targets
    are deep and unbalanced too.

    The order is identical to the recursive version -- a node, then the
    whole of its first child's subtree, then its second child's, and so on
    -- which matters because unlabelled nodes are named by their position
    in this traversal.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        # Reversed, so the first child is popped first.
        stack.extend(reversed(current.children))


# Credit: Guillem Cucurull http://gcucurull.github.io/deep-learning/2020/06/03/jax-sparse-matrix-multiplication/
@functools.partial(jax.jit, static_argnums=(2))
def sp_matmul(A, B, shape):
    """
    Arguments:
        A: (N, M) sparse matrix represented as a tuple (indexes, values)
        B: (M,K) dense matrix
        shape: value of N
    Returns:
        (N, K) dense matrix
    """
    # In theory this performs an unnecessary multiplication by 1,
    # (unnecessary for our purposes)
    # but it probably gets removed in the XLA compilation step.
    # Nevertheless we should ultimately refactor this.
    assert B.ndim == 2
    indexes, values = A
    rows, cols = indexes
    in_ = B.take(cols, axis=0)
    prod = in_ * values[:, None]
    res = jax.ops.segment_sum(prod, rows, shape)
    return res


def do_branch_matmul(rows, cols, branch_lengths_array, final_size):
    A = ((rows, cols), jnp.ones_like(cols))
    B = branch_lengths_array.reshape((branch_lengths_array.shape[0], 1))
    calc_dates = sp_matmul(A, B, final_size).squeeze()
    return calc_dates
