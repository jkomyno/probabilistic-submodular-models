import numpy as np
from nptyping import NDArray, Float64
from objective import Objective


def polyhedron_greedy(f: Objective, x: NDArray[Float64]) -> NDArray[Float64]:
    """
    Implementation of Edmonds'71 greedy algorithm.
    :param f: value-oracle for a polymatroid sumodular function
    :param x: vector in \mathbb{R}^{f.n}_{+}
    """

    # get the indexes of a descent sort of x
    I = x.argsort()[::-1][:f.n]

    # y will be the result, a vertex in the base polyhedron of f
    y = np.zeros((f.n, ), np.float64)

    # evaluation of f(\varnothing)
    A_prev = set()
    f_prev = 0.0

    for i in range(f.n):
        mask = I[i]
        A = A_prev | { f.V[mask] }
        y[mask] = f.value(A) - f_prev
        A_prev = A
        f_prev = f_prev + y[mask]

    return y
