import numpy as np
from functools import reduce
from typing import Union, Tuple, List, Tuple, Set
from nptyping import NDArray
from ..objective import Objective
from ... import common


def polyhedron_greedy(f: Objective, x: NDArray[float],
                      with_grad=False) -> Union[NDArray[float], Tuple[NDArray[float], bool]]:
    """
    Implementation of Edmonds'71 greedy algorithm.
    :param f: value-oracle for a polymatroid sumodular function
    :param x: vector in \mathbb{R}^{f.n}_{+}
    :param with_grad: return also the subgradient of f w.r.t. x
    """
    set_to_vector = common.set_to_vector(f.V)

    # get the indexes of a descent sort of x
    I = x.argsort()[::-1][:f.n]

    def reduce_fn(acc: Tuple[List[float], Set[int], float, NDArray[float]], i: int) -> Tuple[List[float], Set[int], float]:
        (y, A, f_prev, subgrad) = acc
        mask = I[i]
        A.add(f.V[mask])
        f_curr = f.value(A)
        y[mask] = f_curr - f_prev
        subgrad[i] = y[mask] * set_to_vector({ f.V[mask] })

        return (y, A, f_curr, subgrad)

    initializer = (
        np.zeros(f.n),        # y
        set(),                # A
        f.value(set()),       # f(A)
        np.zeros((f.n, f.n))  # empty matrix for the subgradient
    )
    y, _, _, subgrad = reduce(reduce_fn, range(f.n), initializer)

    if with_grad:
        return y, np.sum(subgrad, axis=0)
    
    return y
