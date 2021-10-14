import numpy as np
from typing import Tuple
from nptyping import NDArray
from ..objective import Objective
from . import polyhedron_greedy


def lovasz(f: Objective):
    """
    Create a closure that returns the value of the Lovasz extension
    F of f and a subgradient of F w.r.t. x.
    """
    def helper(x: NDArray[float]) -> Tuple[NDArray[float], NDArray[float]]:
        w, grad_f_x = polyhedron_greedy(f, x, with_grad=True)
        value = np.dot(x, w)
        return value, grad_f_x

    return helper
