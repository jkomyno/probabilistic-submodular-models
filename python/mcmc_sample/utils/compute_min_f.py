from typing import Tuple, AbstractSet
from ..objective import Objective
from .powerset import powerset
from .snd import snd


def compute_min_f(f: Objective) -> Tuple[AbstractSet[int], float]:
    """
    Compute the exact minimum of the given set-submodular function using brute-force.
    Time: O(2^n)
    :param f: submodular function to minimize
    :return: (S, f(S)), where S is the set argument that minimizes f
    """
    return min(((S, f.value(S)) for S in powerset(f.V)), key=snd)
