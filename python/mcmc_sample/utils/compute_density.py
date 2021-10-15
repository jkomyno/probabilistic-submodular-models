import numpy as np
from typing import AbstractSet
from ..objective import Objective
from ... import common

def compute_density_numerator(f: Objective):
    def density_numerator(S: AbstractSet[int]) -> float:
        """"
        Compute the numerator of a density function
        """
        value = f.value(S)
        return np.exp(-value)

    return density_numerator


def compute_normalizing_constant(f: Objective):
    """"
    Compute the normalizing constant of f.
    Time: O(2^n)
    """
    density_numerator = compute_density_numerator(f)
    powerset = common.powerset(f.V)
    Z: float = sum(density_numerator(S) for S in powerset())
    return Z


def compute_density(f):
    density_numerator: float = compute_density_numerator(f)
    Z: float = compute_normalizing_constant(f)

    def density(S: AbstractSet[int]) -> float:
        """"
        Compute the probability density function
        """
        return density_numerator(S) / Z

    return density
