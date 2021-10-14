import numpy as np
from typing import AbstractSet
from objective import Objective


def set_to_vector(f: Objective, S: AbstractSet[int]):
    return np.isin(f.V, list(S), assume_unique=True).astype(np.float64)
