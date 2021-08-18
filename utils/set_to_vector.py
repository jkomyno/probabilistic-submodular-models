import numpy as np
from typing import List


def set_to_vector(V: List[int], S: List[int]):
    return (np.in1d(V, S) * 1).astype(np.int64)
