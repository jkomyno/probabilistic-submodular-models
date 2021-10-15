import numpy as np
from typing import List, AbstractSet
from nptyping import NDArray


def set_to_vector(V: List[int]):
    def inner(S: AbstractSet[int]) -> NDArray[int]:
        return np.isin(V, list(S), assume_unique=True).astype(int)

    return inner
