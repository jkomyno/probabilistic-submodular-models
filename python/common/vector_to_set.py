from typing import FrozenSet, List
from nptyping import NDArray


def vector_to_set(V: List[int]):
    n = len(V)

    def inner(x: NDArray[int]) -> FrozenSet[int]:
        return frozenset((V[i] for i in range(n) if x[i] == 1))

    return inner
