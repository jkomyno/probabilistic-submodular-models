from typing import Tuple, TypeVar, Set, Any
T = TypeVar('T', int, float, Set[int])


def fst(x: Tuple[T, Any]) -> T:
    return x[0]
