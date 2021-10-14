from typing import Tuple, TypeVar, Set, Any
T = TypeVar('T', int, float, Set[int])


def snd(x: Tuple[Any, T]) -> T:
    return x[1]
