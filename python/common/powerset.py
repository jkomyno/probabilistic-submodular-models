import itertools
from typing import Sequence, Iterator, AbstractSet


def powerset(V: Sequence[int]):
    """
    Generate the powerset of V, i.e. the 2^V subsets of V
    :param V: ground set of a submodular function
    """
    n = len(V)

    def inner(as_frozenset: bool = False) -> Iterator[AbstractSet[int]]:
        return map(frozenset if as_frozenset else set,
                   itertools.chain.from_iterable(
                       itertools.combinations(V, r) for r in range(n + 1)))

    return inner
