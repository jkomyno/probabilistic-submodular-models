import itertools
from typing import Iterator, Sequence, AbstractSet


def powerset(V: Sequence[int]) -> Iterator[AbstractSet[int]]:
    """
    Generate the powerset of V, i.e. the 2^V subsets of V
    :param V: ground set of a submodular function
    """
    return map(set,
               itertools.chain.from_iterable(
                   itertools.combinations(V, r) for r in range(len(V) + 1)))
