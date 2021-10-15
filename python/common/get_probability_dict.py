from collections import Counter
from typing import Dict, FrozenSet, Callable, Iterator, AbstractSet


def get_probability_dict(counter: Counter,
                         powerset: Callable[[bool], Iterator[AbstractSet[int]]]) -> Dict[FrozenSet[int], float]:
    denominator = sum((v for _, v in counter.items()))

    return {
        S: counter[S] / denominator if S in counter else 0 
        for S in powerset(as_frozenset=True)
    }
