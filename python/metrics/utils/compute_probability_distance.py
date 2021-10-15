import numpy as np
from collections import Counter
from typing import Dict, FrozenSet, AbstractSet, List, Callable, Iterator


def compute_probability_distance(ground_truth_density_map: Dict[FrozenSet[int], float],
                                 empirical_probabilities: List[Dict[FrozenSet[int], float]],
                                 powerset: Callable[[bool], Iterator[AbstractSet[int]]]) -> List[float]:
    """
    :param M: number of samples
    :param step: number of steps for which to compute the cumulative mixing rate
    :param ground_truth_history: chronological history of samples mixed from the ground truth distribution
    :return:
    """
    def mixing_rate_partial(S: FrozenSet[int], empirical_probability: Counter) -> float:
        ground_truth_frequency: float = ground_truth_density_map[S]
        empirical_frequency: float = empirical_probability[S] if S in empirical_probability else 0.0
        return abs(ground_truth_frequency - empirical_frequency)

    return [
        np.sum([
            mixing_rate_partial(S, empirical_probability)
            for S in powerset(as_frozenset=True)
        ])
        for empirical_probability in empirical_probabilities
    ]
