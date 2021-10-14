import numpy as np
from dataclasses import dataclass
from collections import Counter
from typing import Dict, FrozenSet, AbstractSet, List
from ..utils import powerset
from ..objective import Objective


@dataclass
class MixingRates:
    gibbs: List[float]
    metropolis: List[float]
    lovasz_projection: List[float]


def split_in_cumulative_lists(history: List[AbstractSet[int]], step: int):
    return (
        list(map(frozenset, history[0:i + step]))
        for i in range(0, len(history), step)
    )


def compute_cumulative_probability_distance(f: Objective, M: int, n_cumulative: int,
                                            ground_truth_density_map: Dict[FrozenSet[int], float],
                                            gibbs_history: List[AbstractSet[int]],
                                            metropolis_history: List[AbstractSet[int]],
                                            lovasz_projection_history: List[AbstractSet[int]]) -> MixingRates:
    """
    :param f: submodular function
    :param n_cumulative: number of steps for which to compute the cumulative mixing rate
    :param ground_truth_history: chronological history of samples mixed from the ground truth distribution
    :param gibbs_history: chronological history of samples mixed from the Gibbs sampler
    :param metropolis_history: chronological history of samples mixed from the Metropolis sampler
    :param lovasz_projection_history: chronological history of samples mixed from the Lovasz-Projection sampler
    :return:
    """
    gibbs_cumulative              = split_in_cumulative_lists(gibbs_history, n_cumulative)
    metropolis_cumulative         = split_in_cumulative_lists(metropolis_history, n_cumulative)
    lovasz_projection_cumulative  = split_in_cumulative_lists(lovasz_projection_history, n_cumulative)

    gibbs_counters              = list(map(Counter, gibbs_cumulative))
    metropolis_counters         = list(map(Counter, metropolis_cumulative))
    lovasz_projection_counters  = list(map(Counter, lovasz_projection_cumulative))

    # compute the powerset of f only once
    P_V = list(utils.powerset(f.V))

    def mixing_rate_partial(S: FrozenSet[int], empirical_counter: Counter) -> float:
        ground_truth_frequency: float = ground_truth_density_map[S]
        empirical_frequency: float = empirical_counter[S] / len(empirical_counter)
        return abs(ground_truth_frequency - empirical_frequency)

    def mixing_rates(empirical_counters: List[Counter]) -> List[float]:
        return [
            np.sum([
                mixing_rate_partial(frozenset(S), empirical_counter)
                for S in P_V
            ])
            for empirical_counter in empirical_counters
        ]

    return MixingRates(
        gibbs=mixing_rates(gibbs_counters),
        metropolis=mixing_rates(metropolis_counters),
        lovasz_projection=mixing_rates(lovasz_projection_counters)
    )
