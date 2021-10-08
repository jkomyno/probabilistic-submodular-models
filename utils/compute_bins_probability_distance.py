import numpy as np
from dataclasses import dataclass
from collections import Counter
from typing import Dict, FrozenSet, AbstractSet, List
import utils
from objective import Objective


@dataclass
class MixingRates:
    gibbs: List[float]
    metropolis: List[float]
    gibbs_gotovos: List[float]
    metropolis_gotovos: List[float]
    # lovasz_projection: List[float]


def split_in_bins(history: List[AbstractSet[int]], n_bins: int) -> List[List[FrozenSet[int]]]:
    return list(map(lambda v: v.tolist(), np.array_split(list(map(frozenset, history)), n_bins)))


def compute_bins_probability_distance(f: Objective, M: int, n_bins: int,
                                      ground_truth_density_map: Dict[FrozenSet[int], float],
                                      gibbs_history: List[AbstractSet[int]],
                                      metropolis_history: List[AbstractSet[int]],
                                      gibbs_gotovos_history: List[AbstractSet[int]],
                                      metropolis_gotovos_history: List[AbstractSet[int]],
                                      lovasz_projection_history: List[AbstractSet[int]]) -> MixingRates:
    """
    :param f: submodular function
    :param n_bins: number of bins for which to compute the mixing rate
    :param ground_truth_history: chronological history of samples mixed from the ground truth distribution
    :param gibbs_history: chronological history of samples mixed from the Gibbs sampler
    :param metropolis_history: chronological history of samples mixed from the Metropolis sampler
    :param gibbs_gotovos_history: chronological history of samples mixed from the Gibbs-Gotovos sampler
    :param metropolis_gotovos_history: chronological history of samples mixed from the Metropolis-Gotovos sampler
    :param lovasz_projection_history: chronological history of samples mixed from the Lovasz-Projection sampler
    :return:
    """
    gibbs_bins              = split_in_bins(gibbs_history, n_bins)
    metropolis_bins         = split_in_bins(metropolis_history, n_bins)
    gibbs_gotovos_bins      = split_in_bins(gibbs_gotovos_history, n_bins)
    metropolis_gotovos_bins = split_in_bins(metropolis_gotovos_history, n_bins)
    # lovasz_projection_bins  = split_in_bins(lovasz_projection_history, n_bins)

    gibbs_counters              = list(map(Counter, gibbs_bins))
    metropolis_counters         = list(map(Counter, metropolis_bins))
    gibbs_gotovos_counters      = list(map(Counter, gibbs_gotovos_bins))
    metropolis_gotovos_counters = list(map(Counter, metropolis_gotovos_bins))
    # lovasz_projection_counters  = list(map(Counter, lovasz_projection_bins))

    # compute the powerset of f only once
    P_V = list(utils.powerset(f.V))

    # print(f'gibbs_history:\n{gibbs_history}\n')
    # print(f'metropolis_history:\n{metropolis_history}\n')

    n_elements_per_bin = M / n_bins

    def mixing_rate_partial(S: FrozenSet[int], empirical_counter: Counter) -> float:
        ground_truth_frequency: float = ground_truth_density_map[S]
        empirical_frequency: float = empirical_counter[S] / n_elements_per_bin

        print(f'p_i: {ground_truth_frequency}; f_i: {empirical_frequency}; S: {set(S)}')

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
        gibbs_gotovos=mixing_rates(gibbs_gotovos_counters),
        metropolis_gotovos=mixing_rates(metropolis_gotovos_counters),
        # lovasz_projection=mixing_rates(lovasz_projection_counters)
    )
