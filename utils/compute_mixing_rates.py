import numpy as np
from dataclasses import dataclass
from collections import Counter
from typing import FrozenSet, AbstractSet, List
import utils
from objective import Objective


@dataclass
class MixingRates:
    gibbs: List[float]
    metropolis: List[float]
    lovasz_projection: List[float]


def split_in_bins(history: List[AbstractSet[int]], n_bins: int) -> List[List[FrozenSet[int]]]:
    return list(map(lambda v: v.tolist(), np.array_split(list(map(frozenset, history)), n_bins)))


def compute_mixing_rates(f: Objective, n_bins: int,
                         ground_truth_history: List[AbstractSet[int]],
                         gibbs_history: List[AbstractSet[int]],
                         metropolis_history: List[AbstractSet[int]],
                         lovasz_projection_history: List[AbstractSet[int]]) -> MixingRates:
    """
    :param f: submodular function
    :param n_bins: number of bins for which to compute the mixing rate
    :param ground_truth_history: chronological history of samples mixed from the ground truth distribution
    :param gibbs_history: chronological history of samples mixed from the Gibbs sampler
    :param metropolis_history: chronological history of samples mixed from the Metropolis sampler
    :param lovasz_projection_history: chronological history of samples mixed from the Frank-Wolfe sampler
    :return:
    """
    ground_truth_bins = split_in_bins(ground_truth_history, n_bins)
    gibbs_bins        = split_in_bins(gibbs_history, n_bins)
    metropolis_bins   = split_in_bins(metropolis_history, n_bins)
    lovasz_projection_bins  = split_in_bins(lovasz_projection_history, n_bins)

    ground_truth_counters = list(map(Counter, ground_truth_bins))
    gibbs_counters        = list(map(Counter, gibbs_bins))
    metropolis_counters   = list(map(Counter, metropolis_bins))
    lovasz_projection_counters  = list(map(Counter, lovasz_projection_bins))

    # compute the powerset of f
    P_V = list(utils.powerset(f.V))

    def mixing_rate_partial(S: FrozenSet[int], ground_truth_counter: Counter, empirical_counter: Counter) -> float:
        f_S: float = empirical_counter[S]
        p_S: float = ground_truth_counter[S]
        return abs(f_S - p_S)

    def mixing_rates(empirical_counters: List[Counter]) -> List[float]:
        return [
            sum((
                mixing_rate_partial(frozenset(S), ground_truth_counter, empirical_counter)
                for S in P_V
            ))
            for ground_truth_counter, empirical_counter in zip(ground_truth_counters, empirical_counters)
        ]

    return MixingRates(
        gibbs=mixing_rates(gibbs_counters),
        metropolis=mixing_rates(metropolis_counters),
        lovasz_projection=mixing_rates(lovasz_projection_counters)
    )
