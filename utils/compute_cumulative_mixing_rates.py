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
    gibbs_gotovos: List[float]
    metropolis_gotovos: List[float]
    # lovasz_projection: List[float]


def split_in_cumulative_lists(history: List[AbstractSet[int]], step: int):
    return (
        list(map(frozenset, history[0:i + step]))
        for i in range(0, len(history), step)
    )


def compute_cumulative_mixing_rates(f: Objective, n_cumulative: int,
                                    ground_truth_history: List[AbstractSet[int]],
                                    gibbs_history: List[AbstractSet[int]],
                                    metropolis_history: List[AbstractSet[int]],
                                    gibbs_gotovos_history: List[AbstractSet[int]],
                                    metropolis_gotovos_history: List[AbstractSet[int]],
                                    lovasz_projection_history: List[AbstractSet[int]]) -> MixingRates:
    """
    :param f: submodular function
    :param n_cumulative: number of steps for which to compute the cumulative mixing rate
    :param ground_truth_history: chronological history of samples mixed from the ground truth distribution
    :param gibbs_history: chronological history of samples mixed from the Gibbs sampler
    :param metropolis_history: chronological history of samples mixed from the Metropolis sampler
    :param gibbs_gotovos_history: chronological history of samples mixed from the Gibbs-Gotovos sampler
    :param metropolis_gotovos_history: chronological history of samples mixed from the Metropolis-Gotovos sampler
    :param lovasz_projection_history: chronological history of samples mixed from the Lovasz-Projection sampler
    :return:
    """
    ground_truth_cumulative       = split_in_cumulative_lists(ground_truth_history, n_cumulative)
    gibbs_cumulative              = split_in_cumulative_lists(gibbs_history, n_cumulative)
    metropolis_cumulative         = split_in_cumulative_lists(metropolis_history, n_cumulative)
    gibbs_gotovos_cumulative      = split_in_cumulative_lists(gibbs_gotovos_history, n_cumulative)
    metropolis_gotovos_cumulative = split_in_cumulative_lists(metropolis_gotovos_history, n_cumulative)
    # lovasz_projection_cumulative  = split_in_cumulative_lists(lovasz_projection_history, n_cumulative)

    ground_truth_counters       = list(map(Counter, ground_truth_cumulative))
    gibbs_counters              = list(map(Counter, gibbs_cumulative))
    metropolis_counters         = list(map(Counter, metropolis_cumulative))
    gibbs_gotovos_counters      = list(map(Counter, gibbs_gotovos_cumulative))
    metropolis_gotovos_counters = list(map(Counter, metropolis_gotovos_cumulative))
    # lovasz_projection_counters  = list(map(Counter, lovasz_projection_cumulative))

    # compute the powerset of f only once
    P_V = list(utils.powerset(f.V))

    def mixing_rate_partial(S: FrozenSet[int], ground_truth_counter: Counter, empirical_counter: Counter) -> float:
        f_S: float = empirical_counter[S]
        p_S: float = ground_truth_counter[S]
        return abs(f_S - p_S)

    def mixing_rates(empirical_counters: List[Counter]) -> List[float]:
        return [
            np.sum([
                mixing_rate_partial(frozenset(S), ground_truth_counter, empirical_counter)
                for S in P_V
            ])
            for ground_truth_counter, empirical_counter in zip(ground_truth_counters, empirical_counters)
        ]

    return MixingRates(
        gibbs=mixing_rates(gibbs_counters),
        metropolis=mixing_rates(metropolis_counters),
        gibbs_gotovos=mixing_rates(gibbs_gotovos_counters),
        metropolis_gotovos=mixing_rates(metropolis_gotovos_counters),
        # lovasz_projection=mixing_rates(lovasz_projection_counters)
    )
