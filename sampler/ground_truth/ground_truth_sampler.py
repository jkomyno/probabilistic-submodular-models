import numpy as np
from omegaconf import DictConfig
from collections import Counter
from objective import Objective
from typing import FrozenSet, Tuple, List
import utils


def ground_truth_sampler(f: Objective, rng: np.random.Generator, cfg: DictConfig) -> Tuple[Counter, List[FrozenSet[int]]]:
    """
    :param f: submodular function
    :param rng: numpy random generator instance
    :param cfg: Hydra configuration dictionary
    """

    # number of samples
    M = cfg.selected.M

    density_f = utils.compute_density(f)
    set_densities_f_as_dict = dict(((frozenset(S), density_f(S)) for S in utils.powerset(f.V)))

    # chronological history of ground truth samples
    ground_truth_history = rng.choice(list(set_densities_f_as_dict.keys()), size=(M, ), replace=True,
                                      p=list(set_densities_f_as_dict.values()))

    # aggregate the samples from the known distribution
    ground_samples_f = Counter(ground_truth_history)
    return ground_samples_f, ground_truth_history.tolist()
