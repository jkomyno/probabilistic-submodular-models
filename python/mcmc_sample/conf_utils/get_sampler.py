import numpy as np
from omegaconf import DictConfig
from collections import Counter
from .. import sampler
from ..objective import Objective
from typing import Tuple, List, Set


SAMPLER_MAP = {
    'gibbs': lambda *args: load_gibbs(*args),
    'metropolis': lambda *args: load_metropolis(*args),
    'lovasz_projection': lambda *args: load_lovasz_projection(*args),
}


def load_gibbs(f: Objective, rng: np.random.Generator,
               cfg: DictConfig):
    def load() -> Tuple[Counter, List[Set[int]]]:
        samples_f, history = sampler.gibbs(f, rng, cfg)
        return samples_f, history

    return load


def load_metropolis(f: Objective, rng: np.random.Generator,
                    cfg: DictConfig):
    def load() -> Tuple[Counter, List[Set[int]]]:
        samples_f, history = sampler.metropolis(f, rng, cfg)
        return samples_f, history

    return load


def load_lovasz_projection(f: Objective, rng: np.random.Generator,
                           cfg: DictConfig):
    def load() -> Tuple[Counter, List[Set[int]]]:
        samples_f, history = sampler.lovasz_projection(f, rng, cfg)
        return samples_f, history

    return load


def get_sampler(f: Objective, rng: np.random.Generator,
                cfg: DictConfig):
    """
    Return an instance of the selected sampler.
    :param f: probabilistic submodular model
    :param rng: numpy random generator instance
    :param cfg: Hydra configuration dictionary
    """
    sampler_name = cfg.sampler.name
    print(f'Loading {sampler_name}...')
    return SAMPLER_MAP[sampler_name](f, rng, cfg)
