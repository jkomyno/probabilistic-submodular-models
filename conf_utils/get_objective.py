import numpy as np
from omegaconf import DictConfig
from objective import Objective, DemoMonotone, DemoNonMonotone


OBJ_MAP = {
    'demo_monotone': lambda *args: load_demo_monotone(*args),
    'demo_non_monotone': lambda *args: load_demo_non_monotone(*args),
}


def load_demo_monotone(rng: np.random.Generator, params):
    """
    Generate a random set-modular, monotone function
    :param rng: numpy random generator instance
    :param params: 'params.demo_monotone' dictionary entry in conf/config.yaml
    """
    return DemoMonotone(rng, n=params.n)


def load_demo_non_monotone(rng: np.random.Generator, params):
    """
    Generate a random set-modular, non_monotone function
    :param rng: numpy random generator instance
    :param params: 'params.demo_non_monotone' dictionary entry in conf/config.yaml
    """
    return DemoNonMonotone(rng, n=params.n)


def get_objective(rng: np.random.Generator, cfg: DictConfig) -> Objective:
    """
    Return an instance of the selected set-submodular objective
    :param rng: numpy random generator instance
    :param cfg: Hydra configuration dictionary
    """
    objective_name = cfg.selected.objective
    return OBJ_MAP[objective_name](rng, cfg.objectives[objective_name])
