from ..rng import rng
import os
import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from .objective import Objective
from . import conf_utils
from . import utils
from .. import common
from typing import Dict, FrozenSet, List, Set, Any


def compute_probabilty_df(f: Objective,
                          density_map: Dict[FrozenSet[int], float]) -> pd.DataFrame:
    set_to_vector = common.set_to_vector(f.V)

    return pd.DataFrame(
        [{ 'i': i, 'x': set_to_vector(S), 'probability': probability}
        for i, (S, probability) in enumerate(sorted(density_map.items()))]
    )


def compute_history_df(f: Objective,
                       history: List[Set[int]]) -> pd.DataFrame:
    set_to_vector = common.set_to_vector(f.V)
    
    return pd.DataFrame(
        [
            { 'i': i, 'x': set_to_vector(S) }
            for i, S in enumerate(history)
        ]
    )


def compute_sampler_folder(sampler_name: str, sampler_params: Any) -> str:
    if sampler_name == 'lovasz_projection' or sampler_name == 'lovasz_projection_descending':
        std, eta = sampler_params
        std_as_str = common.float_to_str(std)
        eta_as_str = common.float_to_str(eta) if type(eta) == float else eta
        return f'lovasz_projection/std-{std_as_str}/eta-{eta_as_str}'
    elif sampler_name == 'lovasz_projection_continuous' or sampler_name == 'lovasz_projection_continuous_descending':
        std, eta = sampler_params
        std_as_str = common.float_to_str(std)
        eta_as_str = common.float_to_str(eta) if type(eta) == float else eta
        return f'lovasz_projection_continuous/std-{std_as_str}/eta-{eta_as_str}'
    else:
        return sampler_name


@hydra.main(config_path="../conf", config_name="config")
def mcmc_sample(cfg: DictConfig) -> None:
    # boolean switch for verbose messages
    is_verbose = bool(cfg.selected.verbose)

    # basedir w.r.t. main.py
    basedir = os.path.join(hydra.utils.get_original_cwd(), Path(__file__).parent.parent.parent)

    # probabilistic submodular model name
    f_name = cfg.obj.name

    # number of samples
    M = cfg.sample_size.M

    for f in conf_utils.get_objective(rng=rng, cfg=cfg):
        powerset = common.powerset(f.V)

        # output folder for the ground truth probabilities CSVs
        out_ground_truth = f'{basedir}/out/{f_name}/n-{f.n}'

        # Only compute the ground truth density map once
        if not Path(f'{out_ground_truth}/ground_truth.csv').exists():
            ground_truth_density_f = utils.compute_density(f)
            ground_truth_density_map: Dict[FrozenSet[int], float] = \
                dict(((S, ground_truth_density_f(S)) for S in powerset(as_frozenset=True)))

            print(f'')
            print(f'Ground Truth probabilities')
            df = compute_probabilty_df(f, ground_truth_density_map)
            common.to_csv(out_ground_truth, df, name='ground_truth', create_path=True)

        # import the selected sampler, distinguishing among the different parameters used
        sampler_name = cfg.sampler.name
        for sample, sampler_params in conf_utils.get_sampler(f=f, rng=rng, cfg=cfg):
            samples_f, history = sample()

            # output folder for the samplers history CSVs
            sampler_folder = compute_sampler_folder(sampler_name, sampler_params)
            out_history = f'{basedir}/out/{f_name}/n-{f.n}/M-{M}/{sampler_folder}'

            # run the selected sampler
            print(f'')
            print(f'{sampler_name} samples ({len(samples_f)})')
            df = compute_history_df(f, history)
            common.to_csv(out_history, df, name=f'history', create_path=True)

            print(f'')
