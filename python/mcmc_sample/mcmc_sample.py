from ..rng import rng
import os
import hydra
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
from .objective import Objective
from . import conf_utils
from . import utils
from typing import Dict, FrozenSet, List, Set


def compute_probabilty_df(f: Objective,
                          density_map: Dict[FrozenSet[int], float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{ 'i': i, 'x': utils.set_to_vector(f, S), 'probability': probability}
        for i, (S, probability) in enumerate(sorted(density_map.items()))]
    )


def compute_history_df(f: Objective,
                       history: List[Set[int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            { 'i': i, 'x': utils.set_to_vector(f, S) }
            for i, S in enumerate(history)
        ]
    )


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

        # output folder for the ground truth probabilities CSVs
        out_ground_truth = f'{basedir}/out/{f_name}/n-{f.n}'

        # Only compute the ground truth density map once
        if not Path(f'{out_ground_truth}/ground_truth.csv').exists():
            ground_truth_density_f = utils.compute_density(f)
            ground_truth_density_map: Dict[FrozenSet[int], float] = \
                dict(((frozenset(S), ground_truth_density_f(S)) for S in utils.powerset(f.V)))

            print(f'')
            print(f'Ground Truth probabilities')
            df = compute_probabilty_df(f, ground_truth_density_map)
            utils.to_csv(out_ground_truth, df, name='ground_truth', create_path=True)

        # import the selected sampler
        sampler_name = cfg.sampler.name
        sample = conf_utils.get_sampler(f=f, rng=rng, cfg=cfg)
        samples_f, history = sample()

        # output folder for the samplers history CSVs
        out_history = f'{basedir}/out/{f_name}/n-{f.n}/M-{M}/{sampler_name}'

        # run the selected sampler
        print(f'')
        print(f'{sampler_name} samples ({len(samples_f)})')
        df = compute_history_df(f, history)
        utils.to_csv(out_history, df, name=f'history', create_path=True)

        print(f'')
