import os
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
import conf_utils
import utils
import sampler
from typing import Dict, FrozenSet, List, Set


# numpy random generator instance
rng: np.random.Generator = np.random.default_rng(2022)


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    # boolean switch for verbose messages
    is_verbose = bool(cfg.selected.verbose)

    # number of samples
    M = cfg.selected.M

    # basedir w.r.t. main.py
    basedir = os.path.join(hydra.utils.get_original_cwd(), Path(__file__).parent)

    # output folder for the samplers history CSVs
    out_history = f'{basedir}/out/history'

    # objective submodular function
    f = conf_utils.get_objective(rng, cfg=cfg)

    ######################
    #  Run the samplers  #
    ######################

    # Samples from the Gibbs sampler
    print(f'Computing Gibbs samples...')
    gibbs_samples_f, gibbs_history = sampler.gibbs(f, rng, cfg)

    # Samples from the Metropolis sampler
    print(f'Computing Metropolis samples...')
    metropolis_samples_f, metropolis_history = sampler.metropolis(f, rng, cfg)

    # Samples from the Lovasz-Projection sampler
    print(f'Computing Lovasz-Projection samples...')
    lovasz_projection_samples_f, lovasz_projection_history = sampler.lovasz_projection(f, rng, cfg)

    # Ground truth density map 
    ground_truth_density_f = utils.compute_density(f)
    ground_truth_density_map: Dict[FrozenSet[int], float] = \
        dict(((frozenset(S), ground_truth_density_f(S)) for S in utils.powerset(f.V)))

    def compute_history_df(history: List[Set[int]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                { 'i': i, 'x': utils.set_to_vector(f, S) }
                for i, S in enumerate(history)
            ]
        )

    def compute_probabilty_df(density_map: Dict[FrozenSet[int], float]) -> pd.DataFrame:
        return pd.DataFrame(
            [{ 'i': i, 'x': utils.set_to_vector(f, S), 'probability': probability}
            for i, (S, probability) in enumerate(sorted(density_map.items()))]
        )

    print(f'')
    print(f'Gibbs samples ({len(gibbs_samples_f)})')
    df = compute_history_df(gibbs_history)
    df.to_csv(f'{out_history}/gibbs_history.csv', mode='w', header=True, index=False)

    print(f'')
    print(f'Metropolis samples ({len(metropolis_samples_f)})')
    df = compute_history_df(metropolis_history)
    df.to_csv(f'{out_history}/metropolis_history.csv', mode='w', header=True, index=False)

    print(f'')
    print(f'Lovasz-Projection samples ({len(lovasz_projection_samples_f)})')
    df = compute_history_df(lovasz_projection_history)
    df.to_csv(f'{out_history}/lovasz_projection_history.csv', mode='w', header=True, index=False)

    print(f'')
    print(f'Ground Truth probabilities')
    df = compute_probabilty_df(ground_truth_density_map)
    df.to_csv(f'{out_history}/ground_truth_probabilities.csv', mode='w', header=True, index=False)

    print(f'')

if __name__ == '__main__':
    run()
