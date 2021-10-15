import os
import hydra
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
from typing import Dict, FrozenSet, List
from . import utils
from .. import common


def read_csv(filepath: str):
    return pd.read_csv(filepath, sep=',', decimal='.', encoding='utf-8',
                       index_col=None)


def compute_probabilty_distance_df(probability_distances: List[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{ 'i': i, 'distance': distance }
        for i, distance in enumerate(probability_distances)]
    )


@hydra.main(config_path="../conf", config_name="config")
def metrics(cfg: DictConfig) -> None:
    # boolean switch for verbose messages
    is_verbose = bool(cfg.selected.verbose)

    # basedir w.r.t. main.py
    basedir = os.path.join(hydra.utils.get_original_cwd(), Path(__file__).parent.parent.parent)
    
    for history_path in Path(os.path.join(basedir, 'out')).rglob('**/history.csv'):
        ground_truth_path = f'{history_path.parent.parent.parent}/ground_truth.csv'

        f_name, n_str, M_str, sampler_name = history_path.parent.parts[-4:]
        n = int(n_str.split('-')[1])
        M = int(M_str.split('-')[1])

        # ground set
        V = list(range(n))

        # instantiate closures that use the ground set
        powerset = common.powerset(V)
        vector_to_set = common.vector_to_set(V)

        print(f'f_name: {f_name}, sampler_name: {sampler_name}')

        ground_truth_df = read_csv(ground_truth_path)
        ground_truth_df = utils.add_array(ground_truth_df)

        ground_truth_density_map: Dict[FrozenSet[int], float] = {
            S: ground_truth_df['probability'][i] for i, S in enumerate(ground_truth_df['array'].map(vector_to_set))
        }

        history_df = read_csv(history_path)
        history_df = utils.add_array(history_df)

        empirical_cumulative_probabilities = utils.get_cumulative_probabilities(history_df,
                                                                                powerset=powerset,
                                                                                vector_to_set=vector_to_set,
                                                                                step=50)
        
        cumulative_probability_distances = utils.compute_probability_distance(ground_truth_density_map,
                                                                              empirical_cumulative_probabilities,
                                                                              powerset=powerset)

        print(f'cumulative_probability_distances:')
        print(cumulative_probability_distances)
        print(f'\n\n')

        cumulative_probability_distances_df = compute_probabilty_distance_df(cumulative_probability_distances)
        common.to_csv(out_dir=history_path.parent, df=cumulative_probability_distances_df,
                      name='cumulative_probability_distances', create_path=False)
