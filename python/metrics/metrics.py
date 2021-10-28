import os
import hydra
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
from typing import List
from .. import common
from . import utils


def compute_probabilty_distance_df(probability_distances: List[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{ 'i': i, 'distance': distance }
        for i, distance in enumerate(probability_distances)]
    )


def metrics_sampler(traverser: utils.Traverser):
    # size of the cumulative step
    n_steps = 50

    empirical_cumulative_probabilities = common.get_cumulative_probabilities(
        traverser.history_df,
        powerset=traverser.powerset,
        vector_to_set=traverser.vector_to_set,
        step=n_steps)
    
    cumulative_probability_distances = common.compute_probability_distance(
        traverser.ground_truth_density_map,
        empirical_cumulative_probabilities,
        powerset=traverser.powerset)

    print(f'cumulative_probability_distances:')
    print(cumulative_probability_distances)
    print(f'\n\n')

    cumulative_probability_distances_df = compute_probabilty_distance_df(cumulative_probability_distances)
    common.to_csv(out_dir=traverser.out_dir, df=cumulative_probability_distances_df,
                    name='cumulative_probability_distances', create_path=False)


@hydra.main(config_path="../conf", config_name="config")
def metrics(cfg: DictConfig) -> None:
    # boolean switch for verbose messages
    is_verbose = bool(cfg.selected.verbose)

    # basedir w.r.t. main.py
    basedir = os.path.join(hydra.utils.get_original_cwd(), Path(__file__).parent.parent.parent)
    
    ###############################
    #  lovasz_projection sampler  #
    ###############################

    sampler_name = 'lovasz_projection'
    for traverser in utils.traverse_lovasz_projection(basedir, sampler_name=sampler_name):
        metrics_sampler(traverser)

    ###################
    #  gibbs sampler  #
    ###################

    sampler_name = 'gibbs'
    for traverser in utils.traverse_sampler(basedir, sampler_name=sampler_name):
        metrics_sampler(traverser)

    ########################
    #  metropolis sampler  #
    ########################

    sampler_name = 'metropolis'
    for traverser in utils.traverse_sampler(basedir, sampler_name=sampler_name):
        metrics_sampler(traverser)
