import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
from .. import common
from . import utils


def plot_sampler(sampler_name: str, traverser: utils.Traverser):
    # size of the cumulative step
    n_steps = 50

    cumulative_probability_distances_df = common.read_csv(traverser.cumulative_probability_distances_path)
    # empirical_cumulative_probabilities = common.get_cumulative_probabilities(
    #     traverser.history_df,
    #     powerset=traverser.powerset,
    #     vector_to_set=traverser.vector_to_set,
    #     step=n_steps)

    # plot empirical probability vs ground truth probability of subsets of V 
    # utils.plot_empirical_subset_probabilities(
    #     empirical_cumulative_probabilities,
    #     sampler_name=sampler_name,
    #     step=n_steps,
    #     traverser=traverser)

    # plot cumulative probability distances  
    utils.plot_cumulative_probability_distances(
        cumulative_probability_distances_df,
        sampler_name=sampler_name,
        traverser=traverser)


@hydra.main(config_path="../conf", config_name="config")
def plotter(cfg: DictConfig) -> None:
    # boolean switch for verbose messages
    is_verbose = bool(cfg.selected.verbose)

    # basedir w.r.t. main.py
    basedir = os.path.join(hydra.utils.get_original_cwd(), Path(__file__).parent.parent.parent)

    ###############################
    #  lovasz_projection sampler  #
    ###############################

    sampler_name = 'lovasz_projection'
    for traverser in utils.traverse_lovasz_projection(basedir, sampler_name=sampler_name):
        plot_sampler(sampler_name, traverser)

    ###################
    #  gibbs sampler  #
    ###################

    sampler_name = 'gibbs'
    for traverser in utils.traverse_sampler(basedir, sampler_name=sampler_name):
        plot_sampler(sampler_name, traverser)

    ########################
    #  metropolis sampler  #
    ########################

    sampler_name = 'metropolis'
    for traverser in utils.traverse_sampler(basedir, sampler_name=sampler_name):
        plot_sampler(sampler_name, traverser)
