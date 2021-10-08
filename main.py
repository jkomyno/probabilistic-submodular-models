import hydra
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import operator
import conf_utils
import utils
import sampler
from typing import Dict, FrozenSet


# numpy random generator instance
rng: np.random.Generator = np.random.default_rng(2022)


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    # boolean switch for verbose messages
    is_verbose = bool(cfg.selected.verbose)

    # number of samples
    M = cfg.selected.M

    # number of bins for computing the mixing rate
    n_bins = cfg.selected.n_bins

    # number of steps for which to compute the cumulative mixing rate
    n_cumulative = cfg.selected.n_cumulative

    # basedir w.r.t. main.py
    basedir = f'{Path(__file__).parent}'

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

    # Samples from the Gibbs-Govotos sampler
    print(f'Computing Gibbs-Govotos samples...')
    gibbs_gotovos_samples_f, gibbs_gotovos_history = sampler.gibbs_gotovos(f, rng, cfg)

    # Samples from the Metropolis-Gotovos sampler
    print(f'Computing Metropolis-Gotovos samples...')
    metropolis_gotovos_samples_f, metropolis_gotovos_history = sampler.metropolis_gotovos(f, rng, cfg)

    # Samples from the Lovasz-Projection sampler
    # print(f'Computing Lovasz-Projection samples...')
    # lovasz_projection_samples_f, lovasz_projection_history = sampler.lovasz_projection(f, rng, cfg)

    # Ground truth density map 
    ground_truth_density_f = utils.compute_density(f)
    ground_truth_density_map: Dict[FrozenSet[int], float] = \
        dict(((frozenset(S), ground_truth_density_f(S)) for S in utils.powerset(f.V)))

    if is_verbose:
        print(f'Gibbs samples ({len(gibbs_samples_f)})')
        for S, frequency in sorted(gibbs_samples_f.items()):
            print(f'{S}: {frequency}')

        print(f'')
        print(f'Metropolis samples ({len(metropolis_samples_f)})')
        for S, frequency in sorted(metropolis_samples_f.items()):
            print(f'{S}: {frequency}')

        print(f'')
        print(f'Gibbs-Gotovos samples ({len(gibbs_samples_f)})')
        for S, frequency in sorted(gibbs_gotovos_samples_f.items()):
            print(f'{S}: {frequency}')

        print(f'')
        print(f'Metropolis-Gotovos samples ({len(metropolis_samples_f)})')
        for S, frequency in sorted(metropolis_gotovos_samples_f.items()):
            print(f'{S}: {frequency}')

        # print(f'')
        # print(f'Lovasz-Projection samples ({len(lovasz_projection_samples_f)})')
        # for S, frequency in sorted(lovasz_projection_samples_f.items()):
        #     print(f'{S}: {frequency}')

    ##########################################
    #  Compute probability distances by bin  #
    ##########################################

    gibbs_mixing_rates, \
        metropolis_mixing_rates, \
        gibbs_gotovos_mixing_rates, \
        metropolis_gotovos_mixing_rates = operator.attrgetter('gibbs', 'metropolis', 'gibbs_gotovos', 'metropolis_gotovos')(
            utils.compute_bins_probability_distance(f, M, n_bins,
                                       ground_truth_density_map=ground_truth_density_map,
                                       gibbs_history=gibbs_history,
                                       metropolis_history=metropolis_history,
                                       gibbs_gotovos_history=gibbs_gotovos_history,
                                       metropolis_gotovos_history=metropolis_gotovos_history,
                                       lovasz_projection_history=None))

    print(f'')
    print(f'Gibbs probability distances: \n{gibbs_mixing_rates}')

    print(f'')
    print(f'Metropolis probability distances: \n{metropolis_mixing_rates}')

    print(f'')
    print(f'Gibbs-Gotovos probability distances: \n{gibbs_gotovos_mixing_rates}')

    print(f'')
    print(f'Metropolis-Gotovos probability distances: \n{metropolis_gotovos_mixing_rates}')

    # print(f'')
    # print(f'Lovasz-Projection probability distances: \n{lovasz_projection_mixing_rates}')

    ##############################################
    #  Compute cumulative probability distances  #
    ##############################################

    print(f'\nComputing cumulative probability distances...')
    
    gibbs_cumulative_mixing_rates, \
        metropolis_cumulative_mixing_rates, \
        gibbs_gotovos_cumulative_mixing_rates, \
        metropolis_gotovos_cumulative_mixing_rates = operator.attrgetter('gibbs', 'metropolis', 'gibbs_gotovos', 'metropolis_gotovos')(
            utils.compute_cumulative_probability_distance(f, M, n_cumulative,
                                                  ground_truth_density_map=ground_truth_density_map,
                                                  gibbs_history=gibbs_history,
                                                  metropolis_history=metropolis_history,
                                                  gibbs_gotovos_history=gibbs_gotovos_history,
                                                  metropolis_gotovos_history=metropolis_gotovos_history,
                                                  lovasz_projection_history=None))

    print(f'')
    print(f'Gibbs cumulative probability distances: \n{gibbs_cumulative_mixing_rates}')

    print(f'')
    print(f'Metropolis cumulative probability distances: \n{metropolis_cumulative_mixing_rates}')

    print(f'')
    print(f'Gibbs-Gotovos cumulative probability distances: \n{gibbs_gotovos_cumulative_mixing_rates}')

    print(f'')
    print(f'Metropolis-Gotovos cumulative probability distances: \n{metropolis_gotovos_cumulative_mixing_rates}')

    # print(f'')
    # print(f'Lovasz-Projection cumulative probability distances: \n{lovasz_projection_mixing_rates}')


    """
    ##########################################
    #  Compute probability distances by bin  #
    ##########################################

    print(f'\nComputing probability distances by bin...')

    # We create n_bins for each sample, and we want to compare each bin to the ground truth's bin
    # f_S: empirical frequency of S
    # p_S: ground truth probability of S
    # \sum_{S \in P(S)} |f_S - p_S|

    gibbs_mixing_rates, \
        metropolis_mixing_rates, \
        gibbs_gotovos_mixing_rates, \
        metropolis_gotovos_mixing_rates = operator.attrgetter('gibbs', 'metropolis', 'gibbs_gotovos', 'metropolis_gotovos')(
            utils.compute_mixing_rates(f, n_bins,
                                       ground_truth_history=ground_truth_history,
                                       gibbs_history=gibbs_history,
                                       metropolis_history=metropolis_history,
                                       gibbs_gotovos_history=gibbs_gotovos_history,
                                       metropolis_gotovos_history=metropolis_gotovos_history,
                                       lovasz_projection_history=None))

    print(f'')
    print(f'Gibbs probability distances: \n{gibbs_mixing_rates}')

    print(f'')
    print(f'Metropolis probability distances: \n{metropolis_mixing_rates}')

    print(f'')
    print(f'Gibbs-Gotovos probability distances: \n{gibbs_gotovos_mixing_rates}')

    print(f'')
    print(f'Metropolis-Gotovos probability distances: \n{metropolis_gotovos_mixing_rates}')

    # print(f'')
    # print(f'Lovasz-Projection probability distances: \n{lovasz_projection_mixing_rates}')

    ##############################################
    #  Compute cumulative probability distances  #
    ##############################################

    print(f'\nComputing cumulative probability distances...')
    
    gibbs_cumulative_mixing_rates, \
        metropolis_cumulative_mixing_rates, \
        gibbs_gotovos_cumulative_mixing_rates, \
        metropolis_gotovos_cumulative_mixing_rates = operator.attrgetter('gibbs', 'metropolis', 'gibbs_gotovos', 'metropolis_gotovos')(
            utils.compute_cumulative_mixing_rates(f, n_cumulative,
                                                  ground_truth_history=ground_truth_history,
                                                  gibbs_history=gibbs_history,
                                                  metropolis_history=metropolis_history,
                                                  gibbs_gotovos_history=gibbs_gotovos_history,
                                                  metropolis_gotovos_history=metropolis_gotovos_history,
                                                  lovasz_projection_history=None))

    print(f'')
    print(f'Gibbs cumulative probability distances: \n{gibbs_cumulative_mixing_rates}')

    print(f'')
    print(f'Metropolis cumulative probability distances: \n{metropolis_cumulative_mixing_rates}')

    print(f'')
    print(f'Gibbs-Gotovos cumulative probability distances: \n{gibbs_gotovos_cumulative_mixing_rates}')

    print(f'')
    print(f'Metropolis-Gotovos cumulative probability distances: \n{metropolis_gotovos_cumulative_mixing_rates}')

    # print(f'')
    # print(f'Lovasz-Projection cumulative probability distances: \n{lovasz_projection_mixing_rates}')
    """

if __name__ == '__main__':
    run()
