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

    # Samples from the Lovasz-Projection sampler
    print(f'Computing Lovasz-Projection samples...')
    lovasz_projection_samples_f, lovasz_projection_history = sampler.lovasz_projection(f, rng, cfg)

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
        print(f'Lovasz-Projection samples ({len(lovasz_projection_samples_f)})')
        for S, frequency in sorted(lovasz_projection_samples_f.items()):
            print(f'{S}: {frequency}')

        print(f'')
        print(f'Ground Truth')
        for S, probability in sorted(ground_truth_density_map.items()):
            print(f'{S}: {probability}')            

    print(f'')

    ##########################################
    #  Compute probability distances by bin  #
    ##########################################

    gibbs_mixing_rates, \
        metropolis_mixing_rates, \
        lovasz_projection_mixing_rates = operator.attrgetter('gibbs', 'metropolis', 'lovasz_projection')(
            utils.compute_bins_probability_distance(f, M, n_bins,
                                       ground_truth_density_map=ground_truth_density_map,
                                       gibbs_history=gibbs_history,
                                       metropolis_history=metropolis_history,
                                       lovasz_projection_history=lovasz_projection_history))

    print(f'')
    print(f'Gibbs probability distances: \n{gibbs_mixing_rates}')

    print(f'')
    print(f'Metropolis probability distances: \n{metropolis_mixing_rates}')

    print(f'')
    print(f'Lovasz-Projection probability distances: \n{lovasz_projection_mixing_rates}')

    ##############################################
    #  Compute cumulative probability distances  #
    ##############################################

    print(f'\nComputing cumulative probability distances...\n')
    
    gibbs_cumulative_mixing_rates, \
        metropolis_cumulative_mixing_rates, \
        lovasz_projection_cumulative_mixing_rates = operator.attrgetter('gibbs', 'metropolis', 'lovasz_projection')(
            utils.compute_cumulative_probability_distance(f, M, n_cumulative,
                                                  ground_truth_density_map=ground_truth_density_map,
                                                  gibbs_history=gibbs_history,
                                                  metropolis_history=metropolis_history,
                                                  lovasz_projection_history=lovasz_projection_history))

    print(f'')
    print(f'Gibbs cumulative probability distances: \n{gibbs_cumulative_mixing_rates}')

    print(f'')
    print(f'Metropolis cumulative probability distances: \n{metropolis_cumulative_mixing_rates}')

    print(f'')
    print(f'Lovasz-Projection cumulative probability distances: \n{lovasz_projection_cumulative_mixing_rates}')

if __name__ == '__main__':
    run()
