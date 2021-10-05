import hydra
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import operator
import conf_utils
import utils
import sampler


# numpy random generator instance
rng: np.random.Generator = np.random.default_rng(2022)


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    # boolean switch for verbose messages
    is_verbose = bool(cfg.selected.verbose)

    # number of bins for computing the mixing rate
    n_bins = cfg.selected.n_bins

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

    # Samples from the known distribution
    print(f'Computing Ground-Truth samples...')
    ground_truth_samples_f, ground_truth_history = sampler.ground_truth(f, rng, cfg)
    
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
        print(f'Ground truth samples ({len(ground_truth_samples_f)})')
        for S, frequency in sorted(ground_truth_samples_f.items()):
            print(f'{S}: {frequency}')

    #########################
    #  Compute mixing rate  #
    #########################

    print(f'\nComputing mixing rates...')

    # We create n_bins for each sample, and we want to compare each bin to the ground truth's bin
    # f_S: empirical frequency of S
    # p_S: ground truth probability of S
    # \sum_{S \in P(S)} |f_S - p_S|

    gibbs_mixing_rates, \
        metropolis_mixing_rates, \
        lovasz_projection_mixing_rates = operator.attrgetter('gibbs', 'metropolis', 'lovasz_projection')(
            utils.compute_mixing_rates(f, n_bins,
                                       ground_truth_history=ground_truth_history,
                                       gibbs_history=gibbs_history,
                                       metropolis_history=metropolis_history,
                                       lovasz_projection_history=lovasz_projection_history))

    print(f'')
    print(f'Gibbs mixing rates: \n{gibbs_mixing_rates}')

    print(f'')
    print(f'Metropolis mixing rates: \n{metropolis_mixing_rates}')

    print(f'')
    print(f'Lovasz-Projection mixing rates: \n{lovasz_projection_mixing_rates}')


if __name__ == '__main__':
    run()
