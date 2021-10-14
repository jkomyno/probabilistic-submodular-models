import numpy as np
import itertools
from omegaconf import DictConfig
from collections import Counter
from typing import Set, Iterator, Tuple, List
from nptyping import NDArray, Float64
from objective import Objective


def gibbs_sampler(f: Objective, rng: np.random.Generator, cfg: DictConfig) -> Tuple[Counter, List[Set[int]]]:
    """
    :param f: submodular function
    :param rng: numpy random generator instance
    :param cfg: Hydra configuration dictionary
    """

    # number of samples, excluding the burn-in
    M = cfg.selected.M

    # percentage of initial samples to discard
    burn_in_ratio = cfg.sampler['gibbs'].burn_in_ratio

    # elements dedicated to the burn-in
    n_burn_in = int(M * burn_in_ratio)

    print(f'Running Gibbs sampler with M={M}, burn-in ratio={burn_in_ratio}, n_burn_in={n_burn_in}')

    # run the Gibbs sampler, skipping the initial n_burn_in results
    it: Iterator[Set[int]] = itertools.islice(
        gibbs_inner(f=f, rng=rng, M=M + n_burn_in),
        n_burn_in,
        None)

    # chronological history of Gibbs samples
    gibbs_history = list(it)

    # aggregate the Gibbs samples
    gibbs_samples_f = Counter((frozenset(X) for X in gibbs_history))
    return gibbs_samples_f, gibbs_history


def gibbs_inner(f: Objective, rng: np.random.Generator, M: int) -> Iterator[Set[int]]:
    """
    :param f: submodular function
    :param rng: numpy random generator instance
    :param M: number of samples, excluding the burn-in
    """

    # size of the ground set
    n = len(f.V)

    # mean of the uniform distribution in R^n
    mean = 0.5

    # we use the uniform distribution as the proposed distribution
    def q() -> NDArray[Float64]:
        return rng.uniform(low=0.0, high=1.0, size=(n, ))

    # probabilistic marginal gain
    def delta_f(I: Set[int], S: Set[int]) -> float:
        return int(i not in S) * f.marginal_gain(I, S) + \
            int(i in S) * f.marginal_gain(I, S - I)

    # deterministically discretize the sample to an initial sampled set
    X: Set[int] = set(np.where(q() >= mean)[0])

    for _ in range(M):
        # select uniformly at random an element in the ground set
        i = rng.choice(f.V)

        # z is the threshold for the acceptance of the new candidate
        z = rng.uniform(low=0.0, high=1.0)

        # Add or remove i to the current state X: the new candidate differs
        # by exactly one element from the previous one.
        I: Set[int] = { i }
        exp_delta_f = np.exp(-delta_f(I, X))
        p_add = exp_delta_f / (1 + exp_delta_f)

        if z <= p_add:
            X = X | I
        else:
            X = X - I

        yield X
