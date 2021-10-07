import numpy as np
import itertools
from omegaconf import DictConfig
from collections import Counter
from typing import Set, Iterator, Tuple, List
from nptyping import NDArray, Float64
from objective import Objective


def metropolis_gotovos_sampler(f: Objective, rng: np.random.Generator, cfg: DictConfig) -> Tuple[Counter, List[Set[int]]]:
    """
    :param f: submodular function
    :param rng: numpy random generator instance
    :param cfg: Hydra configuration dictionary
    """

    # number of samples, excluding the burn-in
    M = cfg.selected.M

    # percentage of initial samples to discard
    burn_in_ratio = cfg.sampler['metropolis_gotovos'].burn_in_ratio

    # probability of removing an element v \in S if v is also \in X
    p_remove = cfg.sampler['metropolis_gotovos'].p_remove

    # elements dedicated to the burn-in
    n_burn_in = int(M * burn_in_ratio)

    print(f'Running Metropolis-Gotovos sampler with M={M}, burn-in ratio={burn_in_ratio}, n_burn_in={n_burn_in}')

    # run the Metropolis-Gotovos sampler, skipping the initial n_burn_in results
    it: Iterator[Set[int]] = itertools.islice(
        metropolis_gotovos_inner(f=f, rng=rng, M=M + n_burn_in, p_remove=p_remove),
        n_burn_in,
        None)

    # chronological history of Metropolis-Gotovos samples
    metropolis_gotovos_history = list(it)

    # aggregate the Metropolis-Gotovos samples
    metropolis_gotovos_samples_f = Counter((frozenset(X) for X in metropolis_gotovos_history))
    return metropolis_gotovos_samples_f, metropolis_gotovos_history


def metropolis_gotovos_inner(f: Objective, rng: np.random.Generator, M: int, p_remove: float) -> Iterator[Set[int]]:
    """
    :param f: submodular function
    :param rng: numpy random generator instance
    :param M: number of samples, excluding the burn-in
    :param p_remove: probability of removing an element v \in S if v is also \in X
    """

    # size of the ground set
    n = len(f.V)

    # mean of the uniform distribution
    mean = 0.5

    # we use the uniform distribution as the proposed distribution
    def q() -> NDArray[Float64]:
        return rng.uniform(low=0.0, high=1.0, size=(n,))

    # X initially is a random subset of V
    X: Set[int] = set(np.where(q() >= mean)[0])

    for _ in range(M):
        # We draw the next candidate S, with S ~ q(. | X).
        S = set(X)
        threshold = rng.uniform(low=0.0, high=1.0, size=(n, ))

        for i, p in zip(f.V, threshold):
            # with low probability, either add or remove some elements i from S
            if p <= p_remove:
                if i in S:
                    S.remove(i)
                else:
                    S.add(i)

        # p_acc is the probability of accepting the proposal sample S.
        # the conditional probabilities in the fraction cancel each other out
        p_acc = min(1, np.exp(-f.value(S)) / np.exp(-f.value(X)))

        # z is the threshold for the acceptance of the new candidate
        z = rng.uniform(low=0.0, high=1.0)

        if z <= p_acc:
            X = S

        yield X
