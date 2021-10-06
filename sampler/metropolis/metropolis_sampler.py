import numpy as np
import itertools
from omegaconf import DictConfig
from collections import Counter
from typing import Set, Iterator, Tuple, List
from nptyping import NDArray, Float64
from objective import Objective


def metropolis_sampler(f: Objective, rng: np.random.Generator, cfg: DictConfig) -> Tuple[Counter, List[Set[int]]]:
    """
    :param f: submodular function
    :param rng: numpy random generator instance
    :param cfg: Hydra configuration dictionary
    """

    # number of samples, excluding the burn-in
    M = cfg.selected.M

    # percentage of initial samples to discard
    burn_in_ratio = cfg.sampler['metropolis'].burn_in_ratio

    # probability of removing an element v \in S if v is also \in X
    p_remove = cfg.sampler['metropolis'].p_remove

    # elements dedicated to the burn-in
    n_burn_in = int(M * burn_in_ratio)

    print(f'Running Metropolis sampler with M={M}, burn-in ratio={burn_in_ratio}, n_burn_in={n_burn_in}')

    # run the Metropolis sampler, skipping the initial n_burn_in results
    it: Iterator[Set[int]] = itertools.islice(
        metropolis_inner(f=f, rng=rng, M=M + n_burn_in, p_remove=p_remove),
        n_burn_in,
        None)

    # chronological history of Metropolis samples
    metropolis_history = list(it)

    # aggregate the Metropolis samples
    metropolis_samples_f = Counter((frozenset(X) for X in metropolis_history))
    return metropolis_samples_f, metropolis_history


def metropolis_inner(f: Objective, rng: np.random.Generator, M: int, p_remove: float) -> Iterator[Set[int]]:
    """
    :param f: submodular function
    :param rng: numpy random generator instance
    :param M: number of samples, excluding the burn-in
    :param p_remove: probability of removing an element v \in S if v is also \in X
    """

    # size of the ground set
    n = len(f.V)

    # average of the uniform distribution in R^n
    mean = np.full(n, 0.5)

    # we use the uniform distribution as the proposed distribution
    def q() -> NDArray[Float64]:
        return rng.uniform(low=0.0, high=1.0, size=(n,))

    # deterministically discretize the sample to an initial sampled set
    X: Set[int] = set(np.where(q() >= mean)[0])
    # yield X

    for _ in range(M):
        # we use the previous distribution where the mean is the previous iteration
        # draw S ~ q(. | X).
        # We sample S randomly with uniform distribution.
        # For each element v in X, add v to S if v \notin S.
        # Otherwise, if v \in S, remove v with a small probability p
        S = set(np.where(q() >= mean)[0])

        # we sample the probabilities p in batch, it's most likely faster than
        # generating them on demand
        ps = rng.uniform(low=0.0, high=1.0, size=n)

        for i, v in enumerate(X):
            if ps[i] <= p_remove:
                if v in S:
                    S.remove(v)
                else:
                    S.add(v)

        # for i in X:
        #     if i in S:
        #         S.remove(i)
        #     else:
        #         S.add(i)

        # the conditional probabilities in the fraction cancel each other out
        p_acc = min(1, np.exp(-f.value(S)) / np.exp(-f.value(X)))

        # z is the threshold for the acceptance of the new candidate
        z = rng.uniform(low=0.0, high=1.0)

        if z <= p_acc:
            X = S

        yield X
