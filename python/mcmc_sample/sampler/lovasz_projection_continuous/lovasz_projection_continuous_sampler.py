import numpy as np
import itertools
from omegaconf import DictConfig
from collections import Counter
from collections import Counter
from typing import Set, Iterator, Tuple, List
from nptyping import NDArray
from ...objective import Objective
from ...utils import lovasz
from .... import common


def lovasz_projection_continuous_sampler(f: Objective, rng: np.random.Generator,
                                         std: float, eta: float,
                                         cfg: DictConfig) -> Tuple[Counter, List[Set[int]]]:
    """
    :param f: submodular function
    :param rng: numpy random generator instance
    :param eta: step size of the subgradient projected descent
    :param std: standard deviation of the noise
    :param cfg: Hydra configuration dictionary
    """

    # number of samples, excluding the burn-in
    M = cfg.sample_size.M

    """
    # percentage of initial samples to discard
    burn_in_ratio = cfg.selected.burn_in_ratio

    # acceleration rate of the subgradient projected descent
    # momentum = cfg.sampler.momentum

    # elements dedicated to the burn-in
    n_burn_in = int(M * burn_in_ratio)

    print(f'Running Lovasz-Projection sampler with M={M}, burn-in ratio={burn_in_ratio}, n_burn_in={n_burn_in}, eta="descending" std={std}')

    # run the Lovasz-Projection sampler, skipping the initial n_burn_in results
    it: Iterator[Set[int]] = itertools.islice(
        lovasz_projection_continuous_inner(f=f, rng=rng, M=M + n_burn_in, eta=eta, std=std),
        n_burn_in,
        None)

    # chronological history of Lovasz-Projection samples
    lovasz_projection_continuous_history = list(it)
    """

    print(f'Running Lovasz-Projection (continuous) sampler with M={M}, no burn-in, eta={eta}, std={std}')
    lovasz_projection_continuous_history = list(lovasz_projection_continuous_inner(f=f, rng=rng, M=M, eta=eta, std=std))

    # aggregate the Lovasz-Projection samples
    lovasz_projection_continuous_samples_f = Counter((frozenset(X) for X in lovasz_projection_continuous_history))
    return lovasz_projection_continuous_samples_f, lovasz_projection_continuous_history


def lovasz_projection_continuous_inner(f: Objective, rng: np.random.Generator,
                            M: int, eta: float, std: float) -> Iterator[Set[int]]:   
    # F is the submodular convex closure of f
    F = lovasz(f)

    # convert a given set to vector representation
    set_to_vector = common.set_to_vector(f.V)

    def project(y_i: NDArray[float]) -> NDArray[float]:
        """
        Perform a Euclidean projection step
        """
        if y_i < 0:
            return 0
        elif y_i > 1:
            return 1
        else:
            return y_i

    # closure that applies the project function component-wise
    project_step = np.vectorize(project)

    # size of the ground set
    n = f.n

    # mean of the uniform distribution
    mean = 0.5

    # we use the uniform distribution as the proposed distribution
    def q() -> NDArray[float]:
        return rng.uniform(low=0.0, high=1.0, size=(n,))

    # X initially is a random subset of V
    X: Set[int] = set(np.where(q() >= mean)[0])

    # x is the vector representation of X
    x = set_to_vector(X)

    change = np.full((n, ) , fill_value=0.0)
    zero = np.full((n, ), fill_value=0.0)
    eta_sqrt = np.sqrt(eta)

    powerset = common.powerset(f.V)
    entries = { S: [] for S in powerset(True) }
    V_set = set(f.V)

    for _ in range(M):
        _, grad_f_x = F(x)
        noise = rng.normal(loc=zero, scale=std, size=(n, ))

        change = zero - (eta * grad_f_x) - (eta_sqrt * noise)
        y = x + change

        # project y back to [0, 1]^n
        x = project_step(y)

        # entries[{ 0 }]       = x[0]*(1-x[1])*(1-x[2])
        # entries[{ 1 }]       = (1-x[0])*(x[1])*(1-x[2])

        entries[frozenset()].append(1 - np.max(x))
        for S in itertools.islice(powerset(True), 1, None):
            S_idx = np.array(list(S), dtype=int)
            S_entry = np.prod(x[S_idx])
            others_entry = 1 - x[list(V_set - S)]
            entries[S].append(S_entry * np.prod(others_entry, initial=1.0))

    empirical_frequencies = { S: sum(values) / M for S, values in entries.items() }

    # transform dictionary of values to a dictionary of probabilities
    empirical_frequencies = { S: empirical_frequencies[S] / sum(empirical_frequencies.values()) for S in empirical_frequencies }

    # chronological history of ground truth samples
    Xs = rng.choice(list(empirical_frequencies.keys()), size=(M, ), replace=True,
                         p=list(empirical_frequencies.values()))

    return Xs.tolist()
