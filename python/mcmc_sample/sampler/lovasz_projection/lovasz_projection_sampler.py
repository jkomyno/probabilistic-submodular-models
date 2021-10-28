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


def lovasz_projection_sampler(f: Objective, rng: np.random.Generator,
                              std: float, eta: float,
                              cfg: DictConfig) -> Tuple[Counter, List[Set[int]]]:
    """
    :param f: submodular function
    :param rng: numpy random generator instance
    :param std: standard deviation of the noise
    :param eta: step size of the subgradient projected descent
    :param cfg: Hydra configuration dictionary
    """

    # number of samples, excluding the burn-in
    M = cfg.sample_size.M

    # percentage of initial samples to discard
    burn_in_ratio = cfg.selected.burn_in_ratio

    # acceleration rate of the subgradient projected descent
    # momentum = cfg.sampler.momentum

    # elements dedicated to the burn-in
    n_burn_in = int(M * burn_in_ratio)

    print(f'Running Lovasz-Projection sampler with M={M}, burn-in ratio={burn_in_ratio}, n_burn_in={n_burn_in}, eta={eta}, std={std}')

    # run the Lovasz-Projection sampler, skipping the initial n_burn_in results
    it: Iterator[Set[int]] = itertools.islice(
        lovasz_projection_inner(f=f, rng=rng, M=M + n_burn_in, eta=eta, std=std),
        n_burn_in,
        None)

    # chronological history of Lovasz-Projection samples
    lovasz_projection_history = list(it)

    # aggregate the Lovasz-Projection samples
    lovasz_projection_samples_f = Counter((frozenset(X) for X in lovasz_projection_history))
    return lovasz_projection_samples_f, lovasz_projection_history


def lovasz_projection_inner(f: Objective, rng: np.random.Generator,
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

    is_eta_descending = eta == 'descending'
    if is_eta_descending:
        eta = 1.0

    for t in range(1, M + 1):
        _, grad_f_x = F(x)
        noise = rng.normal(loc=zero, scale=std, size=(n, ))

        if is_eta_descending:
            eta = (1 / (1 + t))
            print(f'eta: {common.float_to_str(eta)}')

        change = zero - (eta * grad_f_x) - (np.sqrt(eta) * noise)
        y = x + change

        # project y back to [0, 1]^n
        s = project_step(y)

        # round current iterate to a set
        S: Set[int] = set(np.where(q() >= s)[0])

        # p_acc is the probability of accepting the proposal sample S.
        # the conditional probabilities in the fraction cancel each other out
        p_acc = min(1, np.exp(-f.value(S)) / np.exp(-f.value(X)))

        # z is the threshold for the acceptance of the new candidate
        z = rng.uniform(low=0.0, high=1.0)

        # the iterate set could be updated based on p_acc
        if z <= p_acc:
            X = S
            x = s

        yield X
