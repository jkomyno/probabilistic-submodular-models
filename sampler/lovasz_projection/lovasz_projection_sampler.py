import numpy as np
import itertools
import cvxpy as cvx
from omegaconf import DictConfig
from collections import Counter
from objective import Objective
from collections import Counter
from typing import Set, Iterator, Tuple, List
from nptyping import NDArray
import utils


def lovasz_projection_sampler(f: Objective, rng: np.random.Generator,
                        cfg: DictConfig) -> Tuple[Counter, List[Set[int]]]:
    """
    :param f: submodular function
    :param rng: numpy random generator instance
    :param cfg: Hydra configuration dictionary
    """

    # number of samples, excluding the burn-in
    M = cfg.selected.M

    # percentage of initial samples to discard
    burn_in_ratio = cfg.sampler['lovasz_projection'].burn_in_ratio

    # acceleration rate of the subgradient projected descent
    eta = cfg.sampler['lovasz_projection'].eta

    # standard deviation of the normal noise
    std = cfg.sampler['lovasz_projection'].std

    # elements dedicated to the burn-in
    n_burn_in = int(M * burn_in_ratio)

    print(f'Running Lovasz-Projection sampler with M={M}, burn-in ratio={burn_in_ratio}, n_burn_in={n_burn_in}, eta={eta}, std={std}')

    # run the Lovasz-Projection sampler, skipping the initial n_burn_in results
    it: Iterator[Set[int]] = itertools.islice(
        lovasz_projection_inner(f=f, rng=rng, M=M + n_burn_in, eta=eta, std=std),
        n_burn_in,
        None)

    # chronological history of Lovasz-Projection-Metropolis samples
    lovasz_projection_history = list(it)

    # aggregate the Gibbs samples
    lovasz_projection_samples_f = Counter((frozenset(X) for X in lovasz_projection_history))
    return lovasz_projection_samples_f, lovasz_projection_history


def lovasz_projection_inner(f: Objective, rng: np.random.Generator,
                            M: int, eta=0.01, std=1.0) -> Iterator[Set[int]]:
    # F is the submodular convex closure of f
    F = utils.lovasz(f)

    # initialize x
    x = np.full((f.n, ), fill_value=1.0)

    # normal noise scale
    eta_sqrt = np.sqrt(eta)

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

    for t in range(M):
        _, grad_f_x = F(x)
        noise = rng.normal(loc=0.0, scale=std)

        # y = x - eta * grad_f_x
        y = x - (eta * grad_f_x + eta_sqrt * noise)

        # project y back to [0, 1]^n
        x = project_step(y)

        # round iterate to a set
        threshold = rng.uniform(low=0.0, high=1.0, size=(f.n, ))
        S = set(np.where(x > threshold)[0])
        yield S
        
        # round x to match S
        x = utils.set_to_vector(f, S)
