import numpy as np
import itertools
from omegaconf import DictConfig
from collections import Counter
from objective import Objective
from collections import Counter
from typing import Set, Iterator, Tuple, List
from dataclasses import dataclass
from nptyping import NDArray, Float64
import utils


def affine_minimizer(S: List[NDArray[Float64]],
                     f: Objective,
                     rng: np.random.Generator) -> Tuple[NDArray[float], NDArray[float]]:
    """
    Precondition: any vector in S should be linearly independent
    """
    m = len(S)

    # B is a (n, m) matrix where each column is a point in S
    B = np.array(S).T

    # the column vectors in B might not be linearly independent.
    # We can make sure to turn them by applying a Gram-Schmidt decomposition.
    # TODO: turn S into a Set of n-dimensional vectors
    # _, R = np.linalg.qr(B.T)
    # B = B[:, np.abs(np.diag(R)) >= 1e-12]

    # alpha is a (m, )-dimensional vector
    alpha = rng.uniform(low=0.0, high=1.0, size=(m, ))
    np.sort(alpha)

    # y is a (n, )-dimensional vector
    y = B @ alpha

    return y, alpha


@dataclass
class FWStats:
    # number of outer loop iterations
    outer_it: int = 0

    # number of inner loop iterations
    inner_it: int = 0

    # number of times the else branch was hit
    else_it: int = 0

    # maximum length reached by S
    max_len_S: int = 0


def round_fujishige_wolfe(x: NDArray[Float64]) -> Set[int]:
    """
    Convert a Fujishige-Wolfe continuous solution into a discrete set.
    :param x: n-dimensional vector
    """
    return set(np.where(x < 0)[0])


def fujishige_wolfe_sampler(f: Objective, rng: np.random.Generator,
                            cfg: DictConfig) -> Tuple[Counter, List[Set[int]]]:
    """
    :param f: submodular function
    :param rng: numpy random generator instance
    :param cfg: Hydra configuration dictionary
    """

    # number of samples, excluding the burn-in
    M = cfg.selected.M

    # percentage of initial samples to discard
    burn_in_ratio = cfg.sampler['fujishige_wolfe'].burn_in_ratio

    # elements dedicated to the burn-in
    n_burn_in = int(M * burn_in_ratio)

    print(f'Running Fujishige-Wolfe sampler with M={M}, burn-in ratio={burn_in_ratio}, n_burn_in={n_burn_in}')

    # run the Fujishige-Wolfe sampler, skipping the initial n_burn_in results
    it: Iterator[Set[int]] = itertools.islice(
        fujishige_wolfe_inner(f=f, rng=rng, M=M + n_burn_in),
        n_burn_in,
        None)

    # chronological history of Fujishige-Wolfe-Metropolis samples
    fujishige_wolfe_history = list(it)

    # aggregate the Gibbs samples
    fujishige_wolfe_samples_f = Counter((frozenset(X) for X in fujishige_wolfe_history))
    return fujishige_wolfe_samples_f, fujishige_wolfe_history


def fujishige_wolfe_inner(f: Objective, rng: np.random.Generator,
                          M: int, eps: float = 0) -> Iterator[Set[int]]:
    """
    :param f: polymatroid set-submodular function
    :param rng: numpy random generator instance
    :param M: number of samples, excluding the burn-in
    :params eps: error threshold
    Fujishige-Wolfe algorithm, as presented in [1].

    References:
    - [1]: https://arxiv.org/pdf/1411.0095.pdf
    """
    def LO() -> NDArray[float]:
        """
        Linear optimization oracle.
        :return: q = argmin_{p \in \mathcal{B}_f} x^t p
        """
        x = np.zeros((f.n,), dtype=np.float64)
        return utils.polyhedron_greedy(f, x)

    # q is an arbitrary vertex of B_f
    q = LO()
    x = np.copy(q)
    S: List[NDArray[Float64]] = [q]
    lambdas: List[float] = [1.0]

    ###########################
    #  Initialize statistics  #
    ###########################

    fw_stats = FWStats()

    for t in range(M):
        # increase number of outer loop iterations
        fw_stats.outer_it += 1

        # if fw_stats.outer_it % 100 == 0:
        #     print(f'\n(outer) Fujishige-Wolfe stats:\n{fw_stats}')

        # q is an arbitrary vertex of B_f
        q = LO()

        S.append(q)
        y = np.zeros((f.n,))

        while True:
            # increase number of inner loop iterations
            fw_stats.inner_it += 1

            # if fw_stats.inner_it % 100 == 0:
            #     print(f'\n(inner) Fujishige-Wolfe stats:\n{fw_stats}')

            y, alphas = affine_minimizer(S, f=f, rng=rng)

            if np.min(alphas) > 0:
                # y is in conv(S), end minor loop
                # print(f'breaking inner loop')
                break
            else:
                # increase number of times the else branch was hit
                fw_stats.else_it += 1

            # y is not in conv(S)
            theta = min((lambdas[i] / (lambdas[i] - alphas[i]) for i in f.V if alphas[i] < 0))

            # x now lies in conv(S)
            x = theta * y + (1 - theta) * x
            yield round_fujishige_wolfe(x)

            # set the coefficients of the new x
            lambdas = [
                theta * alpha + (1 - theta) * lambda_
                for lambda_, alpha in zip(lambdas, alphas)
            ]

            # delete the points i where lambdas[i] is 0.
            # At least one point is removed
            prev_len = len(S)

            S = list((i for i, lambda_ in enumerate(lambdas) if lambda_ > 0))

            curr_len = len(S)
            assert curr_len < prev_len

            # improve maximum length reached by S
            fw_stats.max_len_S = max([fw_stats.max_len_S, prev_len, curr_len])

        # this is executed only if the inner loop did break
        # print('this is executed only if the inner loop did break')
        # print(f'update x=y')
        x = y
        yield round_fujishige_wolfe(x)

    print(f'\nFujishige-Wolfe stats:\n{fw_stats}')
