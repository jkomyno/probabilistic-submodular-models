import numpy as np
from dataclasses import dataclass
from typing import List, Set, Tuple
from nptyping import NDArray, Float64
from objective import Objective
import utils


def affine_minimizer(S: List[NDArray[Float64]]) -> Tuple[NDArray[float], NDArray[float]]:
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

    # compute the matrix inverse of (B^T B), which has shape (m, m)
    BB_inv = np.linalg.inv(B.T @ B)

    # (m, )-dimensional unit vector
    one = np.ones((m,))

    # alpha is a (m, )-dimensional vector
    alpha = BB_inv @ one / (one.T @ BB_inv @ one)

    # y is a (n, )-dimensional vector
    y = B @ alpha

    return y, alpha


def round_fujishige_wolfe(x: NDArray[Float64]) -> Set[int]:
    """
    Convert a Fujishige-Wolfe continuous solution into a discrete set.
    :param x: n-dimensional vector
    """
    return set(np.where(x < 0)[0])


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


def fujishige_wolfe(f: Objective, eps: float = None) -> Set[int]:
    """
    Fujishige-Wolfe algorithm, as presented in [1].

    References:
    - [1]: https://arxiv.org/pdf/1411.0095.pdf

    :params f: polymatroid set-submodular function
    :params eps: error threshold
    """

    if eps is None:
        # set default value for eps
        eps = 1 / (4 * f.n)

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

    while True:
        # increase number of outer loop iterations
        fw_stats.outer_it += 1

        # q is an arbitrary vertex of B_f
        q = LO()

        threshold = x.T @ q + (eps ** 2)

        if np.linalg.norm(x) <= threshold:
            # termination condition
            print(f'np.linalg.norm(x): {np.linalg.norm(x)}')
            print(f'threshold norm:    {threshold}')
            break

        S.append(q)
        y = np.zeros((f.n,))

        while True:
            # increase number of inner loop iterations
            fw_stats.inner_it += 1

            y, alphas = affine_minimizer(S)

            if np.min(alphas) > 0:
                # y is in conv(S), end minor loop
                break
            else:
                # increase number of times the else branch was hit
                fw_stats.else_it += 1

            # y is not in conv(S)
            theta = min((lambdas[i] / (lambdas[i] - alphas[i]) for i in f.V if alphas[i] < 0))

            # x now lies in conv(S)
            x = theta * y + (1 - theta) * x

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

        # update the solution vector with y \in conv(S)
        x = y

    # S_star is the global minimum
    S_star: Set[int] = round_fujishige_wolfe(x)

    return S_star, fw_stats
