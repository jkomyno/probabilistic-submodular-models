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


def frank_wolfe_sampler(f: Objective, rng: np.random.Generator,
                        cfg: DictConfig) -> Tuple[Counter, List[Set[int]]]:
    """
    :param f: submodular function
    :param rng: numpy random generator instance
    :param cfg: Hydra configuration dictionary
    """

    # number of samples, excluding the burn-in
    M = cfg.selected.M

    # percentage of initial samples to discard
    burn_in_ratio = cfg.sampler['frank_wolfe'].burn_in_ratio

    # elements dedicated to the burn-in
    n_burn_in = int(M * burn_in_ratio)

    print(f'Running Frank-Wolfe sampler with M={M}, burn-in ratio={burn_in_ratio}, n_burn_in={n_burn_in}')

    # run the Frank-Wolfe sampler, skipping the initial n_burn_in results
    it: Iterator[Set[int]] = itertools.islice(
        frank_wolfe_inner(f=f, rng=rng, M=M + n_burn_in),
        n_burn_in,
        None)

    # chronological history of Frank-Wolfe-Metropolis samples
    frank_wolfe_history = list(it)

    # aggregate the Gibbs samples
    frank_wolfe_samples_f = Counter((frozenset(X) for X in frank_wolfe_history))
    return frank_wolfe_samples_f, frank_wolfe_history


def argmin_s_in_base_polytope(f: Objective):
    # y variables (to be found with optimization)
    y = cvx.Variable(shape=(f.n, ))

    # gradient parameter, to be updated
    grad = cvx.Parameter(shape=(f.n, ))

    # objective function
    objective = cvx.Minimize(y.T @ grad)

    # constraints
    equality_constraints = [sum(y) == f(f.V)]
    inequality_constraints = [
        y.T @ utils.set_to_vector(f, A) <= f(A) for A in itertools.islice(utils.powerset(f.V), 1, None)
    ]
    constraints = [*equality_constraints,
                  *inequality_constraints]

    # use cvxpy to solve the objective
    problem = cvx.Problem(objective, constraints)

    # number of runs of the same model
    runs = 0

    def helper(grad_f_x: NDArray[float]) -> NDArray[float]:
        """
        :param grad_f_x: the gradient of the convex function F w.r.t. the vector x.
        """
        nonlocal runs

        grad.value = grad_f_x

        problem.solve(verbose=True, warm_start=runs > 0)
        runs += 1
        print(f'({runs}) \t solve time: {problem.solver_stats.solve_time}')

        # retrieve the value of y
        y_values = y.value
        return y_values
  
    return helper


def frank_wolfe_inner(f: Objective, rng: np.random.Generator,
                      M: int, std=1.0) -> Iterator[Set[int]]:
    # F is the submodular convex closure of f
    F = utils.lovasz(f)

    # initialize linear program inside the base polytope
    # with warm start after the first run
    argmin_lp = argmin_s_in_base_polytope(f)

    # initialize x
    x = np.full((f.n, ), fill_value=1.0)

    for t in range(M):
        _, grad_f_x = F(x)
        noise = rng.normal(loc=0.0, scale=std)

        # compute vector s such that <s, grad_f_x + noise> is minimized
        s = argmin_lp(grad_f_x=grad_f_x + noise)

        # fixed step size
        gamma = 2 / (t + 2)

        # update iterate
        x = (1 - gamma) * x + gamma * s

        # round iterate to a set
        threshold = rng.uniform(low=0.0, high=1.0, size=(f.n, ))
        S = set(np.where(x > threshold)[0])
        yield S
        
        # round x to match S
        x = utils.set_to_vector(f, S)
