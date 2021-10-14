import numpy as np
from typing import AbstractSet
from .Objective import Objective
from ..utils import set_to_vector


class DemoNonMonotone(Objective):
    def __init__(self, rng: np.random.Generator, n: int):
        """
        Generate a random set-modular, non-monotone function
        :param rng: numpy random generator instance
        :param n: size of the ground set
        """
        super().__init__(list(range(n)))

        # generate n random weights
        self.w = rng.integers(low=-10, high=10, size=n)

    def value(self, S: AbstractSet[int]) -> int:
        """
        Value oracle for the revenue maximization problem
        """
        return set_to_vector(self, list(S)) @ self.w
