import numpy as np
from typing import AbstractSet
from .Objective import Objective
from ... import common


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

        # convert a given set to vector representation
        self.set_to_vector = common.set_to_vector(self.V)

    def value(self, S: AbstractSet[int]) -> int:
        """
        Value oracle for the revenue maximization problem
        """
        return self.set_to_vector(S) @ self.w
