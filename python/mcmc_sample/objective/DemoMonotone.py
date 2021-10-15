import numpy as np
from typing import AbstractSet
from .Objective import Objective
from ... import common


class DemoMonotone(Objective):
    def __init__(self, rng: np.random.Generator, n: int):
        """
        Generate a random set-modular, monotone function
        :param rng: numpy random generator instance
        :param n: size of the ground set
        """
        super().__init__(list(range(n)))

        # Decimal context to avoid floating-point issues
        # self.ctx = decimal.Context(prec=12)

        # generate n random weights
        self.w = rng.uniform(low=0.0, high=1.0, size=n)
        np.sort(self.w)

        # convert a given set to vector representation
        self.set_to_vector = common.set_to_vector(self.V)

    def value(self, S: AbstractSet[int]):
        """
        Value oracle for the demo monotone submodular problem
        """
        x = self.set_to_vector(S) @ self.w
        # return decimal.Decimal(x, context=self.ctx)
        return x
