import numpy as np
from .Objective import Objective


class Delta5(Objective):
    """
    Monotone Submodular, non-negative
    """
    def __init__(self, n, alpha=0.5):
        super().__init__(list(range(n)))
        self.alpha = alpha

    def value(self, S):
        "Value oracle for the Delta5 submodular loss function"
        return 1 - np.exp(-self.alpha * len(S))
