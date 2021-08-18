from abc import ABC
from typing import AbstractSet, List


class Objective(ABC):
    def __init__(self, ground_set: List[int]):
        self._ground_set = ground_set
        self._n = len(ground_set)

    @property
    def V(self) -> List[int]:
        """
        Return the ground set
        """
        return self._ground_set

    @property
    def n(self) -> int:
        """
        Return the size of the ground set
        """
        return self._n

    def value(self, S: AbstractSet[int]) -> float:
        """
        Value oracle for the submodular problem.
        :param S: subset of the ground set
        :return: value oracle for S in the submodular problem
        """
        pass

    def marginal_gain(self, I: AbstractSet[int], S: AbstractSet[int]) -> float:
        """
        Value oracle for f(I | S) := f(S \cup I) - f(S)
        """
        return self.value(S | I) - self.value(S)
