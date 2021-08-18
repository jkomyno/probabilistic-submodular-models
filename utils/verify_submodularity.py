from .powerset import powerset
from objective import Objective


def verify_submodularity(f: Objective):
    for A in powerset(f.V):
        for B in powerset(f.V):
            if f.value(A) + f.value(B) < f.value(A | B) + f.value(A & B):
                print(f'Non set-submodular :(')
                print(f'A: {A};  B: {B}')
                return False

    return True
