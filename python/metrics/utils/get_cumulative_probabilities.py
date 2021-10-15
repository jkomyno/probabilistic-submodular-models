import pandas as pd
from typing import List, AbstractSet, FrozenSet, Dict, Callable, Iterator
from collections import Counter
from . import get_probability_dict


def split_in_cumulative_lists(history: List[AbstractSet[int]],
                              step: int) -> List[List[AbstractSet[int]]]:
    return (
        list(history[0:i + step])
        for i in range(0, len(history), step)
    )


def get_cumulative_probabilities(history_df: pd.DataFrame,
                                 powerset: Callable[[bool], Iterator[AbstractSet[int]]],
                                 vector_to_set: Callable[[], FrozenSet[int]],
                                 step: int = 25) -> List[Dict[FrozenSet[int], float]]:
    M = len(history_df)
    cumulative = list(
        split_in_cumulative_lists(history_df['array'].map(vector_to_set), step=step)
    )
    counters = list(map(Counter, cumulative))
    probabilities = list(
        map(lambda counter: get_probability_dict(counter, powerset), counters)
    )
    
    return probabilities
