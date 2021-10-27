import pandas as pd
from dataclasses import dataclass
from typing import Iterator, Callable, AbstractSet, FrozenSet
from nptyping import NDArray
 

@dataclass
class Traverser:
    n: int
    M: int
    ground_truth_df: pd.DataFrame
    history_df: pd.DataFrame
    out_dir: str
    cumulative_probability_distances_path: str
    powerset: Iterator[AbstractSet[int]]
    vector_to_set: Callable[[NDArray[int]], FrozenSet[int]]
