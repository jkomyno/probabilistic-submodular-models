import pandas as pd
from typing import Dict, FrozenSet, AbstractSet, Callable, Iterator, Tuple, List


def create_dfs_for_subsets(M: int,
                           ground_truth_df: pd.DataFrame,
                           empirical_cumulative_probabilities: List[Dict[FrozenSet[int], float]],
                           sampler_name: str,
                           powerset: Callable[[bool], Iterator[AbstractSet[int]]],
                           step: int) -> Iterator[Tuple[pd.DataFrame, FrozenSet[int]]]:
    for i, S in enumerate(powerset(as_frozenset=False)):
        FS = frozenset(S)
        df = pd.DataFrame({
            'step': list(range(step, M + 1, step)),
            'ground_truth': [ground_truth_df['probability'][i] for _ in range(M // step)],
            sampler_name: [lst[FS] for lst in empirical_cumulative_probabilities],
        })
        data = pd.melt(df, ['step'], var_name='sampler', value_name='probability')
        yield (data, S)
