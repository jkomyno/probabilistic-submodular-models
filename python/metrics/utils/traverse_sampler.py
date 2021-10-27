import os
from pathlib import Path
from typing import Iterator, Dict, FrozenSet
from ... import common
from .Traverser import Traverser


def traverse_sampler(basedir: str, sampler_name: str) -> Iterator[Traverser]:
    for history_path in Path(os.path.join(basedir, 'out')).rglob(f'**/{sampler_name}/history.csv'):
        ground_truth_path = f'{history_path.parent.parent.parent}/ground_truth.csv'
        
        f_name, n_str, M_str, sampler_name = history_path.parent.parts[-4:]
        n = int(n_str.split('-')[1])
        M = int(M_str.split('-')[1])

        # ground set
        V = list(range(n))

        # instantiate closures that use the ground set
        powerset = common.powerset(V)
        vector_to_set = common.vector_to_set(V)

        print(f'f_name: {f_name}, sampler_name: {sampler_name}')

        ground_truth_df = common.read_csv(ground_truth_path)
        ground_truth_df = common.add_array(ground_truth_df)

        ground_truth_density_map: Dict[FrozenSet[int], float] = {
            S: ground_truth_df['probability'][i] for i, S in enumerate(ground_truth_df['array'].map(vector_to_set))
        }
        
        history_df = common.read_csv(history_path)
        history_df = common.add_array(history_df)

        out_dir = history_path.parent

        yield Traverser(
            n=n,
            M=M,
            ground_truth_df=ground_truth_df,
            history_df=history_df,
            out_dir=out_dir,
            ground_truth_density_map=ground_truth_density_map,
            powerset=powerset,
            vector_to_set=vector_to_set,
        )
