import os
from pathlib import Path
from typing import Iterator
from ... import common
from .Traverser import Traverser


def traverse_sampler(basedir: str, sampler_name: str) -> Iterator[Traverser]:
    for history_path in Path(os.path.join(basedir, 'out')).rglob(f'**/{sampler_name}/history.csv'):
        ground_truth_path = f'{history_path.parent.parent.parent}/ground_truth.csv'
        cumulative_probability_distances_path = f'{history_path.parent}/cumulative_probability_distances.csv'
        
        # create 'plots' subfolder
        out_dir = f'{history_path.parent}/plots'
        Path(out_dir).mkdir(parents=True, exist_ok=True)

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
        
        history_df = common.read_csv(history_path)
        history_df = common.add_array(history_df)

        yield Traverser(
            n=n,
            M=M,
            ground_truth_df=ground_truth_df,
            history_df=history_df,
            out_dir=out_dir,
            cumulative_probability_distances_path=cumulative_probability_distances_path,
            powerset=powerset,
            vector_to_set=vector_to_set,
        )
