import os
import hydra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from pathlib import Path
from typing import Dict, FrozenSet, AbstractSet, Callable, Iterator, Tuple, List
from .. import common


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


@hydra.main(config_path="../conf", config_name="config")
def plotter(cfg: DictConfig) -> None:
    # boolean switch for verbose messages
    is_verbose = bool(cfg.selected.verbose)

    # basedir w.r.t. main.py
    basedir = os.path.join(hydra.utils.get_original_cwd(), Path(__file__).parent.parent.parent)
    
    for history_path in Path(os.path.join(basedir, 'out')).rglob('**/history.csv'):
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

        cumulative_probability_distances_df = common.read_csv(cumulative_probability_distances_path)

        history_df = common.read_csv(history_path)
        history_df = common.add_array(history_df)

        empirical_cumulative_probabilities = common.get_cumulative_probabilities(history_df,
                                                                                 powerset=powerset,
                                                                                 vector_to_set=vector_to_set,
                                                                                 step=50)
        
        ################################################
        #  Plot empirical probability vs ground truth  #
        #  probability of subsets of V                 #
        ################################################

        sns.set_style('white')
        sns.set_context('paper', font_scale=1.7)
        fig, ax = plt.subplots(1, figsize=(16, 9))

        for data, S in create_dfs_for_subsets(M, ground_truth_df, empirical_cumulative_probabilities,
                                              sampler_name=sampler_name, powerset=powerset, step=50):
            palette = sns.color_palette('Set2', len(data['sampler'].value_counts()))
            sns.lineplot(x='step', y='probability', hue='sampler', data=data,
                         ax=ax, palette=palette)
            ax.set_title(f'S={S}')

            # { 0, 1, 2 } -> '0_1_2'
            set_name = '_'.join(map(str, S))
            figure_name = f'{out_dir}/vs_ground_truth-{set_name}.pdf'
            plt.savefig(figure_name, dpi=300, bbox_inches='tight')
            plt.cla()

        plt.close(fig)

        ###########################################
        #  Plot cumulative probability distances  #
        ###########################################

        sns.set_style('white')
        sns.set_context('paper', font_scale=1.7)
        fig, ax = plt.subplots(1, figsize=(16, 9))

        sns.lineplot(x='i', y='distance', data=cumulative_probability_distances_df, ax=ax)
        ax.set_title(f'Cumulative probability distance ({sampler_name}, M={M})')

        figure_name = f'{out_dir}/cumulative_probability_distances.pdf'
        plt.savefig(figure_name, dpi=300, bbox_inches='tight')
        plt.cla()
