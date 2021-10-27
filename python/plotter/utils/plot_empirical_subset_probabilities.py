import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, FrozenSet, List
from . import create_dfs_for_subsets
from . import Traverser


def plot_empirical_subset_probabilities(empirical_cumulative_probabilities: List[Dict[FrozenSet[int], float]],
                                        sampler_name: str,
                                        step: int,
                                        traverser: Traverser):
    ################################################
    #  Plot empirical probability vs ground truth  #
    #  probability of subsets of V                 #
    ################################################
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.7)
    fig, ax = plt.subplots(1, figsize=(16, 9))

    for data, S in create_dfs_for_subsets(traverser.M,
                                          traverser.ground_truth_df,
                                          empirical_cumulative_probabilities,
                                          sampler_name=sampler_name,
                                          powerset=traverser.powerset,
                                          step=step):
        palette = sns.color_palette('Set2', len(data['sampler'].value_counts()))
        sns.lineplot(x='step', y='probability', hue='sampler', data=data,
                      ax=ax, palette=palette)
        ax.set_title(f'S={S}')

        # { 0, 1, 2 } -> '0_1_2'
        set_name = '_'.join(map(str, S))
        figure_name = f'{traverser.out_dir}/vs_ground_truth-{set_name}.pdf'
        plt.savefig(figure_name, dpi=300, bbox_inches='tight')
        plt.cla()

    plt.close(fig)
