import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from . import Traverser


def plot_cumulative_probability_distances(cumulative_probability_distances_df: pd.DataFrame,
                                          sampler_name: str,
                                          traverser: Traverser):
    ###########################################
    #  Plot cumulative probability distances  #
    ###########################################
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.7)
    fig, ax = plt.subplots(1, figsize=(16, 9))

    sns.lineplot(x='i', y='distance', data=cumulative_probability_distances_df, ax=ax)
    ax.set_title(f'Cumulative probability distance ({sampler_name}, M={traverser.M})')

    figure_name = f'{traverser.out_dir}/cumulative_probability_distances.pdf'
    plt.savefig(figure_name, dpi=300, bbox_inches='tight')
    plt.cla()
    plt.close(fig)
