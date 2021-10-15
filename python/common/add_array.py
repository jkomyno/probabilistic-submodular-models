import numpy as np
import pandas as pd


def x_str_to_array(x_str: str):
    return np.fromiter(map(int, x_str[1:-1].split()), dtype=int)


def add_array(df: pd.DataFrame) -> pd.DataFrame:
    df['array'] = df['x'].map(x_str_to_array)
    return df
