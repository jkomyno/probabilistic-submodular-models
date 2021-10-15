import pandas as pd


def read_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, sep=',', decimal='.', encoding='utf-8',
                       index_col=None)
