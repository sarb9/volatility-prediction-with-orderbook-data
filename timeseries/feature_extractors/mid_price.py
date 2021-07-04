import pandas as pd


def extract_mid_price(df: pd.DataFrame):
    return (df[:, 1, 0, 0] + df[:, 0, 0, 0]) / 2
