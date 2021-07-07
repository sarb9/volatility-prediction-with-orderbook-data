import numpy as np
import pandas as pd

from timeseries.features.feature import register_feature


@register_feature("mid_price", input_functions=["orderbook"])
def extract_mid_price(orderbook: np.ndarray) -> np.ndarray:
    return (orderbook[:, 1, 0, 0] + orderbook[:, 0, 0, 0]) / 2


@register_feature("price_change", input_functions=["mid_price"])
def extract_price_change(mid_prices: np.ndarray) -> np.ndarray:
    price_change: pd.Series = pd.Series(mid_prices).pct_change()
    price_change[0] = price_change[1]

    return price_change.to_numpy()


@register_feature("volatility", input_functions=["price_change"])
def extract_volatility(
    price_change: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    period = 60
    number_of_samples = len(price_change)

    volatilities = np.zeros((number_of_samples,))

    for i in range(number_of_samples):
        start = max(0, i - window)
        volatilities[i] = np.var(price_change[start:i])
    volatilities[0] = volatilities[1]

    return volatilities
