from typing import Tuple
import numpy as np

from data_service.orderbook_dataset import OrderbookDataset
from timeseries.feature_extractors.features import register_feature

# TODO: cannot input extra argument to specify input file
@register_feature("orderbook")
def read_orderbook_from_file():
    dataset: OrderbookDataset = OrderbookDataset()
    dataset.load_from_file()

    return dataset.books


@register_feature("spread", inputs=["orderbook", "mid_price"])
def extract_spread(orderbook, mid_price):
    return abs(orderbook[:, 1, 0, 0] - orderbook[:, 0, 0, 0]) / mid_price


@register_feature("bid_volume", inputs=["orderbook"])
def extract_bid_volume(orderbook):
    return orderbook[:, 0, 1, :].sum(axis=1)


@register_feature("sell_volume", inputs=["orderbook"])
def extract_sell_volume(orderbook):
    return orderbook[:, 1, 1, :].sum(axis=1)
