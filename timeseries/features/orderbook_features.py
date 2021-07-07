from typing import Dict, Tuple
from pathlib import Path

from data_service.orderbook_dataset import OrderbookDataset
from timeseries.features.feature import register_feature


@register_feature("orderbook")
def read_orderbook_from_file(dataset_file: Dict = {}):
    path: Path = Path(dataset_file["directory"]) / dataset_file["file_name"]
    dataset: OrderbookDataset = OrderbookDataset()
    dataset.load_from_file(path)

    return dataset.books


@register_feature("spread", input_functions=["orderbook", "mid_price"])
def extract_spread(orderbook, mid_price):
    return abs(orderbook[:, 1, 0, 0] - orderbook[:, 0, 0, 0]) / mid_price


@register_feature("bid_volume", input_functions=["orderbook"])
def extract_bid_volume(orderbook):
    return orderbook[:, 0, 1, :].sum(axis=1)


@register_feature("ask_volume", input_functions=["orderbook"])
def extract_ask_volume(orderbook):
    return orderbook[:, 1, 1, :].sum(axis=1)
