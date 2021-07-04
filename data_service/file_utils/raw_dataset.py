from pathlib import Path
from typing import Tuple, List
import json

import numpy as np


def load_from_file(file_path: Path) -> int:
    def transform(orders) -> Tuple[List[float], List[float]]:
        prices = [order["price"] for order in orders]
        quantities = [order["quantity"] for order in orders]
        return prices, quantities

    length = 0
    books = []

    with open(file_path) as data:
        for line in data:
            row = json.loads(line)

            bids = transform(row["bids"])
            asks = transform(row["asks"])
            snapshot = [bids, asks]

            try:
                books.append(snapshot)
            except ValueError as e:
                print("Error reading order-book number")
                books.append(books[-1])

            length += 1

    books = np.array(books, dtype=np.float32)

    return books, length
