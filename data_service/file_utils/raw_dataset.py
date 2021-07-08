from pathlib import Path
from typing import Tuple, List
import json

import numpy as np
from tqdm import tqdm

# TODO: infer this quantity from file
DEPTH = 100


def load_from_file(file_path: Path) -> int:
    def transform(orders) -> Tuple[List[float], List[float]]:
        prices = [order["price"] for order in orders]
        quantities = [order["quantity"] for order in orders]
        return prices, quantities

    length = 0
    num_lines: int = sum(1 for _ in open(file_path))
    books: np.ndarray = np.zeros((num_lines, 2, 2, DEPTH), dtype=np.float32)

    with open(file_path) as data:
        for i, line in enumerate(tqdm(data)):
            row = json.loads(line)

            bids = transform(row["bids"])
            asks = transform(row["asks"])

            try:
                books[i, 0, 0, :] = bids[0]
                books[i, 0, 1, :] = bids[1]
                books[i, 1, 0, :] = asks[0]
                books[i, 1, 1, :] = asks[1]
            except ValueError as e:
                print("Error reading order-book number")
                books[i, ...] = books[i - 1, ...]

            length += 1

    books = np.array(books, dtype=np.float32)

    return books, length
