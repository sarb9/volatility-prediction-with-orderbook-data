from pathlib import Path
import numpy as np

from data_service.file_utils.raw_dataset import load_from_file

KEYWORD_SAVE_DIRECTORY = "directory"
KEYWORD_SAVE_FILE_NAME = "file_name"


class OrderbookDataset:
    def __init__(
        self,
    ) -> None:
        self.books = np.zeros((1, 2, 2, 1), dtype=np.float32)
        self._len = 0

    def __len__(self):
        return self._len

    def load_from_file(
        self,
        file: Path,
        save: None = None,
        load: None = None,
    ):
        if load:
            self.books = np.load(
                load[KEYWORD_SAVE_DIRECTORY] + load[KEYWORD_SAVE_FILE_NAME]
            )
        elif save:
            self.books, self._len = load_from_file(file)
            np.save(
                save[KEYWORD_SAVE_DIRECTORY] + save[KEYWORD_SAVE_FILE_NAME], self.books
            )
