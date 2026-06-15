from enum import StrEnum

import datasets
from torch.utils.data import Dataset as TorchDataset


class DatasetSplit(StrEnum):
    TEST = "test"
    TRAIN = "train"
    VALIDATION = "validation"


class WikitextDataset(TorchDataset):
    def __init__(self, split: DatasetSplit):
        self._data = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        self._split = split

    def __getitem__(self, i: int) -> str:
        return self._data[str(self._split)][i]["text"]

    def __len__(self) -> int:
        return len(self._data[str(self._split)])
