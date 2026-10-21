import random
from enum import StrEnum

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, Dataset

from gpt2.models.gpt2_model import GPT2Tokenizer


class DatasetSplit(StrEnum):
    TEST = "test"
    TRAIN = "train"
    VALIDATION = "validation"


class WikitextDataset(IterableDataset):
    @staticmethod
    def _flatten_dataset(dataset: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        concatted = torch.cat(dataset["text"]).reshape((1, -1))
        return {"tokens": concatted}

    @staticmethod
    def _tokenize_example(
        example: dict[str, str], tokenizer: GPT2Tokenizer
    ) -> dict[str, torch.Tensor]:
        example["text"] = tokenizer.encode(example["text"])
        return example

    def __init__(self, split: DatasetSplit, tokenizer: GPT2Tokenizer):
        self._split = split

        self._data = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split=split
        ).with_format("torch")
        self._data = self._data.map(lambda ds: self._tokenize_example(ds, tokenizer))
        self._data = self._data.map(
            self._flatten_dataset, batched=True, batch_size=None, remove_columns="text"
        )

        self._token_tensor = self._data["tokens"][0]

    def __iter__(self):
        while True:
            offset = random.randint(0, self._token_tensor.shape[0] - 256)
            yield self._token_tensor[offset : offset + 256]


class OpusDataset(Dataset):
    @staticmethod
    def _format_pair(record: dict) -> dict:
        """
        Format each english-spanish pair as <en>text</en><es>text</es>.
        """
        en = record["translation"]["en"]
        es = record["translation"]["es"]
        record["translation"] = f"<en>{en}</en><es>{es}</es>"
        return record

    @staticmethod
    def _create_minibatches(examples: dict, seq_len: int) -> dict:
        concatenated = torch.concat(examples["tokenized"])
        max_upper = seq_len * (concatenated.shape[-1] // seq_len)
        # TODO: The GPT-2 paper actually randomly samples 512-token sequences from the dataset. Replicate that here.
        tokens = torch.reshape(concatenated[:max_upper], (-1, seq_len))
        return {"tokens": tokens}

    @staticmethod
    def _tokenize(tokenizer: GPT2Tokenizer, record: dict) -> dict:
        eos = torch.tensor([tokenizer.end_of_text_token()])
        record["tokenized"] = torch.concat(
            (eos, tokenizer.encode(record["translation"]), eos)
        )
        return record

    def __init__(self, lang: str, split: str, tokenizer: GPT2Tokenizer) -> None:
        self._dataset = load_dataset(
            "Helsinki-NLP/opus-100", lang, split=split
        ).with_format("torch")

        # flatten pairs into strings, tokenize, then reshape into minibatches
        self._dataset = self._dataset.map(self._format_pair, desc="format")
        self._dataset = self._dataset.map(
            lambda example: self._tokenize(tokenizer, example), desc="tokenize"
        )
        self._dataset = self._dataset.map(
            lambda examples: self._create_minibatches(examples, 256),
            batched=True,
            batch_size=None,
            remove_columns=["tokenized", "translation"],
            desc="chunk",
        )

    def __getitem__(self, i: int) -> str:
        return self._dataset[i]

    def __len__(self) -> int:
        return len(self._dataset)
