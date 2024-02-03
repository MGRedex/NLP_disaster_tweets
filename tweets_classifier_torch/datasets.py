import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from functools import partial
from typing import Callable, List, Tuple, Union  

class Tweets(Dataset):
    """Dataset created from .parquet file that consists of data from:
    - https://www.kaggle.com/datasets/szelee/disasters-on-social-media
    - https://www.kaggle.com/competitions/nlp-getting-started/data

    All transformations done in __getitem__ when get one data instance.
    """

    def __init__(
            self,
            file: str,
            vectorize: Callable[..., List[int]] = None,
            transform = None,
            concat_cols: List[str] = None
        ):
        """
        Loads data from parquet file.

        Args:
            file: .parquet file path.
            vectorize: Callable that returns vector for a sentence.
            transform: Sentence pytorch transfomations.
            concat_cols: List of columns ("keyword","location","text") to concatenate.
        """
        columns = ["keyword", "location", "text", "target"] 
        self.data = pd.read_parquet(file, columns = columns)
        self.classes = list(self.data["target"].unique())
        self.vectorize = vectorize
        self.transform = transform
        self.concat_cols = concat_cols

    def __getitem__(
            self,
            index: int
        ) -> Union[Tuple[torch.Tensor, int], Tuple[List[int], int], Tuple[List[str], int]]:
        """Gets one data instance. Sentence vectorization occurs before transfomation.

        Args:
            index: Index of data instance.

        Returns:
           Data instance as (sentence, target) tuple,
           where sentence can be token indexes vector(whether list or tensor) or tokens(list).
        """
        if self.concat_cols:
            sentence, target = self.data[self.concat_cols].agg(''.join, axis = 1).iloc[index], self.data["target"].iloc[index]
        else:
            sentence, target = self.data[["text", "target"]].iloc[index]
        if self.vectorize:
            sentence = self.vectorize(sentence)
        if self.transform:
            sentence = self.transform(sentence)

        return sentence, target

    def __len__(self):
        return len(self.data)

class TweetsV2(Dataset):
    """Dataset created from .parquet file that consists of data from:
    - https://www.kaggle.com/datasets/szelee/disasters-on-social-media
    - https://www.kaggle.com/competitions/nlp-getting-started/data
    """

    def __init__(
            self,
            file: str,
            transform = None,
            target_transform = None,
            concat_cols: bool = False,
            vectorize: bool = False,
            max_vector_length = 256
        ):
        """
        Loads data from parquet file.
        Columns concatenation and vectorization occurs for a whole dataset.

        Args:
            file: .parquet file path.
            transform: Sentence pytorch transfomations.
            target_transform: Target pytorch transfomations.
            concat_cols: Whether concatenate columns ("keyword","location","text") or not.
            vectorize: Whether vectorize sentence or not.
            max_vector_length: Length of output vector if vectorize = True.
        """
        columns = ["keyword", "location", "text", "target"]
        data = pd.read_parquet(file, columns = columns)

        if concat_cols:
            data = pd.DataFrame({"text": data[["keyword","location","text"]].agg(''.join, axis = 1), "target": data["target"]})

        if vectorize:
            pretrained_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            text_vect = partial(
                pretrained_tokenizer,
                max_length = max_vector_length,
                truncation = True,
                padding = "max_length",
            )
            data["text"] = data["text"].apply(lambda x: text_vect(x)["input_ids"])

        self.classes = list(data["target"].unique())
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(
            self,
            index: int
        ) -> Union[Tuple[torch.Tensor, int], Tuple[List[int], int], Tuple[List[str], int]]:
        """Gets one data instance.

        Args:
            index: Index of data instance.

        Returns:
           Data instance as (sentence, target) tuple,
           where sentence can be token indexes vector(whether list or tensor) or tokens(list).
        """
        sentence, target = self.data[["text", "target"]].iloc[index]
        if self.transform:
            sentence = self.transform(sentence)

        if self.target_transform:
            target = self.target_transform(target)

        return sentence, target

    def __len__(self):
        return len(self.data)
