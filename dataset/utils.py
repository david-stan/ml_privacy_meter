"""This file contains functions for loading the dataset"""

import os
import pickle
from typing import List, Tuple, Any

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from datasets import Dataset as HFDataset

from dataset import TextDataset, load_agnews, load_swallow_code


class InfinitelyIndexableDataset(Dataset):
    """
    A PyTorch Dataset that is able to index the given dataset infinitely.
    This is a helper class to allow easier and more efficient computation later when repeatedly indexing the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be indexed repeatedly.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        # If the index is out of range, wrap it around
        return self.dataset[idx % len(self.dataset)]


def get_dataset(dataset_name: str, data_dir: str, logger: Any, **kwargs: Any) -> Any:
    """
    Function to load the dataset from the pickle file or download it from the internet.

    Args:
        dataset_name (str): Dataset name.
        data_dir (str): Indicate the log directory for loading the dataset.
        logger (logging.Logger): Logger object for the current run.

    Raises:
        NotImplementedError: If the dataset is not implemented.

    Returns:
        Any: Loaded dataset.
    """
    path = f"{data_dir}/{dataset_name}"
    if os.path.exists(f"{path}.pkl"):
        with open(f"{path}.pkl", "rb") as file:
            all_data = pickle.load(file)
        logger.info(f"Data loaded from {path}.pkl")
        if os.path.exists(f"{path}_population.pkl"):
            with open(f"{path}_population.pkl", "rb") as file:
                test_data = pickle.load(file)
            logger.info(f"Population data loaded from {path}_population.pkl")
    else:
        if dataset_name == "agnews":
            tokenizer = kwargs.get("tokenizer")
            if tokenizer is None:
                agnews = load_agnews(tokenize=False)
                agnews_test = load_agnews(split="test", tokenize=False)
            else:
                agnews = load_agnews(
                    tokenize=True,
                    tokenizer=AutoTokenizer.from_pretrained(
                        tokenizer, clean_up_tokenization_spaces=True
                    ),
                )
                agnews_test = load_agnews(
                    split="test",
                    tokenize=True,
                    tokenizer=AutoTokenizer.from_pretrained(
                        tokenizer, clean_up_tokenization_spaces=True
                    ),
                )
            all_data = TextDataset(agnews, target_column="labels", text_column="text")
            test_data = TextDataset(
                agnews_test, target_column="labels", text_column="text"
            )
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
            with open(f"{path}_population.pkl", "wb") as file:
                pickle.dump(test_data, file)
            logger.info(f"Save population data to {path}_population.pkl")
        elif dataset_name == "swallow_code":
            tokenizer = kwargs.get("tokenizer")
            if tokenizer is None:
                swallow_code = load_swallow_code(tokenize=False)
            else:
                swallow_code = load_swallow_code(
                    tokenize=True,
                    tokenizer=AutoTokenizer.from_pretrained(
                        tokenizer, clean_up_tokenization_spaces=True
                    ),
                )

            logger.info("Downloading Swallow Code train dataset")
            swallow_code = HFDataset.from_list(list(swallow_code.take(200000)))
            logger.info("Downloading Swallow Code test dataset")
            shallow_code_test = HFDataset.from_list(list(swallow_code.take(20000)))

            all_data = TextDataset(swallow_code, target_column="labels", text_column="text")
            test_data = TextDataset(
                shallow_code_test, target_column="labels", text_column="text"
            )
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
            with open(f"{path}_population.pkl", "wb") as file:
                pickle.dump(test_data, file)
            logger.info(f"Save population data to {path}_population.pkl")
        else:
            raise NotImplementedError(f"{dataset_name} is not implemented")

    logger.info(f"The whole dataset size: {len(all_data)}")
    return all_data, test_data


def load_dataset_subsets(
    dataset: torchvision.datasets,
    index: List[int],
    model_type: str,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to divide dataset into subsets and load them into device (GPU).

    Args:
        dataset (torchvision.datasets): The whole dataset.
        index (List[int]): List of sample indices.
        model_type (str): Type of the model.
        batch_size (int): Batch size for getting signals.
        device (str): Device used for loading models.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Loaded samples and their labels.
    """
    assert max(index) < len(dataset) and min(index) >= 0, "Index out of range"
    input_list = []
    targets_list = []
    if batch_size == 1:
        # This happens with range dataset. Need to set num_workers to 0 to avoid CUDA error
        data_loader = get_dataloader(
            torch.utils.data.Subset(dataset, index),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        data_loader = get_dataloader(
            torch.utils.data.Subset(dataset, index),
            batch_size=batch_size,
            shuffle=False,
        )
    for inputs, targets in data_loader:
        input_list.append(inputs)
        targets_list.append(targets)
    inputs = torch.cat(input_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    return inputs, targets


def get_dataloader(
    dataset: torchvision.datasets,
    batch_size: int,
    loader_type: str = "torch",
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Function to get DataLoader.

    Args:
        dataset (torchvision.datasets): The whole dataset.
        batch_size (int): Batch size for getting signals.
        loader_type (str): Loader type.
        shuffle (bool): Whether to shuffle dataset or not.

    Returns:
        DataLoader: DataLoader object.
    """
    if loader_type == "torch":
        repeated_data = InfinitelyIndexableDataset(dataset)
        return DataLoader(
            repeated_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=16 if num_workers > 0 else None,
        )
    else:
        raise NotImplementedError(f"{loader_type} is not supported")
