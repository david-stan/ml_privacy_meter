import os.path
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, AutoTokenizer

from dataset.utils import load_dataset_subsets


def get_softmax(
    model: Union[PreTrainedModel, torch.nn.Module],
    samples: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: str,
    temp: float = 1.0,
    pad_token_id: Optional[int] = None,
) -> np.ndarray:
    """
    Get the model's softmax probabilities for the given inputs and expected outputs.

    Args:
        model (PreTrainedModel or torch.nn.Module): Model instance.
        samples (torch.Tensor): Model input.
        labels (torch.Tensor): Model expected output.
        batch_size (int): Batch size for getting signals.
        device (str): Device used for computing signals.
        temp (float): Temperature used in softmax computation.
        pad_token_id (Optional[int]): Padding token ID to ignore in aggregation.

    Returns:
        all_softmax_list (np.array): softmax value of all samples
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        softmax_list = []
        batched_samples = torch.split(samples, batch_size)
        batched_labels = torch.split(labels, batch_size)

        for x, y in tqdm(
            zip(batched_samples, batched_labels),
            total=len(batched_samples),
            desc="Computing softmax",
        ):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            logits = pred.logits
            logit_signals = torch.div(logits, temp)
            log_probs = torch.log_softmax(logit_signals, dim=-1)
            true_class_log_probs = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)
            # Mask out padding tokens
            mask = (
                y != pad_token_id
                if pad_token_id is not None
                else torch.ones_like(y, dtype=torch.bool)
            )
            true_class_log_probs = true_class_log_probs * mask
            sequence_probs = torch.exp(
                true_class_log_probs.sum(1) / mask.sum(1)
            )
            softmax_list.append(sequence_probs.to("cpu").view(-1, 1))
        all_softmax_list = np.concatenate(softmax_list)
    model.to("cpu")
    return all_softmax_list


def get_model_signals(models_list, dataset, configs, logger, is_population=False):
    """Function to get models' signals (softmax, loss, logits) on a given dataset.

    Args:
        models_list (list): List of models for computing (softmax, loss, logits) signals from them.
        dataset (torchvision.datasets): The whole dataset.
        configs (dict): Configurations of the tool.
        logger (logging.Logger): Logger object for the current run.
        is_population (bool): Whether the signals are computed on population data.

    Returns:
        signals (np.array): Signal value for all samples in all models
    """
    # Check if signals are available on disk
    signal_file_name = f"rmia_signals"
    signal_file_name += "_pop.npy" if is_population else ".npy"

    if os.path.exists(
        f"outputs/{configs['run']['log_dir']}/signals/{signal_file_name}",
    ):
        signals = np.load(
            f"outputs/{configs['run']['log_dir']}/signals/{signal_file_name}",
        )
        expected_size = len(dataset)

        if signals.shape[0] == expected_size:
            logger.info("Signals loaded from disk successfully.")
            return signals
        else:
            logger.warning(
                f"Signals shape ({signals.shape[0]}) does not match the expected size ({expected_size}). "
                f"This mismatch is likely due to a change in the training data size."
            )
            logger.info("Ignoring the signals on disk and recomputing.")

    batch_size = configs["audit"]["batch_size"]  # Batch size used for inferring signals
    model_name = configs["train"]["model_name"]  # Algorithm used for training models
    device = configs["audit"]["device"]  # GPU device used for inferring signals
    if "tokenizer" in configs["data"].keys():
        tokenizer = AutoTokenizer.from_pretrained(
            configs["data"]["tokenizer"], clean_up_tokenization_spaces=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = None

    dataset_samples = np.arange(len(dataset))
    data, targets = load_dataset_subsets(
        dataset, dataset_samples, model_name, batch_size, device
    )

    signals = []
    logger.info("Computing signals for all models.")
    for model in models_list:
        signals.append(
            get_softmax(
                model, data, targets, batch_size, device, pad_token_id=pad_token_id
            )
        )

    signals = np.concatenate(signals, axis=1)
    os.makedirs(f"outputs/{configs['run']['log_dir']}/signals", exist_ok=True)
    np.save(
        f"outputs/{configs['run']['log_dir']}/signals/{signal_file_name}",
        signals,
    )
    logger.info("Signals saved to disk.")
    return signals

def get_target_model_signals(model, dataset, configs, logger, is_population=False):
    """Function to get models' signals (softmax, loss, logits) on a given dataset.

    Args:
        model (hf.Model): Model for computing signals from.
        dataset (torchvision.datasets): The whole dataset.
        configs (dict): Configurations of the tool.
        logger (logging.Logger): Logger object for the current run.
        is_population (bool): Whether the signals are computed on population data.

    Returns:
        signals (np.array): Signal value for all samples in all models
    """
    # Check if signals are available on disk
    signal_file_name = "target_rmia_signals"
    signal_file_name += "_pop.npy" if is_population else ".npy"

    if os.path.exists(
        f"outputs/{configs['run']['log_dir']}/signals/{signal_file_name}",
    ):
        signals = np.load(
            f"outputs/{configs['run']['log_dir']}/signals/{signal_file_name}",
        )
        expected_size = len(dataset)

        if signals.shape[0] == expected_size:
            logger.info("Signals loaded from disk successfully.")
            return signals
        else:
            logger.warning(
                f"Signals shape ({signals.shape[0]}) does not match the expected size ({expected_size}). "
                f"This mismatch is likely due to a change in the data size."
            )
            logger.info("Ignoring the signals on disk and recomputing.")

    batch_size = configs["audit"]["batch_size"]  # Batch size used for inferring signals
    model_name = configs["audit"]["model_name"]  # Algorithm used for training models
    device = configs["audit"]["device"]  # GPU device used for inferring signals
    if "tokenizer" in configs["audit"].keys():
        tokenizer = AutoTokenizer.from_pretrained(
            configs["audit"]["tokenizer"], clean_up_tokenization_spaces=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = None

    batched_samples = dataset.torch_batch(batch_size, device=device)

    model.to(device)
    model.eval()
    softmax_list = []
    for batch in tqdm(
            batched_samples,
            total=len(batched_samples),
            desc="Computing softmax",
    ):
        with torch.no_grad():
            pred = model.forward(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        logits = pred.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        true_class_log_probs = log_probs.gather(2, batch["labels"].unsqueeze(-1)).squeeze(-1)
        # Mask out padding tokens
        true_class_log_probs = true_class_log_probs * batch["attention_mask"]
        sequence_probs = torch.exp(
            true_class_log_probs.sum(1) / batch["attention_mask"].sum(1)
        )
        softmax_list.append(sequence_probs.to("cpu").view(-1, 1))

    signals = np.concatenate(softmax_list)
    os.makedirs(f"outputs/{configs['run']['log_dir']}/signals", exist_ok=True)
    np.save(
        f"outputs/{configs['run']['log_dir']}/signals/{signal_file_name}",
        signals,
    )
    logger.info("Target signals saved to disk.")

    return signals
