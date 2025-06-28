"""This module defines functions for model handling, including model definition, loading, and training."""

import json
import logging

import pickle

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from trainers.train_transformers import *
from peft import get_peft_model


INPUT_OUTPUT_SHAPE = {
    "cifar10": [3, 10],
    "cifar100": [3, 100],
    "purchase100": [600, 100],
    "texas100": [6169, 100],
}


def get_model(model_type: str, dataset_name: str, configs: dict):
    """
    Instantiate and return a model based on the given model type and dataset name.

    Args:
        model_type (str): Type of the model to be instantiated.
        dataset_name (str): Name of the dataset the model will be used for.
        configs (dict): Configuration dictionary containing information about the model.
    Returns:
        torch.nn.Module or PreTrainedModel: An instance of the specified model, ready for training or inference.
    """
    train_configs = configs["train"]

    if train_configs.get("peft", None) is None:
        return AutoModelForCausalLM.from_pretrained(model_type)
    else:
        peft_config = get_peft_model_config(configs)
        orig_model = AutoModelForCausalLM.from_pretrained(model_type)
        return get_peft_model(
            orig_model, peft_config
        )

def load_existing_model(
    model_metadata: dict, dataset, device: str, config: dict
):
    """Load an existing model from disk based on the provided metadata.

    Args:
        model_metadata (dict): Metadata dictionary containing information about the model.
        dataset (datasets): Dataset object used to instantiate the model.
        device (str): The device on which to load the model, such as 'cpu' or 'cuda'.
        config (dict): Configuration dictionary containing information about the model.
    Returns:
        model (torch.nn.Module): Loaded model object with weights restored from disk.
    """
    model_name = model_metadata["model_name"]
    dataset_name = model_metadata["dataset"]

    model = get_model(model_name, dataset_name, config)

    model_checkpoint_extension = os.path.splitext(model_metadata["model_path"])[1]
    if model_checkpoint_extension == ".pkl":
        with open(model_metadata["model_path"], "rb") as file:
            model_weight = pickle.load(file)
        model.load_state_dict(model_weight)
    elif model_checkpoint_extension == ".pt" or model_checkpoint_extension == ".pth":
        model.load_state_dict(torch.load(model_metadata["model_path"]))
    elif model_checkpoint_extension == "":
        if isinstance(model, PreTrainedModel):
            model = model.from_pretrained(model_metadata["model_path"])
        else:
            raise ValueError(f"Model path is invalid.")
    else:
        raise ValueError(f"Model path is invalid.")
    return model


def load_models(log_dir, dataset, num_models, configs, logger):
    """
    Load trained models from disk if available.

    Args:
        log_dir (str): Path to the directory containing model logs and metadata.
        dataset (datasets): Dataset object used for model training.
        num_models (int): Number of models to be loaded from disk.
        configs (dict): Dictionary of configuration settings, including device information.
        logger (logging.Logger): Logger object for logging the model loading process.

    Returns:
        model_list (list of nn.Module): List of loaded model objects.
        all_memberships (np.array): Membership matrix for all loaded models, indicating training set membership.
    """
    experiment_dir = f"{log_dir}/models"
    if os.path.exists(f"{experiment_dir}/models_metadata.json"):
        with open(f"{experiment_dir}/models_metadata.json", "r") as f:
            model_metadata_dict = json.load(f)
        all_memberships = np.load(f"{experiment_dir}/memberships.npy")
        if len(model_metadata_dict) < num_models:
            return None, None
    else:
        return None, None

    model_list = []
    for model_idx in range(len(model_metadata_dict)):
        logger.info(f"Loading model {model_idx}")
        model_obj = load_existing_model(
            model_metadata_dict[str(model_idx)],
            dataset,
            configs["audit"]["device"],
            configs,
        )
        model_list.append(model_obj)
        if len(model_list) == num_models:
            break
    return model_list, all_memberships
