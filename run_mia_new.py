"""This file is the main entry point for running the privacy auditing tool."""

import argparse
import time

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from audit import audit_models, sample_auditing_dataset
from get_signals import get_model_signals
from models.RMIAModelTrainer import RMIAModelTrainer
from models.utils import load_models
from util import (
    setup_log,
    initialize_seeds,
    create_directories,
    load_dataset,
)

# Enable benchmark mode in cudnn to improve performance when input sizes are consistent
torch.backends.cudnn.benchmark = True


def main():
    print(20 * "-")
    print("Privacy Meter Tool!")
    print(20 * "-")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run privacy auditing tool.")
    parser.add_argument(
        "--cf",
        type=str,
        default="configs/agnews.yaml",
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    # Load configuration file
    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    # Initialize seeds for reproducibility
    initialize_seeds(configs["run"]["random_seed"])

    # Create necessary directories
    log_dir = f"outputs/{configs['run']['log_dir']}"
    directories = {
        "log_dir": log_dir,
        "report_dir": f"{log_dir}/report",
        "signal_dir": f"{log_dir}/signals",
        "data_dir": configs["data"]["data_dir"],
    }
    create_directories(directories)

    # Set up logger
    logger = setup_log(
        directories["report_dir"], "time_analysis", configs["run"]["time_log"]
    )

    start_time = time.time()

    # Load the dataset
    baseline_time = time.time()
    dataset, population = load_dataset(configs, directories["data_dir"], logger)
    logger.info("Loading dataset took %0.5f seconds", time.time() - baseline_time)

    # Define experiment parameters
    num_reference_models = configs["audit"]["num_ref_models"]
    num_model_pairs = num_reference_models + 1

    # Load or train models
    baseline_time = time.time()
    models_list, memberships = load_models(
        log_dir, dataset, num_model_pairs * 2, configs, logger
    )
    if models_list is None:
        trainer = RMIAModelTrainer(
            log_dir=log_dir,
            dataset=dataset,
            num_model_pairs=num_model_pairs,
            configs=configs,
            logger=logger,
        )
        models_list, memberships = trainer.train_and_save_models()

    logger.info(
        "Model loading/training took %0.1f seconds", time.time() - baseline_time
    )

    auditing_dataset, auditing_membership = sample_auditing_dataset(
        configs, dataset, logger, memberships
    )

    population = Subset(
        population,
        np.random.choice(
            len(population),
            configs["audit"].get("population_size", len(population)),
            replace=False,
        ),
    )

    # Generate signals (softmax outputs) for all models
    baseline_time = time.time()
    signals = get_model_signals(models_list, auditing_dataset, configs, logger)
    population_signals = get_model_signals(
        models_list, population, configs, logger, is_population=True
    )
    logger.info("Preparing signals took %0.5f seconds", time.time() - baseline_time)

    # Perform the privacy audit
    baseline_time = time.time()
    target_model_indices = [0]



    mia_score_list, membership_list = audit_models(
        f"{directories['report_dir']}/exp",
        target_model_indices,
        auditing_dataset,
        signals,
        population_signals,
        auditing_membership,
        num_reference_models,
        logger,
        configs,
    )

    logger.info("Total runtime: %0.5f seconds", time.time() - start_time)


if __name__ == "__main__":
    main()
