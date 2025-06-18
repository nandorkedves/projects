import argparse
import logging

import pandas as pd

from config import load_experiment_config
from model import SKLearnModel
from preprocessing import preprocess_data
from train import train

logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser(
        "Simple experiment pipeline for SKLearn based predictions"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to an experiment configuration",
    )

    return parser.parse_args()


def main():
    args = get_arguments()
    experiment_config = load_experiment_config(args.config)

    logger.info("Loading data")
    data = pd.read_csv(experiment_config.data.csv_path)

    logger.info("Building model")
    model = SKLearnModel(**experiment_config.model.model_dump())

    logger.info("Starting training")
    train(
        experiment_config.experiment_name,
        model,
        data,
        experiment_config.data.target_column,
    )


if __name__ == "__main__":
    main()
