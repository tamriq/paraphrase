import argparse
import json

import torch
import utils.training as tutils
import log_wrapper as logflow


def run(config):
    if config["meta"]["mlflow_log"]:
        # MLflowWrapper is a logging class that inherits from TrainHelper.
        trainer = logflow.MLflowWrapper(config)
    else:
        trainer = tutils.TrainHelper(config)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training parameters provided at run time from the CLI'
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='Learning parameters for the dataset and the model',
    )
    args, unknown = parser.parse_known_args()
    if args.config is None:
        raise argparse.ArgumentError(args.config, "Training parameters config file name is not specified.")
    try:
        config = json.load(open(args.config))
    except FileNotFoundError:
        raise
    except json.decoder.JSONDecodeError:
        raise
    # Add the current computation method to the config.
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(config)
