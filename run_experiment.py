import datetime
import os
import sys

import numpy as np
import torch
import wandb

# args
from absl import app, flags
from ml_collections.config_flags import config_flags

import dataset
import tester
import trainer
from model import get_model
from path_handler import model_path

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", default="config.py")


def main(_):

    if "absl.logging" in sys.modules:
        import absl.logging

        absl.logging.set_verbosity("info")
        absl.logging.set_stderrthreshold("info")

    config = FLAGS.config
    print(config)

    # Set the seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Check if in the correct branch
    # group_name = config["model"][: config["model"].find("sa")]
    # if group_name not in ["z2", "mz2", "p4", "p4m"]:
    #     raise ValueError(
    #         "Mlp_encoding is required for rotations finer than 90 degrees. Please change to the mlp_encoding branch."
    #     )

    # initialize weight and bias
    os.environ["WANDB_API_KEY"] = "691777d26bb25439a75be52632da71d865d3a671"
    if not config.train:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(
        project="equivariant-attention",
        config=config,
        group=config["dataset"],
        entity="equivatt_team",
    )

    # Define the device to be used and move model to that device
    config["device"] = (
        "cuda:0" if (config.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    model = get_model(config)

    # Define transforms and create dataloaders
    dataloaders = dataset.get_dataset(config, num_workers=4)

    # Create model directory and instantiate config.path
    model_path(config)

    if config.pretrained:
        # Load model state dict
        model.module.load_state_dict(torch.load(config.path), strict=False)

    # Train the model
    if config.train:
        # Print arguments (Sanity check)
        print(config)
        print(datetime.datetime.now())
        # Train the model
        trainer.train(model, dataloaders, config)

    # Test model
    tester.test(model, dataloaders["test"], config)


if __name__ == "__main__":
    app.run(main)
