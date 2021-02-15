import torch
import wandb

import g_selfatt.groups as groups
import models
from g_selfatt.utils import num_params


def get_model(config):
    # Define number of channels
    in_channels = 1 if "mnist" in config.dataset.lower() else 3
    num_classes = 2 if "pcam" in config.dataset.lower() else 10

    # Define the input size
    if "mnist" in config.dataset.lower():
        image_size = 28
    elif "pcam" in config.dataset.lower():
        image_size = 96
    else:
        image_size = 32
    # And the patch_size, if Local
    patch_size = config.patch_size if config.attention_type == "Local" else None

    # Build the model
    if config.model == "z2cnn":
        model = models.CNN(
            in_channels=in_channels,
            num_channels=20,
            bn_epsilon=2e-5,
            dropout_rate=0.3,
            use_bias=False,
        )
    else:
        # Parse the desired group
        group_name = config.model[: config.model.find("sa")]
        group = {
            "z2": groups.SE2(num_elements=1),
            "p4": groups.SE2(num_elements=4),
            "p8": groups.SE2(num_elements=8),
            "p12": groups.SE2(num_elements=12),
            "p16": groups.SE2(num_elements=16),
            "mz2": groups.E2(num_elements=2),
            "p4m": groups.E2(num_elements=8),
            "p8m": groups.E2(num_elements=16),
        }[group_name]

        # Create model
        if config.dataset == "rotMNIST":
            model = models.GroupTransformer(
                group=group,
                in_channels=in_channels,
                num_channels=20,
                block_sizes=[2, 3],
                expansion_per_block=1,
                crop_per_layer=[2, 0, 2, 1, 1],
                image_size=image_size,
                num_classes=num_classes,
                dropout_rate_after_maxpooling=0.0,
                maxpool_after_last_block=False,
                normalize_between_layers=False,
                patch_size=patch_size,
                num_heads=9,
                norm_type=config.norm_type,
                activation_function=config.activation_function,
                attention_dropout_rate=config.dropout_att,
                value_dropout_rate=config.dropout_values,
                whitening_scale=config.whitening_scale,
            )
        elif config.dataset == "CIFAR10":
            model = models.GroupTransformer(
                group=group,
                in_channels=in_channels,
                num_channels=96,
                block_sizes=[2, 2, 2],
                expansion_per_block=[1, 2, 1],
                crop_per_layer=0,
                image_size=image_size,
                num_classes=num_classes,
                dropout_rate_after_maxpooling=0.3,
                maxpool_after_last_block=False,
                normalize_between_layers=True,
                patch_size=patch_size,
                num_heads=9,
                norm_type=config.norm_type,
                activation_function=config.activation_function,
                attention_dropout_rate=config.dropout_att,
                value_dropout_rate=config.dropout_values,
                input_dropout_rate=0.2,
                whitening_scale=config.whitening_scale,
            )
        elif config.dataset == "PCam":
            model = models.GroupTransformer(
                group=group,
                in_channels=in_channels,
                num_channels=12,
                block_sizes=[0, 1, 2, 1],
                expansion_per_block=[1, 2, 2, 2],
                crop_per_layer=[0, 2, 1, 1],
                image_size=image_size,
                num_classes=num_classes,
                dropout_rate_after_maxpooling=0.0,
                maxpool_after_last_block=True,
                normalize_between_layers=True,
                patch_size=patch_size,
                num_heads=9,
                norm_type=config.norm_type,
                activation_function=config.activation_function,
                attention_dropout_rate=config.dropout_att,
                value_dropout_rate=config.dropout_values,
                whitening_scale=config.whitening_scale,
            )

    # Check if multi-GPU available and if so, use the available GPU's
    print("GPU's available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)  # Required for multi-GPU
    model.to(config.device)
    torch.backends.cudnn.benchmark = True

    # print number parameters
    no_params = num_params(model)
    print("Number of parameters:", no_params)
    wandb.run.summary["no_params"] = no_params

    return model
