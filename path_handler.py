import os
import pathlib


def model_path(config, root="./saved"):

    root = pathlib.Path(root)
    filename = "{}".format(config.dataset)

    # Dataset-specific keys
    if config.dataset in ["CIFAR10"]:
        filename += "_augm_{}".format(
            config.augment,
        )

    # Model-specific keys
    filename += "_model_{}".format(
        config.model,
    )
    if "sa" in config.model:
        filename += "_type_{}".format(config.attention_type)
        if config.attention_type == "Local":
            filename += "_patch_{}".format(config.patch_size)
        filename += "_dpatt_{}_dpval_{}_activ_{}_norm_{}_white_{}".format(
            config.dropout_att,
            config.dropout_values,
            config.activation_function,
            config.norm_type,
            config.whitening_scale,
        )

    # Optimization arguments
    filename += "_optim_{}".format(config.optimizer)
    if config.optimizer == "SGD":
        filename += "_momentum_{}".format(config.optimizer_momentum)

    filename += "_lr_{}_bs_{}_ep_{}_wd_{}_seed_{}_sched_{}".format(
        config.lr,
        config.batch_size,
        config.epochs,
        config.weight_decay,
        config.seed,
        config.scheduler,
    )
    if config.scheduler not in ["constant", "linear_warmup_cosine"]:
        filename += "_schdec_{}".format(config.sched_decay_factor)
        if config.scheduler == "multistep":
            filename += "_schsteps_{}".format(config.sched_decay_steps)

    # Comment
    if config.comment != "":
        filename += "_comment_{}".format(config.comment)

    # Add correct termination
    filename += ".pt"

    # Check if directory exists and warn the user if the it exists and train is used.
    os.makedirs(root, exist_ok=True)
    path = root / filename
    config.path = str(path)

    if config.train and path.exists():
        print("WARNING! The model exists in directory and will be overwritten")
