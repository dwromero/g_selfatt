import ml_collections


def get_config():
    default_config = dict(
        # --------------------------
        # General parameters
        dataset="",
        # The dataset to be used, e.g., MNIST.
        model="",
        # The model to be used, e.g., p4sa.
        optimizer="",
        # The optimizer to be used, e.g., Adam.
        optimizer_momentum=0.0,
        # If optimizer == SGD, this specifies the momentum of the SGD.
        device="",
        # The device in which the model will be deployed, e.g., cuda.
        scheduler="constant",
        # The lr scheduler to be used, e.g., multistep, constant.
        sched_decay_steps=(400,),
        # If scheduler == multistep, this specifies the steps at which
        # The scheduler should be decreased.
        sched_decay_factor=1.0,
        # The factor with which the lr will be reduced, e.g., 5, 10.
        lr=0.0,
        # The lr to be used, e.g., 0.001.
        norm_type="",
        # The normalization type to be used in the network, e.g., LayerNorm.
        attention_type="",
        # The type of self-attention to be used in the network, e.g., Local, Global.
        activation_function="",
        # The activation function used in the network. e.g., ReLU, Swish.
        patch_size=0,
        # If attention_type == Local, the extension of the receptive field on which self-attention is calculated.
        dropout_att=0.0,
        # Specifies a layer-wise dropout factor applied on the computed attention coefficients, e.g., 0.1.
        dropout_values=0.0,
        # Specifies a layer-wise dropout factor applied on the resulting value coefficients from self-att layers, e.g., 0.1.
        whitening_scale=1.0,
        # Specifies a factor with which the current variance initialization is weighted.
        weight_decay=0.0,
        # Specifies a L2 norm over the magnitude of the weigths in the network, e.g., 1e-4.
        batch_size=0,
        # The batch size to be used, e.g., 64.
        epochs=0,
        # The number of epochs to perform training, e.g., 200.
        seed=0,
        # The seed of the run. e.g., 0.
        comment="",
        # An additional comment to be added to the config.path parameter specifying where
        # the network parameters will be saved / loaded from.
        pretrained=False,
        # Specifies if a pretrained model should be loaded.
        train=True,
        # Specifies if training should be performed.
        augment=False,  # **No augment used in our experiments.**
        path="",
        # This parameter is automatically derived from the other parameters of the run. It specifies
        # the path where the network parameters will be saved / loaded from.
    )
    default_config = ml_collections.ConfigDict(default_config)
    return default_config
