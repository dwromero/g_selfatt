import copy
import datetime
import os

import torch
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import tester
from g_selfatt import utils


def train(model, dataloaders, config):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model.parameters(), config)
    lr_scheduler, scheduler_step_at = get_scheduler(optimizer, dataloaders, config)

    if config.scheduler != "constant":
        print(
            "No AMP will be used. Schedulers other than constant make models trained with AMP diverge. Current: {}".format(
                config.scheduler
            )
        )

    device = config.device
    epochs = config.epochs

    # Creates a GradScaler once at the beginning of training. Scaler handles mixed-precision on backward pass.
    scaler = GradScaler()

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999
    # iterate over epochs
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate: {}".format(param_group["lr"]))
        print("-" * 30)
        # log learning_rate of the epoch
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

        # Each epoch consist of training and validation
        for phase in ["train", "validation"]:
            train = phase == "train"
            if train:
                model.train()
            else:
                model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            total = 0
            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(train):
                    if config.scheduler != "constant":

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        if train:
                            loss.backward()
                            optimizer.step()

                            # Update lr_scheduler
                            if scheduler_step_at == "step":
                                lr_scheduler.step()

                    else:
                        with autocast():  # Sets autocast in the main thread. It handles mixed precision in the forward pass.
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        if phase == "train":
                            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                            scaler.scale(loss).backward()
                            # scaler.step() first unscales the gradients of the optimizer's assigned params.
                            scaler.step(optimizer)
                            # Updates the scale for next iteration.
                            scaler.update()

                            # Update lr_scheduler
                            if scheduler_step_at == "step":
                                lr_scheduler.step()

                    _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            print(datetime.datetime.now())

            # log statistics of the epoch
            wandb.log(
                {"accuracy" + "_" + phase: epoch_acc, "loss" + "_" + phase: epoch_loss},
                step=epoch + 1,
            )

            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "validation" and epoch_acc >= best_acc:

                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Log best results so far and the weights of the model.
                    wandb.run.summary["best_val_accuracy"] = best_acc
                    wandb.run.summary["best_val_loss"] = best_loss

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()
                    # Perform test and log results
                    if config.dataset in ["PCam"]:
                        test_acc, _, _ = tester.test(model, dataloaders["test"], config)
                    else:
                        test_acc = best_acc
                    wandb.run.summary["best_test_accuracy"] = test_acc
                    wandb.log({"accuracy_test": test_acc}, step=epoch + 1)

        # Update scheduler
        if scheduler_step_at == "epoch":
            lr_scheduler.step()

    # Report best results
    print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)

    # save model and log it
    torch.save(model.state_dict(), config.path)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
    torch.save(
        model.module.state_dict(),
        os.path.join(wandb.run.dir, config.path.split("/")[-1]),
    )


def get_optimizer(model_parameters, config):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: optimizer
    """
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=config.lr,
            momentum=config.optimizer_momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError("Unexpected value for optimizer")

    return optimizer


def get_scheduler(optimizer, dataloaders, config):
    """
    Creates a learning rate scheduler for a given model
    :param optimizer: the optimizer to be used
    :return: scheduler
    """
    if config.scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.sched_decay_steps,
            gamma=1.0 / config.sched_decay_factor,
        )
        step_at = "epoch"
    elif config.scheduler == "linear_warmup_cosine":
        max_steps = config.epochs
        max_steps *= len(dataloaders["train"].dataset) // config.batch_size
        lr_scheduler = utils.schedulers.linear_warmup_cosine_lr_scheduler(
            optimizer, 10.0 / config.epochs, T_max=max_steps  # Perform linear warmup for 10 epochs.
        )
        step_at = "step"
    elif config.scheduler == "constant":
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        step_at = "step"
    else:
        raise ValueError(f"Unknown scheduler '{config.scheduler}'")

    return lr_scheduler, step_at
