from typing import Dict

import ml_collections
import torch
import torchvision

from datasets import MNIST_rot, PCam


def get_dataset(
    config: ml_collections.ConfigDict, num_workers: int = 4, data_root: str = "./data"
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create dataloaders for the chosen datasets
    :return: {'train': training_loader, 'validation': validation_loader, 'test': test_loader}
    """
    dataset = {
        "cifar10": torchvision.datasets.CIFAR10,
        "mnist": torchvision.datasets.MNIST,
        "rotmnist": MNIST_rot,
        "pcam": PCam,
    }[config["dataset"].lower()]

    if "cifar" in config.dataset.lower():
        data_mean = (0.4914, 0.4822, 0.4465)
        data_stddev = (0.2023, 0.1994, 0.2010)
        if config.augment:
            transform_train = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(data_mean, data_stddev),
                ]
            )
        else:
            transform_train = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(data_mean, data_stddev),
                ]
            )
    elif "mnist" in config.dataset.lower():
        data_mean = (0.1307,)
        data_stddev = (0.3081,)
        transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )
    elif "pcam" in config.dataset.lower():
        data_mean = (0.701, 0.538, 0.692)
        data_stddev = (0.235, 0.277, 0.213)
        transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )
    else:
        raise ValueError(f"Unkown preprocessing for datasets '{config.dataset}'")

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    training_set = dataset(root=data_root, train=True, download=True, transform=transform_train)
    test_set = dataset(root=data_root, train=False, download=True, transform=transform_test)

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    dataloaders = {"train": training_loader, "test": test_loader}
    if "pcam" in config.dataset.lower():
        validation_set = dataset(
            root=data_root, train=False, valid=True, download=False, transform=transform_test
        )
        val_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        dataloaders["validation"] = val_loader
    else:
        dataloaders["validation"] = test_loader

    return dataloaders
