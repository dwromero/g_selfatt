import os
import os.path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
    extract_archive,
    verify_str_arg,
)


class MNIST_rot(VisionDataset):
    """Rotated MNIST datasets.

    Download the datasets from https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits
    and preprocess it as in (Cohen and Welling) https://github.com/tscohen/gconv_experiments/blob/master/gconv_experiments/MNIST_ROT/mnist_rot.py
    """

    resources = [
        (
            "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip",
            "0f9a947ff3d30e95cd685462cbf3b847",
        ),
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition("/")[2]
            download_and_extract_archive(
                url, download_root=self.raw_folder, filename=filename, md5=md5
            )

        # process and save as torch files
        print("Processing...")

        train_filename = os.path.join(
            self.raw_folder, "mnist_all_rotation_normalized_float_train_valid.amat"
        )
        test_filename = os.path.join(
            self.raw_folder, "mnist_all_rotation_normalized_float_test.amat"
        )

        train_val = torch.from_numpy(np.loadtxt(train_filename))
        test = torch.from_numpy(np.loadtxt(test_filename))

        train_val_data = train_val[:, :-1].reshape(-1, 28, 28)
        train_val_data = (train_val_data * 256).round().type(torch.uint8)
        train_val_labels = train_val[:, -1].type(torch.uint8)
        training_set = (train_val_data[:10000], train_val_labels[:10000])
        # we ignore the validation test

        test_data = test[:, :-1].reshape(-1, 28, 28)
        test_data = (test_data * 256).round().type(torch.uint8)
        test_labels = test[:, -1].type(torch.uint8)
        test_set = (test_data, test_labels)

        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
