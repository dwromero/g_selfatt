import pathlib

from torchvision.datasets import ImageFolder


class PCam(ImageFolder):
    """
    PCam dataset.

    Download the dataset from https://drive.google.com/file/d/1PcPdBOyImivBz3IMYopIizGvJOnfgXGD/view?usp=sharing

    For more information, please refer to the README.md of the repository.
    """

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False, valid=False
    ):
        if train and valid:
            raise ValueError("PCam 'valid' split available only when train=False.")

        root = pathlib.Path(root) / "PCam"
        split = "train" if train else ("valid" if valid else "test")
        directory = root / split
        if not (root.exists() and directory.exists()):
            raise FileNotFoundError(
                "Please download the PCam dataset. How to download it can be found in 'README.md'"
            )

        super().__init__(root=directory, transform=transform, target_transform=target_transform)
