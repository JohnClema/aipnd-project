import pathlib
from typing import Dict, Tuple
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transforms.transforms import training_transforms, standard_transforms
from folders.folders import Folders


def organise(
    training: ImageFolder | DataLoader,
    validation: ImageFolder | DataLoader,
    testing: ImageFolder | DataLoader,
) -> Dict[Folders, ImageFolder | DataLoader]:
    """
    Organises the datasets into a dictionary.

    :param training: Training dataset
    :param validation: Validation dataset
    :param testing: Testing dataset
    :return: Dictionary with datasets organized by type
    """
    return {
        Folders.TRAINING.value: training,
        Folders.VALIDATION.value: validation,
        Folders.TESTING.value: testing,
    }


def get_data_loaders(
    data_dir: str,
) -> Tuple[Dict[Folders, ImageFolder], Dict[Folders, DataLoader]]:
    """
    Creates data loaders for training, validation, and testing datasets.

    :param data_dir: Directory containing the data
    :return: Tuple containing dictionaries of image datasets and data loaders
    :raises FileNotFoundError: If the data directory does not exist
    """
    data_path = pathlib.Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    train_dir = data_path / Folders.TRAINING.value
    valid_dir = data_path / Folders.VALIDATION.value
    test_dir = data_path / Folders.TESTING.value

    training_folder = ImageFolder(train_dir, transform=training_transforms())
    validation_folder = ImageFolder(valid_dir, transform=standard_transforms())
    testing_folder = ImageFolder(test_dir, transform=standard_transforms())

    training_loader = DataLoader(training_folder, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_folder, batch_size=32)
    testing_loader = DataLoader(testing_folder, batch_size=32)

    image_datasets = organise(training_folder, validation_folder, testing_folder)
    dataloaders = organise(training_loader, validation_loader, testing_loader)

    return image_datasets, dataloaders
