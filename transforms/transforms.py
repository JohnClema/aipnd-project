from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
)

# Constants for normalization
MEANS_NORMALIZATION: list = [0.485, 0.456, 0.406]
STD_DEVIATION_NORMALIZATION: list = [0.229, 0.224, 0.225]


def preprocessing_transforms() -> Compose:
    """
    Returns a composition of transformations for training data.

    :return: Compose object with training transformations
    """
    return Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(MEANS_NORMALIZATION, STD_DEVIATION_NORMALIZATION),
        ]
    )


def training_transforms() -> Compose:
    """
    Returns a composition of transformations for training data.

    :return: Compose object with training transformations
    """
    return Compose(
        [
            RandomRotation(30),
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(MEANS_NORMALIZATION, STD_DEVIATION_NORMALIZATION),
        ]
    )


def standard_transforms() -> Compose:
    """
    Returns a composition of standard transformations for validation/testing data.

    :return: Compose object with standard transformations
    """
    return Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(MEANS_NORMALIZATION, STD_DEVIATION_NORMALIZATION),
        ]
    )
