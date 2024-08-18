import enum


class Folders(str, enum.Enum):
    """
    Enum representing different dataset folders.
    """

    TRAINING = "train"
    VALIDATION = "valid"
    TESTING = "test"
