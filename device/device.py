from torch.cuda import is_available
from logging import info, INFO, basicConfig

# Configure logging
basicConfig(level=INFO)


def get_device(gpu: bool) -> str:
    """
    Determines the device to be used for computation.

    :param gpu: Boolean indicating whether to use GPU if available
    :return: 'cuda' if GPU is to be used and available, otherwise 'cpu'
    """
    if gpu and not is_available():
        info("Using CPU - CUDA is not available")
    return "cuda" if gpu and is_available() else "cpu"
