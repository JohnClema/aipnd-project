import numpy as np
from matplotlib.pyplot import subplots, Axes
from PIL.Image import Image
from typing import Optional


def process_image(image: Image) -> np.ndarray:
    """Preprocesses a PIL image for a PyTorch model and returns a Numpy array."""

    # Resize the image, keeping the aspect ratio
    aspect_ratio = min(image.size) / 256
    new_size = (int(image.size[0] / aspect_ratio), int(image.size[1] / aspect_ratio))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Crop the image to 224x224
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Convert image to numpy array and normalize
    image = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # Transpose the image to match PyTorch's expected input shape
    image = image.transpose((2, 0, 1))

    return image


def show_image(
    image: np.ndarray, ax: Optional[Axes] = None, title: Optional[str] = None
) -> Axes:
    """Show subplots for Tensor."""
    if ax is None:
        fig, ax = subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
