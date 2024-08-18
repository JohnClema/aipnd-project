from torch import no_grad, max, Model
from typing import Dict
from utils import DataLoader
from device import get_device
from folders import Folders


def validate(model: Model, dataloaders: Dict[Folders, DataLoader], gpu: bool) -> None:
    correct = 0
    total = 0
    model.eval()
    device = get_device()

    with no_grad():
        for images, labels in dataloaders[Folders.Test]:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.2f}%")
