from torch import load, save, exp, mean, FloatTensor, no_grad, Tensor
from torch.nn import Module, NLLLoss, Sequential, Linear, ReLU, Dropout, LogSoftmax
from torch.optim import Adam
from torchvision.models import list_models, get_model, get_model_weights
from pathlib import Path
from typing import Optional

from data_loading.data_loading import get_data_loaders
from device.device import get_device
from categories.categories import get_category_count


def train(
    arch: str,
    learning_rate: float,
    data_dir: str,
    hidden_units: int,
    epochs: int,
    save_dir: Optional[str] = None,
    gpu: bool = True,
) -> Module:
    device = get_device(gpu)
    dataloaders, datasets = get_data_loaders(data_dir)

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    if arch not in list_models():
        raise RuntimeError(f"No such model {arch}. Available models: {list_models()}")

    model = initialize_model(arch, hidden_units, device)
    criterion = NLLLoss()
    optimizer = Adam(model.classifier.parameters(), lr=learning_rate)

    model, optimizer = train_model(
        model, optimizer, criterion, dataloaders, epochs, device, print_every=5
    )

    model.arch = arch
    model.hidden_units = hidden_units
    model.optimizer = optimizer
    model.epochs = epochs
    model.gpu = gpu
    model.class_to_idx = datasets["train"].class_to_idx

    if save_dir is not None:
        save_model(model, save_dir)

    return model


def initialize_model(arch: str, hidden_units: int, device) -> Module:
    model = get_model(arch, weights=get_model_weights(arch))
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = get_classifier(hidden_units)
    model.to(device)
    return model


def train_model(
    model: Module,
    optimizer: Adam,
    criterion: NLLLoss,
    dataloaders,
    epochs: int,
    device,
    print_every=5,
) -> (Module, Adam):
    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in dataloaders["train"]:
            steps += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                validate_model(model, criterion, dataloaders["test"], device)
                running_loss = 0
                model.train()

    return model, optimizer


def validate_model(model: Module, criterion: NLLLoss, dataloader, device) -> None:
    test_loss = 0
    accuracy = 0
    model.eval()
    with no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()
            ps = exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += mean(equals.type(FloatTensor)).item()

    print(
        f"Test loss: {test_loss / len(dataloader):.3f}.. "
        f"Test accuracy: {accuracy / len(dataloader):.3f}"
    )


def get_classifier(hidden_units: int) -> Sequential:
    category_count = get_category_count()
    return Sequential(
        Linear(1024, hidden_units),
        ReLU(),
        Dropout(0.2),
        Linear(hidden_units, category_count),
        LogSoftmax(dim=1),
    )


def save_model(model: Module, save_dir: str) -> None:
    checkpoint = {
        "arch": model.arch,
        "hidden_units": model.hidden_units,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
        "epochs": model.epochs,
        "gpu": model.gpu,
        "class_to_idx": model.class_to_idx,
    }
    save(checkpoint, f"{save_dir}/checkpoint.pth")


def load_model(save_dir: str) -> Module:
    if not Path(save_dir).exists():
        raise FileNotFoundError(f"Save directory {save_dir} not found.")

    checkpoint = load(f"{save_dir}/checkpoint.pth")
    device = get_device(checkpoint["gpu"])

    model = get_model(checkpoint["arch"], weights=get_model_weights(checkpoint["arch"]))
    model.classifier = get_classifier(checkpoint["hidden_units"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.optimizer = Adam(model.classifier.parameters(), lr=0.003)
    model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for state in model.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, Tensor):
                state[k] = v.to(device)

    model.arch = checkpoint["arch"]
    model.hidden_units = checkpoint["hidden_units"]
    model.epochs = checkpoint["epochs"]
    model.gpu = checkpoint["gpu"]
    model.class_to_idx = checkpoint["class_to_idx"]

    model.to(device)
    model.eval()

    return model
