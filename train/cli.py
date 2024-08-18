from argparse import _SubParsersAction
from typing import Dict, Any


def get_train_parser_config() -> Dict[str, Any]:
    """
    Returns the configuration for the train subparser.

    The configuration includes the name of the subparser, help text, and a list of arguments
    with their respective flags and keyword arguments.

    Returns:
        dict: A dictionary containing the configuration for the train subparser.
    """
    return {
        "name": "train",
        "help": "Train the model",
        "arguments": [
            {
                "flags": ["--data_dir"],
                "kwargs": {
                    "type": str,
                    "required": True,
                    "help": "Directory of training data",
                },
            },
            {
                "flags": ["--save_dir"],
                "kwargs": {
                    "type": str,
                    "required": True,
                    "help": "Directory to save the trained model",
                },
            },
            {
                "flags": ["--arch"],
                "kwargs": {
                    "type": str,
                    "default": "vgg16",
                    "help": "Model architecture",
                },
            },
            {
                "flags": ["--learning_rate"],
                "kwargs": {"type": float, "default": 0.001, "help": "Learning rate"},
            },
            {
                "flags": ["--hidden_units"],
                "kwargs": {
                    "type": int,
                    "default": 512,
                    "help": "Number of hidden units",
                },
            },
            {
                "flags": ["--epochs"],
                "kwargs": {"type": int, "default": 20, "help": "Number of epochs"},
            },
            {
                "flags": ["--gpu"],
                "kwargs": {"action": "store_true", "help": "Use GPU for training"},
            },
        ],
    }


def add_train_subparser(subparsers: _SubParsersAction) -> None:
    """
    Adds the train subparser to the provided subparsers object.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to which the train subparser will be added.
    """
    config = get_train_parser_config()
    train_parser = subparsers.add_parser(config["name"], help=config["help"])
    for arg in config["arguments"]:
        train_parser.add_argument(*arg["flags"], **arg["kwargs"])
