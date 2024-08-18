from argparse import _SubParsersAction
from typing import Dict, Any


def get_predict_parser_config() -> Dict[str, Any]:
    """
    Returns the configuration for the predict subparser.

    The configuration includes the name of the subparser, help text, and a list of arguments
    with their respective flags and keyword arguments.

    Returns:
        dict: A dictionary containing the configuration for the predict subparser.
    """
    return {
        "name": "predict",
        "help": "Predict using the trained model",
        "args": [
            {
                "flags": ["image_path"],
                "kwargs": {"type": str, "help": "Path to the input image"},
            },
            {
                "flags": ["checkpoint"],
                "kwargs": {"type": str, "help": "Path to the checkpoint file"},
            },
            {
                "flags": ["--top_k"],
                "kwargs": {
                    "type": int,
                    "default": 1,
                    "help": "Return top K most likely classes",
                },
            },
            {
                "flags": ["--category_names"],
                "kwargs": {
                    "type": str,
                    "default": "cat_to_name.json",
                    "help": "Path to the category names mapping file",
                },
            },
            {
                "flags": ["--gpu"],
                "kwargs": {"action": "store_true", "help": "Use GPU for inference"},
            },
        ],
    }


def add_predict_subparser(subparsers: _SubParsersAction) -> None:
    """
    Adds the predict subparser to the main parser.

    Args:
        subparsers (_SubParsersAction): The subparsers action object to add the subparser to.
    """
    config = get_predict_parser_config()
    predict_parser = subparsers.add_parser(config["name"], help=config["help"])
    for arg in config["args"]:
        predict_parser.add_argument(*arg["flags"], **arg["kwargs"])
