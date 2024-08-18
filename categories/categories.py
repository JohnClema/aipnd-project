from json import load, JSONDecodeError
from typing import Dict


def get_categories(json_file: str) -> Dict[str, str]:
    """
    Reads a JSON file and returns a dictionary mapping category IDs to category names.

    :param json_file: Path to the JSON file containing category mappings
    :return: Dictionary with category IDs as keys and category names as values
    :raises FileNotFoundError: If the JSON file does not exist
    :raises json.JSONDecodeError: If the JSON file is not properly formatted
    """
    try:
        with open(json_file, "r") as f:
            cat_to_name = load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {json_file}") from e
    except JSONDecodeError as e:
        raise JSONDecodeError(f"Error decoding JSON from file: {json_file}") from e

    return cat_to_name


def get_category_count() -> int:
    """
    Reads a JSON file and returns the number of categories.

    :param json_file: Path to the JSON file containing category mappings
    :return: Number of categories
    """
    return len(get_categories("./cat_to_name.json"))
