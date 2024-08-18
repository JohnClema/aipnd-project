import argparse
from train.model import train
from predict.predict import predict
from train.cli import add_train_subparser
from predict.cli import add_predict_subparser


def get_parser_config():
    """
    Returns the configuration for the main parser and its subparsers.

    Returns:
        dict: A dictionary containing the configuration for the main parser and its subparsers.
    """
    return {
        "description": "Train or Predict using the Image Classifier",
        "subparsers": [
            {"func": add_train_subparser, "name": "train"},
            {"func": add_predict_subparser, "name": "predict"},
        ],
    }


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    config = get_parser_config()
    parser = argparse.ArgumentParser(description=config["description"])
    subparsers = parser.add_subparsers(dest="command")

    # Add subparsers
    for subparser in config["subparsers"]:
        subparser["func"](subparsers)

    return parser.parse_args()


def main():
    """
    Main function to execute the appropriate command based on parsed arguments.
    """
    args = parse_args()

    if args.command == "train":
        try:
            train(
                data_dir=args.data_dir,
                save_dir=args.save_dir,
                arch=args.arch,
                learning_rate=args.learning_rate,
                hidden_units=args.hidden_units,
                epochs=args.epochs,
                gpu=args.gpu,
            )
        except Exception as e:
            print(f"Error during training: {e}")
    elif args.command == "predict":
        try:
            predict(
                image_path=args.image_path,
                checkpoint=args.checkpoint,
                top_k=args.top_k,
                category_names=args.category_names,
                gpu=args.gpu,
            )
        except Exception as e:
            print(f"Error during prediction: {e}")
    else:
        print("Invalid command. Use 'train' or 'predict'.")


if __name__ == "__main__":
    main()
