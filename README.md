# AI Programming with Python Project - Flower Identification

This project involves training a neural network to identify different species of flowers using a dataset of images. The trained model can then be used to predict the species of flowers in new images.

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/JohnClema/flower-classification.git
cd flower-classification
pip install -r requirements.txt
```

## Running

To train the model, use the following command:

```bash
python train.py data_directory --save_directory save_directory --arch vgg13 --learning_rate 0.01 --hidden_units 512 --epochs 5 --gpu
```

To make predictions with the trained model, use the following command:

```bash
python predict.py image_path checkpoint --top_k 5 --category_names cat_to_name.json --gpu
```

## Options

### Train

- `data_directory`: Path to the data directory containing the training, validation, and testing datasets.
- `--save_directory`: Directory to save checkpoints (optional).
- `--arch`: Model architecture (default: `vgg13`).
- `--learning_rate`: Learning rate (default: `0.01`).
- `--hidden_units`: Number of hidden units in the classifier (default: `512`).
- `--epochs`: Number of epochs to train (default: `20`).
- `--gpu`: Use GPU for training (default: `True`).

### Predict

- `image_path`: Path to the image file to be predicted.
- `checkpoint`: Path to the saved model checkpoint.
- `--top_k`: Return top K most likely classes (default: `5`).
- `--category_names`: Path to a JSON file mapping categories to real names (optional).
- `--gpu`: Use GPU for inference (default: `True`).

## Example Usage

### Training

```bash
python train.py flowers --save_directory checkpoints --arch resnet50 --learning_rate 0.003 --hidden_units 256 --epochs 10 --gpu
```

### Prediction

```bash
python predict.py flowers/test/1/image_06743.jpg checkpoints/checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
```

## Project Entrypoints

- [`train.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fjohnclema%2FDevelopment%2Faipnd-project%2Ftrain.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/johnclema/Development/aipnd-project/train.py"): Script to train the model.
- [`predict.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fjohnclema%2FDevelopment%2Faipnd-project%2Fpredict.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/johnclema/Development/aipnd-project/predict.py"): Script to make predictions using the trained model.

## License

This project is licensed under the MIT License - see the [`LICENSE`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fjohnclema%2FDevelopment%2Faipnd-project%2FLICENSE%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/johnclema/Development/aipnd-project/LICENSE") file for details.
