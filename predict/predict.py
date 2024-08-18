from torch import no_grad, exp
from PIL.Image import Image
from device.device import get_device
from transforms.transforms import preprocessing_transforms

# def load_checkpoint(filepath):
#     # Load the saved checkpoint
#     checkpoint = load(filepath)

#     # Load the pre-trained model
#     model = models.__dict__[checkpoint["arch"]](pretrained=True)

#     # Freeze the parameters so we don't backpropagate through them
#     for param in model.parameters():
#         param.requires_grad = False

#     # Load the saved classifier
#     model.classifier = checkpoint["classifier"]

#     # Load the class_to_idx mapping
#     model.class_to_idx = checkpoint["class_to_idx"]

#     return model


def process_image(image: Image) -> Image:
    # Preprocess the image
    preprocess = preprocessing_transforms()
    processed_image = preprocess(image)
    return processed_image


def predict(image_path, model, topk=5):
    # Load and process the image
    image = Image.open(image_path)
    processed_image = process_image(image)

    # Add a batch dimension to the image
    processed_image = processed_image.unsqueeze(0)

    # Move the image tensor to the device
    device = get_device()
    processed_image = processed_image.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation
    with no_grad():
        # Forward pass through the model
        output = model(processed_image)

        # Calculate the probabilities
        probabilities = exp(output)

        # Get the top k probabilities and indices
        top_probabilities, top_indices = probabilities.topk(topk)

        # Convert indices to classes
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]

    return top_probabilities, top_classes


# Example usage
# model = load_checkpoint("checkpoint.pth")
# image_path = "flower.jpg"
# top_probabilities, top_classes = predict(image_path, model)

# print(top_probabilities)
# print(top_classes)
