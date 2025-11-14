import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

MODEL_PATH = "best_model2.pth"
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_image(image_path):
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at '{image_path}'")
        return

    image_tensor = preprocess(image)
    image_batch = image_tensor.unsqueeze(0).to(DEVICE)

    class_names = ['Female', 'Male']

    with torch.no_grad():
        output = model(image_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    for i, class_name in enumerate(class_names):
        prob = probabilities[i].item()
        print(f"{class_name}: {prob:.2%}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
    else:
        if len(sys.argv) != 2:
            print("Usage: python predict.py <path_to_your_image>")
        else:
            input_image_path = sys.argv[1]
            predict_image(input_image_path)