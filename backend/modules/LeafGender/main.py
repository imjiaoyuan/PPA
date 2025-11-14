import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

MODULE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODULE_DIR, "best_model2.pth")
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = get_model()

def predict_gender_from_path(image_path: str) -> dict:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise
    
    image_tensor = preprocess(image)
    image_batch = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    female_prob = probabilities[0].item()
    male_prob = probabilities[1].item()
    
    return {"female_prob": female_prob, "male_prob": male_prob}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_your_image>")
    else:
        input_image_path = sys.argv[1]
        try:
            result = predict_gender_from_path(input_image_path)
            print(f"Female: {result['female_prob']:.2%}")
            print(f"Male: {result['male_prob']:.2%}")
        except FileNotFoundError:
            print(f"Error: Image not found at '{input_image_path}'")
        except Exception as e:
            print(f"An error occurred: {e}")