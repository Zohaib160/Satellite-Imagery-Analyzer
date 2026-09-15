import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Dict, Any, List

from src.config import BEST_MODEL_PATH, CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
from src.model.classifier import get_model

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads model checkpoint and returns model in eval mode.
    """
    model = get_model(pretrained=False, freeze=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_single(model: torch.nn.Module, image_path: str, device: torch.device) -> Dict[str, Any]:
    """
    Predicts class for a single image.
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output[0], dim=0)
        confidence, class_idx = torch.max(probs, dim=0)
        
    return {
        'class': CLASS_NAMES[class_idx.item()],
        'confidence': confidence.item(),
        'probabilities': probs.cpu().numpy().tolist()
    }

def predict_batch(model: torch.nn.Module, image_dir: str, device: torch.device) -> List[Dict[str, Any]]:
    """
    Predicts classes for all images in a directory.
    """
    predictions = []
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in valid_extensions]
    
    for img_path in image_paths:
        pred = predict_single(model, str(img_path), device)
        pred['file'] = img_path.name
        predictions.append(pred)
        
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Predict satellite image class")
    parser.add_argument('--image', type=str, required=True, help='Path to image or directory of images')
    parser.add_argument('--model-path', type=str, default=str(BEST_MODEL_PATH), help='Path to model checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)
    
    path = Path(args.image)
    if path.is_file():
        result = predict_single(model, str(path), device)
        print(f"Prediction for {path.name}:")
        print(f"  Class: {result['class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
    elif path.is_dir():
        results = predict_batch(model, str(path), device)
        for res in results:
            print(f"{res['file']} -> {res['class']} (Conf: {res['confidence']:.4f})")
    else:
        print("Provided path does not exist.")

if __name__ == '__main__':
    main()
