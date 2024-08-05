import argparse
import torch
from PIL import Image
from torchvision import transforms
import json
import os

# Function to load a checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Function to process the image
def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

# Function to make predictions
def predict(image_path, model, top_k=5):
    model.eval()
    image = process_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(top_k, dim=1)
    return probs[0].tolist(), classes[0].tolist()

def main():
    parser = argparse.ArgumentParser(description='Predict image class')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()
    
    model = load_checkpoint(args.checkpoint)
    
    if args.gpu and torch.cuda.is_available():
        model.to('cuda')
    
    probs, classes = predict(args.image_path, model, top_k=args.top_k)
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(c)] for c in classes]
    
    print(f"Top {args.top_k} classes:")
    for prob, cls in zip(probs, classes):
        print(f"{cls}: {prob:.3f}")

if __name__ == '__main__':
    main()
