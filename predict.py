import argparse
import torch
from PIL import Image
from torchvision import transforms
import json
import os

# Function to load a checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    in_features = model.classifier[0].in_features if hasattr(model.classifier, 'in_features') else model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(checkpoint['hidden_units'], 102),
        nn.LogSoftmax(dim=1)
    )
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

# Function to print predictions
def print_predictions(image_path, checkpoint_path, top_k=5, category_names_path=None, use_gpu=False):
    model = load_checkpoint(checkpoint_path)
    
    if use_gpu and torch.cuda.is_available():
        model.to('cuda')
    
    probs, classes = predict(image_path, model, top_k=top_k)
    
    if category_names_path:
        with open(category_names_path, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(c)] for c in classes]
    
    print(f"Top {top_k} predictions:")
    for prob, cls in zip(probs, classes):
        print(f"{cls}: {prob:.3f}")

# Main function to parse arguments and call print_predictions
def main():
    parser = argparse.ArgumentParser(description='Predict image class and print results')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()
    
    print_predictions(args.image_path, args.checkpoint, top_k=args.top_k, 
                      category_names_path=args.category_names, use_gpu=args.gpu)

if __name__ == '__main__':
    main()
