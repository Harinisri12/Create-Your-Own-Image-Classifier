import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import json
import os

# Utility function to save the checkpoint
def save_checkpoint(model, optimizer, epochs, train_data, filepath='checkpoint.pth'):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'epochs': epochs,
        'optim_stat_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

# Function to load the checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Define training function
def train_model(model, criterion, optimizer, dataloaders, num_epochs=5):
    model.to('cuda')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(dataloaders['train'].dataset)
        valid_loss, accuracy = validate_model(model, criterion, dataloaders)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.3f}")
        print(f"Valid Loss: {valid_loss:.3f}")
        print(f"Accuracy: {accuracy:.3f}%\n")

def validate_model(model, criterion, dataloaders):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    valid_loss = running_loss / len(dataloaders['valid'].dataset)
    accuracy = (running_corrects.double() / len(dataloaders['valid'].dataset)) * 100
    return valid_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Train a deep learning model')
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    image_datasets = {
        'train': datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=data_transforms['train']),
        'valid': datasets.ImageFolder(root=os.path.join(args.data_dir, 'valid'), transform=data_transforms['valid']),
        'test': datasets.ImageFolder(root=os.path.join(args.data_dir, 'test'), transform=data_transforms['test'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=32, shuffle=False),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False)
    }
    
    model = models.__dict__[args.arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
        nn.Linear(25088, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    if args.gpu and torch.cuda.is_available():
        model.to('cuda')
    
    train_model(model, criterion, optimizer, dataloaders, num_epochs=args.epochs)
    
    save_checkpoint(model, optimizer, args.epochs, image_datasets['train'], filepath=os.path.join(args.save_dir, 'checkpoint.pth'))

if __name__ == '__main__':
    main()
