import argparse
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch import nn, optim
from collections import OrderedDict

# Function to get the model
def get_model(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    else:
        raise ValueError("Architecture not supported.")
    return model

# Function to save the checkpoint
def save_checkpoint(model, optimizer, epochs, arch, hidden_units, path):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx
    }
    torch.save(checkpoint, path)

# Training function
def train_model(model, criterion, optimizer, trainloader, validloader, epochs, gpu):
    device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0
        corrects = 0
        total = 0
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                valid_loss += criterion(output, labels).item()
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                corrects += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {running_loss/len(trainloader):.3f}")
        print(f"Validation Loss: {valid_loss/len(validloader):.3f}")
        print(f"Validation Accuracy: {corrects/total:.3f}")
    
# Main function to parse arguments and start training
def main():
    parser = argparse.ArgumentParser(description='Train a neural network on image data')
    parser.add_argument('data_directory', type=str, help='Directory of the image dataset')
    parser.add_argument('save_dir', type=str, help='Directory to save the model checkpoint')
    parser.add_argument('--arch', type=str, choices=['vgg16', 'densenet121'], default='vgg16', help='Model architecture')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    # Data loading
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(root=args.data_directory + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(root=args.data_directory + '/valid', transform=train_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    # Model, criterion, optimizer
    model = get_model(args.arch, args.hidden_units)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_model(model, criterion, optimizer, trainloader, validloader, args.epochs, args.gpu)
    save_checkpoint(model, optimizer, args.epochs, args.arch, args.hidden_units, os.path.join(args.save_dir, 'checkpoint.pth'))

if __name__ == '__main__':
    main()
