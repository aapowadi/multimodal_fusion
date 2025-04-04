import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model import MultiModalViT
import torch
import torch.nn as nn

# Example custom dataset (replace with your actual dataset)
class MultiModalDataset(Dataset):
    def __init__(self, data, transform=None):
        # data: list of tuples, each containing (list_of_5_images, label)
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images, label = self.data[idx]
        # Assume images are PIL Images or tensors
        if self.transform:
            images = [self.transform(img) for img in images]
        # Stack 5 modalities into (5, 3, 224, 224)
        x = torch.stack(images, dim=0)
        return x, label

# Preprocessing for ViT (ImageNet normalization)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training, validation, and testing function
def train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        val_accuracy = correct / total
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Checkpointing: save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy
            }, 'best_model.pth')
            print(f"Saved best model with Val Accuracy: {val_accuracy:.4f}")

    # Testing phase
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            test_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    test_accuracy = correct / total
    print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.4f}")

# Main execution
if __name__ == "__main__":
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model setup (set num_classes based on your dataset)
    num_classes = 10  # Example: replace with your number of classes
    model = MultiModalViT(num_classes=num_classes, pretrained=True).to(device)

    # Dummy data for demonstration (replace with your actual data)
    # Format: list of (list_of_5_images, label)
    dummy_data = [([torch.rand(3, 224, 224) for _ in range(5)], i % num_classes) for i in range(100)]
    train_data = dummy_data[:80]
    val_data = dummy_data[80:90]
    test_data = dummy_data[90:]

    # Create datasets and dataloaders
    train_dataset = MultiModalDataset(train_data, transform=preprocess)
    val_dataset = MultiModalDataset(val_data, transform=preprocess)
    test_dataset = MultiModalDataset(test_data, transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Train, validate, and test
    num_epochs = 10  # Adjust as needed
    train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, device)