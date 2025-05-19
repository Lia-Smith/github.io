from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms (resize and normalize as needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = datasets.ImageFolder("chest_xray/train", transform=transform)
test_dataset = datasets.ImageFolder("chest_xray/test", transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)