import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
from models.resnet_wrapper import get_resnet_model
from tqdm import tqdm


class ChestXRayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx]['filename'])

        try:
            image = Image.open(img_name).convert('RGB')
        except:
            # If image missing, use blank image instead of crashing
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        label = torch.tensor([float(self.df.iloc[idx]['label'])])

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model():
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = ChestXRayDataset(
        csv_file='data/subset_metadata.csv',
        img_dir='data/images',
        transform=transform
    )

    # Train-validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Model
    model = get_resnet_model(pretrained=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Starting training on {device}...")

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Loss: {epoch_loss:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/resnet18_weights.pth')
    print("✅ Model saved to models/resnet18_weights.pth")


if __name__ == "__main__":
    train_model()