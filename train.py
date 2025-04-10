from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# Load and preprocess the dataset
def load_and_preprocess_data():
    # Load the dataset
    dataset = load_dataset("cifar10")

    # Define a function to add a "pixel_values" key (this converts the PIL image to a NumPy array)
    def add_pixel_values(example):
        example["pixel_values"] = np.array(example["img"])
        return example

    # Map the function to the dataset
    dataset = dataset.map(add_pixel_values)

    # Set the format to output PyTorch tensors for pixel_values and labels
    dataset.set_format(type="torch", columns=["pixel_values", "label"])
    
    return dataset

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # Convolutional layers:
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer: kernel size 2, which halves the dimensions.
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer with a dropout probability of 50%
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers:
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Forward pass through conv layers with ReLU activations and pooling
        x = F.relu(self.conv1(x))   # (batch, 32, 32, 32)
        x = self.pool(x)            # (batch, 32, 16, 16)
        x = F.relu(self.conv2(x))   # (batch, 64, 16, 16)
        x = self.pool(x)            # (batch, 64, 8, 8)
        x = F.relu(self.conv3(x))   # (batch, 128, 8, 8)
        x = self.pool(x)            # (batch, 128, 4, 4)
        
        # Flatten the output for fully connected layers
        x = x.reshape(x.size(0), -1)  # (batch, 128*4*4)
        
        # Fully connected layers with dropout applied after the first two layers
        x = F.relu(self.fc1(x))     # (batch, 256)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))     # (batch, 128)
        x = self.dropout(x)
        x = self.fc3(x)             # (batch, num_classes)
        
        return x

# Training function that encapsulates the training loop
def train_model(dataset, num_epochs=5, batch_size=512, learning_rate=0.001):
    # 1. Decide which device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    
    model = CNN(num_classes=10).to(device)  # 2. Move model weights to the device (GPU if available)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        for batch_idx, batch in enumerate(train_loader, start=1):
            # 3. Move inputs/labels to the same device
            images = batch["pixel_values"].float().to(device)
            labels = batch["label"].to(device)

            # Normalize images to [0,1]
            images = images / 255.0
            images = images.permute(0, 3, 1, 2)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass & optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")
    
    print("Training complete!")
    return model

if __name__ == "__main__":
    # Load and preprocess data
    dataset = load_and_preprocess_data()

    # Inspect one sample from training set
    sample = dataset["train"][0]
    print("Pixel values shape:", sample["pixel_values"].shape)  # Expected: (32, 32, 3)
    print("Label:", sample["label"])
    
    # Train the model
    trained_model = train_model(dataset, num_epochs=5, batch_size=512, learning_rate=0.01)
    # After training is complete, export the model's state dict.
    local_model_path = "/model/cnn_model.pth"
    torch.save(trained_model, local_model_path)
    print(f"Model saved locally at: {local_model_path}")
    
    
    


