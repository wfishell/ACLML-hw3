import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.nn as nn
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# The CIFAR-10 class names are fixed.
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

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


def load_model(model_path):
    """
    Loads the trained model.
    If the model file does not exist, a dummy model is created that returns
    a random output (by picking a random index between 0 and 9) for testing purposes.
    Utilizes the GPU if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    return model, device

def preprocess_image(image_path):
    """
    Loads and preprocesses an image.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32))
    img_array = np.array(img).astype('float32')
    img_array /= 255.0
    tensor = torch.tensor(img_array)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)
    return tensor

def predict(image_tensor, model, device):
    """
    Runs inference on the image tensor.
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_idx = outputs.argmax(dim=1).item()
    
    class_name = CIFAR10_CLASSES[predicted_idx]
    return predicted_idx, class_name


@app.route('/')
def index():
    return render_template('pred.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the file temporarily
    temp_filename = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(temp_filename)
    print(temp_filename)
    # Run the inference using the uploaded file
    model_path = "/model/cnn_model.pth"  # Update path if needed
    model, device = load_model(model_path)
    image_tensor = preprocess_image(temp_filename)
    predicted_idx, class_name = predict(image_tensor, model, device)

    # Optionally, remove the temporary file
    os.remove(temp_filename)

    return jsonify({'predicted_index': predicted_idx, 'predicted_class': class_name})

if __name__ == '__main__':
    # Decide which mode to run: Web server or CLI.
    # For example, you could check for an environment variable:
   
    app.run(debug=True, host='0.0.0.0',port=8080
)
