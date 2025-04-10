import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# The CIFAR-10 class names are fixed.
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_model(model_path):
    """
    Loads the trained model.
    If the model file does not exist, a dummy model is created that returns
    a random output (by picking a random index between 0 and 9) for testing purposes.
    Utilizes the GPU if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=device)
        model.to(device)
        model.eval()
        print(f"Loaded model from {model_path}")
        return model, device
    else:
        print(f"Model file {model_path} not found. Using a dummy model for testing.")
        import random
        class DummyModel:
            def __init__(self, num_classes=10):
                self.num_classes = num_classes
            def to(self, device):
                self.device = device
                return self
            def eval(self):
                pass
            def __call__(self, x):
                # Instead of using torch.rand, pick a random integer between 0 and num_classes-1
                batch_size = x.shape[0]
                # Create a tensor of size (batch_size, ) with random integers
                random_indices = [random.randint(0, self.num_classes - 1) for _ in range(batch_size)]
                # Build a one-hot like tensor such that argmax returns the random index.
                outputs = torch.zeros(batch_size, self.num_classes, device=self.device)
                for i, idx in enumerate(random_indices):
                    outputs[i, idx] = 1.0
                return outputs
        dummy_model = DummyModel().to(device)
        return dummy_model, device

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
   
    app.run(debug=True, host='0.0.0.0',port=8111)
