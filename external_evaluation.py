from flask import Flask, request, jsonify, render_template
import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

class TrafficConeClassifier(nn.Module):
    def __init__(self):
        super(TrafficConeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # For 128x128 input size
        self.fc2 = nn.Linear(128, 2)  # Binary classification (2 classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# # Load the trained model
# model = TrafficConeClassifier()  # Ensure this matches your architecture
# model.load_state_dict(torch.load("traffic_cone_classifier.pth"))
# model.eval()
#
# # Define the same transformations as used during training
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
#
# def preprocess_image(image):
#     """
#     Preprocess the input image for the model.
#     """
#     image = Image.open(image).convert("RGB")
#     return transform(image).unsqueeze(0)  # Add batch dimension
#
# def predict(image, model):
#     """
#     Perform inference on the input image.
#     """
#     with torch.no_grad():
#         image_tensor = preprocess_image(image)
#         output = model(image_tensor)
#         _, predicted = torch.max(output, 1)  # Get the predicted class index
#     class_labels = {0: "Contains Cone", 1: "No Cone"}
#     return class_labels[predicted.item()]
#
# # Flask app
# app = Flask(__name__)
#
# @app.route("/predict", methods=["POST"])
# def classify_image():
#     """
#     API endpoint to classify an uploaded image.
#     """
#     if "file" not in request.files:
#         return jsonify({"error": "No file part in the request"}), 400
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#     try:
#         prediction = predict(file, model)
#         print(prediction)
#         return jsonify({"prediction": prediction})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# if __name__ == "__main__":
#     app.run(host='127.0.0.1', port=5000, debug=True)
# Load model
model = TrafficConeClassifier()
model.load_state_dict(torch.load("traffic_cone_classifier.pth"))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict(image_file, model):
    image = Image.open(image_file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    class_labels = {0: "Contains Cone", 1: "No Cone"}
    return class_labels[predicted.item()]

# Flask App
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file selected"
        file = request.files["file"]
        if file.filename == "":
            return "No file selected"
        try:
            result = predict(file, model)
            return render_template("result.html", prediction=result)
        except Exception as e:
            return f"Error: {e}"
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)