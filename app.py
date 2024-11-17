# Import necessary modules from Flask for creating the web server and handling requests
from flask import Flask, request, jsonify
# Import the torch module for PyTorch functionality
import torch
# Import functional module from torch.nn to access the softmax function
import torch.nn.functional as F

# Create a Flask application object
app = Flask(__name__)

# Load your trained TorchScript model from a file
model = torch.jit.load('iris_model.pt')


# Define a route for the predict endpoint, which accepts POST requests
@app.route('/predict', methods=['POST'])
def predict():
    # Extract JSON data from the incoming request
    data = request.get_json()
    # Convert the data into a PyTorch tensor with the specified data type
    inputs = torch.tensor(data['inputs'], dtype=torch.float32)

    # Disable gradient calculations to improve performance and reduce memory usage during inference
    with torch.no_grad():
        # Feed the inputs to the model and get raw output logits
        logits = model(inputs)
        # Apply the softmax function to convert logits to probabilities
        probabilities = F.softmax(logits, dim=1)
        # Find the class with the highest probability
        predicted_class = probabilities.argmax(1)

    # Return the prediction as a JSON response
    return jsonify({'prediction': predicted_class.item()})


# Check if the script is executed directly (i.e., not imported), and run the app
if __name__ == '__main__':
    # Start the Flask application with debugging enabled and bound to all network interfaces on port 8080
    app.run(debug=True, host='0.0.0.0', port=8080)
