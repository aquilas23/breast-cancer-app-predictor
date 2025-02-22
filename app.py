from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Path to Model File
model_path = "breast_cancer_model.pkl"

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f" ERROR: Model file '{model_path}' not found!")

# Load Model and Scaler
try:
    with open(model_path, "rb") as f:
        scaler, model = pickle.load(f)
except (EOFError, pickle.UnpicklingError) as e:
    raise ValueError(f" ERROR: Model file '{model_path}' is corrupted or empty! Please retrain the model.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get Form Data
        features = []
        for key in request.form.keys():
            value = request.form[key].strip()
            if value == "":  # Check for empty input
                return jsonify({'error': 'All fields must be filled!'})

            features.append(float(value))  # Convert to float

        # Reshape, transform, and predict
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)

        result = "Malignant" if prediction[0] == 1 else "Benign"
        return jsonify({'prediction': result})

    except ValueError:
        return jsonify({'error': 'Invalid input! Enter numeric values only.'})

if __name__ == "__main__":
    app.run(debug=True)
