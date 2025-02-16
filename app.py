<<<<<<< HEAD
<<<<<<< HEAD
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load Model
with open("breast_cancer_model.pkl", "rb") as f:
    scaler, model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and check for empty fields
        features = []
        for key in request.form.keys():
            value = request.form[key]
            if value.strip() == "":  # Check for empty input
                 return jsonify({'error': 'All fields must be filled!'})

            features.append(float(value))  # Convert to float

        # Reshape, transform, and make a prediction
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)

        result = "Malignant" if prediction[0] == 1 else "Benign"
        return jsonify({'prediction': result})

    except ValueError:
        return jsonify({'error': 'Please enter valid numeric values!'})
=======
from flask import Flask, request, render_template
=======
from flask import Flask, request, render_template, jsonify
>>>>>>> f69dac6 (Update User Interface)
import pickle
import numpy as np
import os

app = Flask(__name__)

# Check if model file exists
model_path = "breast_cancer_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f" ERROR: Model file '{model_path}' not found!")

# Load model and scaler safely
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
<<<<<<< HEAD
    features = [float(x) for x in request.form.values()]
    prediction = model.predict(np.array(features).reshape(1, -1))
    return render_template("index.html", prediction_text=f"Prediction: {'Malignant' if prediction[0] == 4 else 'Benign'}")
>>>>>>> 387dfb0 (first commit)
=======
    try:
        features = [float(request.form[key]) for key in request.form.keys()]
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)
        return jsonify({'prediction': "Malignant" if prediction[0] == 1 else "Benign"})
    except ValueError:
        return jsonify({'error': 'Invalid input! Enter numeric values only.'})
>>>>>>> f69dac6 (Update User Interface)

if __name__ == "__main__":
    app.run(debug=True)
