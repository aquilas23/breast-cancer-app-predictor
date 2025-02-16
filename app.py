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
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict(np.array(features).reshape(1, -1))
    return render_template("index.html", prediction_text=f"Prediction: {'Malignant' if prediction[0] == 4 else 'Benign'}")
>>>>>>> 387dfb0 (first commit)

if __name__ == "__main__":
    app.run(debug=True)
