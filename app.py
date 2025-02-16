from flask import Flask, request, render_template
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
    features = [float(request.form[key]) for key in request.form.keys()]
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)

    result = "Malignant" if prediction[0] == 1 else "Benign"
    return f"<h3>The tumor is {result}</h3>"

if __name__ == "__main__":
    app.run(debug=True)
