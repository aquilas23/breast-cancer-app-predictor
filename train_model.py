import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Assuming RandomForest performed best
import pickle
import os

# Load dataset from URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
columns = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
           'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
           'Normal Nucleoli', 'Mitoses', 'Class']

df = pd.read_csv(url, names=columns)

# Data Preprocessing
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df['Bare Nuclei'] = df['Bare Nuclei'].astype(int)  # Convert column to integer

# Features & Target
X = df.drop(columns=['ID', 'Class'])
y = df['Class'].map({2: 0, 4: 1})  # Convert labels to 0 (benign) and 1 (malignant)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train best model (RandomForest as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ensure model folder exists
model_path = "breast_cancer_model.pkl"  # Save model in root directory
if not os.path.exists(model_path):
    open(model_path, 'a').close()

# Save the model and scaler
with open(model_path, 'wb') as f:
    pickle.dump((scaler, model), f)

print("Model training complete. Saved as 'breast_cancer_model.pkl'.")
