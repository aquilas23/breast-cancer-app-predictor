import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
columns = ["Sample_code_number", "Clump_Thickness", "Uniformity_of_Cell_Size", "Uniformity_of_Cell_Shape",
           "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei", "Bland_Chromatin",
           "Normal_Nucleoli", "Mitoses", "Class"]

df = pd.read_csv(url, names=columns)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df = df.astype(float)

# Prepare data
X = df.drop(["Sample_code_number", "Class"], axis=1)
y = (df["Class"] == 4).astype(int)  # Convert to binary: Malignant (1), Benign (0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the scaler and model
with open("breast_cancer_model.pkl", "wb") as f:
    pickle.dump((scaler, model), f)

print("✅ Model training completed. 'breast_cancer_model.pkl' saved successfully.")
