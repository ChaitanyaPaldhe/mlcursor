import joblib
import numpy as np

# Load all
model = joblib.load("G:\Codes\mlcursor\outputs\models\logistic_regression_iris.pkl")
encoder = joblib.load("G:\Codes\mlcursor\outputs\models\logistic_regression_iris_encoders.pkl")   # If you have categorical features
scaler = joblib.load("G:\Codes\mlcursor\outputs\models\logistic_regression_iris_scaler.pkl")     # If you scaled features

# Example raw input (sepal length, sepal width, petal length, petal width)
raw_features = [[6.7,2.5,5.8,1.8]]

# Step 1: Encode categorical columns (skip if none in iris dataset)
# raw_features = encoder.transform(raw_features)  # only if applicable

# Step 2: Scale numeric features
features_scaled = scaler.transform(raw_features)

# Step 3: Predict
prediction = model.predict(features_scaled)
print("Predicted:", prediction)
