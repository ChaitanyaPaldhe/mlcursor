# Common imports (always needed)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Framework-specific imports based on model config

import lightgbm as lgb


# Config from prompt
model_name = "LightGBMClassifier"
dataset = "penguins"
optimizer = "adam"
lr = 0.1
epochs = 30

print(f"Training {model_name} on {dataset} for {epochs} epochs using lightgbm...")

# Generic dataset loading logic
try:
    df = pd.read_csv(f"data/{dataset}.csv")
    print(f"Loaded dataset from data/{dataset}.csv")
except Exception as e:
    try:
        import seaborn as sns
        df = sns.load_dataset(dataset).dropna()
        print(f"Loaded {dataset} from seaborn datasets")
    except Exception as e2:
        print(f"Error loading dataset: {e2}")
        exit(1)

# Preprocessing
label_encoders = {}
for col in df.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Assume last column is target
X = df.drop(columns=[df.columns[-1]])
y = df[df.columns[-1]]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for certain models


print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Training based on framework and model

# LightGBM training  
model_params = {
    'n_estimators': epochs,
    'learning_rate': lr,
    'random_state': 42
}


model_params["objective"] = 'binary'



model_params["metric"] = 'auc'



print(f"LightGBM parameters: {model_params}")
model = lgb.LGBMClassifier(**model_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")



print("\nTraining completed successfully!")

# Save model info
import json
import os

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Prepare model info
model_info = {
    "model_name": model_name,
    "framework": "lightgbm",
    "dataset": dataset,
}


model_info["config"] = None


with open(f"outputs/model_info_{model_name}.json", "w", encoding='utf-8') as f:
    json.dump(model_info, f, indent=2)

print(f"Model info saved to outputs/model_info_{model_name}.json")