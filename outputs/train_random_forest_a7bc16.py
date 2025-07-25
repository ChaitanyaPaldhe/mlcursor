# Common imports
import pandas as pd
import numpy as np

# Framework-specific imports based on model config


# Config from prompt
model_name = "random_forest"
dataset = "penguins"
optimizer = "sgd"
lr = 0.01
epochs = 20

print(f"Training {model_name} on {dataset} for {epochs} epochs using scikit-learn...")

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


print("\nTraining completed successfully!")

# Save model info
import json
model_info = {
    "model_name": model_name,
    "framework": "scikit-learn",
    "dataset": dataset,
    "config": {"class": "RandomForestClassifier", "default_params": {"max_depth": null, "n_estimators": 100, "random_state": 42}, "framework": "sklearn", "import": "from sklearn.ensemble import RandomForestClassifier"}
}

with open(f"outputs/model_info_{model_name}.json", "w", encoding='utf-8') as f:
    json.dump(model_info, f, indent=2)

print(f"Model info saved to outputs/model_info_{model_name}.json")