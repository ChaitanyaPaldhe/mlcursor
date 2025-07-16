# Common imports
import pandas as pd
import numpy as np

# Framework-specific imports


# Config from prompt
model_name = "RandomForestClassifier"
dataset = "penguins"
optimizer = "None"
lr = 0.0
epochs = None

print(f"Training {model_name} on {dataset} for {epochs} epochs using scikit-learn...")

# Generic tabular dataset loading logic (CSV assumed)
try:
    df = pd.read_csv(f"data/{dataset}.csv")
except Exception as e:
    import seaborn as sns
    df = sns.load_dataset(dataset).dropna()

# Preprocessing
label_encoders = {}
for col in df.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df.drop(columns=[df.columns[-1]])
y = df[df.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training per framework


