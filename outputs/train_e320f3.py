# Common imports
import pandas as pd
import numpy as np

# Framework-specific imports

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# Config from prompt
model_name = "RandomForestClassifier"
dataset = "penguins"
optimizer = "SGD"
lr = 0.01
epochs = 10

print(f"Training {model_name} on {dataset} for {epochs} epochs using sklearn...")

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

model = RandomForestClassifier(n_estimators=epochs)
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))


