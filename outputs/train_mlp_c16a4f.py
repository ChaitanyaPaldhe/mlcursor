# Common imports
import pandas as pd
import numpy as np

# Framework-specific imports based on model config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Config from prompt
model_name = "mlp"
dataset = "penguin"
optimizer = "adam"
lr = 0.001
epochs = 100

print(f"Training {model_name} on {dataset} for {epochs} epochs using pytorch...")

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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train, X_test = X_train_scaled, X_test_scaled


print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Training based on framework and model

# PyTorch model building

# Default MLP
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

model = MLP(X_train.shape[1])


# Prepare data
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

print(f"Model: {model}")
print(f"Training on {len(dataloader)} batches...")

# Training loop
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred = model(X_test_tensor)
    y_pred_binary = (y_pred > 0.5).float().squeeze()
    accuracy = (y_pred_binary == torch.tensor(y_test.values, dtype=torch.float32)).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")



print("\nTraining completed successfully!")

# Save model info
import json
model_info = {
    "model_name": model_name,
    "framework": "pytorch",
    "dataset": dataset,
    "config": {"architecture": "mlp", "framework": "tensorflow", "layers": [{"activation": "relu", "type": "dense", "units": 128}, {"rate": 0.3, "type": "dropout"}, {"activation": "relu", "type": "dense", "units": 64}, {"activation": "sigmoid", "type": "dense", "units": 1}]}
}

with open(f"outputs/model_info_{model_name}.json", "w", encoding='utf-8') as f:
    json.dump(model_info, f, indent=2)

print(f"Model info saved to outputs/model_info_{model_name}.json")