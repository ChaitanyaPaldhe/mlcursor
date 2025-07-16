import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.ensemble import RandomForestClassifier
ModelClass = RandomForestClassifier


# Load dataset
try:
    df = pd.read_csv("data/penguins.csv")
except:
    import seaborn as sns
    df = sns.load_dataset("penguins").dropna()

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical columns
categoricals = df.select_dtypes(include=['object', 'category']).columns
for col in categoricals:
    df[col] = df[col].astype('category').cat.codes
X = df.drop(columns=['target']) if 'target' in df.columns else df.iloc[:, :-1]
y = df['target'] if 'target' in df.columns else df.iloc[:, -1]

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        
        "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
    }

    scores = []
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        scores.append(accuracy_score(y_valid, preds))

    return sum(scores) / len(scores)


    model = ModelClass(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc



study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best hyperparameters:", study.best_params)
print("Best accuracy:", study.best_value)