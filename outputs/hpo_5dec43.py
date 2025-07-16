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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        
    }

    model = ModelClass(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best hyperparameters:", study.best_params)
print("Best accuracy:", study.best_value)