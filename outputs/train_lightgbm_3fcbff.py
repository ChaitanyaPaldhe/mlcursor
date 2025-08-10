# Common imports (always needed)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Cross-validation imports

from sklearn.model_selection import train_test_split


# Visualization imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.visualizer import create_visualizations
    VISUALIZATIONS_AVAILABLE = True
    print("🎨 Visualizations enabled")
except ImportError as e:
    VISUALIZATIONS_AVAILABLE = False
    print(f"⚠️  Visualizations not available: {e}")

# Framework-specific imports based on model config


# Config from prompt
model_name = "lightgbm"
dataset = "penguins"
optimizer = "adam"
lr = 0.01
epochs = 50
task_type = "classification"
use_cv = False
cv_folds = 5
cv_type = "auto"

print(f"Training {model_name} on {dataset} dataset")
print(f"Task type: {task_type}")
if use_cv:
    print(f"Using {cv_folds}-fold cross-validation ({cv_type})")
else:
    print("Using train/test split")
print(f"Framework: xgboost|lightgbm|catboost")

# Generic dataset loading logic
try:
    df = pd.read_csv(f"data/{dataset}.csv")
    print(f"[SUCCESS] Loaded dataset from data/{dataset}.csv")
except Exception as e:
    try:
        import seaborn as sns
        df = sns.load_dataset(dataset).dropna()
        print(f"[SUCCESS] Loaded {dataset} from seaborn datasets")
    except Exception as e2:
        try:
            from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
            dataset_loaders = {
                'iris': load_iris,
                'wine': load_wine, 
                'breast_cancer': load_breast_cancer,
                'digits': load_digits,
                'diabetes': load_diabetes
            }
            
            if dataset.lower() in dataset_loaders and dataset_loaders[dataset.lower()]:
                sklearn_data = dataset_loaders[dataset.lower()]()
                df = pd.DataFrame(sklearn_data.data, columns=sklearn_data.feature_names if hasattr(sklearn_data, 'feature_names') else [f'feature_{i}' for i in range(sklearn_data.data.shape[1])])
                df['target'] = sklearn_data.target
                print(f"[SUCCESS] Loaded {dataset} from sklearn datasets")
            else:
                print(f"[ERROR] Dataset '{dataset}' not found. Available: iris, wine, breast_cancer, digits, diabetes")
                exit(1)
        except Exception as e3:
            print(f"[ERROR] Error loading dataset: {e3}")
            exit(1)

# Preprocessing
print(f"Dataset shape: {df.shape}")
print(f"Dataset columns: {list(df.columns)}")

# Handle categorical columns
label_encoders = {}
for col in df.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Assume last column is target (or 'target' column if exists)
if 'target' in df.columns:
    X = df.drop(columns=['target'])
    y = df['target']
else:
    X = df.drop(columns=[df.columns[-1]])
    y = df[df.columns[-1]]

# Auto-detect task type if not specified
if task_type == "auto":
    unique_targets = len(y.unique())
    if unique_targets <= 20 and y.dtype in ['int64', 'object', 'category']:
        task_type = "classification"
        print(f"[INFO] Auto-detected task type: classification ({unique_targets} classes)")
    else:
        task_type = "regression"
        print(f"[INFO] Auto-detected task type: regression (continuous target)")

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
if task_type == "classification":
    print(f"Classes: {sorted(y.unique())}")
else:
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")

# Scale features for certain models


# Cross-validation setup

# Traditional train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, 
                                                   stratify=y if task_type == "classification" else None)
print(f"[SUCCESS] Train/test split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")


# Training based on framework and model


print("\n[SUCCESS] Training completed successfully!")

# ═══════════════════════════════════════════════════════════════
# 🎨 VISUALIZATION GENERATION
# ═══════════════════════════════════════════════════════════════

if VISUALIZATIONS_AVAILABLE:
    print("\n" + "="*60)
    print("🎨 GENERATING VISUALIZATIONS")
    print("="*60)
    
    try:
        # Prepare CV scores if available
        cv_scores_for_viz = cv_scores.tolist() if 'cv_scores' in locals() else None
        
        # Generate visualizations
        saved_plots, visualizer = create_visualizations(
            model=model,
            X=X,
            y=y,
            model_name=model_name,
            task_type=task_type,
            cv_scores=cv_scores_for_viz
        )
        
        # Generate summary report
        model_info = {
            "model_name": model_name,
            "framework": "xgboost|lightgbm|catboost",
            "dataset": dataset,
            "task_type": task_type,
            "cross_validation": {
                "enabled": use_cv,
                "folds": cv_folds if use_cv else None,
                "strategy": cv_type if use_cv else None
            }
        }
        
        if 'cv_scores' in locals():
            model_info["cv_results"] = {
                "mean_score": float(mean_score),
                "std_score": float(std_score),
                "individual_scores": cv_scores_for_viz,
                "scoring_metric": scoring_name
            }
        
        report_path = visualizer.generate_summary_report(model_info, saved_plots, model_name)
        
        print(f"\n✅ Visualization complete! Generated {len(saved_plots)} plots")
        print(f"📁 Plots saved in: {visualizer.output_dir}")
        print(f"📋 Summary report: {report_path}")
        
    except Exception as e:
        print(f"⚠️  Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

# Save model info and results
import json
import os

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Prepare model info for JSON serialization
model_info_json = {
    "model_name": model_name,
    "framework": "xgboost|lightgbm|catboost",
    "dataset": dataset,
    "task_type": task_type,
    "cross_validation": {
        "enabled": use_cv,
        "folds": cv_folds if use_cv else None,
        "strategy": cv_type if use_cv else None
    }
}




# Add model config if available
config_data = {
    "class": "LGBMClassifier",
    "framework": "lightgbm",
    "import": "import lightgbm as lgb",
    "default_params": {
        
        "n_estimators": 100,
        
        "learning_rate": 0.1,
        
        "random_state": 42,
        
    }
}
model_info_json["config"] = config_data


# Add training parameters
model_info_json["training_params"] = {
    "learning_rate": lr,
    "epochs": epochs,
    "optimizer": optimizer,
    "other_params": {
        
        "feature_fraction_seed": 24,
        
        "bagging_seed": 19,
        
        "n_estimators": 100,
        
        "learning_rate": 0.1,
        
        "random_state": 42,
        
    }
}

filename_suffix = "_cv" if use_cv else ""
with open(f"outputs/model_info_{model_name}{filename_suffix}.json", "w", encoding='utf-8') as f:
    json.dump(model_info_json, f, indent=2)

print(f"\n💾 Model info saved to outputs/model_info_{model_name}{filename_suffix}.json")

print("\n🎉 Training and visualization pipeline completed successfully!")