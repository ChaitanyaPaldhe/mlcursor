import subprocess
import json
import re
from typing import Dict, Any, List

# Enhanced Model Registry - Fixed XGBoost entry
MODEL_REGISTRY = {
    # Sklearn models
    "random_forest": {
        "framework": "sklearn",
        "class": "RandomForestClassifier",
        "import": "from sklearn.ensemble import RandomForestClassifier",
        "default_params": {"n_estimators": 100, "random_state": 42, "max_depth": None}
    },
    "logistic_regression": {
        "framework": "sklearn", 
        "class": "LogisticRegression",
        "import": "from sklearn.linear_model import LogisticRegression",
        "default_params": {"random_state": 42, "max_iter": 1000, "C": 1.0}
    },
    "svm": {
        "framework": "sklearn",
        "class": "SVC", 
        "import": "from sklearn.svm import SVC",
        "default_params": {"random_state": 42, "C": 1.0, "gamma": "scale"}
    },
    "gradient_boosting": {
        "framework": "sklearn",
        "class": "GradientBoostingClassifier",
        "import": "from sklearn.ensemble import GradientBoostingClassifier", 
        "default_params": {"random_state": 42, "n_estimators": 100, "learning_rate": 0.1}
    },
    
    # XGBoost - Fixed the key to match what LLM returns
    "xgboost": {
        "framework": "xgboost",
        "class": "XGBClassifier",
        "import": "import xgboost as xgb",
        "default_params": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100, "random_state": 42}
    },
    
    # Add alias for common variations
    "XGBClassifier": {
        "framework": "xgboost",
        "class": "XGBClassifier", 
        "import": "import xgboost as xgb",
        "default_params": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100, "random_state": 42}
    },
    
    # LightGBM  
    "lightgbm": {
        "framework": "lightgbm",
        "class": "LGBMClassifier",
        "import": "import lightgbm as lgb",
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
    },
    
    # CatBoost
    "catboost": {
        "framework": "catboost",
        "class": "CatBoostClassifier",
        "import": "from catboost import CatBoostClassifier", 
        "default_params": {"iterations": 100, "learning_rate": 0.1, "verbose": 0}
    },
    
    # Deep Learning architectures
    "mlp": {
        "framework": "tensorflow",
        "architecture": "mlp",
        "layers": [{"type": "dense", "units": 128, "activation": "relu"},
                  {"type": "dropout", "rate": 0.3},
                  {"type": "dense", "units": 64, "activation": "relu"}, 
                  {"type": "dense", "units": 1, "activation": "sigmoid"}]
    },
    "deep_nn": {
        "framework": "tensorflow", 
        "architecture": "deep",
        "layers": [{"type": "dense", "units": 256, "activation": "relu"},
                  {"type": "dropout", "rate": 0.3},
                  {"type": "dense", "units": 128, "activation": "relu"},
                  {"type": "dropout", "rate": 0.3}, 
                  {"type": "dense", "units": 64, "activation": "relu"},
                  {"type": "dense", "units": 1, "activation": "sigmoid"}]
    },
    "pytorch_mlp": {
        "framework": "pytorch",
        "architecture": "mlp",
        "layers": [128, 64, 32, 1]
    }
}

def query_ollama(prompt: str, model: str = "qwen2.5-coder:3b") -> str:
    """Query Ollama with error handling"""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        print("ERROR: Ollama error:", e.stderr.decode())
        return ""
    except FileNotFoundError:
        print("ERROR: Ollama not found. Install ollama first: https://ollama.ai/")
        return ""

def clean_llm_response(response: str) -> str:
    """Clean LLM response by removing markdown and comments"""
    response = response.strip()
    # Remove markdown code blocks - Fix: Use raw strings for regex
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)
    # Remove inline comments
    response = re.sub(r"//.*", "", response)
    return response

def extract_config_with_ollama(prompt: str, model: str = "qwen2.5-coder:3b") -> dict:
    """Extract config using Ollama LLM with model registry awareness"""
    
    # Get available models for the LLM to choose from
    available_models = list(MODEL_REGISTRY.keys())
    
    system_prompt = f"""
You are an ML code assistant. Extract training configuration from the user prompt.

AVAILABLE MODELS: {', '.join(available_models)}

IMPORTANT: Use the exact model names from the available models list above.

Extract these details and respond ONLY with valid JSON:
{{
    "framework": "sklearn|pytorch|tensorflow|xgboost|lightgbm|catboost",
    "model": "choose from available models above - USE EXACT NAME",
    "dataset": "dataset name mentioned", 
    "optimizer": "adam|sgd|rmsprop (if mentioned)",
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32,
    "task_type": "classification|regression|auto",
    "use_cv": false,
    "cv_folds": 5,
    "cv_type": "auto|stratified|kfold",
    "other_params": {{
        "param_name": "param_value"
    }}
}}

User prompt: {prompt}

Respond only with JSON, no explanations:
"""

    response = query_ollama(system_prompt, model)
    
    if not response:
        print("WARNING: Ollama failed, falling back to regex parsing...")
        return extract_config_fallback(prompt)
    
    response = clean_llm_response(response)
    
    try:
        config = json.loads(response)
        print(f"[SUCCESS] LLM extracted config: {config}")
        
        # Map common model names to registry keys
        model_name = config.get("model", "")
        model_mappings = {
            "XGBClassifier": "xgboost",
            "RandomForestClassifier": "random_forest",
            "LogisticRegression": "logistic_regression",
            "SVC": "svm",
            "GradientBoostingClassifier": "gradient_boosting",
            "LGBMClassifier": "lightgbm",
            "LightGBM": "lightgbm",
            "CatBoostClassifier": "catboost"
        }
        
        if model_name in model_mappings:
            config["model"] = model_mappings[model_name]
        
        return config
    except json.JSONDecodeError as e:
        print(f"WARNING: Failed to parse LLM response: {e}")
        print(f"Raw response: {response}")
        print("Falling back to regex parsing...")
        return extract_config_fallback(prompt)

def extract_config_fallback(prompt: str) -> dict:
    """Fallback regex-based config extraction with CV support"""
    config = {
        "framework": None,
        "model": None,
        "dataset": None,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "task_type": "auto",
        "use_cv": False,
        "cv_folds": 5,
        "cv_type": "auto",
        "other_params": {}
    }
    
    # Extract cross-validation settings
    if re.search(r"\bcv\b|\bcross.?validation\b", prompt, re.IGNORECASE):
        config["use_cv"] = True
    
    # Extract CV folds
    cv_folds_match = re.search(r"(?:folds?|k.?fold):\s*(\d+)", prompt, re.IGNORECASE)
    if not cv_folds_match:
        cv_folds_match = re.search(r"(\d+).?fold", prompt, re.IGNORECASE)
    if cv_folds_match:
        config["cv_folds"] = int(cv_folds_match.group(1))
        config["use_cv"] = True
    
    # Extract CV type
    if re.search(r"stratified", prompt, re.IGNORECASE):
        config["cv_type"] = "stratified"
    elif re.search(r"kfold|k.fold", prompt, re.IGNORECASE):
        config["cv_type"] = "kfold"
    
    # Extract task type
    if re.search(r"regression", prompt, re.IGNORECASE):
        config["task_type"] = "regression"
    elif re.search(r"classification|classify", prompt, re.IGNORECASE):
        config["task_type"] = "classification"
    
    # Extract model name (prioritize exact matches)
    # Handle common class name mappings
    class_to_key = {
        "XGBClassifier": "xgboost",
        "RandomForestClassifier": "random_forest", 
        "LogisticRegression": "logistic_regression",
        "SVC": "svm",
        "GradientBoostingClassifier": "gradient_boosting",
        "LGBMClassifier": "lightgbm",
        "LightGBM": "lightgbm",
        "CatBoostClassifier": "catboost"
    }
    
    # First try class names
    for class_name, key in class_to_key.items():
        if re.search(rf"\b{class_name}\b", prompt, re.IGNORECASE):
            config["model"] = key
            config["framework"] = MODEL_REGISTRY[key]["framework"]
            break
    else:
        # Then try registry keys
        for model_name in MODEL_REGISTRY.keys():
            # Fix: Use raw string and proper escaping
            pattern = rf"\b{re.escape(model_name.replace('_', ' '))}\b|\b{re.escape(model_name)}\b"
            if re.search(pattern, prompt, re.IGNORECASE):
                config["model"] = model_name
                config["framework"] = MODEL_REGISTRY[model_name]["framework"]
                break
    
    # Extract dataset
    dataset_match = re.search(r"(?:dataset|data):\s*([a-zA-Z_][a-zA-Z0-9_]*)", prompt, re.IGNORECASE)
    if not dataset_match:
        dataset_match = re.search(r"on\s+(?:the\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s+dataset", prompt, re.IGNORECASE)
    if dataset_match:
        config["dataset"] = dataset_match.group(1)
    
    # Extract learning rate
    lr_match = re.search(r"(?:learning.?rate|lr):\s*([\d.]+)", prompt, re.IGNORECASE)
    if lr_match:
        config["learning_rate"] = float(lr_match.group(1))
    
    # Extract epochs
    epochs_match = re.search(r"(?:epochs?|iterations?):\s*(\d+)", prompt, re.IGNORECASE)
    if not epochs_match:
        epochs_match = re.search(r"for\s+(\d+)\s+epochs?", prompt, re.IGNORECASE)
    if epochs_match:
        config["epochs"] = int(epochs_match.group(1))
    
    # Extract batch size
    batch_match = re.search(r"batch.?size:\s*(\d+)", prompt, re.IGNORECASE)
    if batch_match:
        config["batch_size"] = int(batch_match.group(1))
    
    return config

def extract_config(prompt: str, use_ollama: bool = True, model: str = "qwen2.5-coder:3b", cv_config: dict = None) -> dict:
    """Main config extraction function with hybrid approach and CV support"""
    
    print(f"[CONFIG] Extracting from prompt: {prompt}")
    
    # Try Ollama first if requested
    if use_ollama:
        config = extract_config_with_ollama(prompt, model)
    else:
        config = extract_config_fallback(prompt)
    
    # Override with CLI CV configuration if provided (CLI takes precedence)
    if cv_config:
        print(f"[CV CONFIG] Applying CLI CV settings: {cv_config}")
        config.update({
            "use_cv": cv_config.get("use_cv", config.get("use_cv", False)),
            "cv_folds": cv_config.get("cv_folds", config.get("cv_folds", 5)),
            "cv_type": cv_config.get("cv_type", config.get("cv_type", "auto"))
        })
    
    # Ensure other_params exists
    if "other_params" not in config:
        config["other_params"] = {}
    
    # Enhance config with model registry info
    if config.get("model") and config["model"] in MODEL_REGISTRY:
        config["model_config"] = MODEL_REGISTRY[config["model"]]
        
        # Ensure framework matches
        if not config.get("framework"):
            config["framework"] = config["model_config"]["framework"]
        
        # Apply default parameters
        default_params = config["model_config"].get("default_params", {})
        
        # Start with defaults, then apply any user overrides
        for key, value in default_params.items():
            if key not in config["other_params"]:
                config["other_params"][key] = value
        
        # Extract user parameter overrides
        user_overrides = extract_parameter_overrides(prompt, default_params.keys())
        config["other_params"].update(user_overrides)
        
        print(f"[REGISTRY] Applied model config for {config['model']}")
        if user_overrides:
            print(f"[OVERRIDES] User parameters: {user_overrides}")
    
    else:
        print(f"WARNING: Model '{config.get('model')}' not found in registry")
        print(f"Available models: {list(MODEL_REGISTRY.keys())}")
    
    # Determine task type automatically if not specified
    if config.get("task_type") == "auto":
        config["task_type"] = determine_task_type(config.get("dataset"), prompt)
    
    print(f"[FINAL CONFIG] CV: {config.get('use_cv')}, Folds: {config.get('cv_folds')}, Type: {config.get('cv_type')}")
    
    return config

def determine_task_type(dataset: str, prompt: str) -> str:
    """Automatically determine if task is classification or regression"""
    
    # Check prompt for explicit indicators
    if re.search(r"regression|regress|predict.*value|continuous", prompt, re.IGNORECASE):
        return "regression"
    elif re.search(r"classification|classify|predict.*class|category", prompt, re.IGNORECASE):
        return "classification"
    
    # Dataset-based heuristics
    classification_datasets = {
        "iris", "wine", "breast_cancer", "digits", "titanic", 
        "mushroom", "heart", "spam", "adult", "sonar", "penguins"
    }
    
    regression_datasets = {
        "diabetes", "california_housing", "auto_mpg",
        "housing", "concrete", "energy", "yacht"
    }
    
    if dataset and dataset.lower() in classification_datasets:
        return "classification"
    elif dataset and dataset.lower() in regression_datasets:
        return "regression"
    
    # Default to classification (most common ML task)
    return "classification"

def extract_parameter_overrides(prompt: str, param_names: List[str]) -> dict:
    """Extract user parameter overrides from prompt"""
    overrides = {}
    
    # Common ML parameters that might be mentioned
    all_params = list(param_names) + [
        "n_estimators", "max_depth", "learning_rate", "random_state", 
        "max_iter", "C", "gamma", "kernel", "iterations", "verbose",
        "reg_alpha", "reg_lambda", "subsample", "colsample_bytree",
        "min_samples_split", "min_samples_leaf", "max_features"
    ]
    
    for param in all_params:
        patterns = [
            rf"\b{param}:\s*([\d.]+)",                    # param: 100
            rf"\b{param}=(\d+\.?\d*)",                    # param=100  
            rf"\bwith\s+{param}:\s*([\d.]+)",             # with param: 100
            rf"\bset\s+{param}\s+to\s+([\d.]+)",          # set param to 100
            rf"\b{param}\s+of\s+([\d.]+)"                 # param of 100
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if match.group(1).isdigit():
                        value = int(match.group(1))
                    overrides[param] = value
                    break
                except ValueError:
                    overrides[param] = match.group(1)
    
    return overrides

# Utility functions
def get_available_models() -> List[str]:
    """Return list of all available models"""
    return list(MODEL_REGISTRY.keys())

def get_models_by_framework(framework: str) -> List[str]:
    """Get all models for a specific framework"""
    return [name for name, config in MODEL_REGISTRY.items() 
            if config["framework"] == framework]

def validate_model_config(config: dict) -> bool:
    """Validate that the extracted config is complete"""
    required_fields = ["model", "framework", "dataset"]
    
    for field in required_fields:
        if not config.get(field):
            print(f"ERROR: Missing required field: {field}")
            return False
    
    if config["model"] not in MODEL_REGISTRY:
        print(f"ERROR: Unknown model: {config['model']}")
        return False
    
    return True