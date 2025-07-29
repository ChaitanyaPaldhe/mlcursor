from core.llm_handler import extract_config, get_available_models, get_models_by_framework
from jinja2 import Environment, FileSystemLoader
from core.deps import ensure_script_dependencies
import os
import subprocess
import uuid
import json

def train_from_prompt(prompt: str, cv_config: dict = None):
    """Enhanced training function with cross-validation support"""
    print(f"[TRAIN] Received prompt: {prompt}")
    
    # Extract enhanced config with CV support
    config = extract_config(prompt, cv_config=cv_config)
    
    # Validate model selection
    if not config.get("model"):
        print("ERROR: No model specified or model not found in registry!")
        print(f"Available models: {', '.join(get_available_models())}")
        return
    
    # Ensure other_params exists - this fixes the main error
    if "other_params" not in config:
        config["other_params"] = {}
    
    # Move any top-level params that aren't template variables to other_params
    template_vars = {
        'framework', 'model', 'dataset', 'optimizer', 'learning_rate', 'epochs', 
        'batch_size', 'model_config', 'task_type', 'use_cv', 'cv_folds', 'cv_type'
    }
    for key in list(config.keys()):
        if key not in template_vars and key != 'other_params':
            config['other_params'][key] = config.pop(key)
    
    print("[PARSED CONFIG]:")
    print(json.dumps(config, indent=2, default=str))
    
    # Set up template environment with custom filters
    env = Environment(loader=FileSystemLoader("templates"))
    
    # Add custom filter for Python boolean conversion
    def python_bool(value):
        if isinstance(value, bool):
            return str(value)
        elif isinstance(value, str):
            return str(value.lower() in ('true', 'yes', '1', 'on'))
        else:
            return str(bool(value))
    
    def python_value(value):
        """Convert values to proper Python representation"""
        if value is None:
            return "None"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, str):
            return f'"{value}"'
        else:
            return str(value)
    
    env.filters['python_bool'] = python_bool
    env.filters['python_value'] = python_value
    
    template = env.get_template("train_template.py.j2")
    
    # Render script from enhanced config
    script = template.render(**config)
    
    # Save script to a file with UTF-8 encoding
    os.makedirs("outputs", exist_ok=True)
    cv_suffix = "_cv" if config.get("use_cv") else ""
    script_path = f"outputs/train_{config['model']}{cv_suffix}_{uuid.uuid4().hex[:6]}.py"
    
    # Fix: Use UTF-8 encoding to handle emoji characters
    with open(script_path, "w", encoding='utf-8') as f:
        f.write(script)
    
    print(f"[SCRIPT GENERATED]: {script_path}")
    
    # Install any missing dependencies based on script imports
    ensure_script_dependencies(script_path)
    
    # Run the script in a subprocess with UTF-8 encoding
    try:
        cv_info = f" with {config.get('cv_folds', 5)}-fold CV" if config.get("use_cv") else ""
        print(f"[EXECUTING] Running {config['model']} training{cv_info}...")
        
        # Set environment variables for UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(["python", script_path], 
                              check=True, 
                              capture_output=True, 
                              text=True,
                              encoding='utf-8',
                              env=env)
        print(result.stdout)
        if result.stderr:
            print("WARNING:", result.stderr)
            
    except subprocess.CalledProcessError as e:
        print("ERROR: Error during script execution:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        print("Return code:", e.returncode)