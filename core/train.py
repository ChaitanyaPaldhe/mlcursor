# train.py - Fixed version
from core.llm_handler import extract_config, get_available_models, get_models_by_framework
from jinja2 import Environment, FileSystemLoader
from core.deps import ensure_script_dependencies
import os
import subprocess
import uuid
import json

def train_from_prompt(prompt: str):
    """Enhanced training function with multi-model support"""
    print(f"[TRAIN] Received prompt: {prompt}")
    
    # Extract enhanced config
    config = extract_config(prompt)
    
    # Validate model selection
    if not config.get("model"):
        print("❌ No model specified or model not found in registry!")
        print(f"Available models: {', '.join(get_available_models())}")
        return
    
    # Ensure other_params exists - this fixes the main error
    if "other_params" not in config:
        config["other_params"] = {}
    
    # Move any top-level params that aren't template variables to other_params
    template_vars = {'framework', 'model', 'dataset', 'optimizer', 'learning_rate', 'epochs', 'batch_size', 'model_config'}
    for key in list(config.keys()):
        if key not in template_vars and key != 'other_params':
            config['other_params'][key] = config.pop(key)
    
    print("[PARSED CONFIG]:")
    print(json.dumps(config, indent=2, default=str))
    
    # Set up template environment
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("train_template.py.j2")
    
    # Render script from enhanced config
    script = template.render(**config)
    
    # Save script to a file with UTF-8 encoding
    os.makedirs("outputs", exist_ok=True)
    script_path = f"outputs/train_{config['model']}_{uuid.uuid4().hex[:6]}.py"
    
    # Fix: Use UTF-8 encoding to handle emoji characters
    with open(script_path, "w", encoding='utf-8') as f:
        f.write(script)
    
    print(f"[SCRIPT GENERATED]: {script_path}")
    
    # Install any missing dependencies based on script imports
    ensure_script_dependencies(script_path)
    
    # Run the script in a subprocess
    try:
        print(f"[EXECUTING] Running {config['model']} training...")
        result = subprocess.run(["python", script_path], 
                              check=True, 
                              capture_output=True, 
                              text=True)
        print(result.stdout)
        if result.stderr:
            print("⚠️  Warnings:", result.stderr)
            
    except subprocess.CalledProcessError as e:
        print("❌ Error during script execution:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        print("Return code:", e.returncode)