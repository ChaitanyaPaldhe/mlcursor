from core.llm_handler import extract_config
from jinja2 import Environment, FileSystemLoader
from core.deps import ensure_script_dependencies
import os
import subprocess
import uuid

def train_from_prompt(prompt: str):
    print(f"[TRAIN] Received prompt: {prompt}")
    config = extract_config(prompt)

    # ✅ Merge other_params into top-level config
    if "other_params" in config:
        config.update(config.pop("other_params"))

    print("[PARSED CONFIG]:", config)

    # Set up template environment
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("train_template.py.j2")

    # Render script from config
    script = template.render(**config)

    # Save script to a file
    os.makedirs("outputs", exist_ok=True)
    script_path = f"outputs/train_{uuid.uuid4().hex[:6]}.py"
    with open(script_path, "w") as f:
        f.write(script)

    print(f"[SCRIPT GENERATED]: {script_path}")

    # ✅ Install any missing dependencies based on script imports
    ensure_script_dependencies(script_path)

    # Run the script in a subprocess
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print("⚠️ Error during script execution:", e)



