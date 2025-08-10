import os
import uuid
import subprocess
from jinja2 import Environment, FileSystemLoader
from core.llm_handler import extract_config
from core.deps import ensure_script_dependencies

def tune_from_prompt(prompt: str):
    print(f"[TUNE] Received prompt: {prompt}")

    # Extract model config from the prompt
    config = extract_config(prompt)
    print("[RAW CONFIG]:", config)

    if not config or "model" not in config or "framework" not in config:
        print("❌ Could not parse config from prompt.")
        return

    # ✅ Merge other_params into top-level config
    if "other_params" in config:
        config.update(config.pop("other_params"))

    print("[CONFIG EXTRACTED]:", config)

    # Prepare template engine
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("hpo_template.py.j2")

    # Render the HPO script using the config
    script = template.render(config=config)

    # Save script
    os.makedirs("outputs", exist_ok=True)
    script_path = f"outputs/hpo_{uuid.uuid4().hex[:6]}.py"
    with open(script_path, "w") as f:
        f.write(script)

    print(f"[HPO SCRIPT GENERATED]: {script_path}")

    # ✅ Install any missing dependencies based on script imports
    ensure_script_dependencies(script_path)

    # Run the HPO script
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print("⚠️ Error during HPO script execution:", e)