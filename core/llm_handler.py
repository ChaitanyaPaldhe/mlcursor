import subprocess
import json
import re

def query_ollama(prompt: str, model: str = "qwen2.5-coder:3b") -> str:
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
        print("Ollama error:", e.stderr.decode())
        return ""

def extract_config(prompt: str, model: str = "qwen2.5-coder:3b") -> dict:
    system_prompt = f"""
You are an ML code assistant. A user will give you a natural language prompt.

Extract these details in JSON:
- framework (e.g., pytorch, sklearn, xgboost)
- model (e.g., ResNet18, XGBClassifier)
- dataset (e.g., cifar10, titanic)
- optimizer (if known)
- learning_rate
- epochs
- batch_size
- other params

Prompt: {prompt}
Respond only with JSON.
"""

    response = query_ollama(system_prompt, model)

    # ✅ Strip markdown-style code blocks from LLM output
    response = response.strip()
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)


    # ✅ Remove inline comments from JSON
    response = re.sub(r"//.*", "", response)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse LLM response:", response)
        return {}

