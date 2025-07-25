# 🧠 MLCursor CLI — AI-Powered ML Developer in Your Terminal

MLCursor is a blazing-fast, privacy-first CLI tool that turns plain English into working machine learning code. It's like Cursor, but for your **terminal** — built specifically for training, tuning, and automating ML workflows with **one command**.

---

## 💡 What It Does

Understand prompts like:

> "Train an XGBoost classifier on the Titanic dataset with 50 epochs and max_depth 4"

✅ Auto-generates full training scripts using:

- PyTorch  
- Scikit-learn  
- XGBoost  
- LightGBM  
- CatBoost  
- TensorFlow (more to come)

✅ Installs all missing libraries so you don’t have to.

✅ Fully local — runs entirely on your system using models from Ollama (e.g., `qwen2.5-codeb3`).

✅ Keeps output scripts in a clean `outputs/` folder so you can inspect or reuse.

---

## ⚡ Quickstart

```bash
# 1. Install dependencies
pip install typer jinja2 pandas scikit-learn

# 2. Pull a local model
ollama pull qwen2.5-codeb3

# 3. Run your first ML pipeline
python cli.py train "Train a RandomForestClassifier on the penguins dataset for 20 epochs"

# 4. Tune 
python cli.py tune "Tune a decision tree on penguins dataset"

📁 Project Structure
mlcursor/
├── cli.py                  # Typer CLI entry
├── core/
│   ├── train.py            # Training logic
│   ├── tune.py             # (coming soon)
│   ├── deps.py             # Dependency handler
│   ├── llm_handler.py      # Query local LLMs
│   └── logs.py             # (coming soon)
├── templates/
│   └── train_template.py.j2  # Jinja2 template for ML training
|   └── hpo_template.py.j2  # HPO template for tuning
├── outputs/                # Auto-generated training scripts
├── data/                   # Place custom CSVs here
├── config.yaml             # Configurations
└── README.md

