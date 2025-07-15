import typer
from core.train import train_from_prompt
from core.tune import tune_from_prompt
from core.logs import show_logs

app = typer.Typer()

@app.command()
def train(prompt: str):
    """Train an ML model from a natural language prompt."""
    train_from_prompt(prompt)

@app.command()
def tune(prompt: str):
    """Tune hyperparameters from a prompt."""
    tune_from_prompt(prompt)

@app.command()
def logs():
    """Show training logs."""
    show_logs()

if __name__ == "__main__":
    app()
