import typer
from typing import List, Optional
from core.train import train_from_prompt
from core.tune import tune_from_prompt
from core.logs import show_logs
from core.llm_handler import get_available_models, get_models_by_framework, MODEL_REGISTRY
import json
import os
import time
import re
import sys
from io import StringIO

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

@app.command()
def compare(
    prompt: str,
    models: Optional[str] = typer.Option(None, "--models", "-m", help="Models to compare (comma-separated)"),
    framework: Optional[str] = typer.Option(None, "--framework", "-f", help="Compare all models from a specific framework"),
    save_results: bool = typer.Option(True, "--save/--no-save", help="Save comparison results to file")
):
    """Compare multiple models on the same dataset."""
    
    # Determine which models to compare
    if models:
        model_list = [m.strip() for m in models.split(",")]
    elif framework:
        model_list = get_models_by_framework(framework)
        if not model_list:
            print(f"❌ No models found for framework: {framework}")
            print(f"Available frameworks: {set(config['framework'] for config in MODEL_REGISTRY.values())}")
            return
    else:
        # Default comparison set
        model_list = ["random_forest", "xgboost", "lightgbm", "logistic_regression"]
    
    # Validate models
    available_models = get_available_models()
    invalid_models = [m for m in model_list if m not in available_models]
    if invalid_models:
        print(f"❌ Invalid models: {', '.join(invalid_models)}")
        print(f"Available models: {', '.join(available_models)}")
        return
    
    print(f"🏁 Model Comparison: Testing {len(model_list)} models")
    print(f"Models: {', '.join(model_list)}")
    print(f"Prompt: {prompt}\n")
    
    results = {}
    detailed_results = {}
    
    for i, model in enumerate(model_list, 1):
        print(f"\n[{i}/{len(model_list)}] Training {model}...")
        
        # Modify prompt to specify current model
        if "model:" in prompt.lower():
            # Replace existing model specification
            model_prompt = re.sub(r'model:\s*\w+', f'model: {model}', prompt, flags=re.IGNORECASE)
        else:
            # Add model specification
            model_prompt = f"model: {model} {prompt}"
        
        start_time = time.time()
        
        try:
            # Capture output by redirecting stdout temporarily
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            train_from_prompt(model_prompt)
            
            sys.stdout = old_stdout
            output = captured_output.getvalue()
            
            # Extract accuracy from output
            accuracy = extract_accuracy_from_output(output)
            training_time = time.time() - start_time
            
            results[model] = {
                "status": "✅ Success",
                "accuracy": accuracy,
                "time": f"{training_time:.2f}s"
            }
            
            detailed_results[model] = {
                "status": "success",
                "accuracy": accuracy,
                "training_time": training_time,
                "framework": MODEL_REGISTRY[model]["framework"],
                "output": output
            }
            
            print(f"✅ {model}: {accuracy:.4f} accuracy in {training_time:.2f}s")
            
        except Exception as e:
            training_time = time.time() - start_time
            results[model] = {
                "status": f"❌ Failed: {str(e)[:50]}...",
                "accuracy": 0.0,
                "time": f"{training_time:.2f}s"
            }
            
            detailed_results[model] = {
                "status": "failed",
                "error": str(e),
                "training_time": training_time,
                "framework": MODEL_REGISTRY[model]["framework"]
            }
            
            print(f"❌ {model}: Failed - {str(e)[:50]}...")
    
    # Display results table
    display_comparison_table(results)
    
    # Save results if requested
    if save_results:
        save_comparison_results(prompt, model_list, detailed_results)

def extract_accuracy_from_output(output: str) -> float:
    """Extract accuracy value from training output."""
    # Look for various accuracy patterns
    patterns = [
        r"Accuracy:\s*([\d.]+)",
        r"Test Accuracy:\s*([\d.]+)", 
        r"accuracy.*?:\s*([\d.]+)",
        r"acc.*?:\s*([\d.]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    return 0.0

def display_comparison_table(results: dict):
    """Display comparison results in a formatted table."""
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'Status':<25} {'Accuracy':<12} {'Time':<10}")
    print("-"*80)
    
    # Sort by accuracy (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    for model, result in sorted_results:
        accuracy_str = f"{result['accuracy']:.4f}" if result['accuracy'] > 0 else "N/A"
        print(f"{model:<20} {result['status']:<25} {accuracy_str:<12} {result['time']:<10}")
    
    print("="*80)
    
    # Show winner
    if sorted_results and sorted_results[0][1]["accuracy"] > 0:
        winner = sorted_results[0][0]
        best_acc = sorted_results[0][1]["accuracy"]
        print(f"🏆 Best Model: {winner} with {best_acc:.4f} accuracy")

def save_comparison_results(prompt: str, models: List[str], results: dict):
    """Save comparison results to JSON file."""
    os.makedirs("outputs", exist_ok=True)
    
    comparison_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "models_compared": models,
        "results": results,
        "summary": {
            "total_models": len(models),
            "successful_models": len([r for r in results.values() if r["status"] == "success"]),
            "best_model": max(results.items(), key=lambda x: x[1].get("accuracy", 0))[0] if results else None
        }
    }
    
    filename = f"outputs/comparison_{int(time.time())}.json"
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")

@app.command()
def list_models(framework: Optional[str] = typer.Option(None, "--framework", "-f", help="Filter by framework")):
    """List all available models."""
    
    if framework:
        models = get_models_by_framework(framework)
        if not models:
            print(f"❌ No models found for framework: {framework}")
            return
        print(f"📋 Models for {framework}:")
        for model in models:
            print(f"  • {model}")
    else:
        print("📋 All Available Models:")
        
        # Group by framework
        frameworks = {}
        for model_name in get_available_models():
            fw = MODEL_REGISTRY[model_name]["framework"]
            if fw not in frameworks:
                frameworks[fw] = []
            frameworks[fw].append(model_name)
        
        for fw, models in frameworks.items():
            print(f"\n🔧 {fw.upper()}:")
            for model in models:
                print(f"  • {model}")
    
    print("\n💡 Usage examples:")
    print("  python cli.py train 'model: random_forest dataset: iris'")
    print("  python cli.py compare 'dataset: wine' --models random_forest,xgboost,svm")
    print("  python cli.py compare 'dataset: iris' --framework sklearn")

@app.command()
def benchmark(
    dataset: str = typer.Argument(..., help="Dataset to benchmark on"),
    frameworks: str = typer.Option("sklearn,xgboost", "--frameworks", "-f", help="Frameworks to include (comma-separated)"),
    top_n: int = typer.Option(3, "--top", "-n", help="Number of top models per framework")
):
    """Run a comprehensive benchmark across frameworks."""
    
    framework_list = [f.strip() for f in frameworks.split(",")]
    
    print(f"🏃‍♂️ Benchmarking on {dataset} dataset")
    print(f"Frameworks: {', '.join(framework_list)}")
    
    all_models = []
    for fw in framework_list:
        fw_models = get_models_by_framework(fw)[:top_n]  # Top N per framework
        all_models.extend(fw_models)
    
    if not all_models:
        print("❌ No models found for specified frameworks")
        return
    
    # Run comparison with enhanced prompt
    benchmark_prompt = f"dataset: {dataset} epochs: 50"
    
    # Call the compare function directly
    compare(benchmark_prompt, models=",".join(all_models), save_results=True)

if __name__ == "__main__":
    app()