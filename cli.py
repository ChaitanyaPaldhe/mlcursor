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
def train(
    prompt: str,
    use_cv: bool = typer.Option(False, "--cv/--no-cv", help="Use cross-validation instead of train/test split"),
    cv_folds: int = typer.Option(5, "--folds", "-k", help="Number of cross-validation folds"),
    cv_type: Optional[str] = typer.Option(None, "--cv-type", help="CV type: 'stratified', 'kfold', or 'auto' (default)")
):
    """Train an ML model from a natural language prompt."""
    
    # Clean the prompt - remove any CLI flags that got mixed in
    cleaned_prompt = re.sub(r'\s*--\w+(?:\s+\d+)?', '', prompt).strip()
    
    # Add CV parameters to the prompt context
    cv_config = {
        "use_cv": use_cv,
        "cv_folds": cv_folds,
        "cv_type": cv_type or "auto"
    }
    
    print(f"🎯 Training with CV: {use_cv}, Folds: {cv_folds}")
    train_from_prompt(cleaned_prompt, cv_config=cv_config)

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
    save_results: bool = typer.Option(True, "--save/--no-save", help="Save comparison results to file"),
    use_cv: bool = typer.Option(False, "--cv/--no-cv", help="Use cross-validation for comparison"),
    cv_folds: int = typer.Option(5, "--folds", "-k", help="Number of cross-validation folds")
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
    
    cv_suffix = f" ({cv_folds}-fold CV)" if use_cv else ""
    print(f"🏁 Model Comparison: Testing {len(model_list)} models{cv_suffix}")
    print(f"Models: {', '.join(model_list)}")
    print(f"Prompt: {prompt}\n")
    
    results = {}
    detailed_results = {}
    
    # CV configuration
    cv_config = {
        "use_cv": use_cv,
        "cv_folds": cv_folds,
        "cv_type": "auto"
    }
    
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
            
            train_from_prompt(model_prompt, cv_config=cv_config)
            
            sys.stdout = old_stdout
            output = captured_output.getvalue()
            
            # Extract metrics from output
            if use_cv:
                accuracy, std_dev = extract_cv_metrics_from_output(output)
                results[model] = {
                    "status": "✅ Success",
                    "accuracy": accuracy,
                    "std_dev": std_dev,
                    "time": f"{time.time() - start_time:.2f}s"
                }
                print(f"✅ {model}: {accuracy:.4f} ± {std_dev:.4f} accuracy")
            else:
                accuracy = extract_accuracy_from_output(output)
                training_time = time.time() - start_time
                results[model] = {
                    "status": "✅ Success",
                    "accuracy": accuracy,
                    "time": f"{training_time:.2f}s"
                }
                print(f"✅ {model}: {accuracy:.4f} accuracy in {training_time:.2f}s")
            
            detailed_results[model] = {
                "status": "success",
                "accuracy": accuracy,
                "std_dev": std_dev if use_cv else None,
                "training_time": time.time() - start_time,
                "framework": MODEL_REGISTRY[model]["framework"],
                "output": output,
                "cross_validation": use_cv,
                "cv_folds": cv_folds if use_cv else None
            }
            
        except Exception as e:
            training_time = time.time() - start_time
            results[model] = {
                "status": f"❌ Failed: {str(e)[:50]}...",
                "accuracy": 0.0,
                "std_dev": 0.0 if use_cv else None,
                "time": f"{training_time:.2f}s"
            }
            
            detailed_results[model] = {
                "status": "failed",
                "error": str(e),
                "training_time": training_time,
                "framework": MODEL_REGISTRY[model]["framework"],
                "cross_validation": use_cv
            }
            
            print(f"❌ {model}: Failed - {str(e)[:50]}...")
    
    # Display results table
    display_comparison_table(results, use_cv)
    
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

def extract_cv_metrics_from_output(output: str) -> tuple:
    """Extract cross-validation mean accuracy and standard deviation from output."""
    # Look for CV results patterns
    patterns = [
        r"Mean CV Accuracy:\s*([\d.]+)\s*±\s*([\d.]+)",
        r"CV Mean:\s*([\d.]+),\s*Std:\s*([\d.]+)",
        r"Cross-validation accuracy:\s*([\d.]+)\s*\(\s*±\s*([\d.]+)\s*\)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2))
    
    # Fallback: look for just mean accuracy
    accuracy = extract_accuracy_from_output(output)
    return accuracy, 0.0

def display_comparison_table(results: dict, use_cv: bool = False):
    """Display comparison results in a formatted table."""
    print("\n" + "="*90)
    print("📊 MODEL COMPARISON RESULTS")
    print("="*90)
    
    if use_cv:
        print(f"{'Model':<20} {'Status':<25} {'CV Accuracy ± Std':<20} {'Time':<10}")
    else:
        print(f"{'Model':<20} {'Status':<25} {'Accuracy':<12} {'Time':<10}")
    print("-"*90)
    
    # Sort by accuracy (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    for model, result in sorted_results:
        if use_cv and result.get("std_dev") is not None:
            accuracy_str = f"{result['accuracy']:.4f} ± {result['std_dev']:.4f}" if result['accuracy'] > 0 else "N/A"
        else:
            accuracy_str = f"{result['accuracy']:.4f}" if result['accuracy'] > 0 else "N/A"
        
        print(f"{model:<20} {result['status']:<25} {accuracy_str:<20} {result['time']:<10}")
    
    print("="*90)
    
    # Show winner
    if sorted_results and sorted_results[0][1]["accuracy"] > 0:
        winner = sorted_results[0][0]
        best_acc = sorted_results[0][1]["accuracy"]
        if use_cv and sorted_results[0][1].get("std_dev"):
            best_std = sorted_results[0][1]["std_dev"]
            print(f"🏆 Best Model: {winner} with {best_acc:.4f} ± {best_std:.4f} CV accuracy")
        else:
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
    print("  python cli.py train 'model: xgboost dataset: wine' --cv --folds 10")
    print("  python cli.py compare 'dataset: wine' --models random_forest,xgboost,svm --cv")
    print("  python cli.py compare 'dataset: iris' --framework sklearn --cv --folds 5")

@app.command()
def benchmark(
    dataset: str = typer.Argument(..., help="Dataset to benchmark on"),
    frameworks: str = typer.Option("sklearn,xgboost", "--frameworks", "-f", help="Frameworks to include (comma-separated)"),
    top_n: int = typer.Option(3, "--top", "-n", help="Number of top models per framework"),
    use_cv: bool = typer.Option(True, "--cv/--no-cv", help="Use cross-validation for benchmarking"),
    cv_folds: int = typer.Option(5, "--folds", "-k", help="Number of cross-validation folds")
):
    """Run a comprehensive benchmark across frameworks."""
    
    framework_list = [f.strip() for f in frameworks.split(",")]
    
    cv_suffix = f" with {cv_folds}-fold CV" if use_cv else ""
    print(f"🏃‍♂️ Benchmarking on {dataset} dataset{cv_suffix}")
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
    compare(benchmark_prompt, models=",".join(all_models), save_results=True, use_cv=use_cv, cv_folds=cv_folds)

if __name__ == "__main__":
    app()