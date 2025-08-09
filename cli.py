import typer
from typing import List, Optional
from typing import Dict, Any
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
    cv_folds: int = typer.Option(5, "--folds", "-k", help="Number of cross-validation folds"),
    generate_viz: bool = typer.Option(True, "--viz/--no-viz", help="Generate comparison visualizations")
):
    """Compare multiple models on the same dataset with comprehensive visualizations."""
    
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
    print(f"Prompt: {prompt}")
    if generate_viz:
        print(f"🎨 Visualizations: Enabled")
    print()
    
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
    
    # Generate comparison visualizations
    if generate_viz:
        print(f"\n🎨 Generating comparison visualizations...")
        try:
            sys.path.append('.')
            from core.visualizer import create_comparison_visualizations
            
            viz_plots, visualizer = create_comparison_visualizations(results)
            
            if viz_plots:
                print(f"✅ Generated {len(viz_plots)} comparison visualizations")
                print(f"📁 Visualizations saved in: {visualizer.output_dir}")
                
                # Generate comparison report
                comparison_info = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt": prompt,
                    "models_compared": model_list,
                    "comparison_type": "cross_validation" if use_cv else "train_test_split",
                    "cv_folds": cv_folds if use_cv else None,
                    "results": detailed_results,
                    "summary": {
                        "total_models": len(model_list),
                        "successful_models": len([r for r in detailed_results.values() if r["status"] == "success"]),
                        "best_model": max(detailed_results.items(), key=lambda x: x[1].get("accuracy", 0))[0] if detailed_results else None,
                        "frameworks_tested": list(set(MODEL_REGISTRY[m]["framework"] for m in model_list))
                    }
                }
                
                report_path = visualizer.generate_summary_report(
                    comparison_info, viz_plots, "model_comparison"
                )
                print(f"📋 Comparison report: {report_path}")
            else:
                print("⚠️  No visualizations generated")
                
        except ImportError:
            print("⚠️  Visualization module not available, skipping visualizations")
        except Exception as e:
            print(f"⚠️  Error generating visualizations: {e}")
    
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
    cv_folds: int = typer.Option(5, "--folds", "-k", help="Number of cross-validation folds"),
    generate_viz: bool = typer.Option(True, "--viz/--no-viz", help="Generate comprehensive benchmark visualizations"),
    save_models: bool = typer.Option(False, "--save-models/--no-save-models", help="Save trained models to disk")
):
    """Run a comprehensive benchmark across frameworks with detailed visualizations."""
    
    framework_list = [f.strip() for f in frameworks.split(",")]
    
    cv_suffix = f" with {cv_folds}-fold CV" if use_cv else ""
    print(f"🏃‍♂️ Comprehensive Benchmark on {dataset} dataset{cv_suffix}")
    print(f"Frameworks: {', '.join(framework_list)}")
    print(f"Models per framework: {top_n}")
    if generate_viz:
        print(f"🎨 Comprehensive visualizations: Enabled")
    print()
    
    all_models = []
    framework_models = {}
    
    for fw in framework_list:
        fw_models = get_models_by_framework(fw)[:top_n]  # Top N per framework
        all_models.extend(fw_models)
        framework_models[fw] = fw_models
        print(f"📊 {fw.upper()}: {', '.join(fw_models)}")
    
    if not all_models:
        print("❌ No models found for specified frameworks")
        return
    
    print(f"\n🎯 Total models to benchmark: {len(all_models)}")
    
    # Run comparison with enhanced prompt
    benchmark_prompt = f"dataset: {dataset} epochs: 100"
    
    # Call the enhanced compare function
    compare(
        benchmark_prompt, 
        models=",".join(all_models), 
        save_results=True, 
        use_cv=use_cv, 
        cv_folds=cv_folds,
        generate_viz=generate_viz
    )
    
    # Additional benchmark-specific analysis
    if generate_viz and len(framework_list) > 1:
        print(f"\n📈 Generating framework comparison analysis...")
        try:
            # Load the latest comparison results
            import glob
            result_files = glob.glob("outputs/comparison_*.json")
            if result_files:
                latest_file = max(result_files, key=os.path.getctime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    comparison_data = json.load(f)
                
                # Create framework-level summary
                framework_summary = {}
                for fw in framework_list:
                    fw_results = [
                        result for model, result in comparison_data["results"].items()
                        if model in framework_models.get(fw, []) and result["status"] == "success"
                    ]
                    
                    if fw_results:
                        accuracies = [r["accuracy"] for r in fw_results]
                        framework_summary[fw] = {
                            "mean_accuracy": sum(accuracies) / len(accuracies),
                            "best_accuracy": max(accuracies),
                            "model_count": len(fw_results),
                            "avg_time": sum(float(r["training_time"]) for r in fw_results) / len(fw_results)
                        }
                
                print(f"\n🏆 Framework Performance Summary:")
                for fw, summary in sorted(framework_summary.items(), 
                                        key=lambda x: x[1]["mean_accuracy"], reverse=True):
                    print(f"  {fw.upper()}:")
                    print(f"    Mean Accuracy: {summary['mean_accuracy']:.4f}")
                    print(f"    Best Accuracy: {summary['best_accuracy']:.4f}")
                    print(f"    Models Tested: {summary['model_count']}")
                    print(f"    Avg Time: {summary['avg_time']:.2f}s")
                
        except Exception as e:
            print(f"⚠️  Error in framework analysis: {e}")
    
    print(f"\n🎉 Comprehensive benchmark completed!")
    print(f"📁 Results and visualizations saved in outputs/ directory")

@app.command()
def advisor(
    dataset: str = typer.Argument(..., help="Dataset to analyze"),  # FIXED: Made dataset a required argument
    detailed: bool = typer.Option(True, "--detailed/--summary", help="Show detailed analysis"),
    auto_compare: bool = typer.Option(False, "--auto-compare", help="Automatically compare top 3 models"),
    save_report: bool = typer.Option(True, "--save/--no-save", help="Save advisor report"),
    prefer_interpretable: bool = typer.Option(False, "--interpretable", help="Prefer interpretable models"),
    prefer_fast: bool = typer.Option(False, "--fast", help="Prefer fast training models")
):
    """Get intelligent model recommendations for your dataset."""
    
    # FIXED: Validate dataset parameter
    if not dataset or dataset.lower() in ['none', 'null', '']:
        print("❌ Dataset name is required for advisor command")
        print("💡 Usage: python cli.py advisor <dataset_name>")
        print("💡 Example: python cli.py advisor penguins")
        return
    
    print(f"🧠 Analyzing dataset: {dataset}")
    print("🔍 Loading and profiling data...")
    
    try:
        # Import the advisor
        from core.model_advisor import IntelligentModelAdvisor, print_advisor_report
        
        # Load dataset (reuse existing logic from train.py)
        df = load_dataset(dataset)  # You'll need to extract this function
        
        # Prepare features and target
        if 'target' in df.columns:
            X = df.drop(columns=['target'])
            y = df['target']
        else:
            X = df.drop(columns=[df.columns[-1]])
            y = df[df.columns[-1]]
        
        # User preferences
        user_prefs = {
            "prefer_interpretable": prefer_interpretable,
            "prefer_fast_training": prefer_fast
        }
        
        # Initialize advisor and get recommendations
        advisor = IntelligentModelAdvisor()
        report = advisor.analyze_and_recommend(X, y, user_prefs)
        
        # Print the report
        print_advisor_report(report, detailed=detailed)
        
        # Save report if requested
        if save_report:
            save_advisor_report(report, dataset)
        
        # Auto-compare top models if requested
        if auto_compare and report["model_recommendations"]:
            top_models = [rec.model_name for rec in report["model_recommendations"][:3]]
            print(f"\n🚀 Auto-comparing top 3 models: {', '.join(top_models)}")
            
            # Call compare function with recommended models
            compare(
                prompt=f"dataset: {dataset}",
                models=",".join(top_models),
                use_cv=True,
                cv_folds=5,
                save_results=True,
                generate_viz=True
            )
        
    except ValueError as e:
        print(f"❌ Dataset error: {e}")
        print("💡 Available datasets: iris, wine, breast_cancer, penguins, titanic, etc.")
        print("💡 Or place your CSV file in the data/ directory")
    except Exception as e:
        print(f"❌ Error in model advisor: {e}")
        import traceback
        traceback.print_exc()

def load_dataset(dataset_name: str):
    """Extract dataset loading logic from train.py"""
    import pandas as pd
    import seaborn as sns
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
    
    # FIXED: Validate dataset_name parameter
    if not dataset_name or dataset_name.lower() in ['none', 'null', '']:
        raise ValueError("Dataset name cannot be empty or None")
    
    # Try CSV file first
    try:
        df = pd.read_csv(f"data/{dataset_name}.csv")
        print(f"[SUCCESS] Loaded dataset from data/{dataset_name}.csv")
        return df
    except:
        pass
    
    # Try seaborn datasets
    try:
        df = sns.load_dataset(dataset_name).dropna()
        print(f"[SUCCESS] Loaded {dataset_name} from seaborn datasets")
        return df
    except:
        pass
    
    # Try sklearn datasets
    try:
        dataset_loaders = {
            'iris': load_iris,
            'wine': load_wine, 
            'breast_cancer': load_breast_cancer,
            'digits': load_digits,
            'diabetes': load_diabetes
        }
        
        if dataset_name.lower() in dataset_loaders:
            sklearn_data = dataset_loaders[dataset_name.lower()]()
            df = pd.DataFrame(
                sklearn_data.data, 
                columns=sklearn_data.feature_names if hasattr(sklearn_data, 'feature_names') else 
                        [f'feature_{i}' for i in range(sklearn_data.data.shape[1])]
            )
            df['target'] = sklearn_data.target
            print(f"[SUCCESS] Loaded {dataset_name} from sklearn datasets")
            return df
    except:
        pass
    
    raise ValueError(f"Dataset '{dataset_name}' not found")

def save_advisor_report(report: Dict[str, Any], dataset_name: str):
    """Save advisor report to JSON file"""
    import os
    import json
    import time
    from dataclasses import asdict
    
    os.makedirs("outputs", exist_ok=True)
    
    # Convert dataclasses to dicts for JSON serialization
    serializable_report = {}
    
    # Convert dataset profile
    profile = report["dataset_profile"]
    serializable_report["dataset_profile"] = {
        "n_samples": profile.n_samples,
        "n_features": profile.n_features,
        "target_type": profile.target_type.value,
        "n_classes": profile.n_classes,
        "missing_percentage": profile.missing_percentage,
        "duplicate_percentage": profile.duplicate_percentage,
        "numerical_features": profile.numerical_features,
        "categorical_features": profile.categorical_features,
        "high_cardinality_features": profile.high_cardinality_features,
        "class_imbalance_ratio": profile.class_imbalance_ratio,
        "feature_correlation_max": profile.feature_correlation_max,
        "target_skewness": profile.target_skewness,
        "complexity": profile.complexity.value,
        "estimated_training_time": profile.estimated_training_time,
        "top_features": profile.top_features,
        "feature_importance_scores": profile.feature_importance_scores
    }
    
    # Convert model recommendations
    serializable_report["model_recommendations"] = []
    for rec in report["model_recommendations"]:
        serializable_report["model_recommendations"].append({
            "model_name": rec.model_name,
            "confidence": rec.confidence,
            "reasoning": rec.reasoning,
            "expected_performance_range": rec.expected_performance_range,
            "training_time_estimate": rec.training_time_estimate,
            "memory_requirements": rec.memory_requirements,
            "hyperparameter_suggestions": rec.hyperparameter_suggestions,
            "pros": rec.pros,
            "cons": rec.cons
        })
    
    # Convert preprocessing recommendations
    serializable_report["preprocessing_recommendations"] = []
    for rec in report["preprocessing_recommendations"]:
        serializable_report["preprocessing_recommendations"].append({
            "step": rec.step,
            "reasoning": rec.reasoning,
            "code_snippet": rec.code_snippet,
            "priority": rec.priority
        })
    
    # Copy other sections
    serializable_report["feature_engineering_suggestions"] = report["feature_engineering_suggestions"]
    serializable_report["training_strategy"] = report["training_strategy"]
    serializable_report["evaluation_strategy"] = report["evaluation_strategy"]
    
    # Add metadata
    serializable_report["metadata"] = {
        "dataset_name": dataset_name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0"
    }
    
    # Save to file
    filename = f"outputs/advisor_report_{dataset_name}_{int(time.time())}.json"
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(serializable_report, f, indent=2)
    
    print(f"💾 Advisor report saved to: {filename}")

if __name__ == "__main__":
    app()