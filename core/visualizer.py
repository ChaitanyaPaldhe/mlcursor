import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MLVisualizer:
    """Comprehensive ML visualization class for training results and model analysis"""
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_plot(self, fig, filename: str, model_name: str = "model") -> str:
        """Save plot with timestamp and proper naming"""
        full_filename = f"{model_name}_{filename}_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, full_filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Saved plot: {filepath}")
        return filepath
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, model_name="model") -> str:
        """Generate and save confusion matrix heatmap"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Determine class names
        if class_names is None:
            unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
            class_names = [f"Class {i}" for i in unique_labels]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title(f'Confusion Matrix - {model_name.title()}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Add accuracy information
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return self.save_plot(fig, "confusion_matrix", model_name)
    
    def plot_feature_importance(self, model, feature_names: List[str], model_name="model", top_n=20) -> Optional[str]:
        """Plot feature importance for tree-based models"""
        try:
            # Extract feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
            else:
                print(f"⚠️  Feature importance not available for {type(model).__name__}")
                return None
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))
            bars = ax.barh(range(len(importance_df)), importance_df['importance'])
            
            # Color bars by importance
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'])
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title(f'Top {len(importance_df)} Feature Importances - {model_name.title()}', 
                        fontsize=16, fontweight='bold')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + max(importance_df['importance']) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            plt.gca().invert_yaxis()
            return self.save_plot(fig, "feature_importance", model_name)
            
        except Exception as e:
            print(f"⚠️  Error plotting feature importance: {e}")
            return None
    
    def plot_cv_scores(self, cv_scores: List[float], model_name="model", cv_type="CV") -> str:
        """Plot cross-validation scores distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot
        ax1.boxplot(cv_scores, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title(f'{cv_type} Score Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add mean and std annotations
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        ax1.text(1.1, mean_score, f'Mean: {mean_score:.3f}\nStd: {std_score:.3f}',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Line plot showing fold-by-fold performance
        ax2.plot(range(1, len(cv_scores) + 1), cv_scores, 'o-', linewidth=2, markersize=8)
        ax2.axhline(y=mean_score, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_score:.3f}')
        ax2.fill_between(range(1, len(cv_scores) + 1), 
                        [mean_score - std_score] * len(cv_scores),
                        [mean_score + std_score] * len(cv_scores),
                        alpha=0.2, color='red', label=f'±1 Std')
        
        ax2.set_xlabel('Fold', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title(f'{cv_type} Scores by Fold', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name.title()} - Cross-Validation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return self.save_plot(fig, "cv_scores", model_name)
    
    def plot_learning_curves(self, train_scores: List[float], val_scores: List[float], 
                           model_name="model", metric_name="Score") -> str:
        """Plot training and validation learning curves"""
        epochs = range(1, len(train_scores) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, train_scores, 'o-', label=f'Training {metric_name}', linewidth=2)
        ax.plot(epochs, val_scores, 's-', label=f'Validation {metric_name}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'Learning Curves - {model_name.title()}', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Highlight best validation score
        best_val_idx = np.argmax(val_scores) if 'accuracy' in metric_name.lower() else np.argmin(val_scores)
        ax.scatter(best_val_idx + 1, val_scores[best_val_idx], 
                  color='red', s=100, zorder=5, 
                  label=f'Best Val: {val_scores[best_val_idx]:.3f}')
        ax.legend()
        
        return self.save_plot(fig, "learning_curves", model_name)
    
    def plot_hyperparameter_importance(self, study, model_name="model", top_n=10) -> str:
        """Plot hyperparameter importance from Optuna study"""
        try:
            import optuna
            
            # Get parameter importance
            importance = optuna.importance.get_param_importances(study)
            
            if not importance:
                print("⚠️  No parameter importance data available")
                return None
            
            # Sort and take top N
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            params, values = zip(*sorted_importance)
            
            fig, ax = plt.subplots(figsize=(10, max(6, len(params) * 0.4)))
            
            bars = ax.barh(range(len(params)), values)
            
            # Color bars
            colors = plt.cm.plasma(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_yticks(range(len(params)))
            ax.set_yticklabels(params)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'Hyperparameter Importance - {model_name.title()}', 
                        fontsize=16, fontweight='bold')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            plt.gca().invert_yaxis()
            return self.save_plot(fig, "hyperparameter_importance", model_name)
            
        except ImportError:
            print("⚠️  Optuna not available for hyperparameter importance plot")
            return None
        except Exception as e:
            print(f"⚠️  Error plotting hyperparameter importance: {e}")
            return None
    
    def plot_optimization_history(self, study, model_name="model") -> str:
        """Plot optimization history showing accuracy vs trials"""
        try:
            import optuna
            
            trials_df = study.trials_dataframe()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Trial values over time
            ax1.plot(trials_df.index, trials_df['value'], 'o-', alpha=0.7)
            
            # Highlight best trials
            best_trials = trials_df.nlargest(5, 'value')
            ax1.scatter(best_trials.index, best_trials['value'], 
                       color='red', s=100, zorder=5, label='Top 5 Trials')
            
            ax1.set_xlabel('Trial', fontsize=12)
            ax1.set_ylabel('Objective Value', fontsize=12)
            ax1.set_title(f'Optimization History - {model_name.title()}', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Running best value
            running_best = trials_df['value'].cummax()
            ax2.plot(trials_df.index, running_best, linewidth=3, color='green', label='Best Value So Far')
            ax2.fill_between(trials_df.index, running_best, alpha=0.3, color='green')
            
            ax2.set_xlabel('Trial', fontsize=12)
            ax2.set_ylabel('Best Objective Value', fontsize=12)
            ax2.set_title('Best Value Progress', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add final best value annotation
            final_best = running_best.iloc[-1]
            ax2.text(0.02, 0.98, f'Final Best: {final_best:.4f}', 
                    transform=ax2.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            return self.save_plot(fig, "optimization_history", model_name)
            
        except Exception as e:
            print(f"⚠️  Error plotting optimization history: {e}")
            return None
    
    def plot_model_comparison(self, results_dict: Dict[str, Dict], metric_name="Accuracy") -> str:
        """Plot comparison between multiple models"""
        models = list(results_dict.keys())
        
        # Extract metrics
        scores = []
        std_devs = []
        times = []
        
        for model in models:
            result = results_dict[model]
            scores.append(result.get('accuracy', 0))
            std_devs.append(result.get('std_dev', 0))
            times.append(float(result.get('time', '0s').replace('s', '')))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison with error bars
        bars1 = ax1.bar(models, scores, yerr=std_devs, capsize=5, alpha=0.7)
        
        # Color bars by performance
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars1)))
        for bar, color in zip(bars1, colors):
            bar.set_color(color)
        
        ax1.set_ylabel(metric_name, fontsize=12)
        ax1.set_title(f'Model {metric_name} Comparison', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score, std in zip(bars1, scores, std_devs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Training time comparison
        bars2 = ax2.bar(models, times, alpha=0.7, color='orange')
        ax2.set_ylabel('Training Time (seconds)', fontsize=12)
        ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time in zip(bars2, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return self.save_plot(fig, "model_comparison", "comparison")
    
    def plot_regression_results(self, y_true, y_pred, model_name="model") -> str:
        """Plot regression results with actual vs predicted"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Values', fontsize=12)
        ax1.set_ylabel('Predicted Values', fontsize=12)
        ax1.set_title(f'Actual vs Predicted - {model_name.title()}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=12, fontweight='bold')
        
        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Predicted Values', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.set_title('Residuals Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.save_plot(fig, "regression_results", model_name)
    
    def generate_summary_report(self, model_info: Dict[str, Any], 
                              saved_plots: List[str], model_name="model") -> str:
        """Generate a summary report of all visualizations"""
        report_filename = f"{model_name}_visualization_report_{self.timestamp}.json"
        report_path = os.path.join(self.output_dir, report_filename)
        
        summary = {
            "timestamp": self.timestamp,
            "model_info": model_info,
            "generated_plots": saved_plots,
            "plot_descriptions": {
                "confusion_matrix": "Shows classification accuracy per class",
                "feature_importance": "Displays most influential features",
                "cv_scores": "Cross-validation performance distribution", 
                "learning_curves": "Training progress over epochs",
                "hyperparameter_importance": "Impact of different hyperparameters",
                "optimization_history": "Hyperparameter tuning progress",
                "model_comparison": "Performance comparison between models",
                "regression_results": "Actual vs predicted values for regression"
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📋 Visualization summary saved: {report_path}")
        return report_path


# Helper functions for integration with existing training code
def create_visualizations(model, X, y, model_name="model", task_type="classification", 
                        cv_scores=None, **kwargs):
    """Main function to create all relevant visualizations for a trained model"""
    
    visualizer = MLVisualizer()
    saved_plots = []
    
    print(f"\n🎨 Generating visualizations for {model_name}...")
    
    try:
        # Feature importance (if supported)
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        importance_plot = visualizer.plot_feature_importance(model, feature_names, model_name)
        if importance_plot:
            saved_plots.append(importance_plot)
        
        # Cross-validation scores (if provided)
        if cv_scores is not None and len(cv_scores) > 1:
            cv_plot = visualizer.plot_cv_scores(cv_scores, model_name)
            saved_plots.append(cv_plot)
        
        # Task-specific plots
        if task_type == "classification":
            # For classification: confusion matrix
            if hasattr(model, 'predict'):
                try:
                    # Use a subset for prediction if dataset is large
                    if len(X) > 1000:
                        sample_idx = np.random.choice(len(X), 1000, replace=False)
                        X_sample, y_sample = X.iloc[sample_idx], y.iloc[sample_idx]
                    else:
                        X_sample, y_sample = X, y
                    
                    y_pred = model.predict(X_sample)
                    cm_plot = visualizer.plot_confusion_matrix(y_sample, y_pred, model_name=model_name)
                    saved_plots.append(cm_plot)
                except Exception as e:
                    print(f"⚠️  Could not generate confusion matrix: {e}")
        
        elif task_type == "regression":
            # For regression: actual vs predicted
            if hasattr(model, 'predict'):
                try:
                    if len(X) > 1000:
                        sample_idx = np.random.choice(len(X), 1000, replace=False)
                        X_sample, y_sample = X.iloc[sample_idx], y.iloc[sample_idx]
                    else:
                        X_sample, y_sample = X, y
                    
                    y_pred = model.predict(X_sample)
                    reg_plot = visualizer.plot_regression_results(y_sample, y_pred, model_name)
                    saved_plots.append(reg_plot)
                except Exception as e:
                    print(f"⚠️  Could not generate regression plots: {e}")
        
        print(f"✅ Generated {len(saved_plots)} visualizations")
        return saved_plots, visualizer
        
    except Exception as e:
        print(f"⚠️  Error in visualization generation: {e}")
        return [], visualizer


def create_tuning_visualizations(study, model_name="model"):
    """Create visualizations specific to hyperparameter tuning"""
    
    visualizer = MLVisualizer()
    saved_plots = []
    
    print(f"\n🔬 Generating tuning visualizations for {model_name}...")
    
    try:
        # Optimization history
        opt_plot = visualizer.plot_optimization_history(study, model_name)
        if opt_plot:
            saved_plots.append(opt_plot)
        
        # Hyperparameter importance
        importance_plot = visualizer.plot_hyperparameter_importance(study, model_name)
        if importance_plot:
            saved_plots.append(importance_plot)
        
        print(f"✅ Generated {len(saved_plots)} tuning visualizations")
        return saved_plots, visualizer
        
    except Exception as e:
        print(f"⚠️  Error in tuning visualization generation: {e}")
        return [], visualizer


def create_comparison_visualizations(results_dict):
    """Create visualizations for model comparison"""
    
    visualizer = MLVisualizer()
    saved_plots = []
    
    print(f"\n📊 Generating comparison visualizations...")
    
    try:
        comp_plot = visualizer.plot_model_comparison(results_dict)
        saved_plots.append(comp_plot)
        
        print(f"✅ Generated comparison visualization")
        return saved_plots, visualizer
        
    except Exception as e:
        print(f"⚠️  Error in comparison visualization: {e}")
        return [], visualizer