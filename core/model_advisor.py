import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import json

class DatasetComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class TaskType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    MULTIOUTPUT = "multioutput"

@dataclass
class DatasetProfile:
    """Comprehensive dataset analysis"""
    # Basic properties
    n_samples: int
    n_features: int
    target_type: TaskType
    n_classes: Optional[int] = None
    
    # Data quality
    missing_percentage: float = 0.0
    duplicate_percentage: float = 0.0
    
    # Feature characteristics
    numerical_features: int = 0
    categorical_features: int = 0
    high_cardinality_features: int = 0
    
    # Statistical properties
    class_imbalance_ratio: Optional[float] = None
    feature_correlation_max: float = 0.0
    target_skewness: Optional[float] = None
    
    # Complexity indicators
    complexity: DatasetComplexity = DatasetComplexity.SIMPLE
    estimated_training_time: str = "fast"
    
    # Feature importance hints
    top_features: List[str] = None
    feature_importance_scores: Dict[str, float] = None

@dataclass
class ModelRecommendation:
    """Single model recommendation with reasoning"""
    model_name: str
    confidence: float  # 0-1 score
    reasoning: List[str]
    expected_performance_range: Tuple[float, float]
    training_time_estimate: str
    memory_requirements: str
    hyperparameter_suggestions: Dict[str, Any]
    pros: List[str]
    cons: List[str]

@dataclass
class PreprocessingRecommendation:
    """Preprocessing step recommendations"""
    step: str
    reasoning: str
    code_snippet: str
    priority: int  # 1=critical, 2=recommended, 3=optional

class IntelligentModelAdvisor:
    """AI-powered model selection and recommendation system"""
    
    def __init__(self):
        self.model_characteristics = self._load_model_knowledge_base()
        self.preprocessing_rules = self._load_preprocessing_rules()
        
    def analyze_and_recommend(self, X: pd.DataFrame, y: pd.Series, 
                            user_preferences: Dict = None) -> Dict[str, Any]:
        """Main entry point - analyze dataset and provide recommendations"""
        
        print("🔍 Analyzing dataset characteristics...")
        
        # 1. Profile the dataset
        profile = self._profile_dataset(X, y)
        
        # 2. Get preprocessing recommendations
        preprocessing_recs = self._recommend_preprocessing(X, y, profile)
        
        # 3. Get model recommendations
        model_recs = self._recommend_models(profile, user_preferences or {})
        
        # 4. Generate feature engineering suggestions
        feature_suggestions = self._suggest_feature_engineering(X, y, profile)
        
        # 5. Create comprehensive report
        report = {
            "dataset_profile": profile,
            "preprocessing_recommendations": preprocessing_recs,
            "model_recommendations": model_recs,
            "feature_engineering_suggestions": feature_suggestions,
            "training_strategy": self._suggest_training_strategy(profile),
            "evaluation_strategy": self._suggest_evaluation_strategy(profile)
        }
        
        return report
    
    def _profile_dataset(self, X: pd.DataFrame, y: pd.Series) -> DatasetProfile:
        """Comprehensive dataset profiling"""
        
        n_samples, n_features = X.shape
        
        # Determine task type
        if pd.api.types.is_numeric_dtype(y):
            if len(y.unique()) == 2:
                task_type = TaskType.BINARY_CLASSIFICATION
                n_classes = 2
            elif len(y.unique()) <= 20 and len(y.unique()) / len(y) < 0.05:
                task_type = TaskType.MULTICLASS_CLASSIFICATION
                n_classes = len(y.unique())
            else:
                task_type = TaskType.REGRESSION
                n_classes = None
        else:
            unique_classes = len(y.unique())
            if unique_classes == 2:
                task_type = TaskType.BINARY_CLASSIFICATION
            else:
                task_type = TaskType.MULTICLASS_CLASSIFICATION
            n_classes = unique_classes
        
        # Data quality metrics
        missing_pct = (X.isnull().sum().sum() / (n_samples * n_features)) * 100
        duplicate_pct = (X.duplicated().sum() / n_samples) * 100
        
        # Feature type analysis
        numerical_features = len(X.select_dtypes(include=[np.number]).columns)
        categorical_features = len(X.select_dtypes(include=['object', 'category']).columns)
        
        # High cardinality categorical features
        high_cardinality = 0
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if X[col].nunique() > 50:
                high_cardinality += 1
        
        # Statistical properties
        class_imbalance = None
        if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            class_counts = y.value_counts()
            class_imbalance = class_counts.min() / class_counts.max()
        
        # Feature correlations
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        max_correlation = 0.0
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr().abs()
            # Get max correlation excluding diagonal
            max_correlation = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).max().max()
        
        # Target skewness for regression
        target_skewness = None
        if task_type == TaskType.REGRESSION:
            target_skewness = stats.skew(y.dropna())
        
        # Complexity assessment
        complexity = self._assess_complexity(n_samples, n_features, task_type, 
                                           missing_pct, max_correlation, class_imbalance)
        
        # Feature importance (quick analysis)
        top_features, importance_scores = self._quick_feature_importance(X, y, task_type)
        
        return DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            target_type=task_type,
            n_classes=n_classes,
            missing_percentage=missing_pct,
            duplicate_percentage=duplicate_pct,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            high_cardinality_features=high_cardinality,
            class_imbalance_ratio=class_imbalance,
            feature_correlation_max=max_correlation,
            target_skewness=target_skewness,
            complexity=complexity,
            estimated_training_time=self._estimate_training_time(n_samples, n_features, complexity),
            top_features=top_features,
            feature_importance_scores=importance_scores
        )
    
    def _assess_complexity(self, n_samples: int, n_features: int, task_type: TaskType,
                          missing_pct: float, max_corr: float, class_imbalance: float) -> DatasetComplexity:
        """Assess dataset complexity based on multiple factors"""
        
        complexity_score = 0
        
        # Sample size factor
        if n_samples < 1000:
            complexity_score += 1
        elif n_samples > 100000:
            complexity_score += 2
        
        # Feature dimensionality
        if n_features > 100:
            complexity_score += 1
        if n_features > 1000:
            complexity_score += 2
        
        # Feature-to-sample ratio
        if n_features / n_samples > 0.1:
            complexity_score += 2
        
        # Data quality issues
        if missing_pct > 20:
            complexity_score += 1
        if max_corr > 0.9:
            complexity_score += 1
        
        # Class imbalance (for classification)
        if class_imbalance and class_imbalance < 0.1:
            complexity_score += 2
        elif class_imbalance and class_imbalance < 0.3:
            complexity_score += 1
        
        # Multiclass complexity
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            complexity_score += 1
        
        # Map score to complexity level
        if complexity_score <= 2:
            return DatasetComplexity.SIMPLE
        elif complexity_score <= 4:
            return DatasetComplexity.MODERATE
        elif complexity_score <= 6:
            return DatasetComplexity.COMPLEX
        else:
            return DatasetComplexity.VERY_COMPLEX
    
    def _recommend_models(self, profile: DatasetProfile, 
                         user_prefs: Dict) -> List[ModelRecommendation]:
        """Generate ranked model recommendations based on dataset profile"""
        
        recommendations = []
        
        # Get applicable models for this task type
        applicable_models = self.model_characteristics[profile.target_type.value]
        
        for model_name, characteristics in applicable_models.items():
            confidence = self._calculate_model_confidence(profile, characteristics, user_prefs)
            
            if confidence > 0.3:  # Only recommend models with decent confidence
                reasoning = self._generate_reasoning(profile, characteristics)
                perf_range = self._estimate_performance_range(profile, characteristics)
                
                recommendation = ModelRecommendation(
                    model_name=model_name,
                    confidence=confidence,
                    reasoning=reasoning,
                    expected_performance_range=perf_range,
                    training_time_estimate=characteristics.get('training_time', 'medium'),
                    memory_requirements=characteristics.get('memory_usage', 'medium'),
                    hyperparameter_suggestions=self._suggest_hyperparameters(profile, model_name),
                    pros=characteristics.get('pros', []),
                    cons=characteristics.get('cons', [])
                )
                recommendations.append(recommendation)
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _calculate_model_confidence(self, profile: DatasetProfile, 
                                  model_chars: Dict, user_prefs: Dict) -> float:
        """Calculate confidence score for a model recommendation"""
        
        confidence = 0.5  # Base confidence
        
        # Sample size suitability
        sample_range = model_chars.get('optimal_sample_range', (100, float('inf')))
        if sample_range[0] <= profile.n_samples <= sample_range[1]:
            confidence += 0.2
        elif profile.n_samples < sample_range[0]:
            confidence -= 0.3
        
        # Feature count suitability
        feature_range = model_chars.get('optimal_feature_range', (1, float('inf')))
        if feature_range[0] <= profile.n_features <= feature_range[1]:
            confidence += 0.1
        
        # Complexity match
        suitable_complexity = model_chars.get('suitable_complexity', [])
        if profile.complexity.value in suitable_complexity:
            confidence += 0.2
        
        # Handle missing data
        if profile.missing_percentage > 10:
            if model_chars.get('handles_missing_data', False):
                confidence += 0.1
            else:
                confidence -= 0.2
        
        # Handle categorical features
        if profile.categorical_features > 0:
            if model_chars.get('handles_categorical', False):
                confidence += 0.1
            else:
                confidence -= 0.1
        
        # Class imbalance handling
        if profile.class_imbalance_ratio and profile.class_imbalance_ratio < 0.3:
            if model_chars.get('handles_imbalance', False):
                confidence += 0.15
            else:
                confidence -= 0.1
        
        # High correlation handling
        if profile.feature_correlation_max > 0.8:
            if model_chars.get('robust_to_correlation', False):
                confidence += 0.1
            else:
                confidence -= 0.1
        
        # User preferences
        if user_prefs.get('prefer_interpretable', False):
            if model_chars.get('interpretable', False):
                confidence += 0.2
        
        if user_prefs.get('prefer_fast_training', False):
            if model_chars.get('training_time', 'medium') == 'fast':
                confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
    
    def _recommend_preprocessing(self, X: pd.DataFrame, y: pd.Series, 
                               profile: DatasetProfile) -> List[PreprocessingRecommendation]:
        """Recommend preprocessing steps based on data characteristics"""
        
        recommendations = []
        
        # Handle missing data
        if profile.missing_percentage > 5:
            if profile.missing_percentage < 20:
                rec = PreprocessingRecommendation(
                    step="Handle Missing Values - Simple Imputation",
                    reasoning=f"Dataset has {profile.missing_percentage:.1f}% missing values. Simple imputation recommended.",
                    code_snippet="from sklearn.impute import SimpleImputer\nimputer = SimpleImputer(strategy='median')",
                    priority=1
                )
            else:
                rec = PreprocessingRecommendation(
                    step="Handle Missing Values - Advanced Imputation",
                    reasoning=f"High missing data ({profile.missing_percentage:.1f}%). Consider advanced imputation.",
                    code_snippet="from sklearn.impute import IterativeImputer\nimputer = IterativeImputer()",
                    priority=1
                )
            recommendations.append(rec)
        
        # Handle categorical encoding
        if profile.categorical_features > 0:
            if profile.high_cardinality_features > 0:
                rec = PreprocessingRecommendation(
                    step="Encode High-Cardinality Categoricals",
                    reasoning="High-cardinality categorical features detected. Use target encoding or embeddings.",
                    code_snippet="from category_encoders import TargetEncoder\nencoder = TargetEncoder()",
                    priority=2
                )
            else:
                rec = PreprocessingRecommendation(
                    step="Encode Categorical Features",
                    reasoning="Standard categorical encoding needed.",
                    code_snippet="from sklearn.preprocessing import LabelEncoder, OneHotEncoder",
                    priority=1
                )
            recommendations.append(rec)
        
        # Feature scaling
        if profile.numerical_features > 0:
            rec = PreprocessingRecommendation(
                step="Scale Numerical Features",
                reasoning="Numerical features may benefit from scaling for certain models.",
                code_snippet="from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()",
                priority=2
            )
            recommendations.append(rec)
        
        # Handle class imbalance
        if profile.class_imbalance_ratio and profile.class_imbalance_ratio < 0.3:
            rec = PreprocessingRecommendation(
                step="Address Class Imbalance",
                reasoning=f"Significant class imbalance detected (ratio: {profile.class_imbalance_ratio:.2f}).",
                code_snippet="from imblearn.over_sampling import SMOTE\nsmote = SMOTE()",
                priority=1
            )
            recommendations.append(rec)
        
        # Feature selection
        if profile.n_features > 50:
            rec = PreprocessingRecommendation(
                step="Feature Selection",
                reasoning=f"High dimensionality ({profile.n_features} features). Consider feature selection.",
                code_snippet="from sklearn.feature_selection import SelectKBest, f_classif",
                priority=2
            )
            recommendations.append(rec)
        
        return sorted(recommendations, key=lambda x: x.priority)
    
    def _suggest_feature_engineering(self, X: pd.DataFrame, y: pd.Series, 
                                   profile: DatasetProfile) -> List[str]:
        """Suggest feature engineering techniques"""
        
        suggestions = []
        
        # Time-based features
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            suggestions.append("Extract time-based features: hour, day_of_week, month, season")
        
        # Polynomial features
        if profile.numerical_features > 1 and profile.numerical_features < 20:
            suggestions.append("Consider polynomial features for non-linear relationships")
        
        # Feature interactions
        if profile.n_features < 50:
            suggestions.append("Create interaction features between important variables")
        
        # Text features
        text_cols = [col for col in X.columns if X[col].dtype == 'object' and 
                    X[col].str.len().mean() > 10]  # Likely text columns
        if text_cols:
            suggestions.append("Apply text preprocessing: TF-IDF, word embeddings")
        
        # Binning
        if profile.numerical_features > 0:
            suggestions.append("Consider binning continuous variables for tree-based models")
        
        return suggestions
    
    def _suggest_training_strategy(self, profile: DatasetProfile) -> Dict[str, str]:
        """Suggest training strategy based on dataset characteristics"""
        
        strategy = {}
        
        # Cross-validation strategy
        if profile.target_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            if profile.class_imbalance_ratio and profile.class_imbalance_ratio < 0.3:
                strategy['cv_strategy'] = 'StratifiedKFold with 10 folds (handles imbalance)'
            else:
                strategy['cv_strategy'] = 'StratifiedKFold with 5 folds'
        else:
            strategy['cv_strategy'] = 'KFold with 5 folds'
        
        # Validation approach
        if profile.n_samples < 1000:
            strategy['validation'] = 'Use cross-validation only (small dataset)'
        else:
            strategy['validation'] = 'Train/validation/test split (80/10/10)'
        
        # Hyperparameter optimization
        if profile.complexity in [DatasetComplexity.SIMPLE, DatasetComplexity.MODERATE]:
            strategy['hyperparameter_tuning'] = 'Grid search with 3-fold CV'
        else:
            strategy['hyperparameter_tuning'] = 'Bayesian optimization with Optuna'
        
        return strategy
    
    def _suggest_evaluation_strategy(self, profile: DatasetProfile) -> Dict[str, str]:
        """Suggest evaluation metrics and strategy"""
        
        strategy = {}
        
        if profile.target_type == TaskType.BINARY_CLASSIFICATION:
            if profile.class_imbalance_ratio and profile.class_imbalance_ratio < 0.3:
                strategy['primary_metric'] = 'F1-score or AUC-ROC'
                strategy['additional_metrics'] = 'Precision, Recall, AUC-PR'
            else:
                strategy['primary_metric'] = 'Accuracy'
                strategy['additional_metrics'] = 'F1-score, AUC-ROC'
        
        elif profile.target_type == TaskType.MULTICLASS_CLASSIFICATION:
            strategy['primary_metric'] = 'Macro F1-score'
            strategy['additional_metrics'] = 'Accuracy, Weighted F1-score, Per-class metrics'
        
        elif profile.target_type == TaskType.REGRESSION:
            if profile.target_skewness and abs(profile.target_skewness) > 1:
                strategy['primary_metric'] = 'MAE (robust to outliers)'
                strategy['additional_metrics'] = 'RMSE, MAPE'
            else:
                strategy['primary_metric'] = 'RMSE'
                strategy['additional_metrics'] = 'MAE, R²'
        
        return strategy
    
    def _quick_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                                task_type: TaskType) -> Tuple[List[str], Dict[str, float]]:
        """Quick feature importance analysis"""
        
        try:
            # Prepare data for mutual information
            X_encoded = X.copy()
            
            # Simple encoding for categorical variables
            le = LabelEncoder()
            for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            
            # Handle missing values
            X_encoded = X_encoded.fillna(X_encoded.median())
            
            # Calculate mutual information
            if task_type == TaskType.REGRESSION:
                mi_scores = mutual_info_regression(X_encoded, y)
            else:
                mi_scores = mutual_info_classif(X_encoded, y)
            
            # Create feature importance dictionary
            importance_dict = dict(zip(X.columns, mi_scores))
            
            # Get top features
            top_features = sorted(importance_dict.keys(), 
                                key=lambda x: importance_dict[x], reverse=True)[:10]
            
            return top_features, importance_dict
            
        except Exception as e:
            # Fallback if mutual information fails
            return list(X.columns[:5]), {}
    
    def _load_model_knowledge_base(self) -> Dict:
        """Load model characteristics knowledge base"""
        
        return {
            "binary_classification": {
                "random_forest": {
                    "optimal_sample_range": (100, 1000000),
                    "optimal_feature_range": (1, 1000),
                    "suitable_complexity": ["simple", "moderate", "complex"],
                    "handles_missing_data": True,
                    "handles_categorical": True,
                    "handles_imbalance": False,
                    "robust_to_correlation": True,
                    "interpretable": True,
                    "training_time": "medium",
                    "memory_usage": "medium",
                    "pros": ["Handles mixed data types", "Built-in feature importance", "Robust to outliers"],
                    "cons": ["Can overfit with small datasets", "Less interpretable than single trees"]
                },
                "xgboost": {
                    "optimal_sample_range": (1000, float('inf')),
                    "optimal_feature_range": (1, 10000),
                    "suitable_complexity": ["moderate", "complex", "very_complex"],
                    "handles_missing_data": True,
                    "handles_categorical": True,
                    "handles_imbalance": True,
                    "robust_to_correlation": True,
                    "interpretable": False,
                    "training_time": "medium",
                    "memory_usage": "medium",
                    "pros": ["Excellent performance", "Handles imbalance well", "Built-in regularization"],
                    "cons": ["Many hyperparameters", "Less interpretable", "Can overfit"]
                },
                "logistic_regression": {
                    "optimal_sample_range": (100, 100000),
                    "optimal_feature_range": (1, 1000),
                    "suitable_complexity": ["simple", "moderate"],
                    "handles_missing_data": False,
                    "handles_categorical": False,
                    "handles_imbalance": False,
                    "robust_to_correlation": False,
                    "interpretable": True,
                    "training_time": "fast",
                    "memory_usage": "low",
                    "pros": ["Fast training", "Interpretable", "Probabilistic output"],
                    "cons": ["Assumes linear relationships", "Sensitive to outliers"]
                },
                "svm": {
                    "optimal_sample_range": (100, 10000),
                    "optimal_feature_range": (1, 1000),
                    "suitable_complexity": ["moderate", "complex"],
                    "handles_missing_data": False,
                    "handles_categorical": False,
                    "handles_imbalance": False,
                    "robust_to_correlation": True,
                    "interpretable": False,
                    "training_time": "slow",
                    "memory_usage": "medium",
                    "pros": ["Effective in high dimensions", "Memory efficient"],
                    "cons": ["Slow on large datasets", "Requires feature scaling"]
                }
            },
            "multiclass_classification": {
                # Same models with adjusted characteristics
                "random_forest": {
                    "optimal_sample_range": (200, 1000000),
                    "suitable_complexity": ["simple", "moderate", "complex"],
                    # ... similar structure
                }
            },
            "regression": {
                "random_forest": {
                    "optimal_sample_range": (100, 1000000),
                    "suitable_complexity": ["simple", "moderate", "complex"],
                    # ... adapted for regression
                }
            }
        }
    
    def _load_preprocessing_rules(self) -> Dict:
        """Load preprocessing recommendation rules"""
        return {}
    
    def _generate_reasoning(self, profile: DatasetProfile, characteristics: Dict) -> List[str]:
        """Generate human-readable reasoning for model recommendation"""
        reasoning = []
        
        # Sample size reasoning
        sample_range = characteristics.get('optimal_sample_range', (0, float('inf')))
        if sample_range[0] <= profile.n_samples <= sample_range[1]:
            reasoning.append(f"Good fit for dataset size ({profile.n_samples:,} samples)")
        
        # Complexity reasoning
        if profile.complexity.value in characteristics.get('suitable_complexity', []):
            reasoning.append(f"Handles {profile.complexity.value} datasets well")
        
        # Missing data
        if profile.missing_percentage > 10 and characteristics.get('handles_missing_data'):
            reasoning.append("Can handle missing data without preprocessing")
        
        # Categorical features
        if profile.categorical_features > 0 and characteristics.get('handles_categorical'):
            reasoning.append("Native support for categorical features")
        
        # Class imbalance
        if (profile.class_imbalance_ratio and profile.class_imbalance_ratio < 0.3 and 
            characteristics.get('handles_imbalance')):
            reasoning.append("Good performance on imbalanced datasets")
        
        return reasoning
    
    def _estimate_performance_range(self, profile: DatasetProfile, 
                                  characteristics: Dict) -> Tuple[float, float]:
        """Estimate expected performance range"""
        
        # Base performance estimates (these would be learned from historical data)
        base_performance = {
            "random_forest": (0.75, 0.95),
            "xgboost": (0.80, 0.98),
            "logistic_regression": (0.70, 0.90),
            "svm": (0.72, 0.92)
        }
        
        # Adjust based on dataset characteristics
        # This is a simplified version - in reality, you'd have more sophisticated models
        
        return base_performance.get(characteristics.get('name', 'random_forest'), (0.70, 0.90))
    
    def _suggest_hyperparameters(self, profile: DatasetProfile, model_name: str) -> Dict[str, Any]:
        """Suggest initial hyperparameters based on dataset characteristics"""
        
        suggestions = {}
        
        if model_name == "random_forest":
            suggestions = {
                "n_estimators": 100 if profile.n_samples < 10000 else 200,
                "max_depth": None if profile.complexity != DatasetComplexity.SIMPLE else 10,
                "min_samples_split": 5 if profile.n_samples < 1000 else 2,
                "class_weight": "balanced" if (profile.class_imbalance_ratio and 
                                             profile.class_imbalance_ratio < 0.3) else None
            }
        
        elif model_name == "xgboost":
            suggestions = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6 if profile.complexity == DatasetComplexity.COMPLEX else 3,
                "subsample": 0.8 if profile.n_samples > 1000 else 1.0,
                "colsample_bytree": 0.8 if profile.n_features > 50 else 1.0
            }
        
        elif model_name == "logistic_regression":
            suggestions = {
                "C": 0.1 if profile.n_features > profile.n_samples else 1.0,
                "max_iter": 1000,
                "class_weight": "balanced" if (profile.class_imbalance_ratio and 
                                             profile.class_imbalance_ratio < 0.3) else None
            }
        
        return suggestions
    
    def _estimate_training_time(self, n_samples: int, n_features: int, 
                              complexity: DatasetComplexity) -> str:
        """Estimate training time category"""
        
        if n_samples < 1000 and n_features < 50:
            return "very_fast"
        elif n_samples < 10000 and n_features < 100:
            return "fast"
        elif n_samples < 100000 and n_features < 1000:
            return "medium"
        elif n_samples < 1000000:
            return "slow"
        else:
            return "very_slow"

def print_advisor_report(report: Dict[str, Any], detailed: bool = True):
    """Pretty print the advisor report"""
    
    profile = report["dataset_profile"]
    
    print("\n" + "="*80)
    print("🧠 INTELLIGENT MODEL ADVISOR REPORT")
    print("="*80)
    
    # Dataset Summary
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  • Samples: {profile.n_samples:,}")
    print(f"  • Features: {profile.n_features}")
    print(f"  • Task Type: {profile.target_type.value.replace('_', ' ').title()}")
    if profile.n_classes:
        print(f"  • Classes: {profile.n_classes}")
    print(f"  • Complexity: {profile.complexity.value.title()}")
    print(f"  • Training Time Estimate: {profile.estimated_training_time.title()}")
    
    # Data Quality Issues
    if profile.missing_percentage > 0 or profile.duplicate_percentage > 0:
        print(f"\n⚠️  DATA QUALITY ISSUES:")
        if profile.missing_percentage > 0:
            print(f"  • Missing Values: {profile.missing_percentage:.1f}%")
        if profile.duplicate_percentage > 0:
            print(f"  • Duplicate Rows: {profile.duplicate_percentage:.1f}%")
    
    # Feature Analysis
    print(f"\n🔍 FEATURE ANALYSIS:")
    print(f"  • Numerical Features: {profile.numerical_features}")
    print(f"  • Categorical Features: {profile.categorical_features}")
    if profile.high_cardinality_features > 0:
        print(f"  • High-Cardinality Categorical: {profile.high_cardinality_features}")
    if profile.feature_correlation_max > 0:
        print(f"  • Max Feature Correlation: {profile.feature_correlation_max:.3f}")
    
    # Classification-specific metrics
    if profile.target_type != TaskType.REGRESSION:
        if profile.class_imbalance_ratio:
            print(f"  • Class Imbalance Ratio: {profile.class_imbalance_ratio:.3f}")
    
    # Regression-specific metrics
    if profile.target_type == TaskType.REGRESSION and profile.target_skewness:
        print(f"  • Target Skewness: {profile.target_skewness:.3f}")
    
    # Top Features
    if profile.top_features and detailed:
        print(f"\n🏆 TOP IMPORTANT FEATURES:")
        for i, feature in enumerate(profile.top_features[:5], 1):
            score = profile.feature_importance_scores.get(feature, 0)
            print(f"  {i}. {feature} (importance: {score:.3f})")
    
    # Preprocessing Recommendations
    preprocessing_recs = report["preprocessing_recommendations"]
    if preprocessing_recs:
        print(f"\n🔧 PREPROCESSING RECOMMENDATIONS:")
        for rec in preprocessing_recs[:3]:  # Show top 3
            priority_emoji = "🔴" if rec.priority == 1 else "🟡" if rec.priority == 2 else "🟢"
            print(f"  {priority_emoji} {rec.step}")
            print(f"     Reason: {rec.reasoning}")
            if detailed:
                print(f"     Code: {rec.code_snippet}")
    
    # Model Recommendations
    model_recs = report["model_recommendations"]
    print(f"\n🤖 RECOMMENDED MODELS (Top {len(model_recs)}):")
    
    for i, rec in enumerate(model_recs, 1):
        confidence_bar = "█" * int(rec.confidence * 10) + "░" * (10 - int(rec.confidence * 10))
        print(f"\n  {i}. {rec.model_name.upper()}")
        print(f"     Confidence: {confidence_bar} {rec.confidence:.1%}")
        print(f"     Expected Performance: {rec.expected_performance_range[0]:.2f} - {rec.expected_performance_range[1]:.2f}")
        print(f"     Training Time: {rec.training_time_estimate.title()}")
        print(f"     Memory Usage: {rec.memory_requirements.title()}")
        
        if rec.reasoning:
            print(f"     Why this model:")
            for reason in rec.reasoning[:2]:  # Show top 2 reasons
                print(f"       • {reason}")
        
        if detailed and rec.hyperparameter_suggestions:
            print(f"     Suggested hyperparameters:")
            for param, value in list(rec.hyperparameter_suggestions.items())[:3]:
                print(f"       • {param}: {value}")
        
        if detailed:
            print(f"     Pros: {', '.join(rec.pros[:2])}")
            print(f"     Cons: {', '.join(rec.cons[:2])}")
    
    # Feature Engineering Suggestions
    feature_suggestions = report["feature_engineering_suggestions"]
    if feature_suggestions:
        print(f"\n💡 FEATURE ENGINEERING SUGGESTIONS:")
        for suggestion in feature_suggestions[:3]:
            print(f"  • {suggestion}")
    
    # Training Strategy
    training_strategy = report["training_strategy"]
    print(f"\n🎯 RECOMMENDED TRAINING STRATEGY:")
    for key, value in training_strategy.items():
        key_display = key.replace('_', ' ').title()
        print(f"  • {key_display}: {value}")
    
    # Evaluation Strategy
    eval_strategy = report["evaluation_strategy"]
    print(f"\n📈 RECOMMENDED EVALUATION STRATEGY:")
    for key, value in eval_strategy.items():
        key_display = key.replace('_', ' ').title()
        print(f"  • {key_display}: {value}")
    
    print("\n" + "="*80)
    print("="*80)