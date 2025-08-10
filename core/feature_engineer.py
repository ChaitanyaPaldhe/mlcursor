import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, 
    mutual_info_regression, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, Ridge
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureEngineeringResult:
    """Container for feature engineering results"""
    transformed_features: pd.DataFrame
    feature_names: List[str]
    feature_importance: Optional[Dict[str, float]] = None
    engineering_log: List[str] = None
    removed_features: List[str] = None
    created_features: List[str] = None
    transformation_pipeline: Optional[Any] = None

class AutomatedFeatureEngineer:
    """Comprehensive automated feature engineering system"""
    
    def __init__(self, task_type: str = "auto"):
        self.task_type = task_type
        self.feature_importance_cache = {}
        self.transformation_history = []
        
        # Feature generation strategies
        self.polynomial_strategies = {
            "conservative": {"degree": 2, "interaction_only": True, "include_bias": False},
            "moderate": {"degree": 2, "interaction_only": False, "include_bias": False},
            "aggressive": {"degree": 3, "interaction_only": False, "include_bias": False}
        }
        
        # Feature selection methods
        self.selection_methods = {
            "univariate": self._univariate_selection,
            "recursive": self._recursive_feature_elimination,
            "model_based": self._model_based_selection,
            "correlation": self._correlation_selection,
            "variance": self._variance_selection
        }
        
    def engineer_features(self, 
                         X: pd.DataFrame, 
                         y: Optional[pd.Series] = None,
                         strategy: str = "comprehensive",
                         **kwargs) -> FeatureEngineeringResult:
        """
        Main feature engineering pipeline
        
        Args:
            X: Input features
            y: Target variable (optional)
            strategy: Engineering strategy ('basic', 'comprehensive', 'custom')
            **kwargs: Additional parameters for specific strategies
        """
        logger.info(f"Starting feature engineering with strategy: {strategy}")
        
        # Detect task type if not specified
        if self.task_type == "auto" and y is not None:
            self.task_type = self._detect_task_type(y)
        
        engineering_log = []
        original_features = list(X.columns)
        
        # Start with original data
        X_engineered = X.copy()
        
        if strategy == "basic":
            X_engineered = self._basic_feature_engineering(X_engineered, y, engineering_log)
        elif strategy == "comprehensive":
            X_engineered = self._comprehensive_feature_engineering(X_engineered, y, engineering_log, **kwargs)
        elif strategy == "custom":
            X_engineered = self._custom_feature_engineering(X_engineered, y, engineering_log, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Calculate feature importance if target is provided
        feature_importance = None
        if y is not None:
            feature_importance = self._calculate_feature_importance(X_engineered, y)
        
        # Track created and removed features
        created_features = [f for f in X_engineered.columns if f not in original_features]
        removed_features = [f for f in original_features if f not in X_engineered.columns]
        
        result = FeatureEngineeringResult(
            transformed_features=X_engineered,
            feature_names=list(X_engineered.columns),
            feature_importance=feature_importance,
            engineering_log=engineering_log,
            removed_features=removed_features,
            created_features=created_features
        )
        
        logger.info(f"Feature engineering completed: {len(X.columns)} → {len(X_engineered.columns)} features")
        return result
    
    def _basic_feature_engineering(self, X: pd.DataFrame, y: Optional[pd.Series], log: List[str]) -> pd.DataFrame:
        """Basic feature engineering pipeline"""
        
        # 1. Handle missing values
        X = self._handle_missing_values(X, strategy="simple")
        log.append("Applied basic missing value imputation")
        
        # 2. Encode categorical variables
        X = self._encode_categorical_features(X, strategy="basic")
        log.append("Applied basic categorical encoding")
        
        # 3. Create basic interaction features for numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) >= 2:
            X = self._create_polynomial_features(X, numerical_cols, degree=2, interaction_only=True)
            log.append("Created basic interaction features")
        
        return X
    
    def _comprehensive_feature_engineering(self, X: pd.DataFrame, y: Optional[pd.Series], 
                                         log: List[str], **kwargs) -> pd.DataFrame:
        """Comprehensive feature engineering pipeline"""
        
        # 1. Advanced missing value handling
        X = self._handle_missing_values(X, strategy="advanced")
        log.append("Applied advanced missing value handling")
        
        # 2. Outlier detection and treatment
        X = self._handle_outliers(X, method="iqr")
        log.append("Applied outlier detection and treatment")
        
        # 3. Advanced categorical encoding
        X = self._encode_categorical_features(X, strategy="advanced", target=y)
        log.append("Applied advanced categorical encoding")
        
        # 4. Create polynomial and interaction features
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) >= 2:
            poly_strategy = kwargs.get("polynomial_strategy", "moderate")
            poly_config = self.polynomial_strategies[poly_strategy]
            X = self._create_polynomial_features(X, numerical_cols, **poly_config)
            log.append(f"Created polynomial features (strategy: {poly_strategy})")
        
        # 5. Mathematical transformations
        X = self._create_mathematical_features(X)
        log.append("Created mathematical transformation features")
        
        # 6. Statistical features
        X = self._create_statistical_features(X)
        log.append("Created statistical aggregation features")
        
        # 7. Clustering-based features
        if len(numerical_cols) >= 3:
            X = self._create_clustering_features(X, numerical_cols)
            log.append("Created clustering-based features")
        
        # 8. Feature selection to prevent overfitting
        if y is not None and len(X.columns) > 100:
            X = self._intelligent_feature_selection(X, y, max_features=min(100, len(X.columns) // 2))
            log.append("Applied intelligent feature selection")
        
        return X
    
    def _custom_feature_engineering(self, X: pd.DataFrame, y: Optional[pd.Series], 
                                  log: List[str], **kwargs) -> pd.DataFrame:
        """Custom feature engineering based on specific parameters"""
        
        # Extract custom parameters
        operations = kwargs.get("operations", [])
        
        for operation in operations:
            if operation["type"] == "polynomial":
                degree = operation.get("degree", 2)
                interaction_only = operation.get("interaction_only", False)
                X = self._create_polynomial_features(X, X.select_dtypes(include=[np.number]).columns.tolist(), 
                                                   degree=degree, interaction_only=interaction_only)
                log.append(f"Created polynomial features (degree={degree}, interaction_only={interaction_only})")
            
            elif operation["type"] == "mathematical":
                X = self._create_mathematical_features(X, functions=operation.get("functions", ["log", "sqrt"]))
                log.append(f"Applied mathematical transformations: {operation.get('functions', ['log', 'sqrt'])}")
            
            elif operation["type"] == "binning":
                columns = operation.get("columns", [])
                n_bins = operation.get("n_bins", 5)
                X = self._create_binned_features(X, columns, n_bins)
                log.append(f"Created binned features for {len(columns)} columns")
            
            elif operation["type"] == "selection":
                method = operation.get("method", "univariate")
                k = operation.get("k", 50)
                X = self.select_features(X, y, method=method, k=k)
                log.append(f"Applied feature selection: {method} (k={k})")
        
        return X
    
    def _create_polynomial_features(self, X: pd.DataFrame, numerical_cols: List[str], 
                                  degree: int = 2, interaction_only: bool = False, 
                                  include_bias: bool = False) -> pd.DataFrame:
        """Create polynomial and interaction features"""
        if not numerical_cols or len(numerical_cols) < 2:
            return X
        
        # Limit to prevent explosion of features
        if len(numerical_cols) > 10:
            numerical_cols = numerical_cols[:10]
        
        X_subset = X[numerical_cols].copy()
        
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            X_poly = poly.fit_transform(X_subset)
        
        # Create feature names
        feature_names = poly.get_feature_names_out(numerical_cols)
        
        # Convert back to DataFrame
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
        
        # Remove original features (they're included in polynomial features)
        X_result = X.drop(columns=numerical_cols)
        
        # Add polynomial features
        X_result = pd.concat([X_result, X_poly_df], axis=1)
        
        return X_result
    
    def _create_mathematical_features(self, X: pd.DataFrame, 
                                    functions: List[str] = None) -> pd.DataFrame:
        """Create mathematical transformation features"""
        if functions is None:
            functions = ["log", "sqrt", "square", "reciprocal"]
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols:
            col_data = X[col]
            
            # Skip if all values are zero or negative for log/sqrt
            if "log" in functions and (col_data > 0).all():
                X[f"{col}_log"] = np.log(col_data)
            
            if "sqrt" in functions and (col_data >= 0).all():
                X[f"{col}_sqrt"] = np.sqrt(col_data)
            
            if "square" in functions:
                X[f"{col}_squared"] = col_data ** 2
            
            if "reciprocal" in functions and (col_data != 0).all():
                X[f"{col}_reciprocal"] = 1 / col_data
        
        return X
    
    def _create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical aggregation features"""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            return X
        
        # Row-wise statistics
        X_numerical = X[numerical_cols]
        X["row_mean"] = X_numerical.mean(axis=1)
        X["row_std"] = X_numerical.std(axis=1)
        X["row_max"] = X_numerical.max(axis=1)
        X["row_min"] = X_numerical.min(axis=1)
        X["row_median"] = X_numerical.median(axis=1)
        X["row_sum"] = X_numerical.sum(axis=1)
        
        # Feature ratios (if we have multiple numeric features)
        if len(numerical_cols) >= 2:
            for i, col1 in enumerate(numerical_cols[:5]):  # Limit to prevent explosion
                for col2 in numerical_cols[i+1:6]:
                    if (X[col2] != 0).all():
                        X[f"{col1}_{col2}_ratio"] = X[col1] / X[col2]
        
        return X
    
    def _create_clustering_features(self, X: pd.DataFrame, numerical_cols: List[str], 
                                  n_clusters: int = 5) -> pd.DataFrame:
        """Create clustering-based features"""
        if len(numerical_cols) < 2:
            return X
        
        # Standardize features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[numerical_cols])
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels as feature
        X[f"cluster_{n_clusters}"] = cluster_labels
        
        # Add distance to cluster centers
        cluster_centers = kmeans.cluster_centers_
        distances = np.linalg.norm(X_scaled[:, np.newaxis] - cluster_centers, axis=2)
        
        for i in range(n_clusters):
            X[f"dist_to_cluster_{i}"] = distances[:, i]
        
        return X
    
    def _create_binned_features(self, X: pd.DataFrame, columns: List[str], n_bins: int = 5) -> pd.DataFrame:
        """Create binned categorical features from numerical columns"""
        for col in columns:
            if col in X.columns and X[col].dtype in [np.number]:
                try:
                    X[f"{col}_binned"] = pd.cut(X[col], bins=n_bins, labels=False)
                except ValueError:
                    # Handle cases where binning fails (e.g., all values are the same)
                    X[f"{col}_binned"] = 0
        
        return X
    
    def _handle_missing_values(self, X: pd.DataFrame, strategy: str = "simple") -> pd.DataFrame:
        """Handle missing values with different strategies"""
        
        if strategy == "simple":
            # Simple imputation
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            
            # Fill numerical with median
            for col in numerical_cols:
                if X[col].isnull().any():
                    X[col].fillna(X[col].median(), inplace=True)
            
            # Fill categorical with mode
            for col in categorical_cols:
                if X[col].isnull().any():
                    X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'unknown', inplace=True)
        
        elif strategy == "advanced":
            # Advanced imputation with missing value indicators
            for col in X.columns:
                if X[col].isnull().any():
                    # Create missing value indicator
                    X[f"{col}_is_missing"] = X[col].isnull().astype(int)
                    
                    # Impute based on type
                    if X[col].dtype in [np.number]:
                        X[col].fillna(X[col].median(), inplace=True)
                    else:
                        X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'unknown', inplace=True)
        
        return X
    
    def _handle_outliers(self, X: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """Handle outliers in numerical features"""
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if method == "iqr":
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                X[col] = X[col].clip(lower_bound, upper_bound)
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame, strategy: str = "basic", 
                                   target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Encode categorical features"""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if strategy == "basic":
                # Simple one-hot encoding (with limitations to prevent explosion)
                unique_values = X[col].nunique()
                if unique_values <= 10:
                    dummies = pd.get_dummies(X[col], prefix=col)
                    X = pd.concat([X, dummies], axis=1)
                    X.drop(col, axis=1, inplace=True)
                else:
                    # Label encode high cardinality features
                    X[col] = pd.Categorical(X[col]).codes
            
            elif strategy == "advanced":
                # Advanced encoding strategies
                unique_values = X[col].nunique()
                
                if unique_values <= 5:
                    # One-hot for low cardinality
                    dummies = pd.get_dummies(X[col], prefix=col)
                    X = pd.concat([X, dummies], axis=1)
                    X.drop(col, axis=1, inplace=True)
                elif unique_values <= 20:
                    # Target encoding if target is provided
                    if target is not None:
                        X[col] = self._target_encode(X[col], target)
                    else:
                        X[col] = pd.Categorical(X[col]).codes
                else:
                    # Label encode high cardinality
                    X[col] = pd.Categorical(X[col]).codes
        
        return X
    
    def _target_encode(self, series: pd.Series, target: pd.Series) -> pd.Series:
        """Target encoding for categorical variables"""
        global_mean = target.mean()
        target_mapping = target.groupby(series).mean()
        
        # Use global mean for unseen categories
        return series.map(target_mapping).fillna(global_mean)
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate feature importance using multiple methods"""
        
        if self.task_type == "classification":
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Fit model and get importance
        rf_model.fit(X, y)
        importance_scores = dict(zip(X.columns, rf_model.feature_importances_))
        
        # Sort by importance
        importance_scores = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
        
        return importance_scores
    
    def _intelligent_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                     max_features: int = 50) -> pd.DataFrame:
        """Intelligent feature selection combining multiple methods"""
        
        if len(X.columns) <= max_features:
            return X
        
        # Method 1: Remove low variance features
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        X_selected = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        X = X[selected_features]
        
        if len(X.columns) <= max_features:
            return X
        
        # Method 2: Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X = X.drop(columns=to_drop)
        
        if len(X.columns) <= max_features:
            return X
        
        # Method 3: Select top features by importance
        importance = self._calculate_feature_importance(X, y)
        top_features = list(importance.keys())[:max_features]
        X = X[top_features]
        
        return X
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = "univariate", 
                       k: int = 50, **kwargs) -> pd.DataFrame:
        """Feature selection using specified method"""
        
        if method in self.selection_methods:
            return self.selection_methods[method](X, y, k, **kwargs)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _univariate_selection(self, X: pd.DataFrame, y: pd.Series, k: int, **kwargs) -> pd.DataFrame:
        """Univariate feature selection"""
        
        if self.task_type == "classification":
            score_func = f_classif
        else:
            score_func = f_regression
        
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        return X[selected_features]
    
    def _recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, k: int, **kwargs) -> pd.DataFrame:
        """Recursive feature elimination"""
        
        if self.task_type == "classification":
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        selector = RFE(estimator=estimator, n_features_to_select=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        return X[selected_features]
    
    def _model_based_selection(self, X: pd.DataFrame, y: pd.Series, k: int, **kwargs) -> pd.DataFrame:
        """Model-based feature selection"""
        
        if self.task_type == "classification":
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = LassoCV(random_state=42)
        
        selector = SelectFromModel(estimator, max_features=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        return X[selected_features]
    
    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series, k: int, 
                             threshold: float = 0.95, **kwargs) -> pd.DataFrame:
        """Remove highly correlated features"""
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find features to remove
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # Remove correlated features
        X_selected = X.drop(columns=to_drop)
        
        # If still too many features, select top k by importance
        if len(X_selected.columns) > k:
            importance = self._calculate_feature_importance(X_selected, y)
            top_features = list(importance.keys())[:k]
            X_selected = X_selected[top_features]
        
        return X_selected
    
    def _variance_selection(self, X: pd.DataFrame, y: pd.Series, k: int, 
                          threshold: float = 0.01, **kwargs) -> pd.DataFrame:
        """Remove low variance features"""
        
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        X_result = X[selected_features]
        
        # If still too many features, select top k by importance
        if len(X_result.columns) > k:
            importance = self._calculate_feature_importance(X_result, y)
            top_features = list(importance.keys())[:k]
            X_result = X_result[top_features]
        
        return X_result
    
    def _detect_task_type(self, y: pd.Series) -> str:
        """Detect if task is classification or regression"""
        
        if y.dtype == 'object' or y.dtype.name == 'category':
            return "classification"
        elif y.nunique() < 20 and len(y) > 100:  # Heuristic for classification
            return "classification"
        else:
            return "regression"
    
    def create_time_series_features(self, df: pd.DataFrame, date_column: str, 
                                  window_sizes: List[str] = None) -> pd.DataFrame:
        """Create time-series specific features"""
        
        if window_sizes is None:
            window_sizes = ["7D", "30D", "90D"]
        
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column).sort_index()
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Rolling statistics
            for window in window_sizes:
                df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window).mean()
                df[f"{col}_rolling_std_{window}"] = df[col].rolling(window).std()
                df[f"{col}_rolling_max_{window}"] = df[col].rolling(window).max()
                df[f"{col}_rolling_min_{window}"] = df[col].rolling(window).min()
            
            # Lag features
            for lag in [1, 7, 30]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            
            # Difference features
            df[f"{col}_diff_1"] = df[col].diff(1)
            df[f"{col}_diff_7"] = df[col].diff(7)
        
        # Time-based features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['quarter'] = df.index.quarter
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        return df.reset_index()


class FeatureEngineeringRecommender:
    """Intelligent feature engineering recommendations"""
    
    def __init__(self):
        self.recommendations = {
            "high_cardinality_categorical": "Consider target encoding or feature hashing",
            "many_missing_values": "Create missing value indicators and use advanced imputation",
            "skewed_distribution": "Apply log or box-cox transformation",
            "many_numerical_features": "Consider PCA or feature clustering",
            "time_series_data": "Create lag features, rolling statistics, and time-based features",
            "imbalanced_classes": "Consider SMOTE or class weight balancing",
            "high_correlation": "Remove redundant features or use regularization"
        }
    
    def analyze_dataset(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Analyze dataset and provide engineering recommendations"""
        
        analysis = {
            "dataset_shape": X.shape,
            "missing_values": X.isnull().sum().sum(),
            "numerical_features": len(X.select_dtypes(include=[np.number]).columns),
            "categorical_features": len(X.select_dtypes(include=['object', 'category']).columns),
            "recommendations": []
        }
        
        # Check for high cardinality categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X[col].nunique() > 50:
                analysis["recommendations"].append({
                    "type": "high_cardinality_categorical",
                    "feature": col,
                    "cardinality": X[col].nunique(),
                    "recommendation": self.recommendations["high_cardinality_categorical"]
                })
        
        # Check for missing values
        missing_cols = X.columns[X.isnull().any()].tolist()
        if missing_cols:
            analysis["recommendations"].append({
                "type": "missing_values",
                "features": missing_cols,
                "recommendation": self.recommendations["many_missing_values"]
            })
        
        # Check for skewed distributions
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            skewness = X[col].skew()
            if abs(skewness) > 2:
                analysis["recommendations"].append({
                    "type": "skewed_distribution",
                    "feature": col,
                    "skewness": skewness,
                    "recommendation": self.recommendations["skewed_distribution"]
                })
        
        # Check for high correlation
        if len(numerical_cols) > 1:
            corr_matrix = X[numerical_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.9:
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]
                        ))
            
            if high_corr_pairs:
                analysis["recommendations"].append({
                    "type": "high_correlation",
                    "feature_pairs": high_corr_pairs[:5],  # Show top 5
                    "recommendation": self.recommendations["high_correlation"]
                })
        
        # Check for class imbalance (if classification target provided)
        if y is not None and y.dtype == 'object' or (y.nunique() < 20 and len(y) > 100):
            class_counts = y.value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            if imbalance_ratio > 5:
                analysis["recommendations"].append({
                    "type": "imbalanced_classes",
                    "imbalance_ratio": imbalance_ratio,
                    "class_distribution": class_counts.to_dict(),
                    "recommendation": self.recommendations["imbalanced_classes"]
                })
        
        # Check for many numerical features (PCA recommendation)
        if len(numerical_cols) > 20:
            analysis["recommendations"].append({
                "type": "many_numerical_features",
                "feature_count": len(numerical_cols),
                "recommendation": self.recommendations["many_numerical_features"]
            })
        
        return analysis


# Utility functions for CLI integration
def engineer_features_from_prompt(prompt: str, X: pd.DataFrame, y: Optional[pd.Series] = None) -> FeatureEngineeringResult:
    """Engineer features based on natural language prompt"""
    
    engineer = AutomatedFeatureEngineer()
    
    # Parse prompt to determine strategy and parameters
    prompt_lower = prompt.lower()
    
    # Determine strategy
    if "basic" in prompt_lower or "simple" in prompt_lower:
        strategy = "basic"
        kwargs = {}
    elif "custom" in prompt_lower or "specific" in prompt_lower:
        strategy = "custom" 
        kwargs = parse_custom_operations_from_prompt(prompt)
    else:
        strategy = "comprehensive"
        kwargs = {}
        
        # Parse polynomial strategy
        if "conservative" in prompt_lower:
            kwargs["polynomial_strategy"] = "conservative"
        elif "aggressive" in prompt_lower:
            kwargs["polynomial_strategy"] = "aggressive"
        else:
            kwargs["polynomial_strategy"] = "moderate"
    
    return engineer.engineer_features(X, y, strategy=strategy, **kwargs)


def parse_custom_operations_from_prompt(prompt: str) -> Dict[str, Any]:
    """Parse custom operations from natural language prompt"""
    
    operations = []
    prompt_lower = prompt.lower()
    
    # Parse polynomial operations
    if "polynomial" in prompt_lower:
        operation = {"type": "polynomial"}
        
        # Extract degree
        import re
        degree_match = re.search(r"degree\s*(\d+)", prompt_lower)
        if degree_match:
            operation["degree"] = int(degree_match.group(1))
        
        if "interaction" in prompt_lower and "only" in prompt_lower:
            operation["interaction_only"] = True
            
        operations.append(operation)
    
    # Parse mathematical operations
    math_functions = []
    if "log" in prompt_lower:
        math_functions.append("log")
    if "sqrt" in prompt_lower:
        math_functions.append("sqrt")
    if "square" in prompt_lower:
        math_functions.append("square")
    if "reciprocal" in prompt_lower:
        math_functions.append("reciprocal")
    
    if math_functions:
        operations.append({
            "type": "mathematical",
            "functions": math_functions
        })
    
    # Parse binning operations
    if "bin" in prompt_lower:
        operation = {"type": "binning"}
        
        # Extract number of bins
        bins_match = re.search(r"(\d+)\s*bins?", prompt_lower)
        if bins_match:
            operation["n_bins"] = int(bins_match.group(1))
        
        operations.append(operation)
    
    # Parse selection operations
    selection_methods = ["univariate", "recursive", "model_based", "correlation", "variance"]
    for method in selection_methods:
        if method in prompt_lower or method.replace("_", " ") in prompt_lower:
            operation = {"type": "selection", "method": method}
            
            # Extract k
            k_match = re.search(r"(?:top|best|select)\s*(\d+)", prompt_lower)
            if k_match:
                operation["k"] = int(k_match.group(1))
            
            operations.append(operation)
            break
    
    return {"operations": operations}


def print_engineering_report(result: FeatureEngineeringResult, show_details: bool = True):
    """Print a comprehensive feature engineering report"""
    
    print("\n" + "="*80)
    print("🔧 FEATURE ENGINEERING REPORT")
    print("="*80)
    
    # Basic statistics
    print(f"📊 TRANSFORMATION SUMMARY:")
    print(f"   Original features: {len(result.feature_names) - len(result.created_features or [])}")
    print(f"   Final features: {len(result.feature_names)}")
    print(f"   Features created: {len(result.created_features or [])}")
    print(f"   Features removed: {len(result.removed_features or [])}")
    
    # Engineering log
    if result.engineering_log and show_details:
        print(f"\n⚙️  ENGINEERING OPERATIONS:")
        for i, operation in enumerate(result.engineering_log, 1):
            print(f"   {i}. {operation}")
    
    # Feature importance (top 10)
    if result.feature_importance and show_details:
        print(f"\n🏆 TOP FEATURE IMPORTANCE:")
        top_features = list(result.feature_importance.items())[:10]
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"   {i:2d}. {feature:<30} {importance:.4f}")
    
    # Created features
    if result.created_features and show_details:
        print(f"\n✨ CREATED FEATURES ({len(result.created_features)}):")
        feature_types = {}
        for feature in result.created_features[:20]:  # Show first 20
            # Categorize feature types
            if "_rolling_" in feature:
                feature_types.setdefault("Rolling Statistics", []).append(feature)
            elif "_lag_" in feature:
                feature_types.setdefault("Lag Features", []).append(feature)
            elif "_ratio" in feature:
                feature_types.setdefault("Ratio Features", []).append(feature)
            elif "_squared" in feature or "_log" in feature or "_sqrt" in feature:
                feature_types.setdefault("Mathematical", []).append(feature)
            elif "cluster" in feature:
                feature_types.setdefault("Clustering", []).append(feature)
            elif "row_" in feature:
                feature_types.setdefault("Statistical", []).append(feature)
            else:
                feature_types.setdefault("Other", []).append(feature)
        
        for category, features in feature_types.items():
            print(f"   {category}: {len(features)} features")
            if show_details:
                for feature in features[:5]:  # Show first 5 in each category
                    print(f"     • {feature}")
                if len(features) > 5:
                    print(f"     • ... and {len(features) - 5} more")
    
    # Removed features
    if result.removed_features and show_details:
        print(f"\n❌ REMOVED FEATURES ({len(result.removed_features)}):")
        for feature in result.removed_features[:10]:  # Show first 10
            print(f"   • {feature}")
        if len(result.removed_features) > 10:
            print(f"   • ... and {len(result.removed_features) - 10} more")
    
    print("="*80)


class AdvancedFeatureEngineer:
    """Advanced feature engineering with domain-specific techniques"""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.transformers = {}
    
    def create_domain_specific_features(self, X: pd.DataFrame, domain: str = "general") -> pd.DataFrame:
        """Create domain-specific features"""
        
        if domain == "finance":
            return self._create_financial_features(X)
        elif domain == "text":
            return self._create_text_features(X)
        elif domain == "image":
            return self._create_image_features(X)
        elif domain == "time_series":
            return self._create_time_series_features(X)
        else:
            return self._create_general_features(X)
    
    def _create_financial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create financial domain-specific features"""
        
        # Assuming common financial columns
        price_columns = [col for col in X.columns if any(keyword in col.lower() 
                        for keyword in ['price', 'value', 'amount', 'cost'])]
        
        for col in price_columns:
            if X[col].dtype in [np.number]:
                # Price-based features
                X[f"{col}_pct_change"] = X[col].pct_change()
                X[f"{col}_volatility"] = X[col].rolling(window=30).std()
                X[f"{col}_momentum"] = X[col] / X[col].shift(10) - 1
                
                # Moving averages
                X[f"{col}_sma_10"] = X[col].rolling(window=10).mean()
                X[f"{col}_sma_30"] = X[col].rolling(window=30).mean()
                X[f"{col}_rsi"] = self._calculate_rsi(X[col])
        
        return X
    
    def _create_text_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create text domain-specific features"""
        
        text_columns = X.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            if X[col].dtype == 'object':
                # Text statistics
                X[f"{col}_length"] = X[col].astype(str).str.len()
                X[f"{col}_word_count"] = X[col].astype(str).str.split().str.len()
                X[f"{col}_char_count"] = X[col].astype(str).str.len()
                X[f"{col}_digit_count"] = X[col].astype(str).str.count(r'\d')
                X[f"{col}_upper_count"] = X[col].astype(str).str.count(r'[A-Z]')
                
                # Sentiment analysis (simplified)
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
                
                X[f"{col}_positive_words"] = X[col].astype(str).str.lower().apply(
                    lambda x: sum(1 for word in positive_words if word in x)
                )
                X[f"{col}_negative_words"] = X[col].astype(str).str.lower().apply(
                    lambda x: sum(1 for word in negative_words if word in x)
                )
        
        return X
    
    def _create_time_series_features(self, X: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
        """Create comprehensive time series features"""
        
        # Find date column if not specified
        if date_col is None:
            date_columns = []
            for col in X.columns:
                if X[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                    date_columns.append(col)
            
            if date_columns:
                date_col = date_columns[0]
            else:
                return X
        
        if date_col not in X.columns:
            return X
        
        # Ensure datetime type
        X[date_col] = pd.to_datetime(X[date_col])
        
        # Cyclical time features
        X['hour_sin'] = np.sin(2 * np.pi * X[date_col].dt.hour / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X[date_col].dt.hour / 24)
        X['day_sin'] = np.sin(2 * np.pi * X[date_col].dt.day / 31)
        X['day_cos'] = np.cos(2 * np.pi * X[date_col].dt.day / 31)
        X['month_sin'] = np.sin(2 * np.pi * X[date_col].dt.month / 12)
        X['month_cos'] = np.cos(2 * np.pi * X[date_col].dt.month / 12)
        X['dayofweek_sin'] = np.sin(2 * np.pi * X[date_col].dt.dayofweek / 7)
        X['dayofweek_cos'] = np.cos(2 * np.pi * X[date_col].dt.dayofweek / 7)
        
        # Holiday indicators (simplified)
        X['is_month_start'] = X[date_col].dt.is_month_start.astype(int)
        X['is_month_end'] = X[date_col].dt.is_month_end.astype(int)
        X['is_quarter_start'] = X[date_col].dt.is_quarter_start.astype(int)
        X['is_quarter_end'] = X[date_col].dt.is_quarter_end.astype(int)
        
        return X
    
    def _create_general_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create general-purpose advanced features"""
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Interaction features with feature selection
        if len(numerical_cols) >= 2:
            # Create interactions only for highly correlated features
            corr_matrix = X[numerical_cols].corr().abs()
            
            for i, col1 in enumerate(numerical_cols):
                for j, col2 in enumerate(numerical_cols[i+1:], i+1):
                    if corr_matrix.iloc[i, j] > 0.3:  # Only create interactions for correlated features
                        X[f"{col1}_{col2}_interaction"] = X[col1] * X[col2]
                        X[f"{col1}_{col2}_ratio"] = X[col1] / (X[col2] + 1e-8)
        
        # Advanced statistical features
        if len(numerical_cols) >= 3:
            X_num = X[numerical_cols]
            X['feature_sum'] = X_num.sum(axis=1)
            X['feature_mean'] = X_num.mean(axis=1)
            X['feature_std'] = X_num.std(axis=1)
            X['feature_skew'] = X_num.skew(axis=1)
            X['feature_kurtosis'] = X_num.kurtosis(axis=1)
        
        return X
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


# Main feature engineering function for CLI
def create_comprehensive_features(X: pd.DataFrame, 
                                y: Optional[pd.Series] = None,
                                strategy: str = "comprehensive",
                                domain: str = "general") -> FeatureEngineeringResult:
    """Create comprehensive feature set with domain-specific enhancements"""
    
    # Basic feature engineering
    basic_engineer = AutomatedFeatureEngineer()
    result = basic_engineer.engineer_features(X, y, strategy=strategy)
    
    # Add domain-specific features
    advanced_engineer = AdvancedFeatureEngineer()
    X_enhanced = advanced_engineer.create_domain_specific_features(
        result.transformed_features, domain=domain
    )
    
    # Update result
    new_features = [col for col in X_enhanced.columns if col not in result.transformed_features.columns]
    result.transformed_features = X_enhanced
    result.feature_names = list(X_enhanced.columns)
    result.created_features.extend(new_features)
    result.engineering_log.append(f"Added {len(new_features)} domain-specific features ({domain})")
    
    # Recalculate feature importance
    if y is not None:
        result.feature_importance = basic_engineer._calculate_feature_importance(X_enhanced, y)
    
    return result