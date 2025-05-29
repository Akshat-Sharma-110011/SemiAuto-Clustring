"""
Data Preprocessing Module for Clustering (API Version)

This module handles the preprocessing of data for clustering models, including:
- Handling missing values
- Handling duplicate values
- Handling outliers
- Handling skewed data
- Scaling numerical features
- Encoding categorical features
- Dimensionality reduction for correlated columns

The preprocessing steps are configured based on API requests and feature_store.yaml file.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import dill as cloudpickle
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OneHotEncoder
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.stats as stats
from pydantic import BaseModel
from collections import defaultdict

# Set up the logger
from src.logger import section, configure_logger

# Configure logger
configure_logger()
logger = logging.getLogger("Data Preprocessing")


def get_dataset_name():
    """Lazily load dataset name when needed"""
    try:
        with open('intel.yaml', 'r') as f:
            config = yaml.safe_load(f)
            return config['dataset_name']
    except FileNotFoundError:
        return "default_dataset"


dataset_name = get_dataset_name()


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling outliers using either IQR or Z-Score method.
    """

    def __init__(self, method: str = 'IQR', columns: List[str] = None):
        self.method = method
        self.columns = columns
        self.thresholds = {}

        if method not in ['IQR', 'Z-Score']:
            raise ValueError("Method must be either 'IQR' or 'Z-Score'")

        logger.info(f"Initialized OutlierHandler with method: {method}")

    def fit(self, X, y=None):
        if not self.columns:
            logger.warning("No columns provided for outlier handling")
            return self

        for col in self.columns:
            if col not in X.columns:
                continue

            if self.method == 'IQR':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.thresholds[col] = {'lower': Q1 - 1.5 * IQR, 'upper': Q3 + 1.5 * IQR}
            elif self.method == 'Z-Score':
                mean = X[col].mean()
                std = X[col].std()
                self.thresholds[col] = {'mean': mean, 'std': std}
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            if col not in X_transformed.columns or col not in self.thresholds:
                continue

            if self.method == 'IQR':
                lower = self.thresholds[col]['lower']
                upper = self.thresholds[col]['upper']
                X_transformed[col] = X_transformed[col].clip(lower, upper)
            elif self.method == 'Z-Score':
                mean = self.thresholds[col]['mean']
                std = self.thresholds[col]['std']
                z_scores = (X_transformed[col] - mean) / std
                X_transformed[col] = np.where(z_scores > 3, mean + 3 * std,
                                              np.where(z_scores < -3, mean - 3 * std, X_transformed[col]))
        return X_transformed


class IDColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, id_cols: List[str]):
        self.id_columns = id_cols
        self.columns_to_drop = []
        logger.info(f"Initialized ID column dropper with columns: {id_cols}")

    def fit(self, X, y=None):
        try:
            # Only drop columns that actually exist
            self.columns_to_drop = [col for col in self.id_columns if col in X.columns]
            if self.columns_to_drop:
                logger.info(f"Will drop ID columns: {self.columns_to_drop}")
            else:
                logger.warning(f"No ID columns found in data: {self.id_columns}")
            return self
        except Exception as e:
            logger.error(f"Error in IDColumnDropper fit: {str(e)}")
            raise

    def transform(self, X):
        try:
            if not self.columns_to_drop:
                return X.copy()

            logger.info(f"Dropping ID columns: {self.columns_to_drop}")
            return X.drop(columns=self.columns_to_drop, errors='ignore')
        except Exception as e:
            logger.error(f"Error in IDColumnDropper transform: {str(e)}")
            raise


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling missing values using specified method.
    """

    def __init__(self, method: str = 'mean', columns: List[str] = None):
        self.method = method
        self.columns = columns
        self.fill_values = {}

        if method not in ['mean', 'median', 'mode', 'drop']:
            raise ValueError("Method must be one of 'mean', 'median', 'mode', or 'drop'")

        logger.info(f"Initialized MissingValueHandler with method: {method}")

    def fit(self, X, y=None):
        if self.method == 'drop':
            return self

        for col in self.columns:
            if col not in X.columns:
                continue

            if self.method in ['mean', 'median']:
                if pd.api.types.is_numeric_dtype(X[col]):
                    self.fill_values[col] = X[col].mean() if self.method == 'mean' else X[col].median()
                else:
                    self.fill_values[col] = X[col].mode()[0]
            elif self.method == 'mode':
                self.fill_values[col] = X[col].mode()[0]
        return self

    def transform(self, X):
        if self.method == 'drop':
            return X.dropna(subset=self.columns)

        for col in self.columns:
            if col in X.columns and col in self.fill_values:
                X[col] = X[col].fillna(self.fill_values[col])
        return X


class SkewedDataHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling skewed data using power transformers.
    """

    def __init__(self, method: str = 'yeo-johnson', columns: List[str] = None):
        self.method = method
        self.columns = columns
        self.transformers = {}

        if method not in ['yeo-johnson', 'box-cox']:
            raise ValueError("Method must be either 'yeo-johnson' or 'box-cox'")

        logger.info(f"Initialized SkewedDataHandler with method: {method}")

    def fit(self, X, y=None):
        for col in self.columns:
            if col not in X.columns:
                continue

            transformer = PowerTransformer(method=self.method, standardize=True)
            if self.method == 'box-cox' and X[col].min() <= 0:
                shift_value = abs(X[col].min()) + 1.0
                self.transformers[col] = {'transformer': transformer, 'shift': shift_value}
                transformer.fit(X[col].add(shift_value).values.reshape(-1, 1))
            else:
                self.transformers[col] = {'transformer': transformer, 'shift': 0.0}
                transformer.fit(X[col].values.reshape(-1, 1))
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            if col not in X_transformed.columns or col not in self.transformers:
                continue

            t_info = self.transformers[col]
            if t_info['shift'] > 0:
                transformed = t_info['transformer'].transform(
                    X_transformed[col].add(t_info['shift']).values.reshape(-1, 1))
            else:
                transformed = t_info['transformer'].transform(X_transformed[col].values.reshape(-1, 1))
            X_transformed[col] = transformed.flatten()
        return X_transformed


class NumericalScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for scaling numerical features.
    """

    def __init__(self, method: str = 'standard', columns: List[str] = None):
        self.method = method
        self.columns = columns
        self.scalers = {}

        if method not in ['standard', 'robust', 'minmax']:
            raise ValueError("Method must be one of 'standard', 'robust', or 'minmax'")

        logger.info(f"Initialized NumericalScaler with method: {method}")

    def fit(self, X, y=None):
        for col in self.columns:
            if col not in X.columns:
                continue

            if self.method == 'standard':
                self.scalers[col] = StandardScaler().fit(X[col].values.reshape(-1, 1))
            elif self.method == 'robust':
                self.scalers[col] = RobustScaler().fit(X[col].values.reshape(-1, 1))
            elif self.method == 'minmax':
                self.scalers[col] = MinMaxScaler().fit(X[col].values.reshape(-1, 1))
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            if col in X_transformed.columns and col in self.scalers:
                scaler = self.scalers[col]
                X_transformed[col] = scaler.transform(X_transformed[col].values.reshape(-1, 1)).flatten()
        return X_transformed


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer for encoding categorical features.
    """

    def __init__(self, method: str = 'onehot', columns: List[str] = None, drop_first: bool = True):
        self.method = method
        self.columns = columns
        self.drop_first = drop_first
        self.encoders = {}
        self.dummy_columns = {}

        if method not in ['onehot', 'dummies']:
            raise ValueError("Method must be one of 'onehot' or 'dummies'")

        logger.info(f"Initialized CategoricalEncoder with method: {method}")

    def fit(self, X, y=None):
        for col in self.columns:
            if col not in X.columns:
                continue

            if self.method == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, drop='first' if self.drop_first else None)
                encoder.fit(X[col].values.reshape(-1, 1))
                self.encoders[col] = encoder
                self.dummy_columns[col] = encoder.get_feature_names_out([col])
            elif self.method == 'dummies':
                self.dummy_columns[col] = [f"{col}_{val}" for val in X[col].unique()[1:]] if self.drop_first else [
                    f"{col}_{val}" for val in X[col].unique()]
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            if col not in X_transformed.columns:
                continue

            if self.method == 'onehot':
                encoder = self.encoders[col]
                encoded = encoder.transform(X_transformed[col].values.reshape(-1, 1))
                encoded_df = pd.DataFrame(encoded, columns=self.dummy_columns[col], index=X_transformed.index)
                X_transformed = pd.concat([X_transformed.drop(columns=[col]), encoded_df], axis=1)
            elif self.method == 'dummies':
                dummies = pd.get_dummies(X_transformed[col], prefix=col, drop_first=self.drop_first)
                X_transformed = pd.concat([X_transformed.drop(columns=[col]), dummies], axis=1)
        return X_transformed


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    Handles dimensionality reduction for correlated columns
    """

    def __init__(self, method: str = 'PCA', n_components: Union[int, float] = 0.95, groups: List[List[str]] = None):
        self.method = method
        self.n_components = n_components
        self.groups = groups or []
        self.reducers = {}

        if method not in ['PCA', 'TSNE']:
            raise ValueError("Method must be either 'PCA' or 'TSNE'")

    def fit(self, X, y=None):
        for group in self.groups:
            valid_cols = [col for col in group if col in X.columns]
            if len(valid_cols) < 2:
                continue

            if self.method == 'PCA':
                reducer = PCA(n_components=self.n_components)
                reducer.fit(X[valid_cols])
                self.reducers[tuple(valid_cols)] = reducer
            elif self.method == 'TSNE':
                # For t-SNE, we need to determine n_components differently
                n_comp = min(2, len(valid_cols)) if isinstance(self.n_components, float) else self.n_components
                reducer = TSNE(n_components=n_comp, random_state=42)
                # t-SNE doesn't have fit method, so we store the configuration
                self.reducers[tuple(valid_cols)] = {'method': 'TSNE', 'n_components': n_comp}
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for cols, reducer in self.reducers.items():
            cols = list(cols)

            if self.method == 'PCA':
                components = reducer.transform(X_transformed[cols])
                n_components = components.shape[1]
                new_cols = [f"PCA_{'_'.join(cols)}_{i + 1}" for i in range(n_components)]
            elif self.method == 'TSNE':
                tsne = TSNE(n_components=reducer['n_components'], random_state=42)
                components = tsne.fit_transform(X_transformed[cols])
                n_components = components.shape[1]
                new_cols = [f"TSNE_{'_'.join(cols)}_{i + 1}" for i in range(n_components)]

            components_df = pd.DataFrame(components, columns=new_cols, index=X_transformed.index)
            X_transformed = X_transformed.drop(columns=cols)
            X_transformed = pd.concat([X_transformed, components_df], axis=1)
        return X_transformed


class PreprocessingParameters(BaseModel):
    """Pydantic model for preprocessing parameters"""
    missing_values_method: str = 'mean'
    missing_values_columns: List[str] = []
    handle_duplicates: bool = True
    outliers_method: Optional[str] = None
    outliers_columns: List[str] = []
    skewness_method: Optional[str] = None
    skewness_columns: List[str] = []
    scaling_method: Optional[str] = None
    scaling_columns: List[str] = []
    categorical_encoding_method: Optional[str] = None
    categorical_columns: List[str] = []
    drop_first: bool = True
    dr_method: Optional[str] = None
    dr_components: Optional[Union[int, float]] = 0.95


class PreprocessingPipeline:
    def __init__(self, config: Dict[str, Any], params: PreprocessingParameters):
        self.config = config
        self.params = params
        self.dataset_name = config.get('dataset_name')
        self._configured = False

        # Accept either feature_store dictionary or path
        if 'feature_store' in config:
            self.feature_store = config['feature_store']
            logger.info("Loaded feature store from provided dictionary")
        elif 'feature_store_path' in config:
            feature_store_path = config['feature_store_path']
            if os.path.exists(feature_store_path):
                with open(feature_store_path, 'r') as f:
                    self.feature_store = yaml.safe_load(f)
                logger.info(f"Loaded feature store from: {feature_store_path}")
            else:
                logger.warning(f"Feature store not found at: {feature_store_path}")
                self.feature_store = {}
        else:
            logger.warning("No feature store provided in config")
            self.feature_store = {}

        # Initialize handlers
        self.missing_handler = None
        self.outlier_handler = None
        self.skewed_handler = None
        self.numerical_scaler = None
        self.categorical_encoder = None
        self.id_dropper = None
        self.dimensionality_reducer = None
        self.correlated_groups = self._get_correlated_groups()

    # ADD MISSING METHODS HERE
    def handle_missing_values(self, method: str, columns: List[str]):
        """Set missing value handling parameters"""
        self.params.missing_values_method = method
        self.params.missing_values_columns = columns

    def handle_outliers(self, method: str, columns: List[str]):
        """Set outlier handling parameters"""
        self.params.outliers_method = method
        self.params.outliers_columns = columns

    def handle_skewed_data(self, method: str, columns: List[str]):
        """Set skewed data handling parameters"""
        self.params.skewness_method = method
        self.params.skewness_columns = columns

    def scale_numerical_features(self, method: str, columns: List[str]):
        """Set numerical scaling parameters"""
        self.params.scaling_method = method
        self.params.scaling_columns = columns

    def encode_categorical_features(self, method: str, columns: List[str], drop_first: bool):
        """Set categorical encoding parameters"""
        self.params.categorical_encoding_method = method
        self.params.categorical_columns = columns
        self.params.drop_first = drop_first

    def _get_correlated_groups(self):
        """Get correlated column groups from feature store"""
        correlated_cols = self.feature_store.get('correlated_cols', {})
        graph = defaultdict(set)
        for col, correlations in correlated_cols.items():
            for corr in correlations:
                neighbor = corr['column']
                graph[col].add(neighbor)
                graph[neighbor].add(col)

        visited = set()
        groups = []
        for node in graph:
            if node not in visited:
                stack = [node]
                component = []
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(graph[current] - visited)
                if len(component) > 1:
                    groups.append(component)
        return groups

    def configure_pipeline(self):
        """Configure all pipeline components AUTOMATICALLY from parameters"""
        # Configure ID dropper
        id_cols = self.feature_store.get('id_cols', [])
        if id_cols:
            self.id_dropper = IDColumnDropper(id_cols)
            logger.info(f"Configured ID dropper for columns: {id_cols}")

        # Configure other components ONLY if parameters are provided
        if self.params.missing_values_method:
            self.missing_handler = MissingValueHandler(
                method=self.params.missing_values_method,
                columns=self.params.missing_values_columns
            )
            logger.info(f"Configured missing value handler: {self.params.missing_values_method}")

        if self.params.outliers_method:
            self.outlier_handler = OutlierHandler(
                method=self.params.outliers_method,
                columns=self.params.outliers_columns
            )
            logger.info(f"Configured outlier handler: {self.params.outliers_method}")

        if self.params.skewness_method:
            self.skewed_handler = SkewedDataHandler(
                method=self.params.skewness_method,
                columns=self.params.skewness_columns
            )
            logger.info(f"Configured skewed data handler: {self.params.skewness_method}")

        if self.params.scaling_method:
            self.numerical_scaler = NumericalScaler(
                method=self.params.scaling_method,
                columns=self.params.scaling_columns
            )
            logger.info(f"Configured numerical scaler: {self.params.scaling_method}")

        if self.params.categorical_encoding_method:
            self.categorical_encoder = CategoricalEncoder(
                method=self.params.categorical_encoding_method,
                columns=self.params.categorical_columns,
                drop_first=self.params.drop_first
            )
            logger.info(f"Configured categorical encoder: {self.params.categorical_encoding_method}")

        # Configure dimensionality reducer
        if self.params.dr_method and self.correlated_groups:
            self.dimensionality_reducer = DimensionalityReducer(
                method=self.params.dr_method,
                n_components=self.params.dr_components,
                groups=self.correlated_groups
            )

    def fit(self, X: pd.DataFrame) -> None:
        """Fit all pipeline components"""
        logger.info(f"Fitting pipeline on data with columns: {list(X.columns)}")

        if self.id_dropper:
            self.id_dropper.fit(X)
        if self.missing_handler:
            self.missing_handler.fit(X)
        if self.outlier_handler:
            self.outlier_handler.fit(X)
        if self.skewed_handler:
            self.skewed_handler.fit(X)
        if self.numerical_scaler:
            self.numerical_scaler.fit(X)
        if self.categorical_encoder:
            self.categorical_encoder.fit(X)
        if self.dimensionality_reducer:
            self.dimensionality_reducer.fit(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through all pipeline steps"""
        logger.info(f"Starting transformation with columns: {list(X.columns)}")
        transformed = X.copy()

        # Step 1: Drop ID columns FIRST
        if self.id_dropper:
            transformed = self.id_dropper.transform(transformed)
            logger.info(f"After dropping ID columns: {list(transformed.columns)}")
        else:
            logger.warning("ID dropper not configured - ID columns will not be dropped")

        # Step 2: Apply dimensionality reduction (moved here from Step 8)
        if self.dimensionality_reducer:
            transformed = self.dimensionality_reducer.transform(transformed)
            logger.info(f"After dimensionality reduction: {list(transformed.columns)}")

        # Step 3: Handle duplicates
        if self.params.handle_duplicates:
            initial_rows = len(transformed)
            transformed = transformed.drop_duplicates()
            final_rows = len(transformed)
            if initial_rows != final_rows:
                logger.info(f"Removed {initial_rows - final_rows} duplicate rows")

        # Step 4: Handle missing values
        if self.missing_handler:
            transformed = self.missing_handler.transform(transformed)
            logger.info(f"After handling missing values: {list(transformed.columns)}")

        # Step 5: Handle outliers
        if self.outlier_handler:
            transformed = self.outlier_handler.transform(transformed)
            logger.info(f"After handling outliers: {list(transformed.columns)}")

        # Step 6: Handle skewed data
        if self.skewed_handler:
            transformed = self.skewed_handler.transform(transformed)
            logger.info(f"After handling skewness: {list(transformed.columns)}")

        # Step 7: Scale numerical features
        if self.numerical_scaler:
            transformed = self.numerical_scaler.transform(transformed)
            logger.info(f"After scaling: {list(transformed.columns)}")

        # Step 8: Encode categorical features
        if self.categorical_encoder:
            transformed = self.categorical_encoder.transform(transformed)
            logger.info(f"After categorical encoding: {list(transformed.columns)}")

        logger.info(f"Final transformed data columns: {list(transformed.columns)}")
        return transformed

    def save(self, path: str) -> None:
        """Save the pipeline to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """Load pipeline from disk"""
        with open(path, 'rb') as f:
            return cloudpickle.load(f)


def validate_and_sanitize_parameters(params: PreprocessingParameters, feature_store: Dict) -> PreprocessingParameters:
    """Validate and sanitize parameters based on feature store"""
    validated = params.dict()
    validated['missing_values_columns'] = [col for col in validated['missing_values_columns'] if
                                           col in feature_store.get('contains_null', [])]
    validated['outliers_columns'] = [col for col in validated['outliers_columns'] if
                                     col in feature_store.get('contains_outliers', [])]
    validated['skewness_columns'] = [col for col in validated['skewness_columns'] if
                                     col in feature_store.get('skewed_cols', [])]
    return PreprocessingParameters(**validated)


def preprocess_data(request_params: Dict) -> Dict:
    """
    Main API function to preprocess data based on request parameters

    Args:
        request_params: Dictionary containing preprocessing parameters

    Returns:
        Dictionary with status and file paths
    """
    try:
        section("API PREPROCESSING WORKFLOW", logger)

        # Load configuration
        intel_config = get_intel_config()
        feature_store = load_yaml(intel_config['feature_store_path'])

        # Create preprocessing parameters from request
        params = PreprocessingParameters(**request_params)

        # Validate parameters
        validated_params = validate_and_sanitize_parameters(params, feature_store)

        # Load data
        train_df = pd.read_csv(intel_config['cleaned_train_path'])
        test_df = pd.read_csv(intel_config['cleaned_test_path'])

        # Create preprocessing paths
        preprocessing_paths = create_preprocessing_paths(intel_config)

        # Update intel config
        update_intel_config(preprocessing_paths)

        # Reload updated config
        intel_config = get_intel_config()

        # Create and configure pipeline
        pipeline = PreprocessingPipeline(intel_config, validated_params)
        pipeline.configure_pipeline()

        # Fit and transform
        pipeline.fit(train_df)
        train_preprocessed = pipeline.transform(train_df)
        test_preprocessed = pipeline.transform(test_df)

        # Create directories and save files
        os.makedirs(os.path.dirname(intel_config['train_preprocessed_path']), exist_ok=True)
        os.makedirs(os.path.dirname(intel_config['test_preprocessed_path']), exist_ok=True)
        os.makedirs(os.path.dirname(intel_config['preprocessing_pipeline_path']), exist_ok=True)

        train_preprocessed.to_csv(intel_config['train_preprocessed_path'], index=False)
        test_preprocessed.to_csv(intel_config['test_preprocessed_path'], index=False)
        pipeline.save(intel_config['preprocessing_pipeline_path'])

        logger.info("Preprocessing completed successfully")

        return {
            'status': 'success',
            'message': 'Data preprocessing completed successfully',
            'train_preprocessed_path': intel_config['train_preprocessed_path'],
            'test_preprocessed_path': intel_config['test_preprocessed_path'],
            'pipeline_path': intel_config['preprocessing_pipeline_path'],
            'parameters_used': validated_params.dict()
        }

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return {
            'status': 'error',
            'message': f'Preprocessing failed: {str(e)}'
        }


async def api_preprocessing_workflow(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: PreprocessingParameters,
        config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Async version for API integration"""
    try:
        section("API PREPROCESSING WORKFLOW", logger)
        feature_store = load_yaml(config.get('feature_store_path'))
        validated_params = validate_and_sanitize_parameters(params, feature_store)
        pipeline = PreprocessingPipeline(config, validated_params)
        pipeline.configure_pipeline()
        pipeline.fit(train_df)
        train_preprocessed = pipeline.transform(train_df)
        test_preprocessed = pipeline.transform(test_df)
        return train_preprocessed, test_preprocessed, validated_params.dict()
    except Exception as e:
        logger.error(f"API Preprocessing failed: {str(e)}")
        raise


# Update the get_intel_config function
def get_intel_config():
    """Load intel configuration using absolute path"""
    try:
        # Try to determine project root
        base_path = os.getenv("PROJECT_ROOT", os.getcwd())
        config_path = os.path.join(base_path, "intel.yaml")

        if not os.path.exists(config_path):
            # Try one level up
            config_path = os.path.join(base_path, "..", "intel.yaml")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError(f"intel.yaml not found at: {config_path}")


def load_yaml(file_path: str) -> Dict:
    """Load YAML file"""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def create_preprocessing_paths(intel_config: Dict) -> Dict:
    """Create preprocessing-related paths"""
    dataset_name = intel_config['dataset_name']

    preprocessing_paths = {
        'train_preprocessed_path': f"data/interim/data_{dataset_name}/train_preprocessed.csv",
        'test_preprocessed_path': f"data/interim/data_{dataset_name}/test_preprocessed.csv",
        'preprocessing_pipeline_path': f"model/pipelines/preprocessing_{dataset_name}/preprocessing.pkl",
        'preprocessing_report_path': f"reports/readme/preprocessing_report_{dataset_name}.md",
        'preprocessing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return preprocessing_paths


def update_intel_config(new_paths: Dict):
    """Update intel.yaml file with new paths"""
    try:
        with open('intel.yaml', 'r') as f:
            intel_config = yaml.safe_load(f)

        intel_config.update(new_paths)

        with open('intel.yaml', 'w') as f:
            yaml.dump(intel_config, f, default_flow_style=False)

        logger.info("Updated intel.yaml with preprocessing paths")

    except Exception as e:
        logger.error(f"Failed to update intel.yaml: {str(e)}")
        raise


def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Get numerical columns from dataframe"""
    return df.select_dtypes(include=['number']).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get categorical columns from dataframe"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def run_preprocessing_from_config():
    """Run preprocessing using default configuration (for CLI usage)"""
    try:
        section("DATA PREPROCESSING", logger)
        intel = get_intel_config()
        feature_store = load_yaml(intel['feature_store_path'])

        train_df = pd.read_csv(intel['cleaned_train_path'])
        test_df = pd.read_csv(intel['cleaned_test_path'])

        # Default parameters
        default_params = {
            'missing_values_method': 'mean',
            'missing_values_columns': feature_store.get('contains_null', []),
            'handle_duplicates': True,
            'outliers_method': 'IQR',
            'outliers_columns': feature_store.get('contains_outliers', []),
            'skewness_method': 'yeo-johnson',
            'skewness_columns': feature_store.get('skewed_cols', []),
            'scaling_method': 'standard',
            'scaling_columns': get_numerical_columns(train_df),
            'categorical_encoding_method': 'onehot',
            'categorical_columns': get_categorical_columns(train_df),
            'drop_first': True,
            'dr_method': 'PCA',
            'dr_components': 0.95
        }

        result = preprocess_data(default_params)
        print(f"Preprocessing result: {result}")

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    run_preprocessing_from_config()