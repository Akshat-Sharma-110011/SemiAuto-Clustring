"""
Data Preprocessing Module for Clustering (API Version)

This module handles the preprocessing.html of data for clustering models, including:
- Handling missing values
- Handling duplicate values
- Handling outliers
- Handling skewed data
- Scaling numerical features
- Encoding categorical features
- Dimensionality reduction for correlated columns

The preprocessing.html steps are configured based on API requests and feature_store.yaml file.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import cloudpickle
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
from semiauto_clustering.logger import configure_logger, section
from semiauto_clustering.custom_transformers import (IDColumnDropper, MissingValueHandler, OutlierHandler, SkewedDataHandler,
                                 NumericalScaler, DimensionalityReducer, CategoricalEncoder)

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


class PreprocessingParameters(BaseModel):
    """Pydantic model for preprocessing.html parameters"""
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
        self.dataset_name = config.get('dataset_name', 'unknown_dataset')

        # FIX: Load the actual feature store file instead of getting it from config
        feature_store_path = config.get('feature_store_path')
        if feature_store_path and os.path.exists(feature_store_path):
            with open(feature_store_path, 'r') as f:
                self.feature_store = yaml.safe_load(f)
            logger.info(f"Loaded feature store from: {feature_store_path}")
        else:
            logger.warning(f"Feature store not found at: {feature_store_path}")
            self.feature_store = {}

        self.missing_handler = None
        self.outlier_handler = None
        self.skewed_handler = None
        self.numerical_scaler = None
        self.categorical_encoder = None
        self.id_dropper = None
        self.dimensionality_reducer = None
        self.correlated_groups = self._get_correlated_groups()

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
        """Configure all pipeline components"""
        # Configure ID dropper - FIX: Add debug logging
        id_cols = self.feature_store.get('id_cols', [])
        logger.info(f"Feature store id_cols: {id_cols}")

        if id_cols:
            self.id_dropper = IDColumnDropper(id_cols)
            logger.info(f"Configured ID dropper for columns: {id_cols}")
        else:
            logger.warning("No ID columns found in feature store")
            logger.warning(f"Available feature store keys: {list(self.feature_store.keys())}")

        # Configure missing value handler
        if self.params.missing_values_columns:
            self.missing_handler = MissingValueHandler(
                method=self.params.missing_values_method,
                columns=self.params.missing_values_columns
            )

        # Configure outlier handler
        if self.params.outliers_method and self.params.outliers_columns:
            self.outlier_handler = OutlierHandler(
                method=self.params.outliers_method,
                columns=self.params.outliers_columns
            )

        # Configure skewed data handler
        if self.params.skewness_method and self.params.skewness_columns:
            self.skewed_handler = SkewedDataHandler(
                method=self.params.skewness_method,
                columns=self.params.skewness_columns
            )

        # Configure numerical scaler
        if self.params.scaling_method and self.params.scaling_columns:
            self.numerical_scaler = NumericalScaler(
                method=self.params.scaling_method,
                columns=self.params.scaling_columns
            )

        # Configure categorical encoder
        if self.params.categorical_encoding_method and self.params.categorical_columns:
            self.categorical_encoder = CategoricalEncoder(
                method=self.params.categorical_encoding_method,
                columns=self.params.categorical_columns,
                drop_first=self.params.drop_first
            )

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
        request_params: Dictionary containing preprocessing.html parameters

    Returns:
        Dictionary with status and file paths
    """
    try:
        section("API PREPROCESSING WORKFLOW", logger)

        # Load configuration
        intel_config = get_intel_config()
        feature_store = load_yaml(intel_config['feature_store_path'])

        # Create preprocessing.html parameters from request
        params = PreprocessingParameters(**request_params)

        # Validate parameters
        validated_params = validate_and_sanitize_parameters(params, feature_store)

        # Load data
        train_df = pd.read_csv(intel_config['cleaned_train_path'])
        test_df = pd.read_csv(intel_config['cleaned_test_path'])

        # Create preprocessing.html paths
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
            'message': 'Data preprocessing.html completed successfully',
            'train_preprocessed_path': intel_config['train_preprocessed_path'],
            'test_preprocessed_path': intel_config['test_preprocessed_path'],
            'pipeline_path': intel_config['preprocessing_pipeline_path'],
            'parameters_used': validated_params.dict()
        }

    except Exception as e:
        logger.error(f"Error in preprocessing.html: {str(e)}")
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


def get_intel_config():
    """Load intel configuration"""
    try:
        with open('intel.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError("intel.yaml not found!")


def load_yaml(file_path: str) -> Dict:
    """Load YAML file"""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def create_preprocessing_paths(intel_config: Dict) -> Dict:
    """Create preprocessing.html-related paths"""
    dataset_name = intel_config['dataset_name']

    preprocessing_paths = {
        'train_preprocessed_path': f"data/interim/data_{dataset_name}/train.csv",
        'test_preprocessed_path': f"data/interim/data_{dataset_name}/test.csv",
        'preprocessing_pipeline_path': f"model/pipelines/preprocessing_{dataset_name}/preprocessing.html.pkl",
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

        logger.info("Updated intel.yaml with preprocessing.html paths")

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
    """Run preprocessing.html using default configuration (for CLI usage)"""
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
        logger.error(f"Error in preprocessing.html: {str(e)}")
        raise


if __name__ == "__main__":
    run_preprocessing_from_config()