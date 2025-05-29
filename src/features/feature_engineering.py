#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering script for clustering automl clone.
This module handles automatic feature generation, transformation pipeline creation,
and integration with preprocessing pipeline.
API-friendly version that can be called from FastAPI endpoints.
"""
import matplotlib

matplotlib.use('Agg')
import os
import sys
import yaml
import numpy as np
import pandas as pd
import dill as cloudpickle
from typing import Dict, List, Union, Tuple, Optional
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.validation import check_is_fitted

# Add parent directory to path for importing custom logger
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import custom logger
import logging
from src.logger import section, configure_logger


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """A transformer that returns the data unchanged."""

    def __init__(self):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info("Initializing IdentityTransformer")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """A transformer that encodes categorical variables before feature generation."""

    def __init__(self):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info("Initializing CategoricalEncoder")
        self.label_encoders = {}
        self.categorical_columns = []

    def fit(self, X, y=None):
        X_copy = X.copy()

        # Identify categorical columns
        self.categorical_columns = []
        for col in X_copy.columns:
            if X_copy[col].dtype == 'object' or X_copy[col].dtype.name == 'category':
                self.categorical_columns.append(col)

        # Fit label encoders for categorical columns
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            # Handle NaN values by converting to string
            col_values = X_copy[col].astype(str)
            self.label_encoders[col].fit(col_values)

        self.logger.info(f"Found {len(self.categorical_columns)} categorical columns: {self.categorical_columns}")
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Transform categorical columns
        for col in self.categorical_columns:
            if col in X_copy.columns:
                # Handle NaN values by converting to string
                col_values = X_copy[col].astype(str)
                X_copy[col] = self.label_encoders[col].transform(col_values)

        return X_copy


class ClusteringFeatureGenerator(BaseEstimator, TransformerMixin):
    """A transformer that generates clustering-based features."""

    def __init__(self, n_clusters_list=[3, 5, 8]):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info("Initializing ClusteringFeatureGenerator")
        self.n_clusters_list = n_clusters_list
        self.kmeans_models = {}
        self.scaler = StandardScaler()
        self.feature_names = []

    def fit(self, X, y=None):
        try:
            X_copy = X.copy()

            # Ensure all data is numeric
            for col in X_copy.columns:
                if not pd.api.types.is_numeric_dtype(X_copy[col]):
                    self.logger.warning(f"Column {col} is not numeric, converting...")
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')

            # Fill any NaN values that might result from conversion
            X_copy = X_copy.fillna(0)

            # Scale the features for clustering
            X_scaled = self.scaler.fit_transform(X_copy)

            # Fit K-means models for different cluster numbers
            for n_clusters in self.n_clusters_list:
                if X_copy.shape[0] >= n_clusters:  # Ensure we have enough samples
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    self.kmeans_models[n_clusters] = kmeans
                    self.logger.info(f"Fitted K-means with {n_clusters} clusters")

            # Generate feature names
            self.feature_names = []
            for n_clusters in self.kmeans_models.keys():
                self.feature_names.extend([
                    f'cluster_{n_clusters}',
                    f'distance_to_centroid_{n_clusters}'
                ])

            self.logger.info(f"Generated {len(self.feature_names)} clustering features")
            return self

        except Exception as e:
            self.logger.error(f"Error in ClusteringFeatureGenerator fit: {str(e)}")
            # Initialize empty models to prevent transform errors
            self.kmeans_models = {}
            self.feature_names = []
            return self

    def transform(self, X):
        try:
            if not self.kmeans_models:
                # If no models were fitted, return original data
                return X

            X_copy = X.copy()

            # Ensure all data is numeric
            for col in X_copy.columns:
                if not pd.api.types.is_numeric_dtype(X_copy[col]):
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')

            # Fill any NaN values
            X_copy = X_copy.fillna(0)

            # Scale the features
            X_scaled = self.scaler.transform(X_copy)

            # Generate clustering features
            clustering_features = []

            for n_clusters, kmeans in self.kmeans_models.items():
                # Get cluster assignments
                cluster_labels = kmeans.predict(X_scaled)
                clustering_features.append(cluster_labels)

                # Calculate distances to centroids
                distances = np.min(kmeans.transform(X_scaled), axis=1)
                clustering_features.append(distances)

            # Combine original features with clustering features
            if clustering_features:
                clustering_df = pd.DataFrame(
                    np.column_stack(clustering_features),
                    columns=self.feature_names,
                    index=X_copy.index
                )
                result = pd.concat([X_copy, clustering_df], axis=1)
            else:
                result = X_copy

            return result

        except Exception as e:
            self.logger.error(f"Error in ClusteringFeatureGenerator transform: {str(e)}")
            return X


class FeatureToolsTransformer(BaseEstimator, TransformerMixin):
    """A transformer that uses featuretools to create new features for clustering."""

    def __init__(self):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info("Initializing FeatureToolsTransformer for clustering")
        self.feature_defs = None
        self.feature_names = None

        try:
            import featuretools as ft
            self.ft = ft
        except ImportError:
            self.logger.error("Featuretools package not found. Please install with: pip install featuretools")
            raise

    def fit(self, X, y=None):
        try:
            X_copy = X.copy()

            # Ensure all data is numeric for featuretools
            for col in X_copy.columns:
                if not pd.api.types.is_numeric_dtype(X_copy[col]):
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')

            # Fill NaN values
            X_copy = X_copy.fillna(0)

            es = self.ft.EntitySet(id="features")

            if X_copy.index.name is None:
                X_copy = X_copy.reset_index(drop=True)
                index_name = "index"
            else:
                index_name = X_copy.index.name

            es.add_dataframe(
                dataframe_name="data",
                dataframe=X_copy,
                index=index_name,
                make_index=True,
                time_index=None
            )

            feature_matrix, feature_defs = self.ft.dfs(
                entityset=es,
                target_dataframe_name="data",
                trans_primitives=["add_numeric", "multiply_numeric", "divide_numeric", "subtract_numeric"],
                max_depth=1,
                features_only=False,
                verbose=True
            )

            self.feature_defs = feature_defs
            self.feature_names = list(feature_matrix.columns)
            self.logger.info(f"Generated {len(self.feature_names)} features using featuretools")
            return self

        except Exception as e:
            self.logger.error(f"Error in FeatureToolsTransformer fit: {str(e)}")
            raise

    def transform(self, X):
        try:
            X_copy = X.copy()

            # Ensure all data is numeric
            for col in X_copy.columns:
                if not pd.api.types.is_numeric_dtype(X_copy[col]):
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')

            # Fill NaN values
            X_copy = X_copy.fillna(0)

            if X_copy.index.name is None:
                X_copy = X_copy.reset_index(drop=True)
                index_name = "index"
            else:
                index_name = X_copy.index.name

            es = self.ft.EntitySet(id="features_transform")
            es.add_dataframe(
                dataframe_name="data",
                dataframe=X_copy,
                index=index_name,
                make_index=True,
                time_index=None
            )

            feature_matrix = self.ft.calculate_feature_matrix(
                features=self.feature_defs,
                entityset=es,
                verbose=True
            )

            feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
            feature_matrix = feature_matrix[self.feature_names]

            for col in feature_matrix.columns:
                if not pd.api.types.is_numeric_dtype(feature_matrix[col]):
                    feature_matrix = feature_matrix.drop(columns=[col])

            return feature_matrix

        except Exception as e:
            self.logger.error(f"Error in FeatureToolsTransformer transform: {str(e)}")
            raise


class VarianceFeatureSelector(BaseEstimator, TransformerMixin):
    """A transformer that selects features based on variance threshold for clustering."""

    def __init__(self, n_features: int = 20, variance_threshold: float = 0.01):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info(f"Initializing VarianceFeatureSelector with n_features={n_features}")
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.selected_features = None
        self.importance_df = None
        self.variance_selector = VarianceThreshold(threshold=variance_threshold)

    def fit(self, X, y=None):
        try:
            X_copy = X.copy()

            # Ensure all data is numeric
            for col in X_copy.columns:
                if not pd.api.types.is_numeric_dtype(X_copy[col]):
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')

            # Fill NaN values
            X_copy = X_copy.fillna(0)

            feature_names = list(X_copy.columns)
            self.n_features = min(self.n_features, len(feature_names))

            if X_copy.shape[0] < 10:
                self.selected_features = feature_names
                return self

            # First, remove low variance features
            X_variance_filtered = self.variance_selector.fit_transform(X_copy)
            variance_selected_features = [feature_names[i] for i in range(len(feature_names))
                                          if self.variance_selector.get_support()[i]]

            # Calculate variance for remaining features
            X_filtered = X_copy[variance_selected_features]
            feature_variances = X_filtered.var().values

            # Create importance dataframe based on variance
            self.importance_df = pd.DataFrame({
                'feature': variance_selected_features,
                'importance': feature_variances
            }).sort_values('importance', ascending=False)

            # Select top N features by variance
            self.selected_features = self.importance_df['feature'].head(self.n_features).tolist()

            self.logger.info(f"Selected {len(self.selected_features)} features based on variance")
            return self

        except Exception as e:
            self.logger.error(f"Error in VarianceFeatureSelector fit: {str(e)}")
            self.selected_features = list(X.columns)
            return self

    def transform(self, X):
        try:
            X_copy = X.copy()
            available_features = [f for f in self.selected_features if f in X_copy.columns]
            return X_copy[available_features]
        except Exception as e:
            self.logger.error(f"Error in VarianceFeatureSelector transform: {str(e)}")
            return X


class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    """A transformer that selects features based on correlation analysis for clustering."""

    def __init__(self, n_features: int = 20, correlation_threshold: float = 0.95):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info(f"Initializing CorrelationFeatureSelector with n_features={n_features}")
        self.n_features = n_features
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.importance_df = None

    def fit(self, X, y=None):
        try:
            X_copy = X.copy()

            # Ensure all data is numeric
            for col in X_copy.columns:
                if not pd.api.types.is_numeric_dtype(X_copy[col]):
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')

            # Fill NaN values
            X_copy = X_copy.fillna(0)

            feature_names = list(X_copy.columns)
            self.n_features = min(self.n_features, len(feature_names))

            if X_copy.shape[0] < 10:
                self.selected_features = feature_names
                return self

            # Calculate correlation matrix
            corr_matrix = X_copy.corr().abs()

            # Find pairs of highly correlated features
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Find features to drop (highly correlated)
            to_drop = [column for column in upper_tri.columns
                       if any(upper_tri[column] > self.correlation_threshold)]

            # Keep features that are not highly correlated
            features_to_keep = [f for f in feature_names if f not in to_drop]

            # If we have more features than needed, select based on variance
            if len(features_to_keep) > self.n_features:
                X_filtered = X_copy[features_to_keep]
                feature_variances = X_filtered.var().values

                self.importance_df = pd.DataFrame({
                    'feature': features_to_keep,
                    'importance': feature_variances
                }).sort_values('importance', ascending=False)

                self.selected_features = self.importance_df['feature'].head(self.n_features).tolist()
            else:
                self.selected_features = features_to_keep
                # Create importance dataframe for remaining features
                X_filtered = X_copy[self.selected_features]
                feature_variances = X_filtered.var().values

                self.importance_df = pd.DataFrame({
                    'feature': self.selected_features,
                    'importance': feature_variances
                }).sort_values('importance', ascending=False)

            self.logger.info(f"Selected {len(self.selected_features)} features after correlation filtering")
            return self

        except Exception as e:
            self.logger.error(f"Error in CorrelationFeatureSelector fit: {str(e)}")
            self.selected_features = list(X.columns)
            return self

    def transform(self, X):
        try:
            X_copy = X.copy()
            available_features = [f for f in self.selected_features if f in X_copy.columns]
            return X_copy[available_features]
        except Exception as e:
            self.logger.error(f"Error in CorrelationFeatureSelector transform: {str(e)}")
            return X


class FeatureEngineer:
    """Main class for feature engineering process for clustering, API-friendly version."""

    def __init__(self, config_path: Union[str, Path] = "intel.yaml"):
        """
        Initialize the FeatureEngineer.

        Args:
            config_path: Path to the config file (intel.yaml)
        """
        self.logger = logging.getLogger("Feature Engineering")
        section("FEATURE ENGINEERING INITIALIZATION", self.logger)

        # Configure logger if not already configured
        try:
            if not self.logger.handlers:
                configure_logger()
        except Exception:
            # If logger configuration fails, continue with default logger
            pass

        self.config_path = Path(config_path)
        self.project_root = self.config_path.parent
        self.intel = self._load_intel()

        # Load dataset name from config
        self.dataset_name = self.intel.get("dataset_name")
        if not self.dataset_name:
            raise ValueError("dataset_name not found in intel.yaml")

        self.feature_store = self._load_feature_store()
        # For clustering, we don't need target column
        self.target_column = self.intel.get("target_column", None)
        self._setup_paths()

    def _load_intel(self) -> Dict:
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading intel.yaml: {str(e)}")
            raise

    def _load_feature_store(self) -> Dict:
        try:
            feature_store_path = self.project_root / self.intel.get("feature_store_path")
            with open(feature_store_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading feature store: {str(e)}")
            raise

    def _setup_paths(self):
        # Construct absolute paths using project root
        self.transformation_pipeline_path = self.project_root / f"model/pipelines/preprocessing_{self.dataset_name}/transformation.pkl"
        self.processor_pipeline_path = self.project_root / f"model/pipelines/preprocessing_{self.dataset_name}/processor.pkl"
        self.train_transformed_path = self.project_root / f"data/processed/data_{self.dataset_name}/train_transformed.csv"
        self.test_transformed_path = self.project_root / f"data/processed/data_{self.dataset_name}/test_transformed.csv"

        # Ensure directories exist
        self.transformation_pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        self.train_transformed_path.parent.mkdir(parents=True, exist_ok=True)

    def _update_intel(self, use_feature_tools: bool, feature_selection_method: str, n_features: int,
                      use_clustering_features: bool):
        self.intel.update({
            "transformation_pipeline_path": str(self.transformation_pipeline_path.relative_to(self.project_root)),
            "processor_pipeline_path": str(self.processor_pipeline_path.relative_to(self.project_root)),
            "train_transformed_path": str(self.train_transformed_path.relative_to(self.project_root)),
            "test_transformed_path": str(self.test_transformed_path.relative_to(self.project_root)),
            "feature_engineering_config": {
                "use_feature_tools": use_feature_tools,
                "use_clustering_features": use_clustering_features,
                "feature_selection_method": feature_selection_method,
                "n_features_selected": n_features if feature_selection_method != "none" else None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        })
        with open(self.config_path, 'w') as f:
            yaml.dump(self.intel, f)
        self.logger.info(f"Updated intel.yaml at {self.config_path}")
        return self.intel

    def run(self, use_feature_tools: bool = False, feature_selection_method: str = "none", n_features: int = 20,
            use_clustering_features: bool = True):
        """
        Run the feature engineering process with the specified parameters.

        Args:
            use_feature_tools: Whether to use FeatureTools for feature generation
            feature_selection_method: Method for feature selection ("none", "variance", "correlation")
            n_features: Number of features to select if using feature selection
            use_clustering_features: Whether to generate clustering-based features
        """
        section("FEATURE ENGINEERING PROCESS", self.logger)
        train_df, test_df = self._load_data()

        # For clustering, we don't need to separate target column
        # But we'll handle it if it exists for compatibility
        if self.target_column and self.target_column in train_df.columns:
            X_train = train_df.drop(columns=[self.target_column])
            X_test = test_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            y_test = test_df[self.target_column]
            has_target = True
        else:
            X_train = train_df.copy()
            X_test = test_df.copy()
            y_train = None
            y_test = None
            has_target = False

        pipeline_steps = []

        # Step 1: Handle categorical encoding first
        pipeline_steps.append(('categorical_encoder', CategoricalEncoder()))

        # Step 2: Feature generation steps
        if use_clustering_features:
            pipeline_steps.append(('clustering_features', ClusteringFeatureGenerator()))

        if use_feature_tools:
            pipeline_steps.append(('feature_tools', FeatureToolsTransformer()))

        if not use_clustering_features and not use_feature_tools:
            pipeline_steps.append(('identity', IdentityTransformer()))

        # Step 3: Feature selection step
        if feature_selection_method == "variance":
            pipeline_steps.append(('variance_selector', VarianceFeatureSelector(n_features=n_features)))
        elif feature_selection_method == "correlation":
            pipeline_steps.append(('correlation_selector', CorrelationFeatureSelector(n_features=n_features)))

        transformation_pipeline = Pipeline(pipeline_steps)
        transformation_pipeline.fit(X_train)

        try:
            X_train_transformed = transformation_pipeline.transform(X_train).reset_index(drop=True)
            X_test_transformed = transformation_pipeline.transform(X_test).reset_index(drop=True)

            # Prepare final dataframes
            if has_target:
                y_train_reset = y_train.reset_index(drop=True)
                y_test_reset = y_test.reset_index(drop=True)
                train_transformed_df = pd.concat([X_train_transformed, y_train_reset], axis=1)
                test_transformed_df = pd.concat([X_test_transformed, y_test_reset], axis=1)
            else:
                train_transformed_df = X_train_transformed.copy()
                test_transformed_df = X_test_transformed.copy()

            self._save_data(train_transformed_df, test_transformed_df)
            self._save_pipelines(transformation_pipeline)
            self._log_feature_info(transformation_pipeline, use_feature_tools, feature_selection_method,
                                   use_clustering_features)
            updated_intel = self._update_intel(use_feature_tools, feature_selection_method, n_features,
                                               use_clustering_features)

            return {
                "status": "success",
                "message": "Feature engineering completed successfully",
                "metadata": {
                    "train_shape": train_transformed_df.shape,
                    "test_shape": test_transformed_df.shape,
                    "train_path": str(self.train_transformed_path),
                    "test_path": str(self.test_transformed_path),
                    "pipeline_path": str(self.transformation_pipeline_path),
                    "processor_path": str(self.processor_pipeline_path),
                    "feature_engineering_config": updated_intel.get("feature_engineering_config", {})
                }
            }

        except Exception as e:
            error_msg = f"Error in transformation process: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }

    def _load_data(self):
        try:
            train_path = self.project_root / self.intel.get("train_preprocessed_path")
            test_path = self.project_root / self.intel.get("test_preprocessed_path")
            self.logger.info(f"Loading train data from {train_path}")
            self.logger.info(f"Loading test data from {test_path}")
            return (
                pd.read_csv(train_path),
                pd.read_csv(test_path)
            )
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _save_data(self, train_df, test_df):
        try:
            self.logger.info(f"Saving transformed train data to {self.train_transformed_path}")
            train_df.to_csv(self.train_transformed_path, index=False)
            self.logger.info(f"Saving transformed test data to {self.test_transformed_path}")
            test_df.to_csv(self.test_transformed_path, index=False)
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise

    def _load_cleaning_pipeline(self):
        try:
            cleaning_path = self.project_root / f"model/pipelines/preprocessing_{self.dataset_name}/cleaning.pkl"
            self.logger.info(f"Loading cleaning pipeline from {cleaning_path}")
            with open(cleaning_path, 'rb') as f:
                return cloudpickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading cleaning pipeline: {str(e)}")
            raise

    def _load_preprocessing_pipeline(self):
        try:
            preprocessing_path = self.project_root / self.intel.get("preprocessing_pipeline_path")
            self.logger.info(f"Loading preprocessing pipeline from {preprocessing_path}")
            with open(preprocessing_path, 'rb') as f:
                return cloudpickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading preprocessing pipeline: {str(e)}")
            raise

    def _save_pipelines(self, transformation_pipeline):
        try:
            self.logger.info(f"Saving transformation pipeline to {self.transformation_pipeline_path}")
            with open(self.transformation_pipeline_path, 'wb') as f:
                cloudpickle.dump(transformation_pipeline, f)

            # Load cleaning and preprocessing pipelines
            cleaning_pipeline = self._load_cleaning_pipeline()
            preprocessing_pipeline = self._load_preprocessing_pipeline()

            # Create a combined pipeline with all three components
            processor_pipeline = Pipeline([
                ('cleaning', cleaning_pipeline),
                ('preprocessing', preprocessing_pipeline),
                ('transformation', transformation_pipeline)
            ])

            self.logger.info(f"Saving processor pipeline to {self.processor_pipeline_path}")
            with open(self.processor_pipeline_path, 'wb') as f:
                cloudpickle.dump(processor_pipeline, f)
        except Exception as e:
            self.logger.error(f"Error saving pipelines: {str(e)}")
            raise

    def _log_feature_info(self, pipeline, use_feature_tools, feature_selection_method, use_clustering_features):
        if use_clustering_features and 'clustering_features' in pipeline.named_steps:
            clustering_gen = pipeline.named_steps['clustering_features']
            self.logger.info(f"Generated {len(clustering_gen.feature_names)} clustering features")

        if use_feature_tools and 'feature_tools' in pipeline.named_steps:
            self.logger.info(f"Generated {len(pipeline.named_steps['feature_tools'].feature_names)} features")

        if feature_selection_method == "variance" and 'variance_selector' in pipeline.named_steps:
            selector = pipeline.named_steps['variance_selector']
            if selector.importance_df is not None:
                top_features = selector.importance_df.head(10)['feature'].tolist()
                self.logger.info(f"Top 10 features by variance: {top_features}")

        if feature_selection_method == "correlation" and 'correlation_selector' in pipeline.named_steps:
            selector = pipeline.named_steps['correlation_selector']
            if selector.importance_df is not None:
                top_features = selector.importance_df.head(10)['feature'].tolist()
                self.logger.info(f"Top 10 features after correlation filtering: {top_features}")

    def get_feature_importance(self):
        """
        Return feature importance if feature selection was used.

        Returns:
            Dict: Feature importance data or error message
        """
        try:
            # Try to load the transformation pipeline
            with open(self.transformation_pipeline_path, 'rb') as f:
                pipeline = cloudpickle.load(f)

            # Check for variance selector
            if 'variance_selector' in pipeline.named_steps:
                selector = pipeline.named_steps['variance_selector']
                if selector.importance_df is not None:
                    return {
                        "status": "success",
                        "selection_method": "variance",
                        "feature_importance": selector.importance_df.to_dict(orient='records')
                    }

            # Check for correlation selector
            if 'correlation_selector' in pipeline.named_steps:
                selector = pipeline.named_steps['correlation_selector']
                if selector.importance_df is not None:
                    return {
                        "status": "success",
                        "selection_method": "correlation",
                        "feature_importance": selector.importance_df.to_dict(orient='records')
                    }

            return {
                "status": "error",
                "message": "Feature selection was not used"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error retrieving feature importance: {str(e)}"
            }

    def get_generated_features(self):
        """
        Return generated features if FeatureTools or clustering features were used.

        Returns:
            Dict: Generated features data or error message
        """
        try:
            # Try to load the transformation pipeline
            with open(self.transformation_pipeline_path, 'rb') as f:
                pipeline = cloudpickle.load(f)

            result = {"status": "success", "generated_features": {}}

            # Check for clustering features
            if 'clustering_features' in pipeline.named_steps:
                clustering_gen = pipeline.named_steps['clustering_features']
                if clustering_gen.feature_names:
                    result["generated_features"]["clustering"] = clustering_gen.feature_names

            # Check for FeatureTools features
            if 'feature_tools' in pipeline.named_steps:
                feature_tools = pipeline.named_steps['feature_tools']
                if feature_tools.feature_names is not None:
                    result["generated_features"]["featuretools"] = feature_tools.feature_names

            if not result["generated_features"]:
                return {
                    "status": "error",
                    "message": "No feature generation was used"
                }

            return result
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error retrieving generated features: {str(e)}"
            }


# Helper function for API usage
def run_feature_engineering(
        config_path: str = "intel.yaml",
        use_feature_tools: bool = False,
        feature_selection_method: str = "none",
        n_features: int = 20,
        use_clustering_features: bool = True
):
    """
    Run feature engineering with the specified parameters for clustering.
    This function can be called from an API endpoint.

    Args:
        config_path: Path to the config file (intel.yaml)
        use_feature_tools: Whether to use FeatureTools for feature generation
        feature_selection_method: Method for feature selection ("none", "variance", "correlation")
        n_features: Number of features to select if using feature selection
        use_clustering_features: Whether to generate clustering-based features

    Returns:
        Dict: Result of the feature engineering process
    """
    try:
        engineer = FeatureEngineer(config_path=config_path)
        result = engineer.run(
            use_feature_tools=use_feature_tools,
            feature_selection_method=feature_selection_method,
            n_features=n_features,
            use_clustering_features=use_clustering_features
        )
        return result
    except Exception as e:
        logger = logging.getLogger("Feature Engineering")
        logger.error(f"Error in feature engineering: {str(e)}")
        return {
            "status": "error",
            "message": f"Feature engineering failed: {str(e)}"
        }


if __name__ == "__main__":
    # This block is for direct script execution (not API call)
    # It demonstrates how to use the API-friendly version
    try:
        # Get parameters from command line arguments if provided
        import argparse

        parser = argparse.ArgumentParser(description='Run feature engineering process for clustering')
        parser.add_argument('--config', default='intel.yaml', help='Path to config file')
        parser.add_argument('--use-feature-tools', action='store_true', help='Use FeatureTools')
        parser.add_argument('--feature-selection', choices=['none', 'variance', 'correlation'],
                            default='none', help='Feature selection method')
        parser.add_argument('--n-features', type=int, default=20, help='Number of features to select')
        parser.add_argument('--use-clustering-features', action='store_true', default=True,
                            help='Use clustering-based feature generation')
        args = parser.parse_args()

        # Configure logger
        configure_logger()
        logger = logging.getLogger("Feature Engineering")

        # Run feature engineering
        result = run_feature_engineering(
            config_path=args.config,
            use_feature_tools=args.use_feature_tools,
            feature_selection_method=args.feature_selection,
            n_features=args.n_features,
            use_clustering_features=args.use_clustering_features
        )

        if result["status"] == "success":
            logger.info("Feature engineering completed successfully")
        else:
            logger.critical(f"Feature engineering failed: {result['message']}")
            sys.exit(1)

    except Exception as e:
        logger = logging.getLogger("Feature Engineering")
        logger.critical(f"Feature engineering failed: {str(e)}")
        sys.exit(1)