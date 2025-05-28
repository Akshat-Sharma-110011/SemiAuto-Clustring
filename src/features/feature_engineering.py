#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering script for clustering automl clone.
This module handles automatic feature generation, transformation pipeline creation,
and integration with preprocessing pipeline specifically for clustering tasks.
API-friendly version that can be called from FastAPI endpoints.
"""
import matplotlib

matplotlib.use('Agg')
import os
import sys
import yaml
import numpy as np
import pandas as pd
import cloudpickle
from typing import Dict, List, Union, Tuple, Optional
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import warnings
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


class ClusteringFeatureGenerator(BaseEstimator, TransformerMixin):
    """A transformer that generates clustering-specific features."""

    def __init__(self, n_clusters_range: List[int] = [3, 5, 8], n_neighbors: int = 5):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info("Initializing ClusteringFeatureGenerator")
        self.n_clusters_range = n_clusters_range
        self.n_neighbors = n_neighbors
        self.cluster_models = {}
        self.knn_model = None
        self.original_columns = None

    def fit(self, X, y=None):
        try:
            self.original_columns = list(X.columns)
            X_scaled = StandardScaler().fit_transform(X)

            # Fit K-means models for different cluster numbers
            for n_clusters in self.n_clusters_range:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                self.cluster_models[n_clusters] = kmeans
                self.logger.info(f"Fitted K-means with {n_clusters} clusters")

            # Fit KNN for density-based features
            self.knn_model = NearestNeighbors(n_neighbors=self.n_neighbors)
            self.knn_model.fit(X_scaled)

            self.logger.info(
                f"Generated clustering features for {len(self.n_clusters_range)} different cluster configurations")
            return self

        except Exception as e:
            self.logger.error(f"Error in ClusteringFeatureGenerator fit: {str(e)}")
            raise

    def transform(self, X):
        try:
            X_scaled = StandardScaler().fit_transform(X)
            feature_df = X.copy()

            # Generate cluster membership features
            for n_clusters in self.n_clusters_range:
                cluster_labels = self.cluster_models[n_clusters].predict(X_scaled)
                cluster_distances = self.cluster_models[n_clusters].transform(X_scaled)

                # Add cluster labels as features
                feature_df[f'cluster_{n_clusters}'] = cluster_labels

                # Add distances to cluster centers
                for i in range(n_clusters):
                    feature_df[f'dist_to_cluster_{n_clusters}_{i}'] = cluster_distances[:, i]

                # Add minimum distance to any cluster center
                feature_df[f'min_cluster_dist_{n_clusters}'] = np.min(cluster_distances, axis=1)

                # Add cluster density (inverse of distance to nearest center)
                feature_df[f'cluster_density_{n_clusters}'] = 1 / (1 + np.min(cluster_distances, axis=1))

            # Generate density-based features using KNN
            distances, indices = self.knn_model.kneighbors(X_scaled)
            feature_df['knn_density'] = 1 / (1 + np.mean(distances, axis=1))
            feature_df['knn_avg_distance'] = np.mean(distances, axis=1)
            feature_df['knn_max_distance'] = np.max(distances, axis=1)
            feature_df['knn_std_distance'] = np.std(distances, axis=1)

            # Generate statistical features for original columns
            for col in self.original_columns:
                if pd.api.types.is_numeric_dtype(feature_df[col]):
                    # Z-score normalization features
                    col_mean = feature_df[col].mean()
                    col_std = feature_df[col].std()
                    if col_std > 0:
                        feature_df[f'{col}_zscore'] = (feature_df[col] - col_mean) / col_std
                        feature_df[f'{col}_abs_zscore'] = np.abs(feature_df[f'{col}_zscore'])

            # Generate interaction features (for clustering patterns)
            numeric_cols = [col for col in self.original_columns if pd.api.types.is_numeric_dtype(feature_df[col])]
            if len(numeric_cols) >= 2:
                # Create ratio features for the first few numeric columns
                for i in range(min(3, len(numeric_cols))):
                    for j in range(i + 1, min(3, len(numeric_cols))):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        if feature_df[col2].std() > 0 and not (feature_df[col2] == 0).any():
                            feature_df[f'{col1}_{col2}_ratio'] = feature_df[col1] / (feature_df[col2] + 1e-8)

                        # Distance features
                        feature_df[f'{col1}_{col2}_euclidean'] = np.sqrt(
                            feature_df[col1] ** 2 + feature_df[col2] ** 2
                        )

            # Remove any infinite or NaN values
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)

            self.logger.info(f"Generated {feature_df.shape[1] - len(self.original_columns)} new clustering features")
            return feature_df

        except Exception as e:
            self.logger.error(f"Error in ClusteringFeatureGenerator transform: {str(e)}")
            raise


class DimensionalityReductionTransformer(BaseEstimator, TransformerMixin):
    """A transformer that applies dimensionality reduction techniques for clustering."""

    def __init__(self, use_pca: bool = True, use_tsne: bool = False,
                 pca_components: int = 10, tsne_components: int = 2):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info("Initializing DimensionalityReductionTransformer")
        self.use_pca = use_pca
        self.use_tsne = use_tsne
        self.pca_components = pca_components
        self.tsne_components = tsne_components
        self.pca_model = None
        self.tsne_model = None
        self.scaler = None

    def fit(self, X, y=None):
        try:
            # Scale the features first
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            if self.use_pca:
                # Adjust PCA components based on data dimensions
                n_components = min(self.pca_components, X.shape[1], X.shape[0] - 1)
                self.pca_model = PCA(n_components=n_components, random_state=42)
                self.pca_model.fit(X_scaled)
                self.logger.info(f"Fitted PCA with {n_components} components")

            if self.use_tsne and X.shape[0] <= 1000:  # t-SNE is computationally expensive
                self.tsne_model = TSNE(n_components=self.tsne_components, random_state=42,
                                       perplexity=min(30, X.shape[0] // 4))
                self.logger.info(f"Will apply t-SNE with {self.tsne_components} components")
            elif self.use_tsne:
                self.logger.warning("Skipping t-SNE due to large dataset size (>1000 samples)")
                self.use_tsne = False

            return self

        except Exception as e:
            self.logger.error(f"Error in DimensionalityReductionTransformer fit: {str(e)}")
            raise

    def transform(self, X):
        try:
            X_scaled = self.scaler.transform(X)
            result_df = X.copy()

            if self.use_pca and self.pca_model:
                pca_features = self.pca_model.transform(X_scaled)
                pca_columns = [f'pca_{i + 1}' for i in range(pca_features.shape[1])]
                pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=X.index)
                result_df = pd.concat([result_df, pca_df], axis=1)

                # Add explained variance ratio as features
                explained_var = self.pca_model.explained_variance_ratio_
                for i, var_ratio in enumerate(explained_var):
                    result_df[f'pca_{i + 1}_var_explained'] = var_ratio

            if self.use_tsne and self.tsne_model:
                tsne_features = self.tsne_model.fit_transform(X_scaled)
                tsne_columns = [f'tsne_{i + 1}' for i in range(tsne_features.shape[1])]
                tsne_df = pd.DataFrame(tsne_features, columns=tsne_columns, index=X.index)
                result_df = pd.concat([result_df, tsne_df], axis=1)

            return result_df

        except Exception as e:
            self.logger.error(f"Error in DimensionalityReductionTransformer transform: {str(e)}")
            return X


class VarianceFeatureSelector(BaseEstimator, TransformerMixin):
    """A transformer that removes low-variance features for clustering."""

    def __init__(self, variance_threshold: float = 0.0):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info(f"Initializing VarianceFeatureSelector with threshold={variance_threshold}")
        self.variance_threshold = variance_threshold
        self.selector = None
        self.selected_features = None

    def fit(self, X, y=None):
        try:
            self.selector = VarianceThreshold(threshold=self.variance_threshold)
            self.selector.fit(X)

            # Get selected feature names
            feature_names = list(X.columns)
            selected_mask = self.selector.get_support()
            self.selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]

            self.logger.info(f"Selected {len(self.selected_features)} features out of {len(feature_names)}")
            return self

        except Exception as e:
            self.logger.error(f"Error in VarianceFeatureSelector fit: {str(e)}")
            self.selected_features = list(X.columns)
            return self

    def transform(self, X):
        try:
            if self.selector:
                X_selected = self.selector.transform(X)
                return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
            else:
                return X[self.selected_features]
        except Exception as e:
            self.logger.error(f"Error in VarianceFeatureSelector transform: {str(e)}")
            return X


class FeatureEngineer:
    """Main class for clustering feature engineering process, API-friendly version."""

    def __init__(self, config_path: Union[str, Path] = "intel.yaml"):
        """
        Initialize the FeatureEngineer.

        Args:
            config_path: Path to the config file (intel.yaml)
        """
        self.logger = logging.getLogger("Feature Engineering")
        section("CLUSTERING FEATURE ENGINEERING INITIALIZATION", self.logger)

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
        # For clustering, we don't need a target column
        self.target_column = self.intel.get("target_column")  # May be None for clustering
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

    def _update_intel(self, use_clustering_features: bool, use_dimensionality_reduction: bool,
                      use_variance_selection: bool, n_clusters_range: List[int],
                      pca_components: int, variance_threshold: float):
        self.intel.update({
            "transformation_pipeline_path": str(self.transformation_pipeline_path.relative_to(self.project_root)),
            "processor_pipeline_path": str(self.processor_pipeline_path.relative_to(self.project_root)),
            "train_transformed_path": str(self.train_transformed_path.relative_to(self.project_root)),
            "test_transformed_path": str(self.test_transformed_path.relative_to(self.project_root)),
            "feature_engineering_config": {
                "use_clustering_features": use_clustering_features,
                "use_dimensionality_reduction": use_dimensionality_reduction,
                "use_variance_selection": use_variance_selection,
                "n_clusters_range": n_clusters_range,
                "pca_components": pca_components,
                "variance_threshold": variance_threshold,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        })
        with open(self.config_path, 'w') as f:
            yaml.dump(self.intel, f)
        self.logger.info(f"Updated intel.yaml at {self.config_path}")
        return self.intel

    def run(self, use_clustering_features: bool = True, use_dimensionality_reduction: bool = False,
            use_variance_selection: bool = True, n_clusters_range: List[int] = [3, 5, 8],
            pca_components: int = 10, variance_threshold: float = 0.01):
        """
        Run the clustering feature engineering process with the specified parameters.
        """
        section("CLUSTERING FEATURE ENGINEERING PROCESS", self.logger)
        train_df, test_df = self._load_data()

        # For clustering, we might not have a target column
        if self.target_column and self.target_column in train_df.columns:
            X_train = train_df.drop(columns=[self.target_column])
            X_test = test_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            y_test = test_df[self.target_column]
        else:
            X_train = train_df.copy()
            X_test = test_df.copy()
            y_train = None
            y_test = None

        pipeline_steps = []

        if use_clustering_features:
            pipeline_steps.append(('clustering_features', ClusteringFeatureGenerator(
                n_clusters_range=n_clusters_range, n_neighbors=5)))
        else:
            pipeline_steps.append(('identity', IdentityTransformer()))

        if use_dimensionality_reduction:
            pipeline_steps.append(('dim_reduction', DimensionalityReductionTransformer(
                use_pca=True, use_tsne=False, pca_components=pca_components)))

        if use_variance_selection:
            pipeline_steps.append(('variance_selector', VarianceFeatureSelector(
                variance_threshold=variance_threshold)))

        transformation_pipeline = Pipeline(pipeline_steps)
        transformation_pipeline.fit(X_train)

        try:
            X_train_transformed = transformation_pipeline.transform(X_train).reset_index(drop=True)
            X_test_transformed = transformation_pipeline.transform(X_test).reset_index(drop=True)

            # Handle target column if it exists
            if y_train is not None:
                y_train_reset = y_train.reset_index(drop=True)
                y_test_reset = y_test.reset_index(drop=True)
                train_transformed_df = pd.concat([X_train_transformed, y_train_reset], axis=1)
                test_transformed_df = pd.concat([X_test_transformed, y_test_reset], axis=1)
            else:
                train_transformed_df = X_train_transformed
                test_transformed_df = X_test_transformed

            self._save_data(train_transformed_df, test_transformed_df)
            self._save_pipelines(transformation_pipeline)
            self._log_feature_info(transformation_pipeline, use_clustering_features,
                                   use_dimensionality_reduction, use_variance_selection)
            updated_intel = self._update_intel(use_clustering_features, use_dimensionality_reduction,
                                               use_variance_selection, n_clusters_range,
                                               pca_components, variance_threshold)

            return {
                "status": "success",
                "message": "Clustering feature engineering completed successfully",
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

    def _log_feature_info(self, pipeline, use_clustering_features, use_dimensionality_reduction,
                          use_variance_selection):
        if use_clustering_features and 'clustering_features' in pipeline.named_steps:
            clustering_step = pipeline.named_steps['clustering_features']
            self.logger.info(
                f"Generated clustering features for {len(clustering_step.n_clusters_range)} cluster configurations")

        if use_dimensionality_reduction and 'dim_reduction' in pipeline.named_steps:
            dim_step = pipeline.named_steps['dim_reduction']
            if dim_step.pca_model:
                self.logger.info(f"Applied PCA with {dim_step.pca_model.n_components_} components")

        if use_variance_selection and 'variance_selector' in pipeline.named_steps:
            variance_step = pipeline.named_steps['variance_selector']
            if variance_step.selected_features:
                self.logger.info(f"Selected {len(variance_step.selected_features)} features after variance filtering")

    def get_cluster_info(self):
        """
        Return cluster information if clustering features were generated.

        Returns:
            Dict: Cluster information or error message
        """
        try:
            # Try to load the transformation pipeline
            with open(self.transformation_pipeline_path, 'rb') as f:
                pipeline = cloudpickle.load(f)

            if 'clustering_features' in pipeline.named_steps:
                clustering_step = pipeline.named_steps['clustering_features']
                cluster_info = {
                    "n_clusters_range": clustering_step.n_clusters_range,
                    "n_neighbors": clustering_step.n_neighbors,
                    "cluster_models_fitted": len(clustering_step.cluster_models)
                }
                return {
                    "status": "success",
                    "cluster_info": cluster_info
                }
            else:
                return {
                    "status": "error",
                    "message": "Clustering features were not generated"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error retrieving cluster info: {str(e)}"
            }

    def get_selected_features(self):
        """
        Return selected features if variance selection was used.

        Returns:
            Dict: Selected features or error message
        """
        try:
            # Try to load the transformation pipeline
            with open(self.transformation_pipeline_path, 'rb') as f:
                pipeline = cloudpickle.load(f)

            if 'variance_selector' in pipeline.named_steps:
                variance_step = pipeline.named_steps['variance_selector']
                if variance_step.selected_features:
                    return {
                        "status": "success",
                        "selected_features": variance_step.selected_features,
                        "n_selected": len(variance_step.selected_features)
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Selected features not available"
                    }
            else:
                return {
                    "status": "error",
                    "message": "Variance selection was not used"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error retrieving selected features: {str(e)}"
            }


# Helper function for API usage
def run_clustering_feature_engineering(
        config_path: str = "intel.yaml",
        use_clustering_features: bool = True,
        use_dimensionality_reduction: bool = False,
        use_variance_selection: bool = True,
        n_clusters_range: List[int] = [3, 5, 8],
        pca_components: int = 10,
        variance_threshold: float = 0.01
):
    """
    Run clustering feature engineering with the specified parameters.
    This function can be called from an API endpoint.

    Args:
        config_path: Path to the config file (intel.yaml)
        use_clustering_features: Whether to generate clustering-specific features
        use_dimensionality_reduction: Whether to apply PCA and other dimensionality reduction
        use_variance_selection: Whether to remove low-variance features
        n_clusters_range: List of cluster numbers to try for feature generation
        pca_components: Number of PCA components to keep
        variance_threshold: Threshold for variance-based feature selection

    Returns:
        Dict: Result of the clustering feature engineering process
    """
    try:
        engineer = FeatureEngineer(config_path=config_path)
        result = engineer.run(
            use_clustering_features=use_clustering_features,
            use_dimensionality_reduction=use_dimensionality_reduction,
            use_variance_selection=use_variance_selection,
            n_clusters_range=n_clusters_range,
            pca_components=pca_components,
            variance_threshold=variance_threshold
        )
        return result
    except Exception as e:
        logger = logging.getLogger("Feature Engineering")
        logger.error(f"Error in clustering feature engineering: {str(e)}")
        return {
            "status": "error",
            "message": f"Clustering feature engineering failed: {str(e)}"
        }


if __name__ == "__main__":
    # This block is for direct script execution (not API call)
    # It demonstrates how to use the API-friendly version
    try:
        # Get parameters from command line arguments if provided
        import argparse

        parser = argparse.ArgumentParser(description='Run clustering feature engineering process')
        parser.add_argument('--config', default='intel.yaml', help='Path to config file')
        parser.add_argument('--use-clustering-features', action='store_true', default=True,
                            help='Generate clustering-specific features')
        parser.add_argument('--use-dimensionality-reduction', action='store_true',
                            help='Apply dimensionality reduction')
        parser.add_argument('--use-variance-selection', action='store_true', default=True,
                            help='Remove low-variance features')
        parser.add_argument('--n-clusters', nargs='+', type=int, default=[3, 5, 8], help='Numbers of clusters to try')
        parser.add_argument('--pca-components', type=int, default=10, help='Number of PCA components')
        parser.add_argument('--variance-threshold', type=float, default=0.01,
                            help='Variance threshold for feature selection')
        args = parser.parse_args()

        # Configure logger
        configure_logger()
        logger = logging.getLogger("Feature Engineering")

        # Run clustering feature engineering
        result = run_clustering_feature_engineering(
            config_path=args.config,
            use_clustering_features=args.use_clustering_features,
            use_dimensionality_reduction=args.use_dimensionality_reduction,
            use_variance_selection=args.use_variance_selection,
            n_clusters_range=args.n_clusters,
            pca_components=args.pca_components,
            variance_threshold=args.variance_threshold
        )

        if result["status"] == "success":
            logger.info("Clustering feature engineering completed successfully")
        else:
            logger.critical(f"Clustering feature engineering failed: {result['message']}")
            sys.exit(1)

    except Exception as e:
        logger = logging.getLogger("Feature Engineering")
        logger.critical(f"Feature engineering failed: {str(e)}")
        sys.exit(1)