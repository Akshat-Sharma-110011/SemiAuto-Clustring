#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script handles model selection, training, and storing for clustering problems.
It provides a selection of clustering algorithms including density-based, centroid-based,
hierarchical, and distribution-based clustering methods, allows for custom hyperparameter
tuning, and stores the trained model.
API-friendly version for integration with FastAPI.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import cloudpickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

# Import clustering algorithms
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN,
    SpectralClustering, MeanShift, OPTICS, Birch, AffinityPropagation
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# Import advanced clustering algorithms
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except ImportError:
    KMEDOIDS_AVAILABLE = False

# Import the custom logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from semiauto_clustering.logger import configure_logger, section

# Configure logger
configure_logger()
logger = logging.getLogger("Model Building")


class ModelBuilder:
    """
    A class to build, tune, and save clustering models for the AutoML pipeline.
    API-friendly version for integration with FastAPI.
    """

    def __init__(self, intel_path: str = "intel.yaml"):
        """
        Initialize ModelBuilder with paths from intel.yaml

        Args:
            intel_path: Path to the intel.yaml file
        """
        section(f"Initializing ModelBuilder with intel file: {intel_path}", logger)
        self.intel_path = intel_path
        self.intel = self._load_intel()
        self.dataset_name = self.intel.get('dataset_name')

        # Load data paths
        self.train_data_path = self.intel.get('train_transformed_path')
        self.test_data_path = self.intel.get('test_transformed_path')

        # Setup model directory - Updated for clustering
        self.model_dir = Path(f"model/model_{self.dataset_name}")
        self.model_path = self.model_dir / "model.pkl"

        # Available model dictionary with their default parameters
        self.available_models = self._get_available_models()

        logger.info(f"ModelBuilder initialized for dataset: {self.dataset_name}")

    def _load_intel(self) -> Dict[str, Any]:
        """Load the intel.yaml file"""
        try:
            with open(self.intel_path, 'r') as file:
                intel = yaml.safe_load(file)
            logger.info(f"Successfully loaded intel from {self.intel_path}")
            return intel
        except Exception as e:
            logger.error(f"Failed to load intel file: {e}")
            raise

    def _get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available clustering algorithms with their default parameters

        Returns:
            Dictionary of model names and their class/default parameters
        """
        models = {
            # Centroid-based clustering
            "KMeans": {
                "class": KMeans,
                "params": {
                    "n_clusters": 8,
                    "init": "k-means++",
                    "n_init": 10,
                    "max_iter": 300,
                    "tol": 1e-4,
                    "random_state": 42,
                    "algorithm": "lloyd"
                },
                "description": "K-Means clustering algorithm using Lloyd's algorithm"
            },
            "MiniBatch KMeans": {
                "class": MiniBatchKMeans,
                "params": {
                    "n_clusters": 8,
                    "init": "k-means++",
                    "n_init": 3,
                    "max_iter": 100,
                    "batch_size": 1024,
                    "tol": 1e-4,
                    "random_state": 42
                },
                "description": "Mini-Batch K-Means for large datasets"
            },

            # Hierarchical clustering
            "Agglomerative Clustering": {
                "class": AgglomerativeClustering,
                "params": {
                    "n_clusters": 8,
                    "linkage": "ward",
                    "metric": "euclidean"
                },
                "description": "Agglomerative hierarchical clustering"
            },

            # Density-based clustering
            "DBSCAN": {
                "class": DBSCAN,
                "params": {
                    "eps": 0.5,
                    "min_samples": 5,
                    "metric": "euclidean",
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "n_jobs": -1
                },
                "description": "Density-Based Spatial Clustering of Applications with Noise"
            },
            "OPTICS": {
                "class": OPTICS,
                "params": {
                    "min_samples": 5,
                    "max_eps": float('inf'),
                    "metric": "euclidean",
                    "cluster_method": "xi",
                    "eps": None,
                    "xi": 0.05,
                    "n_jobs": -1
                },
                "description": "Ordering Points To Identify the Clustering Structure"
            },

            # Spectral clustering
            "Spectral Clustering": {
                "class": SpectralClustering,
                "params": {
                    "n_clusters": 8,
                    "eigen_solver": None,
                    "n_components": None,
                    "random_state": 42,
                    "n_init": 10,
                    "gamma": 1.0,
                    "affinity": "rbf",
                    "n_neighbors": 10,
                    "eigen_tol": 0.0,
                    "assign_labels": "kmeans"
                },
                "description": "Spectral clustering using graph theory"
            },

            # Mean Shift clustering
            "Mean Shift": {
                "class": MeanShift,
                "params": {
                    "bandwidth": None,
                    "seeds": None,
                    "bin_seeding": False,
                    "min_bin_freq": 1,
                    "cluster_all": True,
                    "n_jobs": -1,
                    "max_iter": 300
                },
                "description": "Mean shift clustering using a flat kernel"
            },

            # Birch clustering
            "Birch": {
                "class": Birch,
                "params": {
                    "n_clusters": 8,
                    "threshold": 0.5,
                    "branching_factor": 50,
                    "compute_labels": True,
                    "copy": True
                },
                "description": "Balanced Iterative Reducing and Clustering using Hierarchies"
            },

            # Affinity Propagation
            "Affinity Propagation": {
                "class": AffinityPropagation,
                "params": {
                    "damping": 0.5,
                    "max_iter": 200,
                    "convergence_iter": 15,
                    "copy": True,
                    "preference": None,
                    "affinity": "euclidean",
                    "verbose": False,
                    "random_state": 42
                },
                "description": "Affinity Propagation clustering algorithm"
            },

            # Gaussian Mixture Models
            "Gaussian Mixture": {
                "class": GaussianMixture,
                "params": {
                    "n_components": 8,
                    "covariance_type": "full",
                    "tol": 1e-3,
                    "reg_covar": 1e-6,
                    "max_iter": 100,
                    "n_init": 1,
                    "init_params": "kmeans",
                    "weights_init": None,
                    "means_init": None,
                    "precisions_init": None,
                    "random_state": 42,
                    "warm_start": False,
                    "verbose": 0
                },
                "description": "Gaussian Mixture Model for probabilistic clustering"
            },
            "Bayesian Gaussian Mixture": {
                "class": BayesianGaussianMixture,
                "params": {
                    "n_components": 8,
                    "covariance_type": "full",
                    "tol": 1e-3,
                    "reg_covar": 1e-6,
                    "max_iter": 100,
                    "n_init": 1,
                    "init_params": "kmeans",
                    "weight_concentration_prior_type": "dirichlet_process",
                    "weight_concentration_prior": None,
                    "mean_precision_prior": None,
                    "mean_prior": None,
                    "degrees_of_freedom_prior": None,
                    "covariance_prior": None,
                    "random_state": 42,
                    "warm_start": False,
                    "verbose": 0
                },
                "description": "Variational Bayesian estimation of a finite mixture of Gaussian components"
            }
        }

        # Add HDBSCAN if available
        if HDBSCAN_AVAILABLE:
            models["HDBSCAN"] = {
                "class": hdbscan.HDBSCAN,
                "params": {
                    "min_cluster_size": 5,
                    "min_samples": None,
                    "cluster_selection_epsilon": 0.0,
                    "max_cluster_size": None,
                    "metric": "euclidean",
                    "alpha": 1.0,
                    "algorithm": "best",
                    "leaf_size": 40,
                    "n_jobs": -1,
                    "cluster_selection_method": "eom"
                },
                "description": "Hierarchical Density-Based Spatial Clustering of Applications with Noise"
            }

        # Add K-Medoids if available
        if KMEDOIDS_AVAILABLE:
            models["KMedoids"] = {
                "class": KMedoids,
                "params": {
                    "n_clusters": 8,
                    "metric": "euclidean",
                    "method": "alternate",
                    "init": "heuristic",
                    "max_iter": 300,
                    "random_state": 42
                },
                "description": "K-medoids clustering using actual data points as centroids"
            }

        return models

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the training and test data (clustering doesn't use target variables)

        Returns:
            X_train, X_test
        """
        section("Loading Data", logger)

        try:
            # Load train data
            train_data = pd.read_csv(self.train_data_path)
            logger.info(f"Loaded training data from {self.train_data_path}")
            logger.info(f"Training data shape: {train_data.shape}")

            # Load test data
            test_data = pd.read_csv(self.test_data_path)
            logger.info(f"Loaded test data from {self.test_data_path}")
            logger.info(f"Test data shape: {test_data.shape}")

            # For clustering, we use all columns as features (no target column)
            X_train = train_data
            X_test = test_data

            logger.info(f"X_train shape: {X_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}")

            # Log basic statistics
            logger.info(f"Training data columns: {list(X_train.columns)}")
            logger.info(f"Training data info:\n{X_train.describe()}")

            return X_train, X_test

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def get_available_models(self) -> Dict[str, Dict]:
        """
        API-friendly method to get available models and their default parameters.

        Returns:
            Dictionary with model names, descriptions and default parameters
        """
        models_info = {}

        for model_name, model_data in self.available_models.items():
            # Don't include the class object, just params and description
            models_info[model_name] = {
                "params": model_data["params"],
                "description": model_data["description"]
            }

        return models_info

    def process_model_request(self, model_name: str, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        API-friendly method to process a model request.

        Args:
            model_name: Name of the model to train
            custom_params: Optional dictionary of custom parameters to use

        Returns:
            Dictionary with model information
        """
        section(f"Processing model request for {model_name}", logger)

        # Check if model exists
        if model_name not in self.available_models:
            error_msg = f"Model '{model_name}' not found. Available models: {list(self.available_models.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get model info and apply custom parameters if provided
        model_info = self.available_models[model_name].copy()
        if custom_params:
            # Apply custom parameters with type checking
            for param_name, param_value in custom_params.items():
                # Check if the parameter exists in the default parameters
                if param_name in model_info['params']:
                    # Get the default value's type
                    default_value = model_info['params'][param_name]
                    # Cast the custom value to the same type as the default
                    try:
                        if isinstance(default_value, bool):
                            # Handle booleans which might come as strings
                            if isinstance(param_value, str):
                                # Convert string 'true' or 'false' to boolean
                                cast_value = param_value.lower() == 'true'
                            else:
                                cast_value = bool(param_value)
                        elif isinstance(default_value, int):
                            # Cast to int, allowing for float strings (e.g., '5' -> 5)
                            cast_value = int(float(param_value)) if isinstance(param_value, str) else int(param_value)
                        elif isinstance(default_value, float):
                            cast_value = float(param_value)
                        elif default_value is None:
                            # Handle None values - keep the provided value as is
                            cast_value = param_value
                        else:
                            # For other types (str, etc.), use as is
                            cast_value = param_value
                        model_info['params'][param_name] = cast_value
                    except Exception as e:
                        logger.error(
                            f"Failed to cast parameter '{param_name}' value '{param_value}' to type {type(default_value)}: {e}")
                        raise ValueError(f"Invalid value for parameter '{param_name}': {param_value}")
                else:
                    # Add new parameter not present in defaults
                    model_info['params'][param_name] = param_value
            logger.info(f"Applied custom parameters: {custom_params}")

        try:
            # Load data
            X_train, X_test = self.load_data()

            # Train model
            model = self.train_model(model_name, model_info, X_train)

            # Save model
            model_path = self.save_model(model, model_name)

            # Return result
            result = {
                'model_name': model_name,
                'model_path': model_path,
                'parameters': model_info['params'],
                'status': 'success'
            }

            return result

        except Exception as e:
            error_msg = f"Error processing model request: {str(e)}"
            logger.error(error_msg)
            raise

    def train_model(self, model_name: str, model_info: Dict[str, Any], X_train: pd.DataFrame) -> Any:
        """
        Train the selected clustering model

        Args:
            model_name: Name of the model
            model_info: Dictionary with model class and parameters
            X_train: Training features

        Returns:
            Trained model object
        """
        section(f"Training {model_name}", logger)

        try:
            # Special handling for algorithms that need data-dependent parameter adjustment
            if model_name == "Mean Shift":
                # If bandwidth is None, estimate it from data
                if model_info['params']['bandwidth'] is None:
                    from sklearn.cluster import estimate_bandwidth
                    bandwidth = estimate_bandwidth(X_train.values, quantile=0.2, n_samples=500)
                    model_info['params']['bandwidth'] = bandwidth if bandwidth > 0 else 1.0
                    logger.info(f"Estimated bandwidth for Mean Shift: {model_info['params']['bandwidth']}")

            elif model_name == "Spectral Clustering":
                # Adjust n_components if not specified
                if model_info['params']['n_components'] is None:
                    model_info['params']['n_components'] = model_info['params']['n_clusters']
                    logger.info(f"Set n_components to n_clusters for Spectral Clustering")

            elif model_name == "HDBSCAN" and HDBSCAN_AVAILABLE:
                # If min_samples is None, set it to min_cluster_size
                if model_info['params']['min_samples'] is None:
                    model_info['params']['min_samples'] = model_info['params']['min_cluster_size']
                    logger.info(f"Set min_samples to min_cluster_size for HDBSCAN")

            # Handle algorithms that don't support n_jobs parameter
            if model_name in ["Gaussian Mixture", "Bayesian Gaussian Mixture", "Birch", "Affinity Propagation"]:
                if 'n_jobs' in model_info['params']:
                    del model_info['params']['n_jobs']

            # Instantiate model with parameters
            model = model_info['class'](**model_info['params'])

            # Train the model
            logger.info(f"Starting training for {model_name}...")

            # Some algorithms use fit_predict, others use fit
            if hasattr(model, 'fit_predict') and model_name in ["DBSCAN", "OPTICS", "HDBSCAN"]:
                # For density-based algorithms, fit_predict is often preferred
                labels = model.fit_predict(X_train)
                logger.info(f"Model training completed successfully")
                logger.info(f"Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
                if -1 in labels:
                    logger.info(f"Number of noise points: {list(labels).count(-1)}")
            else:
                # For other algorithms, use fit
                model.fit(X_train)
                logger.info(f"Model training completed successfully")

                # Log cluster information if available
                if hasattr(model, 'labels_'):
                    labels = model.labels_
                    logger.info(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
                    if -1 in labels:
                        logger.info(f"Number of noise points: {list(labels).count(-1)}")
                elif hasattr(model, 'n_components'):
                    logger.info(f"Number of components: {model.n_components}")

            return model

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def save_model(self, model: Any, model_name: str) -> str:
        """
        Save the trained model using cloudpickle

        Args:
            model: Trained model object
            model_name: Name of the model

        Returns:
            Path to the saved model
        """
        section("Saving Model", logger)

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        try:
            # Save model with cloudpickle
            with open(self.model_path, 'wb') as f:
                cloudpickle.dump(model, f)

            logger.info(f"Model saved to {self.model_path}")

            # Update intel.yaml with model path
            model_path_str = str(self.model_path)
            self.intel['model_path'] = model_path_str
            self.intel['model_name'] = model_name
            self.intel['model_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

            with open(self.intel_path, 'w') as f:
                yaml.dump(self.intel, f, default_flow_style=False)

            logger.info(f"Updated intel.yaml with model path: {model_path_str}")

            return model_path_str

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise


# For backward compatibility with command-line usage
if __name__ == "__main__":
    print("This script is intended to be used as a module by the FastAPI application.")
    print("For command-line usage, please use the original model_building.py script.")
    sys.exit(1)