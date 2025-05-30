import os
import yaml
import cloudpickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, Tuple, List, Optional, Union

from sklearn.model_selection import GridSearchCV
import optuna
import atexit

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

from sklearn.cluster import (
    KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN,
    SpectralClustering, MeanShift, OPTICS, Birch, AffinityPropagation
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import hdbscan

from semiauto_clustering.logger import configure_logger, section

INTEL_PATH = "intel.yaml"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

configure_logger()
logger = logging.getLogger("Model Optimization")


class ModelOptimizer:
    def __init__(self, intel_path: str = INTEL_PATH, config_overrides: dict = None):
        self.intel_path = intel_path
        self.intel_config = self._load_intel()
        if config_overrides:
            self.intel_config.update(config_overrides)
        self.dataset_name = self.intel_config.get("dataset_name")
        self.model_name = self.intel_config.get("model_name")
        self.target_column = self.intel_config.get("target_column")

        self.logger = logger

        self.train_path = self.intel_config.get("train_transformed_path")
        self.test_path = self.intel_config.get("test_transformed_path")

        self.optimized_model_dir = os.path.join(ROOT_DIR, "model", f"model_{self.dataset_name}")
        self.optimized_model_path = os.path.join(self.optimized_model_dir, "optimized_model.pkl")

        self.best_params_dir = os.path.join(ROOT_DIR, "reports", "metrics", f"best_params_{self.dataset_name}")
        self.best_params_path = os.path.join(self.best_params_dir, "params.json")

        os.makedirs(self.optimized_model_dir, exist_ok=True)
        os.makedirs(self.best_params_dir, exist_ok=True)

        self.X_train, self.y_train = self._load_data(self.train_path)
        self.X_test, self.y_test = self._load_data(self.test_path)

        self.models = self._get_available_models()
        self.param_spaces = self._get_hyperparameter_spaces()

    def _load_intel(self) -> Dict[str, Any]:
        try:
            with open(self.intel_path, "r") as file:
                config = yaml.safe_load(file)
            logger.info(f"Successfully loaded configuration from {self.intel_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Successfully loaded data from {data_path}")

            X = data.drop(columns=[self.target_column], errors='ignore')
            if self.target_column in data.columns:
                y = data[self.target_column]
            else:
                y = None
                logger.warning(f"Target column '{self.target_column}' not found - using unsupervised clustering")

            logger.info(f"Data shape - X: {X.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _get_available_models(self) -> Dict[str, Any]:
        models = {
            "KMeans": KMeans,
            "MiniBatch KMeans": MiniBatchKMeans,
            "Agglomerative Clustering": AgglomerativeClustering,
            "DBSCAN": DBSCAN,
            "Spectral Clustering": SpectralClustering,
            "Mean Shift": MeanShift,
            "OPTICS": OPTICS,
            "Birch": Birch,
            "Affinity Propagation": AffinityPropagation,
            "Gaussian Mixture": GaussianMixture,
            "Bayesian Gaussian Mixture": BayesianGaussianMixture,
            "HDBSCAN": hdbscan.HDBSCAN
        }

        section("Available Clustering Models", logger)
        descriptions = {
            "KMeans": "K-Means clustering algorithm",
            "MiniBatch KMeans": "Mini-batch K-Means clustering for large datasets",
            "Agglomerative Clustering": "Hierarchical clustering using linkage criterion",
            "DBSCAN": "Density-based spatial clustering of applications with noise",
            "Spectral Clustering": "Spectral clustering for non-convex clusters",
            "Mean Shift": "Mean shift clustering algorithm",
            "OPTICS": "Ordering points to identify the clustering structure",
            "Birch": "Balanced iterative reducing and clustering using hierarchies",
            "Affinity Propagation": "Clustering by passing messages between data points",
            "Gaussian Mixture": "Gaussian mixture model clustering",
            "Bayesian Gaussian Mixture": "Bayesian gaussian mixture model with variational inference",
            "HDBSCAN": "Hierarchical density-based spatial clustering"
        }
        for i, (name, _) in enumerate(models.items(), 1):
            logger.info(f"{i}. {name} - {descriptions.get(name, '')}")

        return models

    def _get_hyperparameter_spaces(self) -> Dict[str, Dict[str, Any]]:
        param_spaces = {
            "KMeans": {
                "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "init": ["k-means++", "random"],
                "n_init": [10, 20, 30],
                "max_iter": [100, 200, 300],
                "algorithm": ["lloyd", "elkan"]
            },
            "MiniBatch KMeans": {
                "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "init": ["k-means++", "random"],
                "n_init": [3, 5, 10],
                "max_iter": [100, 200, 300],
                "batch_size": [100, 256, 512, 1024]
            },
            "Agglomerative Clustering": {
                "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "linkage": ["ward", "complete", "average", "single"],
                "affinity": ["euclidean", "manhattan", "cosine"]
            },
            "DBSCAN": {
                "eps": [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
                "min_samples": [3, 5, 7, 10, 15, 20],
                "metric": ["euclidean", "manhattan", "cosine"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
            },
            "Spectral Clustering": {
                "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "affinity": ["nearest_neighbors", "rbf", "polynomial", "sigmoid"],
                "n_neighbors": [5, 10, 15, 20],
                "gamma": [0.1, 1.0, 10.0],
                "assign_labels": ["kmeans", "discretize", "cluster_qr"]
            },
            "Mean Shift": {
                "bandwidth": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                "bin_seeding": [True, False],
                "cluster_all": [True, False]
            },
            "OPTICS": {
                "min_samples": [3, 5, 7, 10, 15, 20],
                "max_eps": [0.5, 1.0, 2.0, 5.0, 10.0],
                "metric": ["euclidean", "manhattan", "cosine"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "cluster_method": ["xi", "dbscan"]
            },
            "Birch": {
                "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10, None],
                "threshold": [0.1, 0.3, 0.5, 0.7, 1.0],
                "branching_factor": [25, 50, 75, 100]
            },
            "Affinity Propagation": {
                "damping": [0.5, 0.6, 0.7, 0.8, 0.9],
                "max_iter": [100, 200, 300],
                "convergence_iter": [10, 15, 20],
                "preference": [-1, -10, -50, -100]
            },
            "Gaussian Mixture": {
                "n_components": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "covariance_type": ["full", "tied", "diag", "spherical"],
                "init_params": ["kmeans", "k-means++", "random", "random_from_data"],
                "max_iter": [100, 200, 300],
                "n_init": [1, 3, 5]
            },
            "Bayesian Gaussian Mixture": {
                "n_components": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "covariance_type": ["full", "tied", "diag", "spherical"],
                "init_params": ["kmeans", "k-means++", "random", "random_from_data"],
                "max_iter": [100, 200, 300],
                "n_init": [1, 3, 5],
                "weight_concentration_prior_type": ["dirichlet_process", "dirichlet_distribution"]
            },
            "HDBSCAN": {
                "min_cluster_size": [3, 5, 7, 10, 15, 20],
                "min_samples": [1, 3, 5, 7, 10],
                "metric": ["euclidean", "manhattan", "cosine"],
                "cluster_selection_method": ["eom", "leaf"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
            }
        }
        return param_spaces

    def _get_model_class(self) -> Any:
        for model_key, model_class in self.models.items():
            if model_key.lower().replace(" ", "") == self.model_name.lower().replace(" ", ""):
                logger.info(f"Found model class for '{self.model_name}': {model_key}")
                return model_class

        model_mapping = {
            "KMeans": KMeans,
            "MiniBatchKMeans": MiniBatchKMeans,
            "AgglomerativeClustering": AgglomerativeClustering,
            "DBSCAN": DBSCAN,
            "SpectralClustering": SpectralClustering,
            "MeanShift": MeanShift,
            "OPTICS": OPTICS,
            "Birch": Birch,
            "AffinityPropagation": AffinityPropagation,
            "GaussianMixture": GaussianMixture,
            "BayesianGaussianMixture": BayesianGaussianMixture,
            "HDBSCAN": hdbscan.HDBSCAN
        }

        if self.model_name in model_mapping:
            logger.info(f"Found model class for '{self.model_name}' in direct mapping")
            return model_mapping[self.model_name]

        logger.error(f"Model '{self.model_name}' not found in available models")
        raise ValueError(f"Model '{self.model_name}' not found in available models")

    def _get_param_space(self) -> Dict[str, Any]:
        for model_key, param_space in self.param_spaces.items():
            if model_key.lower().replace(" ", "") == self.model_name.lower().replace(" ", ""):
                return param_space

        model_mapping = {
            "KMeans": "KMeans",
            "MiniBatchKMeans": "MiniBatch KMeans",

            "AgglomerativeClustering": "Agglomerative Clustering",
            "DBSCAN": "DBSCAN",
            "SpectralClustering": "Spectral Clustering",
            "MeanShift": "Mean Shift",
            "OPTICS": "OPTICS",
            "Birch": "Birch",
            "AffinityPropagation": "Affinity Propagation",
            "GaussianMixture": "Gaussian Mixture",
            "BayesianGaussianMixture": "Bayesian Gaussian Mixture",
            "HDBSCAN": "HDBSCAN"
        }

        if self.model_name in model_mapping:
            mapped_name = model_mapping[self.model_name]
            if mapped_name in self.param_spaces:
                return self.param_spaces[mapped_name]

        logger.error(f"Parameter space for model '{self.model_name}' not found")
        raise ValueError(f"Parameter space for model '{self.model_name}' not found")

    def _calculate_metric(
            self,
            X: np.ndarray,
            labels: np.ndarray,
            metric_name: str,
            y_true: Optional[np.ndarray] = None
    ) -> float:
        if metric_name == "silhouette":
            if len(np.unique(labels)) > 1:
                return silhouette_score(X, labels)
            else:
                return -1
        elif metric_name == "calinski_harabasz":
            if len(np.unique(labels)) > 1:
                return calinski_harabasz_score(X, labels)
            else:
                return 0
        elif metric_name == "davies_bouldin":
            if len(np.unique(labels)) > 1:
                return davies_bouldin_score(X, labels)
            else:
                return float('inf')
        elif metric_name == "adjusted_rand" and y_true is not None:
            return adjusted_rand_score(y_true, labels)
        elif metric_name == "normalized_mutual_info" and y_true is not None:
            return normalized_mutual_info_score(y_true, labels)
        elif metric_name == "homogeneity" and y_true is not None:
            return homogeneity_score(y_true, labels)
        elif metric_name == "completeness" and y_true is not None:
            return completeness_score(y_true, labels)
        elif metric_name == "v_measure" and y_true is not None:
            return v_measure_score(y_true, labels)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def optimize_with_grid_search(
            self,
            cv: int = 5,
            scoring: str = "silhouette",
            n_jobs: int = -1
    ) -> Tuple[Any, Dict[str, Any]]:
        section("Grid Search Optimization", logger)

        model_class = self._get_model_class()
        param_grid = self._get_param_space()

        logger.info(f"Starting grid search for {self.model_name}")
        logger.info(f"Hyperparameter grid: {param_grid}")

        best_score = float('-inf') if scoring in ["silhouette", "calinski_harabasz"] else float('inf')
        best_params = None
        best_model = None

        from itertools import product
        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())

        logger.info(f"Testing {len(param_combinations)} parameter combinations...")

        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))

            try:
                model = model_class(**params)

                if hasattr(model, 'fit_predict'):
                    labels = model.fit_predict(self.X_train)
                else:
                    model.fit(self.X_train)
                    labels = model.labels_ if hasattr(model, 'labels_') else model.predict(self.X_train)

                if len(np.unique(labels)) <= 1:
                    continue

                score = self._calculate_metric(self.X_train, labels, scoring, self.y_train)

                if scoring in ["silhouette", "calinski_harabasz"]:
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_model = model
                else:
                    if score < best_score:
                        best_score = score
                        best_params = params
                        best_model = model

                logger.info(f"Combination {i + 1}/{len(param_combinations)} - Score: {score:.4f}")

            except Exception as e:
                logger.warning(f"Failed to fit model with params {params}: {str(e)}")
                continue

        logger.info(f"Grid search complete. Best score: {best_score}")
        logger.info(f"Best parameters: {best_params}")

        return best_model, best_params

    def _objective(
            self,
            trial: optuna.Trial,
            model_class: Any,
            param_space: Dict[str, Any],
            metric_name: str,
            maximize: bool
    ) -> float:
        params = {}
        for param_name, param_values in param_space.items():
            if isinstance(param_values, list):
                if all(isinstance(val, (int, float)) for val in param_values) and len(param_values) > 1:
                    if all(isinstance(val, int) for val in param_values):
                        params[param_name] = trial.suggest_int(
                            param_name,
                            min(param_values),
                            max(param_values)
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            min(param_values),
                            max(param_values)
                        )
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)

        try:
            model = model_class(**params)

            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(self.X_train)
            else:
                model.fit(self.X_train)
                labels = model.labels_ if hasattr(model, 'labels_') else model.predict(self.X_train)

            if len(np.unique(labels)) <= 1:
                return float('-inf') if maximize else float('inf')

            metric_value = self._calculate_metric(self.X_train, labels, metric_name, self.y_train)

            logger.info(
                f"Trial {trial.number} - Params: {params}, "
                f"Metric ({metric_name}): {metric_value:.4f}"
            )

            return metric_value if maximize else -metric_value

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {str(e)}")
            return float('-inf') if maximize else float('inf')

    def optimize_with_optuna(
            self,
            n_trials: int,
            metric_name: str = "silhouette",
            maximize: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        section("Optuna Optimization", logger)

        model_class = self._get_model_class()
        param_space = self._get_param_space()

        logger.info(f"Starting Optuna optimization for {self.model_name}")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Metric to {'maximize' if maximize else 'minimize'}: {metric_name}")

        direction = "maximize" if maximize else "minimize"
        study = optuna.create_study(direction=direction)

        study.optimize(
            lambda trial: self._objective(trial, model_class, param_space, metric_name, maximize),
            n_trials=n_trials
        )

        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        if maximize:
            logger.info(f"Optimization complete. Best {metric_name}: {best_value:.4f}")
        else:
            logger.info(f"Optimization complete. Best {metric_name}: {-best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        best_model = model_class(**best_params)
        if hasattr(best_model, 'fit_predict'):
            best_model.fit_predict(self.X_train)
        else:
            best_model.fit(self.X_train)

        return best_model, best_params

    def save_optimized_model(self, model: Any, best_params: Dict[str, Any]) -> None:
        section("Saving Optimized Model", logger)
        try:
            # Save the raw model for evaluation compatibility
            with open(self.optimized_model_path, 'wb') as f:
                cloudpickle.dump(model, f)

            logger.info(f"Optimized model saved to {self.optimized_model_path}")

            # Save parameters separately
            with open(self.best_params_path, "w") as file:
                yaml.dump(best_params, file)
            logger.info(f"Best parameters saved to {self.best_params_path}")

        except Exception as e:
            logger.error(f"Error saving optimized model: {str(e)}")
            raise

    def update_intel_yaml(self) -> None:
        section("Updating Intel YAML", logger)

        try:
            self.intel_config["optimized_model_path"] = self.optimized_model_path
            self.intel_config["best_params_path"] = self.best_params_path
            self.intel_config["optimization_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(self.intel_path, "w") as file:
                yaml.dump(self.intel_config, file)

            logger.info(f"Updated intel.yaml with optimized model information")
        except Exception as e:
            logger.error(f"Error updating intel.yaml: {str(e)}")
            raise


def get_available_metrics():
    return {
        "1": ("silhouette", True, "Silhouette Score"),
        "2": ("calinski_harabasz", True, "Calinski-Harabasz Index"),
        "3": ("davies_bouldin", False, "Davies-Bouldin Index"),
        "4": ("adjusted_rand", True, "Adjusted Rand Index"),
        "5": ("normalized_mutual_info", True, "Normalized Mutual Information"),
        "6": ("homogeneity", True, "Homogeneity Score"),
        "7": ("completeness", True, "Completeness Score"),
        "8": ("v_measure", True, "V-Measure Score")
    }


def get_optimization_methods():
    return {
        "1": "Grid Search",
        "2": "Optuna"
    }


def optimize_model(
        optimize: bool = True,
        method: str = "1",
        n_trials: int = 50,
        metric: str = "1",
        config_overrides: dict = None
) -> dict:
    result = {
        "status": "success",
        "message": "",
        "best_params": None,
        "model_path": None,
        "metrics": {}
    }

    try:
        if not optimize:
            result["message"] = "Optimization skipped by user choice"
            return result

        logger.info("Starting model optimization process")
        optimizer = ModelOptimizer(config_overrides=config_overrides)

        if method == "1":
            optimized_model, best_params = optimizer.optimize_with_grid_search()
        else:
            metric_mapping = get_available_metrics()

            if metric not in metric_mapping:
                logger.warning("Invalid metric choice, defaulting to Silhouette Score")
                metric = "1"

            metric_name, maximize, _ = metric_mapping[metric]
            optimized_model, best_params = optimizer.optimize_with_optuna(
                n_trials=n_trials,
                metric_name=metric_name,
                maximize=maximize
            )

        optimizer.save_optimized_model(optimized_model, best_params)
        optimizer.update_intel_yaml()

        if hasattr(optimized_model, 'fit_predict'):
            labels = optimized_model.fit_predict(optimizer.X_test)
        else:
            optimized_model.fit(optimizer.X_test)
            labels = optimized_model.labels_ if hasattr(optimized_model, 'labels_') else optimized_model.predict(
                optimizer.X_test)

        metrics = {}
        if len(np.unique(labels)) > 1:
            metrics["silhouette"] = silhouette_score(optimizer.X_test, labels)
            metrics["calinski_harabasz"] = calinski_harabasz_score(optimizer.X_test, labels)
            metrics["davies_bouldin"] = davies_bouldin_score(optimizer.X_test, labels)

        if optimizer.y_test is not None:
            metrics["adjusted_rand"] = adjusted_rand_score(optimizer.y_test, labels)
            metrics["normalized_mutual_info"] = normalized_mutual_info_score(optimizer.y_test, labels)
            metrics["homogeneity"] = homogeneity_score(optimizer.y_test, labels)
            metrics["completeness"] = completeness_score(optimizer.y_test, labels)
            metrics["v_measure"] = v_measure_score(optimizer.y_test, labels)

        logger.info("Final Model Performance:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name.upper()}: {metric_value:.4f}")

        result.update({
            "best_params": best_params,
            "model_path": optimizer.optimized_model_path,
            "metrics": metrics,
            "message": "Optimization completed successfully"
        })

    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        result.update({
            "status": "error",
            "message": str(e)
        })

    return result