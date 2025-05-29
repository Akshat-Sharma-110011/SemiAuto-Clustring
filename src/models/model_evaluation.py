"""
Model Evaluation Module for Clustering.

This module evaluates the performance of trained clustering model using various metrics
and stores the results in a YAML file. It also evaluates the optimized model if available.
API-friendly version that can be imported and used in a FastAPI application.
"""
import math
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Clustering metrics imports
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score
)

# Import custom logger
import logging
from src.logger import section, configure_logger

# Configure logger
configure_logger()
logger = logging.getLogger("Model Evaluation")


def load_intel(intel_path: str = "intel.yaml") -> Dict[str, Any]:
    section(f"Loading Intel from {intel_path}", logger)
    try:
        with open(intel_path, "r") as f:
            intel = yaml.safe_load(f)
        logger.info(f"Successfully loaded intel from {intel_path}")
        return intel
    except Exception as e:
        logger.error(f"Failed to load intel file: {e}")
        raise


def load_model(model_path: str) -> Any:
    section(f"Loading Model from {model_path}", logger)
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_test_data(test_path: str, target_column: str = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    section(f"Loading Test Data from {test_path}", logger)
    try:
        test_data = pd.read_csv(test_path)
        logger.info(f"Test data shape: {test_data.shape}")

        # For clustering, target column might not exist
        if target_column and target_column in test_data.columns:
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]
            logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            logger.info(f"Number of unique classes: {y_test.nunique()}")
            logger.info(f"Class distribution: {y_test.value_counts().to_dict()}")
        else:
            X_test = test_data
            y_test = None
            logger.info(f"X_test shape: {X_test.shape}, no target column available")

        return X_test, y_test
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise


def get_cluster_labels(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    try:
        if hasattr(model, 'predict'):
            return model.predict(X_test)
        elif hasattr(model, 'labels_'):
            # For models that need to be fitted again (like DBSCAN, etc.)
            model_copy = model.__class__(**model.get_params())
            return model_copy.fit_predict(X_test)
        elif hasattr(model, 'fit_predict'):
            model_copy = model.__class__(**model.get_params())
            return model_copy.fit_predict(X_test)
        else:
            raise ValueError("Model doesn't have predict, labels_, or fit_predict method")
    except Exception as e:
        logger.error(f"Error getting cluster labels: {e}")
        raise


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: Optional[pd.Series] = None) -> Dict[str, Any]:
    section("Evaluating Clustering Model Performance", logger)
    try:
        # Get cluster labels
        cluster_labels = get_cluster_labels(model, X_test)

        # Remove outliers (-1 labels) for certain metrics if present
        valid_mask = cluster_labels != -1
        X_valid = X_test[valid_mask]
        labels_valid = cluster_labels[valid_mask]

        # Initialize metrics dictionary
        metrics = {}

        # Basic cluster information
        n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
        n_outliers = np.sum(cluster_labels == -1)

        metrics.update({
            "n_clusters": int(n_clusters),
            "n_outliers": int(n_outliers),
            "n_samples": int(len(cluster_labels)),
            "outlier_ratio": float(n_outliers / len(cluster_labels))
        })

        # Internal evaluation metrics (don't require true labels)
        if len(labels_valid) > 1 and n_clusters > 1:
            try:
                metrics["silhouette_score"] = float(silhouette_score(X_valid, labels_valid))
            except Exception as e:
                logger.warning(f"Could not calculate silhouette score: {e}")
                metrics["silhouette_score"] = None

            try:
                metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_valid, labels_valid))
            except Exception as e:
                logger.warning(f"Could not calculate Calinski-Harabasz score: {e}")
                metrics["calinski_harabasz_score"] = None

            try:
                metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_valid, labels_valid))
            except Exception as e:
                logger.warning(f"Could not calculate Davies-Bouldin score: {e}")
                metrics["davies_bouldin_score"] = None

        # External evaluation metrics (require true labels)
        if y_test is not None:
            y_test_valid = y_test[valid_mask] if len(y_test) == len(cluster_labels) else y_test

            try:
                metrics["adjusted_rand_score"] = float(adjusted_rand_score(y_test_valid, labels_valid))
            except Exception as e:
                logger.warning(f"Could not calculate Adjusted Rand Score: {e}")
                metrics["adjusted_rand_score"] = None

            try:
                metrics["adjusted_mutual_info_score"] = float(adjusted_mutual_info_score(y_test_valid, labels_valid))
            except Exception as e:
                logger.warning(f"Could not calculate Adjusted Mutual Info Score: {e}")
                metrics["adjusted_mutual_info_score"] = None

            try:
                metrics["normalized_mutual_info_score"] = float(
                    normalized_mutual_info_score(y_test_valid, labels_valid))
            except Exception as e:
                logger.warning(f"Could not calculate Normalized Mutual Info Score: {e}")
                metrics["normalized_mutual_info_score"] = None

            try:
                metrics["homogeneity_score"] = float(homogeneity_score(y_test_valid, labels_valid))
            except Exception as e:
                logger.warning(f"Could not calculate Homogeneity Score: {e}")
                metrics["homogeneity_score"] = None

            try:
                metrics["completeness_score"] = float(completeness_score(y_test_valid, labels_valid))
            except Exception as e:
                logger.warning(f"Could not calculate Completeness Score: {e}")
                metrics["completeness_score"] = None

            try:
                metrics["v_measure_score"] = float(v_measure_score(y_test_valid, labels_valid))
            except Exception as e:
                logger.warning(f"Could not calculate V-Measure Score: {e}")
                metrics["v_measure_score"] = None

            try:
                metrics["fowlkes_mallows_score"] = float(fowlkes_mallows_score(y_test_valid, labels_valid))
            except Exception as e:
                logger.warning(f"Could not calculate Fowlkes-Mallows Score: {e}")
                metrics["fowlkes_mallows_score"] = None

        # Cluster size statistics
        cluster_sizes = []
        for i in range(n_clusters):
            size = np.sum(labels_valid == i) if n_clusters > 0 else 0
            cluster_sizes.append(int(size))

        metrics.update({
            "cluster_sizes": cluster_sizes,
            "min_cluster_size": int(min(cluster_sizes)) if cluster_sizes else 0,
            "max_cluster_size": int(max(cluster_sizes)) if cluster_sizes else 0,
            "avg_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            "std_cluster_size": float(np.std(cluster_sizes)) if cluster_sizes else 0.0
        })

        # Model-specific information
        metrics["model_type"] = model.__class__.__name__
        metrics["model_params"] = model.get_params()

        # Store cluster labels for further analysis
        metrics["cluster_labels"] = cluster_labels.tolist()

        # Log key metrics
        logger.info(f"Number of Clusters: {metrics['n_clusters']}")
        logger.info(f"Number of Outliers: {metrics['n_outliers']}")
        logger.info(f"Outlier Ratio: {metrics['outlier_ratio']:.4f}")

        if metrics.get("silhouette_score") is not None:
            logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        if metrics.get("calinski_harabasz_score") is not None:
            logger.info(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")
        if metrics.get("davies_bouldin_score") is not None:
            logger.info(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")

        if y_test is not None:
            if metrics.get("adjusted_rand_score") is not None:
                logger.info(f"Adjusted Rand Score: {metrics['adjusted_rand_score']:.4f}")
            if metrics.get("v_measure_score") is not None:
                logger.info(f"V-Measure Score: {metrics['v_measure_score']:.4f}")

        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def save_metrics(metrics: Dict[str, Any], dataset_name: str, filename: str = "performance.yaml") -> str:
    section(f"Saving Performance Metrics to {filename}", logger)
    try:
        metrics_dir = os.path.join("reports", "metrics", f"performance_{dataset_name}")
        os.makedirs(metrics_dir, exist_ok=True)

        metrics["evaluation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cleaned_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                cleaned_metrics[key] = None
            else:
                cleaned_metrics[key] = value

        metrics_file_path = os.path.join(metrics_dir, filename)

        with open(metrics_file_path, "w") as f:
            yaml.dump(cleaned_metrics, f, default_flow_style=False, indent=2)

        logger.info(f"Metrics saved to {metrics_file_path}")
        return metrics_file_path
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise


def update_intel(intel: Dict[str, Any], metrics_path: str, intel_path: str = "intel.yaml",
                 is_optimized: bool = False) -> Dict[str, Any]:
    section("Updating Intel YAML", logger)
    try:
        updated_intel = intel.copy()

        if is_optimized:
            updated_intel["optimized_performance_metrics_path"] = metrics_path
            updated_intel["optimized_evaluation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Intel updated with optimized performance metrics path: {metrics_path}")
        else:
            updated_intel["performance_metrics_path"] = metrics_path
            updated_intel["evaluation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Intel updated with performance metrics path: {metrics_path}")

        with open(intel_path, "w") as f:
            yaml.dump(updated_intel, f, default_flow_style=False)

        return updated_intel

    except Exception as e:
        logger.error(f"Failed to update intel: {e}")
        raise


def check_optimized_model_exists(dataset_name: str) -> Tuple[bool, str]:
    optimized_model_path = os.path.join("model", f"model_{dataset_name}", "optimized_model.pkl")
    exists = os.path.isfile(optimized_model_path)

    if exists:
        logger.info(f"Optimized model found at {optimized_model_path}")
    else:
        logger.info("No optimized model found")

    return exists, optimized_model_path


def evaluate_and_save_model(model: Any, X_test: pd.DataFrame, y_test: Optional[pd.Series],
                            dataset_name: str, is_optimized: bool = False) -> Dict[str, Any]:
    metrics = evaluate_model(model, X_test, y_test)

    filename = "optimized_performance.yaml" if is_optimized else "performance.yaml"

    metrics_path = save_metrics(metrics, dataset_name, filename)

    metrics["metrics_path"] = metrics_path
    return metrics


def run_evaluation(intel_path: str = "intel.yaml") -> Dict[str, Any]:
    section("Starting Clustering Model Evaluation", logger, char="*", length=60)
    results = {
        "success": False,
        "standard_model": None,
        "optimized_model": None,
        "intel": None,
        "error": None
    }

    try:
        intel = load_intel(intel_path)
        results["intel"] = intel

        model_path = intel["model_path"]
        test_path = intel["test_transformed_path"]
        target_column = intel.get("target_column")  # May not exist for clustering
        dataset_name = intel["dataset_name"]

        X_test, y_test = load_test_data(test_path, target_column)

        model = load_model(model_path)

        standard_metrics = evaluate_and_save_model(model, X_test, y_test, dataset_name, is_optimized=False)
        results["standard_model"] = standard_metrics

        intel = update_intel(intel, standard_metrics["metrics_path"], intel_path)
        results["intel"] = intel

        optimized_exists, optimized_model_path = check_optimized_model_exists(dataset_name)

        if optimized_exists:
            section("Evaluating Optimized Clustering Model", logger, char="-", length=50)

            optimized_model = load_model(optimized_model_path)

            optimized_metrics = evaluate_and_save_model(
                optimized_model, X_test, y_test, dataset_name, is_optimized=True)
            results["optimized_model"] = optimized_metrics

            intel = update_intel(intel, optimized_metrics["metrics_path"], intel_path, is_optimized=True)
            results["intel"] = intel

            logger.info("Optimized model evaluation completed successfully")

        section("Clustering Model Evaluation Complete", logger, char="*", length=60)
        results["success"] = True
        return results

    except Exception as e:
        error_msg = f"Clustering model evaluation failed: {str(e)}"
        logger.critical(error_msg)
        section("Clustering Model Evaluation Failed", logger, level=logging.CRITICAL, char="*", length=60)
        results["error"] = error_msg
        return results


def get_evaluation_summary(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "success": evaluation_results["success"],
        "dataset_name": evaluation_results["intel"]["dataset_name"] if evaluation_results["intel"] else None,
        "standard_model": {},
        "optimized_model": {},
        "has_optimized_model": evaluation_results["optimized_model"] is not None
    }

    if evaluation_results["standard_model"]:
        metrics = evaluation_results["standard_model"]
        summary["standard_model"] = {
            "n_clusters": metrics["n_clusters"],
            "n_outliers": metrics["n_outliers"],
            "outlier_ratio": metrics["outlier_ratio"],
            "silhouette_score": metrics.get("silhouette_score"),
            "calinski_harabasz_score": metrics.get("calinski_harabasz_score"),
            "davies_bouldin_score": metrics.get("davies_bouldin_score"),
            "adjusted_rand_score": metrics.get("adjusted_rand_score"),
            "v_measure_score": metrics.get("v_measure_score"),
            "model_type": metrics["model_type"]
        }

    if evaluation_results["optimized_model"]:
        metrics = evaluation_results["optimized_model"]
        summary["optimized_model"] = {
            "n_clusters": metrics["n_clusters"],
            "n_outliers": metrics["n_outliers"],
            "outlier_ratio": metrics["outlier_ratio"],
            "silhouette_score": metrics.get("silhouette_score"),
            "calinski_harabasz_score": metrics.get("calinski_harabasz_score"),
            "davies_bouldin_score": metrics.get("davies_bouldin_score"),
            "adjusted_rand_score": metrics.get("adjusted_rand_score"),
            "v_measure_score": metrics.get("v_measure_score"),
            "model_type": metrics["model_type"]
        }

        if evaluation_results["standard_model"]:
            std_metrics = evaluation_results["standard_model"]
            opt_metrics = evaluation_results["optimized_model"]

            improvement = {}

            if std_metrics.get("silhouette_score") and opt_metrics.get("silhouette_score"):
                improvement["silhouette_score"] = (opt_metrics["silhouette_score"] - std_metrics[
                    "silhouette_score"]) / max(abs(std_metrics["silhouette_score"]), 1e-10) * 100

            if std_metrics.get("calinski_harabasz_score") and opt_metrics.get("calinski_harabasz_score"):
                improvement["calinski_harabasz_score"] = (opt_metrics["calinski_harabasz_score"] - std_metrics[
                    "calinski_harabasz_score"]) / max(std_metrics["calinski_harabasz_score"], 1e-10) * 100

            if std_metrics.get("davies_bouldin_score") and opt_metrics.get("davies_bouldin_score"):
                improvement["davies_bouldin_score"] = (std_metrics["davies_bouldin_score"] - opt_metrics[
                    "davies_bouldin_score"]) / max(std_metrics["davies_bouldin_score"], 1e-10) * 100

            if std_metrics.get("adjusted_rand_score") and opt_metrics.get("adjusted_rand_score"):
                improvement["adjusted_rand_score"] = (opt_metrics["adjusted_rand_score"] - std_metrics[
                    "adjusted_rand_score"]) / max(abs(std_metrics["adjusted_rand_score"]), 1e-10) * 100

            if std_metrics.get("v_measure_score") and opt_metrics.get("v_measure_score"):
                improvement["v_measure_score"] = (opt_metrics["v_measure_score"] - std_metrics[
                    "v_measure_score"]) / max(std_metrics["v_measure_score"], 1e-10) * 100

            if improvement:
                summary["improvement"] = improvement

    return summary


if __name__ == "__main__":
    results = run_evaluation()
    if results["success"]:
        print("Clustering model evaluation completed successfully.")
        summary = get_evaluation_summary(results)
        print(f"\nEvaluation Summary for {summary['dataset_name']}:")
        print(f"Model Type: {summary['standard_model'].get('model_type', 'Unknown')}")
        print(f"Number of Clusters: {summary['standard_model']['n_clusters']}")
        print(f"Number of Outliers: {summary['standard_model']['n_outliers']}")
        print(f"Outlier Ratio: {summary['standard_model']['outlier_ratio']:.4f}")

        if summary['standard_model'].get('silhouette_score'):
            print(f"Standard Model Silhouette Score: {summary['standard_model']['silhouette_score']:.4f}")
        if summary['standard_model'].get('calinski_harabasz_score'):
            print(f"Standard Model Calinski-Harabasz Score: {summary['standard_model']['calinski_harabasz_score']:.4f}")
        if summary['standard_model'].get('davies_bouldin_score'):
            print(f"Standard Model Davies-Bouldin Score: {summary['standard_model']['davies_bouldin_score']:.4f}")
        if summary['standard_model'].get('adjusted_rand_score'):
            print(f"Standard Model Adjusted Rand Score: {summary['standard_model']['adjusted_rand_score']:.4f}")
        if summary['standard_model'].get('v_measure_score'):
            print(f"Standard Model V-Measure Score: {summary['standard_model']['v_measure_score']:.4f}")

        if summary['has_optimized_model']:
            print(f"\nOptimized Model Type: {summary['optimized_model'].get('model_type', 'Unknown')}")
            print(f"Optimized Model Clusters: {summary['optimized_model']['n_clusters']}")
            print(f"Optimized Model Outliers: {summary['optimized_model']['n_outliers']}")

            if summary['optimized_model'].get('silhouette_score'):
                print(f"Optimized Model Silhouette Score: {summary['optimized_model']['silhouette_score']:.4f}")
            if summary['optimized_model'].get('calinski_harabasz_score'):
                print(
                    f"Optimized Model Calinski-Harabasz Score: {summary['optimized_model']['calinski_harabasz_score']:.4f}")
            if summary['optimized_model'].get('davies_bouldin_score'):
                print(f"Optimized Model Davies-Bouldin Score: {summary['optimized_model']['davies_bouldin_score']:.4f}")
            if summary['optimized_model'].get('adjusted_rand_score'):
                print(f"Optimized Model Adjusted Rand Score: {summary['optimized_model']['adjusted_rand_score']:.4f}")
            if summary['optimized_model'].get('v_measure_score'):
                print(f"Optimized Model V-Measure Score: {summary['optimized_model']['v_measure_score']:.4f}")

            if 'improvement' in summary:
                print(f"\nImprovements:")
                for metric, improvement in summary['improvement'].items():
                    print(f"{metric.replace('_', ' ').title()}: {improvement:.2f}%")
    else:
        print(f"Clustering model evaluation failed: {results['error']}")