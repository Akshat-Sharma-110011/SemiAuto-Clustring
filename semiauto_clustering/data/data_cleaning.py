import os
import pandas as pd
import numpy as np
import re
import yaml
import cloudpickle
import hashlib
import joblib
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
from pathlib import Path
import logging
from pydantic import BaseModel, Field, validator, field_validator
import string
from functools import reduce
import warnings
from collections import Counter
from semiauto_clustering.custom_transformers import DataCleaner, CleaningParameters
# Import the logger
from semiauto_clustering.logger import configure_logger, section

# Suppress specific pandas warnings that might clutter logs
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Get module level logger
logger = logging.getLogger(__name__)


def main(dataset_name=None):
    """
    Main entry point for data cleaning.

    Args:
        dataset_name: Optional name of the dataset to clean.
                     If None, will try to load from intel.yaml.
    """
    try:
        logger.info("Starting data cleaning process")
        section("Data Cleaning Process", logger)

        # Initialize the data cleaner
        cleaner = DataCleaner()

        # Load configuration
        if dataset_name is None:
            intel = cleaner._get_intel_config()
            dataset_name = intel.get('dataset_name')

            if not dataset_name:
                logger.error("Dataset name not provided and not found in intel.yaml")
                return

        logger.info(f"Processing dataset: {dataset_name}")

        # Construct paths using intel.yaml values
        intel = cleaner._get_intel_config()
        train_path = intel.get('train_path')
        test_path = intel.get('test_path')
        cleaned_dir = os.path.join('data', 'cleaned', f'data_{dataset_name}')
        pipeline_dir = os.path.join('model', 'pipelines', f'preprocessing_{dataset_name}')

        # Ensure directories exist
        os.makedirs(cleaned_dir, exist_ok=True)
        os.makedirs(pipeline_dir, exist_ok=True)

        # Load raw data
        if not os.path.exists(train_path):
            logger.error(f"Training data not found at {train_path}")
            return

        if not os.path.exists(test_path):
            logger.warning(f"Test data not found at {test_path}, will only process training data")
            has_test = False
        else:
            has_test = True

        # Load training data
        logger.info(f"Loading training data from {train_path}")
        try:
            train_df = pd.read_csv(train_path, low_memory=False)
            logger.info(f"Loaded training data: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            return

        # Load test data if available
        test_df = None
        if has_test:
            logger.info(f"Loading test data from {test_path}")
            try:
                test_df = pd.read_csv(test_path, low_memory=False)
                logger.info(f"Loaded test data: {test_df.shape[0]} rows × {test_df.shape[1]} columns")
            except Exception as e:
                logger.error(f"Failed to load test data: {str(e)}")
                has_test = False

        # Fit the cleaning pipeline on training data
        logger.info("Fitting cleaning pipeline on training data")
        cleaner.fit(train_df, dataset_name)

        # Transform training data
        logger.info("Transforming training data")
        cleaned_train = cleaner.transform(train_df)

        # Transform test data if available
        cleaned_test = None
        if has_test and test_df is not None:
            logger.info("Transforming test data")
            cleaned_test = cleaner.transform(test_df)

        # Save cleaned data
        cleaned_train_path = os.path.join(cleaned_dir, "train.csv")
        logger.info(f"Saving cleaned training data to {cleaned_train_path}")
        cleaned_train.to_csv(cleaned_train_path, index=False)

        if has_test and cleaned_test is not None:
            cleaned_test_path = os.path.join(cleaned_dir, "test.csv")
            logger.info(f"Saving cleaned test data to {cleaned_test_path}")
            cleaned_test.to_csv(cleaned_test_path, index=False)

        # Save the cleaning pipeline
        cleaning_pipeline_path = os.path.join(pipeline_dir, "cleaning.pkl")
        logger.info(f"Saving cleaning pipeline to {cleaning_pipeline_path}")
        cleaner.save(cleaning_pipeline_path)

        # Generate and save cleaning report
        report_path = os.path.join('reports/readme/', "cleaning_report.md")
        logger.info(f"Generating cleaning report at {report_path}")
        cleaner.save_cleaning_report(report_path)

        # Update intel.yaml with cleaning information
        update_dict = {
            'cleaned_train_path': cleaned_train_path,
            'cleaning_pipeline_path': cleaning_pipeline_path,
            'cleaning_report_path': report_path,
            'cleaning_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if has_test:
            update_dict['cleaned_test_path'] = cleaned_test_path

        cleaner._update_intel_config(dataset_name, update_dict)

        logger.info("Data cleaning completed successfully")

    except Exception as e:
        logger.error(f"Error in data cleaning process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean raw data for machine learning.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset to clean")

    args = parser.parse_args()
    main(args.dataset)