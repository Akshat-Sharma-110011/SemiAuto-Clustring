# src/custom_transforms.py
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Counter

import cloudpickle
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OneHotEncoder, LabelEncoder
)
from sklearn.decomposition import PCA
import logging

from src.logger import section

logger = logging.getLogger(__name__)

class CleaningParameters(BaseModel):
    """
    Parameters for data cleaning pipeline configured using pydantic.
    All parameters have sensible defaults but can be overridden.
    """
    # Column handling
    drop_high_null_columns: bool = Field(True, description="Drop columns with high null values")
    null_column_threshold: float = Field(0.6, description="Threshold ratio for dropping high-null columns", ge=0.0,
                                         le=1.0)
    drop_constant_columns: bool = Field(True, description="Drop columns with only one unique value")
    drop_near_constant_columns: bool = Field(True, description="Drop columns with mostly one dominant value")
    dominant_value_threshold: float = Field(0.98, description="Threshold for dropping near-constant columns", ge=0.0,
                                            le=1.0)
    drop_duplicated_columns: bool = Field(True, description="Drop duplicate columns with different names")

    # Row handling
    drop_high_null_rows: bool = Field(True, description="Drop rows with too many nulls")
    null_row_threshold: float = Field(0.5, description="Threshold ratio for dropping high-null rows", ge=0.0, le=1.0)

    # Data type cleaning
    clean_strings: bool = Field(True, description="Clean string columns")
    strip_whitespace: bool = Field(True, description="Strip leading/trailing whitespace")
    lowercase_strings: bool = Field(True, description="Convert strings to lowercase")
    normalize_case: bool = Field(True, description="Normalize inconsistent case in categorical columns")
    fix_mixed_types: bool = Field(True, description="Fix columns with mixed data types")
    convert_binary_strings: bool = Field(True, description="Convert binary strings to boolean")

    # Standardize formatting
    date_format: str = Field("%Y-%m-%d", description="Standard format for dates")
    fix_dates: bool = Field(True, description="Try to fix date columns")
    standardize_formats: bool = Field(True, description="Standardize common formats (currency, percentages)")
    remove_special_chars: bool = Field(True, description="Remove special characters from strings")

    # Miscellaneous
    verbose: bool = Field(True, description="Print detailed logs")
    keep_id_columns: bool = Field(True, description="Preserve ID columns untouched")
    save_cleaning_metadata: bool = Field(True, description="Save metadata about cleaning operations")
    memory_efficient: bool = Field(True, description="Use memory-efficient operations for large datasets")

    # Processing parameters
    chunk_size: Optional[int] = Field(None, description="Process data in chunks of this size (None=no chunking)")
    max_unique_for_categorical: int = Field(100, description="Max unique values to consider a column categorical")

    # Advanced parameters
    null_filling_strategy: str = Field("none", description="Don't fill nulls - leave that to preprocessing")
    hash_threshold: int = Field(50, description="Maximum string length before converting to hash")
    correlation_threshold: float = Field(0.95, description="Threshold for highly correlated columns")

    @field_validator('null_column_threshold', 'null_row_threshold', 'dominant_value_threshold', 'correlation_threshold')
    def validate_thresholds(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {v}")
        return v


class DataCleaner:
    """
    A comprehensive data cleaning pipeline that handles:
    - High-null columns and rows
    - Constant/near-constant columns
    - String cleaning and normalization
    - Mixed types detection and conversion
    - Date standardization
    - Duplicate column detection

    This class follows the fit/transform pattern for sklearn compatibility.
    It can be fitted on training data and then applied to test data.
    """

    def __init__(self, parameters: Optional[CleaningParameters] = None):
        """Initialize the DataCleaner with optional parameters."""
        self.params = parameters if parameters is not None else CleaningParameters()
        self.stats = {
            "columns_dropped": [],
            "rows_dropped": 0,
            "fixed_columns": {},
            "original_dtypes": {},
            "new_dtypes": {},
            "start_shape": None,
            "end_shape": None,
            "cleaning_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {},
            "column_changes": {}
        }
        self.id_columns = []
        self.known_dtypes = {}
        self._is_fitted = False
        self.metadata = {}

    def _log_column_change(self, df: pd.DataFrame, column: str, operation: str):
        """Track changes made to columns for reporting."""
        if column not in self.stats["column_changes"]:
            self.stats["column_changes"][column] = []
        self.stats["column_changes"][column].append(operation)

        # Track the actual impact if verbose
        if self.params.verbose:
            before = df[column].isna().sum()
            null_pct = round((before / len(df)) * 100, 2)
            logger.info(f"Column {column}: {operation} (contains {before} nulls, {null_pct}%)")

    def _get_intel_config(self) -> dict:
        """Load the intel.yaml configuration file."""
        try:
            root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
            intel_path = root_dir / 'intel.yaml'

            if not intel_path.exists():
                logger.warning(f"Intel config not found at {intel_path}")
                return {}

            with open(intel_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading intel config: {str(e)}")
            return {}

    def _get_feature_store_config(self, dataset_name: str) -> dict:
        """Load the feature store configuration for the specific dataset."""
        try:
            root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
            feature_store_path = (
                    root_dir / 'references' /
                    f'feature_store_{dataset_name}' /
                    'feature_store.yaml'
            )

            if not feature_store_path.exists():
                logger.warning(f"Feature store config not found at {feature_store_path}")
                return {}

            with open(feature_store_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading feature store config: {str(e)}")
            return {}

    def _update_intel_config(self, dataset_name: str, update_dict: dict):
        """Update the intel.yaml file with cleaning information."""
        try:
            root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
            intel_path = root_dir / 'intel.yaml'

            if not intel_path.exists():
                logger.warning(f"Intel config not found at {intel_path}, cannot update")
                return

            with open(intel_path, 'r') as f:
                config = yaml.safe_load(f)

            # Update the config
            config.update(update_dict)

            with open(intel_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            logger.info(f"Updated intel config with cleaning information")
        except Exception as e:
            logger.error(f"Error updating intel config: {str(e)}")

    def _setup_from_config(self, dataset_name: str = None):
        """Setup the cleaner based on intel.yaml and feature_store configuration."""
        if dataset_name is None:
            intel = self._get_intel_config()
            dataset_name = intel.get('dataset_name', '')

        # Load feature store config to get column information
        feature_store = self._get_feature_store_config(dataset_name)

        # Extract ID columns that should be preserved untouched
        self.id_columns = feature_store.get('id_cols', [])
        logger.info(f"Identified ID columns to preserve: {self.id_columns}")

        # Remember original columns for filtering
        self.original_cols = feature_store.get('original_cols', [])

        # Store the dataset name
        self.dataset_name = dataset_name

        return dataset_name

    def drop_high_null_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with high percentage of null values."""
        if not self.params.drop_high_null_columns:
            return df

        section(f"Dropping High-Null Columns (threshold: {self.params.null_column_threshold})", logger)

        # Save original shape for comparison
        original_cols = df.shape[1]

        # Calculate null percentage for each column
        null_percentages = df.isnull().mean()

        # Find columns to drop (excluding ID columns)
        high_null_cols = [
            col for col in null_percentages[null_percentages > self.params.null_column_threshold].index
            if col not in self.id_columns
        ]

        # Add to stats
        self.stats["columns_dropped"].extend([(col, "high_null") for col in high_null_cols])

        if high_null_cols:
            logger.info(
                f"Dropping {len(high_null_cols)} columns with more than {self.params.null_column_threshold * 100}% null values")
            for col in high_null_cols:
                null_pct = null_percentages[col] * 100
                logger.info(f"  - {col}: {null_pct:.2f}% null values")

            # Drop the columns
            df = df.drop(columns=high_null_cols)
            logger.info(f"Reduced columns from {original_cols} to {df.shape[1]}")
        else:
            logger.info("No high-null columns found to drop")

        return df

    def drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with only one unique value (excluding nulls)."""
        if not self.params.drop_constant_columns:
            return df

        section("Dropping Constant Columns", logger)

        # Save original shape for comparison
        original_cols = df.shape[1]

        # Find constant columns (excluding ID columns)
        constant_cols = []

        for col in df.columns:
            if col in self.id_columns:
                continue

            # Get unique non-null values
            unique_values = df[col].dropna().nunique()

            if unique_values <= 1:
                constant_cols.append(col)
                self.stats["columns_dropped"].append((col, "constant"))

        if constant_cols:
            logger.info(f"Dropping {len(constant_cols)} constant columns")
            for col in constant_cols:
                try:
                    unique_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    logger.info(f"  - {col}: constant value = {unique_val}")
                except:
                    logger.info(f"  - {col}: constant column (value display error)")

            # Drop the columns
            df = df.drop(columns=constant_cols)
            logger.info(f"Reduced columns from {original_cols} to {df.shape[1]}")
        else:
            logger.info("No constant columns found to drop")

        return df

    def drop_near_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are nearly constant (one value dominates)."""
        if not self.params.drop_near_constant_columns:
            return df

        section("Dropping Near-Constant Columns", logger)

        # Save original shape for comparison
        original_cols = df.shape[1]

        # Find near-constant columns (excluding ID columns)
        near_constant_cols = []

        for col in df.columns:
            if col in self.id_columns or col in near_constant_cols:
                continue

            # Skip if too many nulls to determine
            if df[col].isna().mean() > 0.5:
                continue

            # Get value counts
            value_counts = df[col].value_counts(normalize=True, dropna=True)

            # Check if the most common value exceeds the threshold
            if not value_counts.empty and value_counts.iloc[0] > self.params.dominant_value_threshold:
                near_constant_cols.append(col)
                self.stats["columns_dropped"].append((col, "near_constant"))

        if near_constant_cols:
            logger.info(
                f"Dropping {len(near_constant_cols)} near-constant columns (dominant value > {self.params.dominant_value_threshold * 100}%)")
            for col in near_constant_cols:
                try:
                    value_counts = df[col].value_counts(normalize=True, dropna=True)
                    dominant_val = value_counts.index[0]
                    dominant_pct = value_counts.iloc[0] * 100
                    logger.info(f"  - {col}: dominant value '{dominant_val}' appears in {dominant_pct:.2f}% of records")
                except:
                    logger.info(f"  - {col}: near-constant column (value display error)")

            # Drop the columns
            df = df.drop(columns=near_constant_cols)
            logger.info(f"Reduced columns from {original_cols} to {df.shape[1]}")
        else:
            logger.info("No near-constant columns found to drop")

        return df

    def drop_duplicated_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are duplicates of other columns but with different names."""
        if not self.params.drop_duplicated_columns:
            return df

        section("Dropping Duplicated Columns", logger)

        # Save original shape for comparison
        original_cols = df.shape[1]

        # Dictionary to store duplicate columns
        duplicates = {}

        # Find duplicate columns
        for i, col1 in enumerate(df.columns):
            # Skip ID columns and already identified duplicates
            if col1 in self.id_columns or any(col1 in dupes for dupes in duplicates.values()):
                continue

            duplicates[col1] = []

            # Compare with other columns
            for col2 in df.columns[i + 1:]:
                if col2 in self.id_columns:
                    continue

                # Check if columns are identical
                if df[col1].equals(df[col2]):
                    duplicates[col1].append(col2)
                    self.stats["columns_dropped"].append((col2, f"duplicate_of_{col1}"))

        # Remove empty entries
        duplicates = {k: v for k, v in duplicates.items() if v}

        if duplicates:
            logger.info(f"Found {sum(len(v) for v in duplicates.values())} duplicate columns")

            # List all duplicates
            for original, dupes in duplicates.items():
                logger.info(f"  - {original} has duplicates: {', '.join(dupes)}")

            # Create list of all duplicates to drop
            to_drop = [dup for dupes in duplicates.values() for dup in dupes]

            # Drop the columns
            df = df.drop(columns=to_drop)
            logger.info(f"Reduced columns from {original_cols} to {df.shape[1]}")
        else:
            logger.info("No duplicate columns found")

        return df

    def clean_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean string columns: lowercase, strip whitespace, remove special characters."""
        if not self.params.clean_strings:
            return df

        section("Cleaning String Columns", logger)

        # Process object/string columns
        string_cols = df.select_dtypes(include=['object']).columns

        # Filter out ID columns
        string_cols = [col for col in string_cols if col not in self.id_columns]

        if len(string_cols) > 0:
            logger.info(f"Cleaning {len(string_cols)} string columns")

            for col in string_cols:
                # Skip columns with high null ratios
                if df[col].isna().mean() > 0.5:
                    logger.info(f"  - Skipping {col} (>50% nulls)")
                    continue

                # Make a copy to avoid chained assignment warning
                df_col = df[col].copy()

                # Only process string data
                string_mask = df_col.apply(lambda x: isinstance(x, str))
                if string_mask.sum() == 0:
                    continue

                # Track if column was modified
                modified = False

                # Strip whitespace
                if self.params.strip_whitespace:
                    original_values = df_col[string_mask].copy()
                    df_col[string_mask] = df_col[string_mask].str.strip()
                    if not df_col[string_mask].equals(original_values):
                        modified = True
                        self._log_column_change(df, col, "stripped_whitespace")

                # Convert to lowercase
                if self.params.lowercase_strings:
                    original_values = df_col[string_mask].copy()
                    df_col[string_mask] = df_col[string_mask].str.lower()
                    if not df_col[string_mask].equals(original_values):
                        modified = True
                        self._log_column_change(df, col, "lowercased")

                # Remove special characters
                if self.params.remove_special_chars:
                    original_values = df_col[string_mask].copy()
                    # Only keep alphanumeric and basic punctuation
                    df_col[string_mask] = df_col[string_mask].apply(
                        lambda x: re.sub(r'[^\w\s.,;:!?()-]', '', x) if isinstance(x, str) else x
                    )
                    if not df_col[string_mask].equals(original_values):
                        modified = True
                        self._log_column_change(df, col, "removed_special_chars")

                # Update column in dataframe if modified
                if modified:
                    self.stats["fixed_columns"][col] = "string_cleaning"
                    df[col] = df_col
                    logger.info(f"  - Cleaned string column: {col}")
        else:
            logger.info("No string columns found to clean")

        return df

    def fix_mixed_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix columns with mixed data types by converting to the most appropriate type.
        This handles common cases like numbers stored as strings and boolean values.
        """
        if not self.params.fix_mixed_types:
            return df

        section("Fixing Mixed Types", logger)

        # Only process non-ID columns
        cols_to_process = [col for col in df.columns if col not in self.id_columns]

        # Store original dtypes for reference
        self.stats["original_dtypes"] = {col: str(df[col].dtype) for col in df.columns}

        # Process each column
        for col in cols_to_process:
            # Skip if the column has too many nulls
            if df[col].isna().mean() > 0.5:
                continue

            # Get current dtype
            current_dtype = df[col].dtype

            # Only process object columns
            if current_dtype != 'object':
                continue

            # Make a copy of the column to avoid SettingWithCopyWarning
            series = df[col].copy()

            # Skip columns with long string values
            if series.dropna().astype(str).str.len().max() > 100:
                continue

            # Try converting to numeric
            numeric_series = pd.to_numeric(series, errors='coerce')
            numeric_conversion_success = numeric_series.notna().mean() > 0.8  # 80% success rate

            # Try converting to datetime
            if not numeric_conversion_success and self.params.fix_dates:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        datetime_series = pd.to_datetime(series, errors='coerce')
                    datetime_conversion_success = datetime_series.notna().mean() > 0.8  # 80% success rate

                    if datetime_conversion_success:
                        df[col] = datetime_series
                        logger.info(f"Converted {col} to datetime")
                        self._log_column_change(df, col, "converted_to_datetime")
                        self.stats["fixed_columns"][col] = "datetime_conversion"
                        continue
                except:
                    pass

            # Try converting to numeric if high success rate
            if numeric_conversion_success:
                # Check if values are mostly integers
                if pd.notna(numeric_series).all() and np.floor(numeric_series) == numeric_series:
                    df[col] = numeric_series.astype('Int64')  # Int64 allows NaN values
                    logger.info(f"Converted {col} to integer")
                    self._log_column_change(df, col, "converted_to_integer")
                else:
                    df[col] = numeric_series
                    logger.info(f"Converted {col} to float")
                    self._log_column_change(df, col, "converted_to_float")

                self.stats["fixed_columns"][col] = "numeric_conversion"
                continue

            # Check for boolean values
            if self.params.convert_binary_strings:
                # Define common boolean mappings
                true_values = ['yes', 'y', 'true', 't', '1', 'on']
                false_values = ['no', 'n', 'false', 'f', '0', 'off']

                # Convert to lowercase for comparison
                lowercase_series = series.astype(str).str.lower()

                # Check if values match boolean patterns
                is_bool = lowercase_series.isin(true_values + false_values).all()

                if is_bool:
                    # Convert to boolean
                    df[col] = lowercase_series.isin(true_values)
                    logger.info(f"Converted {col} to boolean")
                    self._log_column_change(df, col, "converted_to_boolean")
                    self.stats["fixed_columns"][col] = "boolean_conversion"

        # Store new dtypes
        self.stats["new_dtypes"] = {col: str(df[col].dtype) for col in df.columns}

        return df

    def remove_high_null_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with too many null values."""
        if not self.params.drop_high_null_rows:
            return df

        section(f"Removing High-Null Rows (threshold: {self.params.null_row_threshold})", logger)

        # Save original shape for comparison
        original_rows = df.shape[0]

        # Calculate null ratio for each row
        null_ratio = df.isnull().sum(axis=1) / df.shape[1]

        # Find rows to drop
        rows_to_drop = null_ratio[null_ratio > self.params.null_row_threshold].index

        if len(rows_to_drop) > 0:
            # Calculate percentage of rows to drop
            drop_percentage = (len(rows_to_drop) / original_rows) * 100

            # Warn if dropping too many rows
            if drop_percentage > 20:
                logger.warning(f"Removing {drop_percentage:.2f}% of rows due to high null values!")

            # Drop the rows
            df = df.drop(index=rows_to_drop)

            # Update stats
            self.stats["rows_dropped"] += len(rows_to_drop)

            logger.info(
                f"Removed {len(rows_to_drop)} rows with more than {self.params.null_row_threshold * 100}% null values")
            logger.info(f"Reduced rows from {original_rows} to {df.shape[0]}")
        else:
            logger.info("No high-null rows found to remove")

        return df

    def normalize_case_in_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize inconsistent case in categorical columns."""
        if not self.params.normalize_case:
            return df

        section("Normalizing Case in Categorical Columns", logger)

        # Process string columns that might be categorical
        obj_cols = df.select_dtypes(include=['object']).columns

        # Filter out ID columns and columns with very high cardinality
        categorical_cols = []
        for col in obj_cols:
            if col in self.id_columns:
                continue

            # Check if column is likely categorical (limited unique values)
            unique_count = df[col].nunique()
            if unique_count <= self.params.max_unique_for_categorical and unique_count > 1:
                categorical_cols.append(col)

        if categorical_cols:
            logger.info(f"Checking {len(categorical_cols)} potential categorical columns for case inconsistencies")

            for col in categorical_cols:
                # Skip if too many nulls
                if df[col].isna().mean() > 0.5:
                    continue

                # Get string values only
                string_values = df[col][df[col].apply(lambda x: isinstance(x, str))]

                if len(string_values) == 0:
                    continue

                # Check for case inconsistencies by comparing lowercase versions
                lowercase_values = string_values.str.lower()
                case_inconsistent = False

                # Group by lowercase and check if same value appears with different cases
                case_groups = {}
                for original, lower in zip(string_values, lowercase_values):
                    if lower not in case_groups:
                        case_groups[lower] = []
                    if original not in case_groups[lower]:
                        case_groups[lower].append(original)

                # Find groups with multiple case variants
                inconsistent_groups = {k: v for k, v in case_groups.items() if len(v) > 1}

                if inconsistent_groups:
                    case_inconsistent = True
                    logger.info(f"  - Found case inconsistencies in {col}:")
                    for lower, variants in inconsistent_groups.items():
                        logger.info(f"    * '{lower}' appears as: {', '.join(variants)}")

                    # Create mapping from inconsistent cases to the most common form
                    mapping = {}
                    for lower, variants in inconsistent_groups.items():
                        # Find most common variant
                        variant_counts = Counter([v for v in string_values if v.lower() == lower])
                        most_common = variant_counts.most_common(1)[0][0]

                        # Map all variants to the most common
                        for variant in variants:
                            if variant != most_common:
                                mapping[variant] = most_common

                    # Apply mapping
                    df[col] = df[col].replace(mapping)
                    logger.info(f"  - Normalized case inconsistencies in {col}")
                    self._log_column_change(df, col, "normalized_case")
                    self.stats["fixed_columns"][col] = "case_normalization"
        else:
            logger.info("No suitable categorical columns found for case normalization")

        return df

    def standardize_date_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize date/datetime columns to a consistent format."""
        if not self.params.fix_dates:
            return df

        section("Standardizing Date Formats", logger)

        # Only process non-ID columns
        cols_to_process = [col for col in df.columns if col not in self.id_columns]

        # Process each column
        for col in cols_to_process:
            # Skip if already datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # Format datetime columns consistently
                df[col] = pd.to_datetime(df[col]).dt.strftime(self.params.date_format)
                self._log_column_change(df, col, "standardized_date_format")
                self.stats["fixed_columns"][col] = "date_standardization"
                continue

            # Skip non-string columns
            if df[col].dtype != 'object':
                continue

            # Check for date patterns in the column
            sample_values = df[col].dropna().astype(str).sample(min(100, len(df[col].dropna())))
            date_patterns = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d']

            # Try to convert sample to datetime using different patterns
            for pattern in date_patterns:
                try:
                    converted = pd.to_datetime(sample_values, format=pattern, errors='coerce')
                    if converted.notna().mean() > 0.8:  # 80% success rate
                        # Apply to full column
                        df[col] = pd.to_datetime(df[col], format=pattern, errors='coerce')
                        df[col] = df[col].dt.strftime(self.params.date_format)
                        logger.info(f"Standardized date format in {col} using pattern {pattern}")
                        self._log_column_change(df, col, f"standardized_date_format_{pattern}")
                        self.stats["fixed_columns"][col] = "date_standardization"
                        break
                except:
                    continue

        return df

    def basic_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps in sequence."""
        section("Starting Basic Data Cleaning", logger)

        # Store original shape
        self.stats["start_shape"] = df.shape

        # Apply cleaning steps in logical order
        df = self.drop_high_null_columns(df)
        df = self.drop_constant_columns(df)
        df = self.drop_near_constant_columns(df)
        df = self.drop_duplicated_columns(df)
        df = self.clean_string_columns(df)
        df = self.normalize_case_in_categorical(df)
        df = self.fix_mixed_types(df)
        df = self.standardize_date_formats(df)
        df = self.remove_high_null_rows(df)

        # Store final shape
        self.stats["end_shape"] = df.shape

        # Calculate summary statistics
        self.stats["summary"] = {
            "columns_dropped": len(self.stats["columns_dropped"]),
            "rows_dropped": self.stats["rows_dropped"],
            "columns_fixed": len(self.stats["fixed_columns"]),
            "start_rows": self.stats["start_shape"][0],
            "start_cols": self.stats["start_shape"][1],
            "end_rows": self.stats["end_shape"][0],
            "end_cols": self.stats["end_shape"][1],
        }

        # Log summary
        section("Data Cleaning Summary", logger)
        logger.info(f"Starting shape: {self.stats['start_shape'][0]} rows × {self.stats['start_shape'][1]} columns")
        logger.info(f"Ending shape: {self.stats['end_shape'][0]} rows × {self.stats['end_shape'][1]} columns")
        logger.info(f"Columns dropped: {self.stats['summary']['columns_dropped']}")
        logger.info(f"Rows dropped: {self.stats['summary']['rows_dropped']}")
        logger.info(f"Columns fixed/modified: {self.stats['summary']['columns_fixed']}")

        return df

    def fit(self, df: pd.DataFrame, dataset_name: str = None) -> 'DataCleaner':
        """
        Fit the cleaning pipeline on training data.
        This captures the state needed to apply the same transformations to test data.

        Args:
            df: Pandas DataFrame with training data
            dataset_name: Optional name of the dataset used to load config

        Returns:
            self: The fitted DataCleaner instance
        """
        section("Fitting Data Cleaning Pipeline", logger)

        # Setup from configuration if dataset name provided
        if dataset_name:
            self._setup_from_config(dataset_name)

        # Store the column information pre-cleaning
        self.original_cols = df.columns.tolist()
        self.original_shape = df.shape

        # Store dtypes for later reference
        self.known_dtypes = {col: df[col].dtype for col in df.columns}

        # Apply the cleaning pipeline
        cleaned_df = self.basic_data_cleaning(df.copy())

        # Save the column state after cleaning
        self.cleaned_cols = cleaned_df.columns.tolist()

        # Store correlation information for high-correlation columns
        if self.params.correlation_threshold < 1.0:
            self._store_correlation_info(cleaned_df)

        # Mark as fitted
        self._is_fitted = True

        # Return self for chaining
        return self

    def _store_correlation_info(self, df: pd.DataFrame) -> None:
        """Store information about highly correlated columns."""
        # Only calculate for numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # Skip if not enough numeric columns
        if len(numeric_cols) < 2:
            logger.info("Not enough numeric columns to calculate correlations")
            return

        # Calculate correlation matrix
        try:
            corr_matrix = df[numeric_cols].corr().abs()

            # Create pairs of highly correlated columns
            self.correlated_cols = []

            # Get upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # Find column pairs with correlation > threshold
            high_corr_pairs = [(upper.index[i], upper.columns[j], upper.iloc[i, j])
                               for i, j in zip(*np.where(upper > self.params.correlation_threshold))]

            if high_corr_pairs:
                logger.info(
                    f"Found {len(high_corr_pairs)} pairs of highly correlated columns (r > {self.params.correlation_threshold})")
                for col1, col2, corr in high_corr_pairs:
                    logger.info(f"  - {col1} and {col2}: r = {corr:.4f}")
                    self.correlated_cols.append((col1, col2, corr))
            else:
                logger.info(f"No highly correlated columns found (threshold: {self.params.correlation_threshold})")

        except Exception as e:
            logger.warning(f"Failed to calculate correlations: {str(e)}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted cleaning pipeline.

        Args:
            df: Pandas DataFrame to clean

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if not self._is_fitted:
            raise ValueError("DataCleaner must be fitted before transform can be called")

        section("Transforming Data with Fitted Cleaning Pipeline", logger)

        # Apply the cleaning steps
        return self.basic_data_cleaning(df.copy())

    def fit_transform(self, df: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        """
        Convenience method to fit and transform in one step.

        Args:
            df: Pandas DataFrame to clean
            dataset_name: Optional name of the dataset used to load config

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        return self.fit(df, dataset_name).transform(df)

    def save(self, path: str) -> None:
        """
        Save the fitted DataCleaner to disk using cloudpickle.

        Args:
            path: Path where to save the pipeline
        """
        if not self._is_fitted:
            raise ValueError("Cannot save an unfitted DataCleaner")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save using cloudpickle for better serialization
            with open(path, 'wb') as f:
                cloudpickle.dump(self, f)

            logger.info(f"Saved data cleaning pipeline to {path}")
        except Exception as e:
            logger.error(f"Failed to save pipeline: {str(e)}")
            raise

    @classmethod
    def load(cls, path: str) -> 'DataCleaner':
        """
        Load a fitted DataCleaner from disk.

        Args:
            path: Path from where to load the pipeline

        Returns:
            DataCleaner: Loaded pipeline
        """
        try:
            with open(path, 'rb') as f:
                pipeline = cloudpickle.load(f)

            logger.info(f"Loaded data cleaning pipeline from {path}")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load pipeline: {str(e)}")
            raise

    def get_statistics(self) -> dict:
        """Get summary statistics of the cleaning process."""
        if not self._is_fitted:
            raise ValueError("DataCleaner must be fitted before statistics can be retrieved")

        return self.stats

    def save_cleaning_report(self, path: str) -> None:
        """
        Save a detailed cleaning report to a file.

        Args:
            path: Path where to save the report
        """
        if not self._is_fitted:
            raise ValueError("DataCleaner must be fitted before generating a report")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Create report as markdown
            with open(path, 'w') as f:
                f.write("# Data Cleaning Report\n\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("## Summary\n\n")
                f.write(
                    f"* Starting shape: {self.stats['start_shape'][0]} rows × {self.stats['start_shape'][1]} columns\n")
                f.write(f"* Ending shape: {self.stats['end_shape'][0]} rows × {self.stats['end_shape'][1]} columns\n")
                f.write(f"* Columns dropped: {self.stats['summary']['columns_dropped']}\n")
                f.write(f"* Rows dropped: {self.stats['summary']['rows_dropped']}\n")
                f.write(f"* Columns fixed/modified: {self.stats['summary']['columns_fixed']}\n\n")

                f.write("## Columns Dropped\n\n")
                if self.stats["columns_dropped"]:
                    f.write("| Column | Reason |\n")
                    f.write("|--------|--------|\n")
                    for col, reason in self.stats["columns_dropped"]:
                        f.write(f"| {col} | {reason} |\n")
                else:
                    f.write("No columns were dropped.\n")

                f.write("\n## Columns Modified\n\n")
                if self.stats["fixed_columns"]:
                    f.write("| Column | Modification |\n")
                    f.write("|--------|-------------|\n")
                    for col, mod in self.stats["fixed_columns"].items():
                        f.write(f"| {col} | {mod} |\n")
                else:
                    f.write("No columns were modified.\n")

                f.write("\n## Data Type Changes\n\n")
                f.write("| Column | Original Type | New Type |\n")
                f.write("|--------|--------------|----------|\n")

                type_changes = {}
                for col in self.stats["original_dtypes"]:
                    if col in self.stats["new_dtypes"] and self.stats["original_dtypes"][col] != \
                            self.stats["new_dtypes"][col]:
                        type_changes[col] = (self.stats["original_dtypes"][col], self.stats["new_dtypes"][col])

                if type_changes:
                    for col, (orig_type, new_type) in type_changes.items():
                        f.write(f"| {col} | {orig_type} | {new_type} |\n")
                else:
                    f.write("| No data type changes were made | | |\n")

                # Add correlation information if available
                if hasattr(self, 'correlated_cols') and self.correlated_cols:
                    f.write("\n## Highly Correlated Columns\n\n")
                    f.write("| Column 1 | Column 2 | Correlation |\n")
                    f.write("|----------|----------|-------------|\n")
                    for col1, col2, corr in self.correlated_cols:
                        f.write(f"| {col1} | {col2} | {corr:.4f} |\n")

            logger.info(f"Saved cleaning report to {path}")
        except Exception as e:
            logger.error(f"Failed to save cleaning report: {str(e)}")
            raise

class IDColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, id_cols: list):
        self.id_columns = id_cols
        self.columns_to_drop = []

    def fit(self, X, y=None):
        self.columns_to_drop = [col for col in self.id_columns if col in X.columns]
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')


class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'mean', columns: list = None):
        self.method = method
        self.columns = columns
        self.fill_values = {}

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


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'IQR', columns: list = None):
        self.method = method
        self.columns = columns
        self.thresholds = {}

    def fit(self, X, y=None):
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


class SkewedDataHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'yeo-johnson', columns: list = None):
        self.method = method
        self.columns = columns
        self.transformers = {}

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
    def __init__(self, method: str = 'standard', columns: list = None):
        self.method = method
        self.columns = columns
        self.scalers = {}

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
    def __init__(self, method: str = 'onehot', columns: list = None, drop_first: bool = True):
        self.method = method
        self.columns = columns
        self.drop_first = drop_first
        self.encoders = {}
        self.dummy_columns = {}

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
    def __init__(self, method: str = 'PCA', n_components: int = 0.95, groups: list = None):
        self.method = method
        self.n_components = n_components
        self.groups = groups or []
        self.reducers = {}

    def fit(self, X, y=None):
        for group in self.groups:
            valid_cols = [col for col in group if col in X.columns]
            if len(valid_cols) < 2:
                continue

            reducer = PCA(n_components=self.n_components)
            reducer.fit(X[valid_cols])
            self.reducers[tuple(valid_cols)] = reducer
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for cols, reducer in self.reducers.items():
            cols = list(cols)
            components = reducer.transform(X_transformed[cols])
            n_components = components.shape[1]
            new_cols = [f"PCA_{'_'.join(cols)}_{i + 1}" for i in range(n_components)]
            components_df = pd.DataFrame(components, columns=new_cols, index=X_transformed.index)
            X_transformed = X_transformed.drop(columns=cols)
            X_transformed = pd.concat([X_transformed, components_df], axis=1)
        return X_transformed


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