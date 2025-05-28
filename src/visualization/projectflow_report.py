import json
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from pathlib import Path
import datetime
import matplotlib

with open('intel.yaml', 'r') as f:
    config = yaml.safe_load(f)
    dataset_name = config['dataset_name']

matplotlib.use('Agg')  # Use non-interactive backend


class CustomPDF(FPDF):
    def __init__(self, bg_color, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bg_color = bg_color

    def header(self):
        """Draw the dark background on every page"""
        # Fill the background with specified color
        self.set_fill_color(*self.bg_color)
        self.rect(0, 0, 210, 297, style='F')  # A4 size in mm


class ProjectFlowReport:
    def __init__(self, intel_yaml_path):
        """Initialize with the path to the intel.yaml file"""
        with open(intel_yaml_path, 'r') as file:
            self.intel = yaml.safe_load(file)

        self.dataset_name = self.intel.get('dataset_name', 'unknown')
        self.output_path = f"reports/pdf/projectflow_report_{self.dataset_name}.pdf"

        # Create the PDF directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Set up color scheme
        # Dark olive background color (HEX: #202216)
        self.bg_color = (32, 34, 22)
        # Pale gold text color (HEX: #F2DE9B)
        self.primary_color = (242, 222, 155)
        # Lighter gold for highlights (HEX: #FFF0C1)
        self.highlight_color = (255, 240, 193)
        # Darker gold for secondary text (HEX: #D4C172)
        self.secondary_color = (212, 193, 114)

        # Initialize PDF with custom class and branding
        self.pdf = CustomPDF(bg_color=self.bg_color)
        self.pdf.set_auto_page_break(auto=True, margin=15)

        # Add the first page
        self.add_page()

    def add_page(self):
        """Add a new page (background handled in header)"""
        self.pdf.add_page()

    def add_title_page(self):
        """Add a title page to the PDF"""
        self.pdf.set_font('Helvetica', 'B', 24)
        self.pdf.set_text_color(*self.primary_color)

        # Add title
        self.pdf.cell(0, 20, "SemiAuto clustering Report", 0, 1, 'C')

        # Add dataset name
        self.pdf.set_font('Helvetica', 'B', 18)
        self.pdf.set_text_color(*self.primary_color)
        self.pdf.cell(0, 15, f"Dataset: {self.dataset_name.upper()}", 0, 1, 'C')

        # Add date
        self.pdf.set_font('Helvetica', '', 12)
        self.pdf.set_text_color(*self.primary_color)
        self.pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')

        # Add project flow diagram
        self.pdf.ln(20)
        self.pdf.set_font('Helvetica', 'B', 16)
        self.pdf.set_text_color(*self.primary_color)
        self.pdf.cell(0, 10, "Project Flow", 0, 1, 'C')

        # Flow steps
        steps = ["Data Ingestion", "Data Preprocessing", "Feature Engineering",
                 "Model Building", "Model Evaluation", "Model Optimization",
                 "Final Evaluation"]

        # Add flow diagram
        self.pdf.ln(5)
        self.pdf.set_draw_color(*self.primary_color)
        self.pdf.set_line_width(0.5)

        y_pos = self.pdf.get_y()
        x_start = 30
        x_end = 180
        y_arrow = y_pos + 20

        # Draw arrow line
        self.pdf.line(x_start, y_arrow, x_end, y_arrow)

        # Draw arrowhead
        self.pdf.line(x_end, y_arrow, x_end - 5, y_arrow - 3)
        self.pdf.line(x_end, y_arrow, x_end - 5, y_arrow + 3)

        # Add each step along the arrow
        step_width = (x_end - x_start) / (len(steps) - 1)
        self.pdf.set_font('Helvetica', '', 9)

        for i, step in enumerate(steps):
            x_pos = x_start + (i * step_width)
            # Draw circle at step position
            self.pdf.set_fill_color(*self.secondary_color)
            self.pdf.ellipse(x_pos - 3, y_arrow - 3, 6, 6, 'F')

            # Add step text
            self.pdf.set_xy(x_pos - 15, y_arrow + 5)
            self.pdf.cell(30, 10, step, 0, 0, 'C')

        self.pdf.ln(40)

        # Add table of contents title
        self.pdf.set_font('Helvetica', 'B', 16)
        self.pdf.set_text_color(*self.primary_color)
        self.pdf.cell(0, 10, "Table of Contents", 0, 1, 'L')

        # Add table of contents items
        self.pdf.set_font('Helvetica', '', 12)
        self.pdf.set_text_color(*self.primary_color)

        toc_items = [
            "1. Data Ingestion",
            "2. Data Preprocessing",
            "3. Feature Engineering",
            "4. Model Building",
            "5. Model Evaluation",
            "6. Model Optimization (if performed)",
            "7. Final Evaluation Results"
        ]

        for item in toc_items:
            self.pdf.cell(0, 8, item, 0, 1, 'L')

        self.add_page()

    def add_section_header(self, title, description=""):
        """Add a section header to the report"""
        self.pdf.set_font('Helvetica', 'B', 16)
        self.pdf.set_text_color(*self.primary_color)
        self.pdf.cell(0, 15, title, 0, 1, 'L')

        if description:
            self.pdf.set_font('Helvetica', '', 11)
            self.pdf.set_text_color(*self.primary_color)
            self.pdf.multi_cell(0, 6, description)
            self.pdf.ln(5)

    def add_subsection_header(self, title):
        """Add a subsection header to the report"""
        self.pdf.set_font('Helvetica', 'B', 14)
        self.pdf.set_text_color(*self.secondary_color)
        self.pdf.cell(0, 10, title, 0, 1, 'L')

    def add_data_ingestion_section(self):
        """Add the data ingestion section to the report"""
        self.add_section_header("1. Data Ingestion",
                                "This step involves loading and analyzing the original dataset to understand its structure and characteristics.")

        # Dataset information
        self.add_subsection_header("Dataset Overview")

        # Try to read feature store YAML
        feature_store_path = self.intel.get('feature_store_path')
        if feature_store_path and os.path.exists(feature_store_path):
            with open(feature_store_path, 'r') as file:
                feature_store = yaml.safe_load(file)

            # Dataset stats
            self.pdf.set_font('Helvetica', 'B', 12)
            self.pdf.cell(0, 8, f"Dataset: {self.dataset_name}", 0, 1, 'L')
            train_rows = feature_store.get('train_rows', 'N/A')
            test_rows = feature_store.get('test_rows', 'N/A')
            self.pdf.cell(0, 8, f"Train samples: {train_rows}, Test samples: {test_rows}", 0, 1, 'L')
            self.pdf.cell(0, 8, f"Target column: {feature_store.get('target_col', 'N/A')}", 0, 1, 'L')
            self.pdf.ln(5)

            # Column types
            self.add_subsection_header("Column Types")
            self.pdf.set_font('Helvetica', '', 10)

            # Format column lists nicely
            def format_column_list(columns):
                if not columns:
                    return "None"
                return ", ".join(columns)

            # Add column information
            column_info = [
                {"title": "Original Columns", "data": feature_store.get('original_cols', [])},
                {"title": "Numerical Columns", "data": feature_store.get('numerical_cols', [])},
                {"title": "Categorical Columns", "data": feature_store.get('categorical_cols', [])},
                {"title": "Skewed Columns", "data": feature_store.get('skewed_cols', [])},
                {"title": "Normal Columns", "data": feature_store.get('normal_cols', [])},
                {"title": "Columns with Nulls", "data": feature_store.get('contains_null', [])},
                {"title": "Columns with Outliers", "data": feature_store.get('contains_outliers', [])}
            ]

            for info in column_info:
                self.pdf.set_font('Helvetica', 'B', 11)
                self.pdf.cell(0, 8, info["title"] + ":", 0, 1, 'L')
                self.pdf.set_font('Helvetica', '', 10)
                self.pdf.multi_cell(0, 6, format_column_list(info["data"]))
                self.pdf.ln(3)

            # Correlation information
            if 'correlated_cols' in feature_store and feature_store['correlated_cols']:
                self.add_subsection_header("Highly Correlated Features")
                self.pdf.set_font('Helvetica', '', 10)

                for col, correlations in feature_store['correlated_cols'].items():
                    if correlations:
                        self.pdf.set_font('Helvetica', 'B', 11)
                        self.pdf.cell(0, 8, f"{col}:", 0, 1, 'L')
                        self.pdf.set_font('Helvetica', '', 10)

                        for corr in correlations:
                            corr_col = corr.get('column', 'N/A')
                            corr_val = corr.get('correlation', 'N/A')
                            corr_str = f"- {corr_col}: {corr_val:.4f}"
                            self.pdf.cell(0, 6, corr_str, 0, 1, 'L')

                        self.pdf.ln(3)

            # Add distribution plots
            plots_dir = self.intel.get('plots_dir')
            if plots_dir and os.path.exists(plots_dir):
                self.add_page()
                self.add_subsection_header("Feature Distributions")

                # Get all distribution plots
                dist_plots = [f for f in os.listdir(plots_dir) if f.startswith('distribution_')]

                # Add plots in pairs
                for i in range(0, len(dist_plots), 2):
                    y_pos = self.pdf.get_y()

                    # First plot
                    plot_path = os.path.join(plots_dir, dist_plots[i])
                    feature_name = dist_plots[i].replace('distribution_', '').replace('.png', '')
                    self.pdf.set_font('Helvetica', 'B', 10)
                    self.pdf.cell(0, 6, feature_name, 0, 1, 'C')
                    self.pdf.image(plot_path, x=25, y=None, w=75)

                    # Second plot (if available)
                    if i + 1 < len(dist_plots):
                        self.pdf.set_xy(110, y_pos)
                        plot_path = os.path.join(plots_dir, dist_plots[i + 1])
                        feature_name = dist_plots[i + 1].replace('distribution_', '').replace('.png', '')
                        self.pdf.cell(0, 6, feature_name, 0, 1, 'C')
                        self.pdf.image(plot_path, x=110, y=None, w=75)

                    # Position for next pair of plots
                    self.pdf.ln(5)

                    # Add new page if needed
                    if self.pdf.get_y() > 240:
                        self.add_page()

                # Add correlation heatmap on a new page
                corr_plot = os.path.join(plots_dir, 'correlation_heatmap.png')
                if os.path.exists(corr_plot):
                    self.add_page()
                    self.add_subsection_header("Correlation Heatmap")
                    self.pdf.image(corr_plot, x=25, y=None, w=160)

        self.add_page()

    def add_data_preprocessing_section(self):
        """Add the data preprocessing section to the report"""
        self.add_section_header("2. Data Preprocessing",
                                "This step involves cleaning the dataset and preparing it for model training.")

        # Preprocessing configuration
        if 'preprocessing_config' in self.intel:
            preproc_config = self.intel['preprocessing_config']

            self.add_subsection_header("Preprocessing Configuration")

            # Handle duplicates
            self.pdf.set_font('Helvetica', 'B', 11)
            self.pdf.cell(0, 8, "Duplicate handling:", 0, 1, 'L')
            self.pdf.set_font('Helvetica', '', 10)
            self.pdf.cell(0, 6, f"Remove duplicates: {preproc_config.get('handle_duplicates', False)}", 0, 1, 'L')
            self.pdf.ln(3)

            # Outlier treatment
            if 'outliers' in preproc_config:
                self.pdf.set_font('Helvetica', 'B', 11)
                self.pdf.cell(0, 8, "Outlier Treatment:", 0, 1, 'L')
                self.pdf.set_font('Helvetica', '', 10)
                self.pdf.cell(0, 6, f"Method: {preproc_config['outliers'].get('method', 'N/A')}", 0, 1, 'L')

                columns = preproc_config['outliers'].get('columns', [])
                self.pdf.cell(0, 6, "Applied to columns:", 0, 1, 'L')
                for col in columns:
                    self.pdf.cell(0, 6, f"- {col}", 0, 1, 'L')
                self.pdf.ln(3)

            # Skewed data transformation
            if 'skewed_data' in preproc_config:
                self.pdf.set_font('Helvetica', 'B', 11)
                self.pdf.cell(0, 8, "Skewed Data Transformation:", 0, 1, 'L')
                self.pdf.set_font('Helvetica', '', 10)
                self.pdf.cell(0, 6, f"Method: {preproc_config['skewed_data'].get('method', 'N/A')}", 0, 1, 'L')

                columns = preproc_config['skewed_data'].get('columns', [])
                self.pdf.cell(0, 6, "Applied to columns:", 0, 1, 'L')
                for col in columns:
                    self.pdf.cell(0, 6, f"- {col}", 0, 1, 'L')
                self.pdf.ln(3)

            # Numerical scaling
            if 'numerical_scaling' in preproc_config:
                self.pdf.set_font('Helvetica', 'B', 11)
                self.pdf.cell(0, 8, "Numerical Scaling:", 0, 1, 'L')
                self.pdf.set_font('Helvetica', '', 10)
                self.pdf.cell(0, 6, f"Method: {preproc_config['numerical_scaling'].get('method', 'N/A')}", 0, 1, 'L')

                columns = preproc_config['numerical_scaling'].get('columns', [])
                self.pdf.cell(0, 6, "Applied to columns:", 0, 1, 'L')
                for col in columns:
                    self.pdf.cell(0, 6, f"- {col}", 0, 1, 'L')
                self.pdf.ln(3)

            # Categorical encoding
            if 'categorical_encoding' in preproc_config:
                self.pdf.set_font('Helvetica', 'B', 11)
                self.pdf.cell(0, 8, "Categorical Encoding:", 0, 1, 'L')
                self.pdf.set_font('Helvetica', '', 10)
                self.pdf.cell(0, 6, f"Method: {preproc_config['categorical_encoding'].get('method', 'N/A')}", 0, 1, 'L')
                self.pdf.cell(0, 6, f"Drop first: {preproc_config['categorical_encoding'].get('drop_first', False)}", 0,
                              1, 'L')

                columns = preproc_config['categorical_encoding'].get('columns', [])
                self.pdf.cell(0, 6, "Applied to columns:", 0, 1, 'L')
                for col in columns:
                    self.pdf.cell(0, 6, f"- {col}", 0, 1, 'L')
                self.pdf.ln(3)

        # Preprocessed data preview
        self.add_subsection_header("Preprocessed Data Preview")

        # Train data preview
        if 'train_preprocessed_path' in self.intel:
            train_path = self.intel['train_preprocessed_path']
            if os.path.exists(train_path):
                try:
                    df_train = pd.read_csv(train_path)

                    self.pdf.set_font('Helvetica', 'B', 11)
                    self.pdf.cell(0, 8, "Training Data Sample (First 5 rows):", 0, 1, 'L')

                    # Create a preview table
                    self.add_dataframe_preview(df_train.head())
                except Exception as e:
                    self.pdf.set_font('Helvetica', '', 10)
                    self.pdf.cell(0, 6, f"Error loading preprocessed train data: {str(e)}", 0, 1, 'L')

        # Test data preview
        if 'test_preprocessed_path' in self.intel:
            test_path = self.intel['test_preprocessed_path']
            if os.path.exists(test_path):
                try:
                    df_test = pd.read_csv(test_path)

                    self.pdf.set_font('Helvetica', 'B', 11)
                    self.pdf.cell(0, 8, "Test Data Sample (First 5 rows):", 0, 1, 'L')

                    # Create a preview table
                    self.add_dataframe_preview(df_test.head())
                except Exception as e:
                    self.pdf.set_font('Helvetica', '', 10)
                    self.pdf.cell(0, 6, f"Error loading preprocessed test data: {str(e)}", 0, 1, 'L')

        self.add_page()

    def add_dataframe_preview(self, df):
        """Add a preview of a dataframe to the report"""
        if df.empty:
            self.pdf.set_font('Helvetica', '', 10)
            self.pdf.cell(0, 6, "No data available", 0, 1, 'L')
            return

        # Calculate column widths
        n_cols = len(df.columns)

        # If there are too many columns, show a subset
        if n_cols > 10:
            df = df.iloc[:, :10]
            n_cols = 10

        col_widths = [min(190 / n_cols, 30) for _ in range(n_cols)]

        # Table header with column names
        self.pdf.set_font('Helvetica', 'B', 8)
        self.pdf.set_fill_color(*self.highlight_color)
        self.pdf.set_text_color(*self.bg_color)

        x_start = self.pdf.get_x()
        y_start = self.pdf.get_y()

        # Print column headers
        for i, col_name in enumerate(df.columns):
            # Truncate column name if too long
            if len(str(col_name)) > 12:
                col_name = str(col_name)[:10] + ".."

            self.pdf.cell(col_widths[i], 6, str(col_name), 1, 0, 'C', True)

        self.pdf.ln()
        self.pdf.set_text_color(*self.primary_color)  # Reset text color for data

        # Table rows with data
        self.pdf.set_font('Helvetica', '', 8)
        for _, row in df.iterrows():
            for i, val in enumerate(row):
                # Truncate value if too long
                cell_value = str(val)
                if len(cell_value) > 12:
                    cell_value = cell_value[:10] + ".."

                self.pdf.cell(col_widths[i], 6, cell_value, 1, 0, 'C')
            self.pdf.ln()

        self.pdf.ln(5)

    def add_feature_engineering_section(self):
        """Add the feature engineering section to the report"""
        self.add_section_header("3. Feature Engineering",
                                "This step involves creating new features or selecting the most important ones.")

        # Feature engineering configuration
        if 'feature_engineering_config' in self.intel:
            feature_config = self.intel['feature_engineering_config']

            self.add_subsection_header("Feature Engineering Configuration")

            self.pdf.set_font('Helvetica', 'B', 11)
            self.pdf.cell(0, 8, "Applied Techniques:", 0, 1, 'L')
            self.pdf.set_font('Helvetica', '', 10)

            # Feature tools
            use_feature_tools = feature_config.get('use_feature_tools', False)
            self.pdf.cell(0, 6, f"Automated Feature Engineering: {'Yes' if use_feature_tools else 'No'}", 0, 1, 'L')

            # SHAP based feature selection
            use_shap_selection = feature_config.get('use_shap_selection', False)
            self.pdf.cell(0, 6, f"SHAP-based Feature Selection: {'Yes' if use_shap_selection else 'No'}", 0, 1, 'L')

            self.pdf.ln(5)

        # Transformed data preview
        self.add_subsection_header("Transformed Data Preview")

        # Train data preview
        if 'train_transformed_path' in self.intel:
            train_path = self.intel['train_transformed_path']
            if os.path.exists(train_path):
                try:
                    df_train = pd.read_csv(train_path)

                    self.pdf.set_font('Helvetica', 'B', 11)
                    self.pdf.cell(0, 8, "Transformed Training Data Sample (First 5 rows):", 0, 1, 'L')

                    # Create a preview table
                    self.add_dataframe_preview(df_train.head())
                except Exception as e:
                    self.pdf.set_font('Helvetica', '', 10)
                    self.pdf.cell(0, 6, f"Error loading transformed train data: {str(e)}", 0, 1, 'L')

        # Test data preview
        if 'test_transformed_path' in self.intel:
            test_path = self.intel['test_transformed_path']
            if os.path.exists(test_path):
                try:
                    df_test = pd.read_csv(test_path)

                    self.pdf.set_font('Helvetica', 'B', 11)
                    self.pdf.cell(0, 8, "Transformed Test Data Sample (First 5 rows):", 0, 1, 'L')

                    # Create a preview table
                    self.add_dataframe_preview(df_test.head())
                except Exception as e:
                    self.pdf.set_font('Helvetica', '', 10)
                    self.pdf.cell(0, 6, f"Error loading transformed test data: {str(e)}", 0, 1, 'L')

        self.add_page()

    def add_model_building_section(self):
        """Add the model building section to the report"""
        self.add_section_header("4. Model Building",
                                "This step involves training the clustering model on the transformed data.")

        # Model information
        if 'model_name' in self.intel:
            model_name = self.intel['model_name']

            self.add_subsection_header("Model Selection")

            self.pdf.set_font('Helvetica', 'B', 11)
            self.pdf.cell(0, 8, "Selected Model:", 0, 1, 'L')
            self.pdf.set_font('Helvetica', '', 10)
            self.pdf.cell(0, 6, model_name, 0, 1, 'L')

            if 'model_timestamp' in self.intel:
                timestamp = self.intel['model_timestamp']
                self.pdf.cell(0, 6, f"Training timestamp: {timestamp}", 0, 1, 'L')

            self.pdf.ln(5)

        self.add_page()

    def add_model_evaluation_section(self):
        """Add the model evaluation section to the report (clustering version)"""
        self.add_section_header("5. Model Evaluation",
                                "This step involves evaluating the performance of the trained clustering model.")

        # Performance metrics
        if 'performance_metrics_path' in self.intel:
            metrics_path = self.intel['performance_metrics_path']
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as file:
                    performance = yaml.safe_load(file)

                self.add_subsection_header("Clustering Metrics")

                metrics_to_display = [
                    ("Silhouette Score", performance.get('silhouette_score', 'N/A')),
                    ("Calinski-Harabasz Score", performance.get('calinski_harabasz_score', 'N/A')),
                    ("Davies-Bouldin Score", performance.get('davies_bouldin_score', 'N/A')),
                    ("Number of Clusters", performance.get('n_clusters', 'N/A')),
                    ("Outlier Ratio", performance.get('outlier_ratio', 'N/A'))
                ]

                # Create performance metrics table
                self.pdf.set_font('Helvetica', 'B', 11)
                self.pdf.cell(0, 8, "Original Model Performance:", 0, 1, 'L')

                self.pdf.set_font('Helvetica', '', 10)
                if 'evaluation_timestamp' in performance:
                    self.pdf.cell(0, 6, f"Evaluation timestamp: {performance['evaluation_timestamp']}", 0, 1, 'L')
                self.pdf.ln(3)

                # Table header
                self.pdf.set_font('Helvetica', 'B', 10)
                self.pdf.set_fill_color(*self.highlight_color)
                self.pdf.set_text_color(*self.bg_color)
                self.pdf.cell(90, 8, "Metric", 1, 0, 'C', True)
                self.pdf.cell(90, 8, "Value", 1, 1, 'C', True)
                self.pdf.set_text_color(*self.primary_color)  # Reset text color

                # Table rows
                self.pdf.set_font('Helvetica', '', 10)
                for metric, value in metrics_to_display:
                    # Format the value to 5 decimal places if it's a number
                    if isinstance(value, (int, float)):
                        value = f"{value:.5f}"

                    self.pdf.cell(90, 8, metric, 1, 0, 'L')
                    self.pdf.cell(90, 8, str(value), 1, 1, 'C')

                self.pdf.ln(5)

        self.add_page()

    def add_model_optimization_section(self):
        """Add the model optimization section to the report"""
        # Check if optimization was performed
        if 'optimized_model_path' not in self.intel:
            self.add_section_header("6. Model Optimization",
                                    "This step was skipped in the current project.")
            self.add_page()
            return

        self.add_section_header("6. Model Optimization",
                                "This step involves tuning the hyperparameters of the model to improve performance.")

        # Best parameters
        if 'best_params_path' in self.intel:
            params_path = self.intel['best_params_path']
            if os.path.exists(params_path):
                try:
                    with open(params_path, 'r') as file:
                        # Check if file is not empty
                        if os.path.getsize(params_path) > 0:
                            params = json.load(file)

                            self.add_subsection_header("Optimized Hyperparameters")

                            # Table header
                            self.pdf.set_font('Helvetica', 'B', 10)
                            self.pdf.set_fill_color(*self.highlight_color)
                            self.pdf.set_text_color(*self.bg_color)
                            self.pdf.cell(90, 8, "Parameter", 1, 0, 'C', True)
                            self.pdf.cell(90, 8, "Value", 1, 1, 'C', True)
                            self.pdf.set_text_color(*self.primary_color)  # Reset text color

                            # Table rows
                            self.pdf.set_font('Helvetica', '', 10)
                            for param, value in params.items():
                                self.pdf.cell(90, 8, param, 1, 0, 'L')
                                self.pdf.cell(90, 8, str(value), 1, 1, 'C')

                            self.pdf.ln(5)
                        else:
                            self.pdf.set_font('Helvetica', '', 10)
                            self.pdf.cell(0, 6, "Hyperparameters file is empty.", 0, 1, 'L')
                except json.JSONDecodeError:
                    self.pdf.set_font('Helvetica', '', 10)
                    self.pdf.cell(0, 6, "Error: Could not decode hyperparameters file.", 0, 1, 'L')
                except Exception as e:
                    self.pdf.set_font('Helvetica', '', 10)
                    self.pdf.cell(0, 6, f"Error loading hyperparameters: {str(e)}", 0, 1, 'L')

        # Optimization timestamp
        if 'optimization_timestamp' in self.intel:
            self.pdf.set_font('Helvetica', '', 10)
            self.pdf.cell(0, 6, f"Optimization timestamp: {self.intel['optimization_timestamp']}", 0, 1, 'L')
            self.pdf.ln(5)

        self.add_page()

    def add_final_evaluation_section(self):
        """Add the final evaluation section to the report (clustering version)"""
        self.add_section_header("7. Final Evaluation Results",
                                "This section presents the final performance of the optimized clustering model.")

        # Check if optimization was performed
        if 'optimized_performance_metrics_path' not in self.intel:
            self.pdf.set_font('Helvetica', '', 10)
            self.pdf.cell(0, 8, "Model optimization was not performed. Please refer to the original model evaluation.",
                          0, 1, 'L')
            return

        # Optimized performance metrics
        metrics_path = self.intel['optimized_performance_metrics_path']
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as file:
                performance = yaml.safe_load(file)

            self.add_subsection_header("Optimized Model Performance")

            metrics_to_display = [
                ("Silhouette Score", performance.get('silhouette_score', 'N/A')),
                ("Calinski-Harabasz Score", performance.get('calinski_harabasz_score', 'N/A')),
                ("Davies-Bouldin Score", performance.get('davies_bouldin_score', 'N/A')),
                ("Number of Clusters", performance.get('n_clusters', 'N/A')),
                ("Outlier Ratio", performance.get('outlier_ratio', 'N/A'))
            ]

            # Table header
            self.pdf.set_font('Helvetica', 'B', 10)
            self.pdf.set_fill_color(*self.highlight_color)
            self.pdf.set_text_color(*self.bg_color)
            self.pdf.cell(90, 8, "Metric", 1, 0, 'C', True)
            self.pdf.cell(90, 8, "Value", 1, 1, 'C', True)
            self.pdf.set_text_color(*self.primary_color)  # Reset text color

            # Table rows
            self.pdf.set_font('Helvetica', '', 10)
            for metric, value in metrics_to_display:
                # Format the value to 5 decimal places if it's a number
                if isinstance(value, (int, float)):
                    value = f"{value:.5f}"

                self.pdf.cell(90, 8, metric, 1, 0, 'L')
                self.pdf.cell(90, 8, str(value), 1, 1, 'C')

            # Evaluation timestamp
            if 'optimized_evaluation_timestamp' in self.intel:
                self.pdf.ln(5)
                self.pdf.set_font('Helvetica', '', 10)
                self.pdf.cell(0, 6, f"Evaluation timestamp: {self.intel['optimized_evaluation_timestamp']}", 0, 1, 'L')

            # Compare with original model
            if 'performance_metrics_path' in self.intel:
                orig_metrics_path = self.intel['performance_metrics_path']
                if os.path.exists(orig_metrics_path):
                    with open(orig_metrics_path, 'r') as file:
                        orig_performance = yaml.safe_load(file)

                    self.pdf.ln(10)
                    self.add_subsection_header("Performance Comparison")

                    # Comparative metrics table
                    key_metrics = [
                        ("Silhouette Score", 'silhouette_score', True),  # Higher is better
                        ("Davies-Bouldin Score", 'davies_bouldin_score', False),  # Lower is better
                        ("Calinski-Harabasz Score", 'calinski_harabasz_score', True)  # Higher is better
                    ]

                    # Table header
                    self.pdf.set_font('Helvetica', 'B', 10)
                    self.pdf.set_fill_color(*self.highlight_color)
                    self.pdf.set_text_color(*self.bg_color)
                    self.pdf.cell(50, 8, "Metric", 1, 0, 'C', True)
                    self.pdf.cell(50, 8, "Original Model", 1, 0, 'C', True)
                    self.pdf.cell(50, 8, "Optimized Model", 1, 0, 'C', True)
                    self.pdf.cell(30, 8, "Improvement", 1, 1, 'C', True)
                    self.pdf.set_text_color(*self.primary_color)  # Reset text color

                    # Table rows
                    self.pdf.set_font('Helvetica', '', 10)

                    for display_name, metric_key, higher_better in key_metrics:
                        orig_val = orig_performance.get(metric_key, 0)
                        opt_val = performance.get(metric_key, 0)

                        # Calculate improvement
                        if isinstance(orig_val, (int, float)) and isinstance(opt_val, (int, float)):
                            if higher_better:
                                # For metrics where higher is better
                                improvement = opt_val - orig_val
                                improvement_pct = (improvement / max(abs(orig_val), 1e-10)) * 100

                                if improvement > 0:
                                    improvement_str = f"+{improvement_pct:.2f}%"
                                    text_color = (46, 204, 113)  # Green for improvement
                                else:
                                    improvement_str = f"{improvement_pct:.2f}%"
                                    text_color = (231, 76, 60)  # Red for deterioration
                            else:
                                # For metrics where lower is better
                                improvement = orig_val - opt_val
                                improvement_pct = (improvement / max(abs(orig_val), 1e-10)) * 100

                                if improvement > 0:
                                    improvement_str = f"+{improvement_pct:.2f}%"
                                    text_color = (46, 204, 113)  # Green for improvement
                                else:
                                    improvement_str = f"{improvement_pct:.2f}%"
                                    text_color = (231, 76, 60)  # Red for deterioration
                        else:
                            improvement_str = "N/A"
                            text_color = self.primary_color

                        # Format values to 5 decimal places
                        if isinstance(orig_val, (int, float)):
                            orig_val_str = f"{orig_val:.5f}"
                        else:
                            orig_val_str = str(orig_val)

                        if isinstance(opt_val, (int, float)):
                            opt_val_str = f"{opt_val:.5f}"
                        else:
                            opt_val_str = str(opt_val)

                        # Add row to table
                        self.pdf.cell(50, 8, display_name, 1, 0, 'L')
                        self.pdf.cell(50, 8, orig_val_str, 1, 0, 'C')
                        self.pdf.cell(50, 8, opt_val_str, 1, 0, 'C')

                        # Set text color for improvement cell
                        self.pdf.set_text_color(*text_color)
                        self.pdf.cell(30, 8, improvement_str, 1, 1, 'C')
                        self.pdf.set_text_color(*self.primary_color)  # Reset text color

    def add_conclusion(self):
        """Add a conclusion section to the report (clustering version)"""
        self.add_page()
        self.add_section_header("Conclusion",
                                "Summary of the clustering model development and performance.")

        # Generate conclusion text
        self.pdf.set_font('Helvetica', '', 11)

        conclusion_text = f"This report summarizes the development of a clustering model for the {self.dataset_name} dataset. "

        # Check if optimization was performed
        if 'optimized_model_path' in self.intel:
            model_name = self.intel.get('model_name', 'unknown')
            conclusion_text += f"A {model_name} clustering model was trained and optimized using hyperparameter tuning. "

            # If we have performance metrics for both
            if ('performance_metrics_path' in self.intel and
                    'optimized_performance_metrics_path' in self.intel):

                try:
                    # Get Silhouette scores
                    with open(self.intel['performance_metrics_path'], 'r') as f:
                        orig_perf = yaml.safe_load(f)
                    with open(self.intel['optimized_performance_metrics_path'], 'r') as f:
                        opt_perf = yaml.safe_load(f)

                    orig_sil = orig_perf.get('silhouette_score', 0)
                    opt_sil = opt_perf.get('silhouette_score', 0)

                    if isinstance(orig_sil, (int, float)) and isinstance(opt_sil, (int, float)):
                        improvement = opt_sil - orig_sil

                        if improvement > 0:
                            conclusion_text += f"The optimization process improved the model's Silhouette Score "
                            conclusion_text += f"from {orig_sil:.5f} to {opt_sil:.5f}, "
                            conclusion_text += f"representing a {(improvement / max(abs(orig_sil), 1e-10)) * 100:.2f}% improvement. "
                except Exception:
                    pass
        else:
            model_name = self.intel.get('model_name', 'unknown')
            conclusion_text += f"A {model_name} clustering model was trained without hyperparameter optimization. "

        conclusion_text += "\n\nThis automatic report was generated to provide insights into the model development process "
        conclusion_text += "and performance metrics. It includes details about data preprocessing, feature engineering, "
        conclusion_text += "model selection, and evaluation results."

        self.pdf.multi_cell(0, 6, conclusion_text)

    def generate_report(self):
        """Generate the complete project flow report"""
        # Add title page
        self.add_title_page()

        # Add each section
        self.add_data_ingestion_section()
        self.add_data_preprocessing_section()
        self.add_feature_engineering_section()
        self.add_model_building_section()
        self.add_model_evaluation_section()
        self.add_model_optimization_section()
        self.add_final_evaluation_section()
        self.add_conclusion()

        # Save the PDF
        self.pdf.output(self.output_path)
        print(f"Report successfully generated at: {self.output_path}")


if __name__ == "__main__":
    # Use the main project directory intel.yaml file
    intel_yaml_path = "intel.yaml"

    # Create and generate the report
    report_generator = ProjectFlowReport(intel_yaml_path)
    report_generator.generate_report()