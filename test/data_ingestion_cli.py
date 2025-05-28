# data_ingestion_cli.py
import argparse
from src.data.data_ingestion import create_data_ingestion

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run data ingestion pipeline')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--project-dir', type=str,
                       help='Custom project directory path')
    args = parser.parse_args()

    # Create and run data ingestion pipeline
    try:
        di = create_data_ingestion(project_dir=args.project_dir)
        results = di.run_ingestion_pipeline(file=args.input)
        print("\nIngestion completed successfully!")
        print(f"Processed data saved to: {results['train_path']}")
        print(f"Feature store metadata: {results['feature_store_path']}")
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()