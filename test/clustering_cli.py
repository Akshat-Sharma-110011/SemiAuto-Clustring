#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for the clustering model builder.
Usage: python clustering_cli.py --model "KMeans" --params '{"n_clusters": 5}'
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add the current directory to path to import model_building
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.model_building import ModelBuilder


def main():
    parser = argparse.ArgumentParser(description='Train clustering models from command line')
    
    # Required arguments
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Name of the clustering model to train')
    
    # Optional arguments
    parser.add_argument('--params', '-p', type=str, default=None,
                        help='Custom parameters as JSON string (e.g., \'{"n_clusters": 5}\')')
    
    parser.add_argument('--intel', '-i', type=str, default='intel.yaml',
                        help='Path to intel.yaml file (default: intel.yaml)')
    
    parser.add_argument('--list-models', '-l', action='store_true',
                        help='List all available models and their parameters')
    
    args = parser.parse_args()
    
    try:
        # Initialize ModelBuilder
        builder = ModelBuilder(intel_path=args.intel)
        
        # List models if requested
        if args.list_models:
            print("\n=== Available Clustering Models ===")
            models = builder.get_available_models()
            for model_name, model_info in models.items():
                print(f"\n{model_name}:")
                print(f"  Description: {model_info['description']}")
                print(f"  Default Parameters:")
                for param, value in model_info['params'].items():
                    print(f"    {param}: {value}")
            return
        
        # Parse custom parameters if provided
        custom_params = None
        if args.params:
            try:
                custom_params = json.loads(args.params)
                print(f"Using custom parameters: {custom_params}")
            except json.JSONDecodeError as e:
                print(f"Error parsing parameters JSON: {e}")
                return
        
        # Check if model exists
        available_models = list(builder.get_available_models().keys())
        if args.model not in available_models:
            print(f"Error: Model '{args.model}' not found.")
            print(f"Available models: {', '.join(available_models)}")
            return
        
        print(f"\n=== Training {args.model} ===")
        
        # Process the model request
        result = builder.process_model_request(args.model, custom_params)
        
        print(f"\n=== Training Completed Successfully ===")
        print(f"Model: {result['model_name']}")
        print(f"Model saved to: {result['model_path']}")
        print(f"Parameters used:")
        for param, value in result['parameters'].items():
            print(f"  {param}: {value}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
