#!/usr/bin/env python3
"""
Terminal interface for the Model Optimizer
Usage examples and command-line script
"""

import argparse
import sys
import os
from pathlib import Path

# Assuming the optimizer code is in a file called model_optimizer.py
from src.models.model_optimization import optimize_model, get_available_metrics, get_optimization_methods

def main():
    parser = argparse.ArgumentParser(description='Clustering Model Optimizer')
    
    # Basic arguments
    parser.add_argument('--optimize', action='store_true', default=True,
                       help='Enable optimization (default: True)')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false',
                       help='Skip optimization')
    
    # Optimization method
    parser.add_argument('--method', type=str, choices=['1', '2'], default='1',
                       help='Optimization method: 1=Grid Search, 2=Optuna (default: 1)')
    
    # Optuna specific
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of trials for Optuna optimization (default: 50)')
    
    # Metric selection
    parser.add_argument('--metric', type=str, choices=['1', '2', '3', '4', '5', '6', '7', '8'], 
                       default='1', help='Metric for optimization (default: 1=Silhouette)')
    
    # Configuration overrides
    parser.add_argument('--dataset-name', type=str, help='Override dataset name')
    parser.add_argument('--model-name', type=str, help='Override model name')
    parser.add_argument('--target-column', type=str, help='Override target column name')
    parser.add_argument('--intel-path', type=str, default='intel.yaml', 
                       help='Path to intel.yaml config file')
    
    # Verbose output
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Build config overrides
    config_overrides = {}
    if args.dataset_name:
        config_overrides['dataset_name'] = args.dataset_name
    if args.model_name:
        config_overrides['model_name'] = args.model_name
    if args.target_column:
        config_overrides['target_column'] = args.target_column
    
    # Display available options if verbose
    if args.verbose:
        print("Available Metrics:")
        metrics = get_available_metrics()
        for key, (name, maximize, description) in metrics.items():
            direction = "↑ (higher is better)" if maximize else "↓ (lower is better)"
            print(f"  {key}: {description} {direction}")
        
        print("\nAvailable Methods:")
        methods = get_optimization_methods() 
        for key, method in methods.items():
            print(f"  {key}: {method}")
        print()
    
    # Run optimization
    try:
        print(f"Starting optimization with method {args.method}, metric {args.metric}")
        if args.method == '2':
            print(f"Using {args.n_trials} trials for Optuna")
        
        result = optimize_model(
            optimize=args.optimize,
            method=args.method,
            n_trials=args.n_trials,
            metric=args.metric,
            config_overrides=config_overrides if config_overrides else None
        )
        
        # Display results
        if result['status'] == 'success':
            print("\n✅ Optimization completed successfully!")
            print(f"Message: {result['message']}")
            
            if result['best_params']:
                print(f"\n🎯 Best Parameters:")
                for param, value in result['best_params'].items():
                    print(f"  {param}: {value}")
            
            if result['model_path']:
                print(f"\n📁 Model saved to: {result['model_path']}")
            
            if result['metrics']:
                print(f"\n📊 Performance Metrics:")
                for metric_name, value in result['metrics'].items():
                    print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"\n❌ Optimization failed: {result['message']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

def interactive_mode():
    """Interactive terminal mode for easier usage"""
    print("🤖 Clustering Model Optimizer - Interactive Mode")
    print("=" * 50)
    
    # Display available metrics
    print("\nAvailable Metrics:")
    metrics = get_available_metrics()
    for key, (name, maximize, description) in metrics.items():
        direction = "↑" if maximize else "↓"
        print(f"  {key}: {description} {direction}")
    
    metric_choice = input("\nSelect metric (1-8, default=1): ").strip() or "1"
    
    # Display available methods
    print("\nOptimization Methods:")
    methods = get_optimization_methods()
    for key, method in methods.items():
        print(f"  {key}: {method}")
    
    method_choice = input("Select method (1-2, default=1): ").strip() or "1"
    
    n_trials = 50
    if method_choice == "2":
        n_trials_input = input("Number of trials for Optuna (default=50): ").strip()
        if n_trials_input:
            try:
                n_trials = int(n_trials_input)
            except ValueError:
                print("Invalid number, using default 50")
    
    # Optional overrides
    print("\nOptional Configuration Overrides (press Enter to skip):")
    dataset_name = input("Dataset name: ").strip() or None
    model_name = input("Model name: ").strip() or None  
    target_column = input("Target column: ").strip() or None
    
    config_overrides = {}
    if dataset_name:
        config_overrides['dataset_name'] = dataset_name
    if model_name:
        config_overrides['model_name'] = model_name
    if target_column:
        config_overrides['target_column'] = target_column
    
    print("\n🚀 Starting optimization...")
    
    try:
        result = optimize_model(
            optimize=True,
            method=method_choice,
            n_trials=n_trials,
            metric=metric_choice,
            config_overrides=config_overrides if config_overrides else None
        )
        
        if result['status'] == 'success':
            print("\n✅ Optimization completed!")
            if result['best_params']:
                print("\n🎯 Best Parameters:")
                for param, value in result['best_params'].items():
                    print(f"  {param}: {value}")
            
            if result['metrics']:
                print("\n📊 Final Metrics:")
                for metric_name, value in result['metrics'].items():
                    print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"\n❌ Failed: {result['message']}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run interactive mode
        interactive_mode()
    else:
        # Arguments provided, run command line mode
        main()
