#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Model Training Script

This script orchestrates the complete model training pipeline:
1. Load experiment data
2. Preprocess features
3. Train model (XGBoost or Random Forest)
4. Evaluate model performance
5. Generate reports and visualizations

Usage:
    python scripts/train_model.py \
        --experiments-dir experiments/ \
        --output-dir models/ \
        --model-type xgboost

Author: Video QoE Assessment System
Date: 2025-11-15
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Set UTF-8 encoding for output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import yaml
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from video_qoe.training import (
    ExperimentDataLoader,
    FeaturePreprocessor,
    XGBoostTrainer,
    RandomForestTrainer,
    ModelEvaluator
)

console = Console(force_terminal=True, legacy_windows=False)

# Version
__version__ = '0.1.0'


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train video QoE assessment models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train XGBoost with default config
  python scripts/train_model.py --experiments-dir experiments/ --output-dir models/ --model-type xgboost
  
  # Train Random Forest with custom config
  python scripts/train_model.py --experiments-dir experiments/ --output-dir models/ --model-type random_forest --config configs/training/rf_custom.yaml
  
  # Override specific hyperparameters
  python scripts/train_model.py --experiments-dir experiments/ --output-dir models/ --model-type xgboost --max-depth 8 --n-estimators 200
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--experiments-dir',
        type=str,
        required=True,
        help='Directory containing experiment data (required)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save trained models and reports (required)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['xgboost', 'random_forest'],
        required=True,
        help='Model type to train (required)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file with hyperparameters'
    )
    parser.add_argument(
        '--class-names',
        type=str,
        nargs='+',
        default=None,
        help='Class names for visualization (e.g., 480p 720p 1080p)'
    )
    
    # Hyperparameter overrides
    parser.add_argument(
        '--max-depth',
        type=int,
        help='Maximum tree depth (overrides config)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        help='Number of estimators (overrides config)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate for XGBoost (overrides config)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    # Output control
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable generating plots'
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Disable generating markdown report'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: output-dir/training.log)'
    )
    
    # Miscellaneous
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    args = parser.parse_args()
    return args


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        console.print(f"[green]OK[/green] Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        console.print(f"[red]ERROR[/red] Config file not found: {config_path}", style="bold red")
        sys.exit(1)
    except yaml.YAMLError as e:
        console.print(f"[red]ERROR[/red] Error parsing YAML config: {e}", style="bold red")
        sys.exit(1)


def get_default_config(model_type: str) -> Dict[str, Any]:
    """Get default configuration for model type.
    
    Args:
        model_type: 'xgboost' or 'random_forest'
        
    Returns:
        Default configuration dictionary
    """
    default_config_path = project_root / 'configs' / 'training' / f'{model_type}_default.yaml'
    
    if default_config_path.exists():
        return load_config(str(default_config_path))
    
    # Fallback to hardcoded defaults
    if model_type == 'xgboost':
        return {
            'hyperparameters': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'early_stopping_rounds': 10,
                'random_state': 42
            },
            'preprocessing': {
                'test_size': 0.15,
                'val_size': 0.15,
                'random_state': 42
            },
            'output': {
                'save_model': True,
                'generate_report': True,
                'plot_confusion_matrix': True,
                'plot_feature_importance': True
            }
        }
    else:  # random_forest
        return {
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'preprocessing': {
                'test_size': 0.15,
                'val_size': 0.15,
                'random_state': 42
            },
            'output': {
                'save_model': True,
                'generate_report': True,
                'plot_confusion_matrix': True,
                'plot_feature_importance': True
            }
        }


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply command line argument overrides to config.
    
    Args:
        config: Base configuration
        args: Command line arguments
        
    Returns:
        Updated configuration
    """
    hyperparams = config.get('hyperparameters', {})
    
    if args.max_depth is not None:
        hyperparams['max_depth'] = args.max_depth
    if args.n_estimators is not None:
        hyperparams['n_estimators'] = args.n_estimators
    if args.learning_rate is not None and args.model_type == 'xgboost':
        hyperparams['learning_rate'] = args.learning_rate
    if args.random_state is not None:
        hyperparams['random_state'] = args.random_state
    
    config['hyperparameters'] = hyperparams
    
    # Apply output control overrides
    if 'output' not in config:
        config['output'] = {}
    
    if args.no_plots:
        config['output']['plot_confusion_matrix'] = False
        config['output']['plot_feature_importance'] = False
    if args.no_report:
        config['output']['generate_report'] = False
    
    return config


def setup_logging(output_dir: Path, log_file: Optional[str] = None, verbose: bool = False):
    """Setup logging configuration.
    
    Args:
        output_dir: Output directory for logs
        log_file: Optional custom log file name
        verbose: Enable verbose logging
    """
    log_path = output_dir / (log_file or 'training.log')
    
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout) if verbose else logging.NullHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger


def validate_inputs(experiments_dir: Path, output_dir: Path):
    """Validate input paths.
    
    Args:
        experiments_dir: Experiment data directory
        output_dir: Output directory
    """
    if not experiments_dir.exists():
        console.print(f"[red]ERROR: Experiments directory not found: {experiments_dir}[/red]", style="bold red")
        sys.exit(1)
    
    if not experiments_dir.is_dir():
        console.print(f"[red]ERROR: Experiments path is not a directory: {experiments_dir}[/red]", style="bold red")
        sys.exit(1)
    
    # Check if directory has any experiment subdirectories
    exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith('exp_')]
    if not exp_dirs:
        console.print(f"[red]ERROR: No experiment directories found in {experiments_dir}[/red]", style="bold red")
        console.print("[yellow]Expected directory structure:[/yellow]")
        console.print("  experiments/")
        console.print("  ├── exp_20251115_143022/")
        console.print("  │   ├── features.csv")
        console.print("  │   └── timeline.json")
        console.print("  └── exp_20251115_150045/")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]OK[/green] Validated inputs")


def print_header():
    """Print script header."""
    console.print(Panel.fit(
        "[bold cyan]Video QoE Model Training Pipeline[/bold cyan]\n"
        f"Version {__version__}",
        border_style="cyan"
    ))


def print_config_summary(config: Dict[str, Any], model_type: str):
    """Print configuration summary.
    
    Args:
        config: Configuration dictionary
        model_type: Model type
    """
    table = Table(title="Training Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model Type", model_type.upper())
    
    hyperparams = config.get('hyperparameters', {})
    for key, value in hyperparams.items():
        table.add_row(key, str(value))
    
    console.print(table)


def create_trainer(model_type: str, hyperparams: Dict[str, Any]):
    """Create model trainer instance.
    
    Args:
        model_type: 'xgboost' or 'random_forest'
        hyperparams: Hyperparameters dictionary
        
    Returns:
        ModelTrainer instance
    """
    if model_type == 'xgboost':
        return XGBoostTrainer(hyperparameters=hyperparams)
    elif model_type == 'random_forest':
        return RandomForestTrainer(hyperparameters=hyperparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_data_summary(X: pd.DataFrame, y: pd.Series, metadata: Dict):
    """Print data loading summary.
    
    Args:
        X: Feature matrix
        y: Target labels
        metadata: Metadata dictionary
    """
    table = Table(title="Data Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Samples", str(len(X)))
    table.add_row("Features", str(X.shape[1]))
    table.add_row("Classes", str(len(np.unique(y))))
    table.add_row("Experiments", str(metadata.get('num_experiments', 'N/A')))
    
    # Class distribution
    class_counts = pd.Series(y).value_counts().sort_index()
    for label, count in class_counts.items():
        table.add_row(f"  Class {label}", f"{count} ({count/len(y)*100:.1f}%)")
    
    console.print(table)


def print_results_summary(train_result, eval_result, training_time: float):
    """Print training results summary.
    
    Args:
        train_result: Training result object
        eval_result: Evaluation result object
        training_time: Total training time in seconds
    """
    table = Table(title="Training Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Training metrics
    table.add_row("Training Accuracy", f"{train_result.train_accuracy:.4f}")
    table.add_row("Training Time", f"{train_result.training_time:.2f}s")
    
    # Evaluation metrics
    table.add_row("Test Accuracy", f"{eval_result.accuracy:.4f}")
    table.add_row("Test Precision", f"{eval_result.precision:.4f}")
    table.add_row("Test Recall", f"{eval_result.recall:.4f}")
    table.add_row("Test F1-Score", f"{eval_result.f1_score:.4f}")
    
    # Total time
    table.add_row("[bold]Total Time[/bold]", f"[bold]{training_time:.2f}s[/bold]")
    
    console.print(table)


def main():
    """Main training pipeline."""
    start_time = time.time()
    
    # Print header
    print_header()
    
    # Parse arguments
    args = parse_arguments()
    
    # Convert paths
    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    validate_inputs(experiments_dir, output_dir)
    
    # Setup logging
    logger = setup_logging(output_dir, args.log_file, args.verbose)
    logger.info("=" * 80)
    logger.info("Starting model training pipeline")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Experiments dir: {experiments_dir}")
    logger.info(f"Output dir: {output_dir}")
    
    try:
        # Load configuration
        console.print("\n[bold cyan]Step 1: Loading Configuration[/bold cyan]")
        if args.config:
            config = load_config(args.config)
        else:
            config = get_default_config(args.model_type)
            console.print(f"[green]OK[/green] Using default config for {args.model_type}")
        
        # Apply CLI overrides
        config = apply_cli_overrides(config, args)
        print_config_summary(config, args.model_type)
        logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Step 2: Load data
        console.print("\n[bold cyan]Step 2: Loading Experiment Data[/bold cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Loading experiments...", total=None)
            
            loader = ExperimentDataLoader(str(experiments_dir))
            df = loader.load_experiments(min_samples_per_exp=1)
            
            progress.update(task, completed=100)
        
        # Create metadata
        metadata_cols = ['timestamp', 'exp_id', 'actual_resolution', 'predicted_resolution', 'confidence', 'scenario']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        metadata = {
            'num_experiments': df['exp_id'].nunique() if 'exp_id' in df.columns else len(loader.experiment_infos),
            'num_samples': len(df),
            'num_features': len(feature_cols),
            'feature_names': feature_cols
        }
        
        console.print(f"[green]OK[/green] Loaded {len(df)} samples from {metadata['num_experiments']} experiments")
        
        # For display purposes, extract X and y
        X_display = df[feature_cols]
        y_display = df['actual_resolution']
        print_data_summary(X_display, y_display, metadata)
        logger.info(f"Data loaded: {df.shape[0]} samples, {len(feature_cols)} features")
        
        # Step 3: Preprocess data
        console.print("\n[bold cyan]Step 3: Preprocessing Features[/bold cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Preprocessing...", total=None)
            
            # Get preprocessing config
            test_ratio = config.get('preprocessing', {}).get('test_size', 0.15)
            val_ratio = config.get('preprocessing', {}).get('val_size', 0.15)
            train_ratio = 1.0 - test_ratio - val_ratio
            random_state = config.get('preprocessing', {}).get('random_state', 42)
            
            # Initialize preprocessor
            preprocessor = FeaturePreprocessor()
            
            # Fit and transform using the entire DataFrame
            data_split = preprocessor.fit_transform(
                df,
                target_col='actual_resolution',
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_state=random_state
            )
            
            progress.update(task, completed=100)
        
        console.print(f"[green]OK[/green] Train: {len(data_split.X_train)}, Val: {len(data_split.X_val)}, Test: {len(data_split.X_test)}")
        logger.info(f"Data split - Train: {len(data_split.X_train)}, Val: {len(data_split.X_val)}, Test: {len(data_split.X_test)}")
        
        # Step 4: Train model
        console.print(f"\n[bold cyan]Step 4: Training {args.model_type.upper()} Model[/bold cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Training model...", total=None)
            
            trainer = create_trainer(args.model_type, config['hyperparameters'])
            train_result = trainer.train(
                data_split.X_train,
                data_split.y_train,
                X_val=data_split.X_val,
                y_val=data_split.y_val
            )
            
            progress.update(task, completed=100)
        
        console.print(f"[green]OK[/green] Training completed in {train_result.training_time:.2f}s")
        console.print(f"[green]OK[/green] Training accuracy: {train_result.train_accuracy:.4f}")
        logger.info(f"Training completed: accuracy={train_result.train_accuracy:.4f}, time={train_result.training_time:.2f}s")
        
        # Step 5: Save model and preprocessor
        console.print("\n[bold cyan]Step 5: Saving Model and Preprocessor[/bold cyan]")
        if config.get('output', {}).get('save_model', True):
            # Save model
            model_filename = f"{args.model_type}_model.pkl"
            model_path = output_dir / model_filename
            trainer.save_model(str(model_path))
            console.print(f"[green]OK[/green] Model saved to {model_path}")
            logger.info(f"Model saved to {model_path}")
            
            # Save preprocessor (CRITICAL for inference!)
            preprocessor_path = output_dir / 'preprocessor.pkl'
            preprocessor.save(preprocessor_path)
            console.print(f"[green]OK[/green] Preprocessor saved to {preprocessor_path}")
            logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Step 6: Evaluate model
        console.print("\n[bold cyan]Step 6: Evaluating Model[/bold cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating...", total=None)
            
            evaluator = ModelEvaluator()
            eval_result = evaluator.evaluate(
                trainer.model,
                data_split.X_test,
                data_split.y_test,
                class_names=args.class_names
            )
            
            progress.update(task, completed=100)
        
        console.print(f"[green]OK[/green] Test accuracy: {eval_result.accuracy:.4f}")
        console.print(f"[green]OK[/green] Test F1-score: {eval_result.f1_score:.4f}")
        logger.info(f"Evaluation: accuracy={eval_result.accuracy:.4f}, f1={eval_result.f1_score:.4f}")
        
        # Step 7: Generate visualizations
        if config.get('output', {}).get('plot_confusion_matrix', True):
            console.print("\n[bold cyan]Step 7: Generating Visualizations[/bold cyan]")
            
            # Confusion matrix
            cm_path = output_dir / 'confusion_matrix.png'
            evaluator.plot_confusion_matrix(eval_result, str(cm_path))
            console.print(f"[green]OK[/green] Confusion matrix saved to {cm_path}")
            logger.info(f"Confusion matrix saved to {cm_path}")
            
            # Feature importance
            if config.get('output', {}).get('plot_feature_importance', True):
                fi_path = output_dir / 'feature_importance.png'
                feature_names = data_split.feature_names
                evaluator.plot_feature_importance(
                    eval_result,
                    feature_names,
                    str(fi_path),
                    top_n=20
                )
                console.print(f"[green]OK[/green] Feature importance saved to {fi_path}")
                logger.info(f"Feature importance saved to {fi_path}")
        
        # Step 8: Generate report
        if config.get('output', {}).get('generate_report', True):
            console.print("\n[bold cyan]Step 8: Generating Report[/bold cyan]")
            
            report_path = output_dir / 'evaluation_report.md'
            evaluator.generate_report(
                eval_result,
                str(report_path),
                include_plots=True,
                cm_plot_path='confusion_matrix.png',
                fi_plot_path='feature_importance.png'
            )
            console.print(f"[green]OK[/green] Report saved to {report_path}")
            logger.info(f"Report saved to {report_path}")
        
        # Print final summary
        total_time = time.time() - start_time
        console.print(f"\n[bold green]{'=' * 60}[/bold green]")
        print_results_summary(train_result, eval_result, total_time)
        console.print(f"[bold green]{'=' * 60}[/bold green]")
        
        console.print(f"\n[bold green]Training pipeline completed successfully![/bold green]")
        console.print(f"[green]  All outputs saved to: {output_dir}[/green]")
        logger.info(f"Training pipeline completed successfully in {total_time:.2f}s")
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]WARNING: Training interrupted by user[/yellow]")
        logger.warning("Training interrupted by user")
        return 130
        
    except Exception as e:
        console.print(f"\n[red]ERROR: Training failed: {e}[/red]", style="bold red")
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

