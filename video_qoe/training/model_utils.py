"""
Model Utilities

Helper functions for loading and managing pretrained models.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from .model_trainer import XGBoostTrainer, RandomForestTrainer, ModelTrainer
from .preprocessor import FeaturePreprocessor


logger = logging.getLogger(__name__)


def load_pretrained_model(
    model_name: str = 'xgboost_v1.0',
    models_dir: str = 'models/'
) -> Tuple[ModelTrainer, FeaturePreprocessor, Dict[str, Any]]:
    """Load a pretrained model with its preprocessor and metadata.
    
    Args:
        model_name: Model name (without extension)
        models_dir: Directory containing model files
        
    Returns:
        Tuple of (model_trainer, preprocessor, metadata)
        
    Raises:
        FileNotFoundError: If model files are not found
        ValueError: If model type is unknown
        
    Example:
        >>> trainer, preprocessor, metadata = load_pretrained_model()
        >>> print(f"Model: {metadata['model_name']}")
        >>> print(f"Accuracy: {metadata['performance']['accuracy']:.2%}")
        >>> 
        >>> # Make predictions
        >>> predictions = trainer.predict(X_test)
    """
    models_path = Path(models_dir)
    model_path = models_path / f'{model_name}.pkl'
    meta_path = models_path / f'{model_name}.json'
    preprocessor_path = models_path / 'preprocessor.pkl'
    
    # Check if files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    
    logger.info(f"Loading pretrained model: {model_name}")
    
    # Load metadata first to determine model type
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    model_type = metadata.get('model_type', '').lower()
    
    # Load model based on type
    if model_type == 'xgboost' or 'xgboost' in model_name.lower():
        trainer = XGBoostTrainer.load_model(str(model_path))
    elif model_type == 'random_forest' or 'random' in model_name.lower():
        trainer = RandomForestTrainer.load_model(str(model_path))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Version: {metadata.get('version', 'unknown')}")
    
    # Load preprocessor
    preprocessor = FeaturePreprocessor.load(preprocessor_path)
    logger.info(f"  Features: {len(preprocessor.feature_names)}")
    logger.info(f"  Classes: {preprocessor.label_names}")
    
    # Display performance
    perf = metadata.get('performance', {})
    if perf:
        logger.info("  Performance:")
        logger.info(f"    Accuracy:  {perf.get('accuracy', 0):.4f}")
        logger.info(f"    Precision: {perf.get('precision', 0):.4f}")
        logger.info(f"    Recall:    {perf.get('recall', 0):.4f}")
        logger.info(f"    F1-Score:  {perf.get('f1_score', 0):.4f}")
    
    logger.info("Model loaded successfully")
    
    return trainer, preprocessor, metadata


def list_available_models(models_dir: str = 'models/') -> List[Dict[str, Any]]:
    """List all available pretrained models in the models directory.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        List of model information dictionaries
        
    Example:
        >>> models = list_available_models()
        >>> for model in models:
        ...     print(f"{model['name']}: {model['accuracy']:.2%}")
    """
    models_path = Path(models_dir)
    
    if not models_path.exists():
        logger.warning(f"Models directory not found: {models_path}")
        return []
    
    models = []
    
    # Find all .json metadata files
    for meta_file in models_path.glob('*.json'):
        # Skip preprocessor metadata
        if meta_file.stem == 'preprocessor':
            continue
        
        try:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if corresponding .pkl file exists
            model_file = models_path / f'{meta_file.stem}.pkl'
            if not model_file.exists():
                continue
            
            # Extract key information
            model_info = {
                'name': meta_file.stem,
                'model_type': metadata.get('model_type', 'unknown'),
                'version': metadata.get('version', 'unknown'),
                'created_date': metadata.get('created_date', 'unknown'),
                'training_samples': metadata.get('training_samples', 0),
                'accuracy': metadata.get('performance', {}).get('accuracy', 0),
                'f1_score': metadata.get('performance', {}).get('f1_score', 0),
                'file_path': str(model_file)
            }
            
            models.append(model_info)
            
        except Exception as e:
            logger.warning(f"Failed to read model metadata {meta_file}: {e}")
            continue
    
    # Sort by creation date (newest first)
    models.sort(key=lambda x: x['created_date'], reverse=True)
    
    return models


def get_model_info(model_name: str, models_dir: str = 'models/') -> Dict[str, Any]:
    """Get detailed information about a specific model.
    
    Args:
        model_name: Model name (without extension)
        models_dir: Directory containing model files
        
    Returns:
        Model metadata dictionary
        
    Raises:
        FileNotFoundError: If model metadata file is not found
    """
    meta_path = Path(models_dir) / f'{model_name}.json'
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {meta_path}")
    
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def save_model_metadata(
    model_name: str,
    model_type: str,
    hyperparameters: Dict[str, Any],
    performance: Dict[str, float],
    training_info: Dict[str, Any],
    models_dir: str = 'models/',
    version: str = '1.0.0'
) -> Path:
    """Save model metadata to JSON file.
    
    Args:
        model_name: Model name (without extension)
        model_type: Model type ('xgboost' or 'random_forest')
        hyperparameters: Model hyperparameters
        performance: Performance metrics dict
        training_info: Training information (samples, experiments, etc.)
        models_dir: Directory to save metadata
        version: Model version string
        
    Returns:
        Path to saved metadata file
        
    Example:
        >>> save_model_metadata(
        ...     model_name='xgboost_v1.0',
        ...     model_type='xgboost',
        ...     hyperparameters={'max_depth': 6, 'n_estimators': 100},
        ...     performance={'accuracy': 0.89, 'f1_score': 0.88},
        ...     training_info={'training_samples': 600, 'training_experiments': 18}
        ... )
    """
    from datetime import datetime
    
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    meta_path = models_path / f'{model_name}.json'
    
    metadata = {
        'model_name': model_name,
        'model_type': model_type,
        'version': version,
        'created_date': datetime.now().strftime('%Y-%m-%d'),
        'hyperparameters': hyperparameters,
        'performance': performance,
        **training_info
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved model metadata to {meta_path}")
    
    return meta_path


def print_model_summary(model_name: str, models_dir: str = 'models/'):
    """Print a formatted summary of a model.
    
    Args:
        model_name: Model name (without extension)
        models_dir: Directory containing model files
    """
    try:
        metadata = get_model_info(model_name, models_dir)
        
        print("=" * 60)
        print(f"Model: {metadata.get('model_name', 'Unknown')}")
        print("=" * 60)
        print(f"Type:          {metadata.get('model_type', 'unknown')}")
        print(f"Version:       {metadata.get('version', 'unknown')}")
        print(f"Created:       {metadata.get('created_date', 'unknown')}")
        print(f"Samples:       {metadata.get('training_samples', 0)}")
        print(f"Experiments:   {metadata.get('training_experiments', 0)}")
        
        print("\nHyperparameters:")
        for key, value in metadata.get('hyperparameters', {}).items():
            print(f"  {key:20s}: {value}")
        
        print("\nPerformance:")
        perf = metadata.get('performance', {})
        print(f"  Accuracy:  {perf.get('accuracy', 0):.4f}")
        print(f"  Precision: {perf.get('precision', 0):.4f}")
        print(f"  Recall:    {perf.get('recall', 0):.4f}")
        print(f"  F1-Score:  {perf.get('f1_score', 0):.4f}")
        
        print("\nClasses:")
        for cls in metadata.get('classes', []):
            count = metadata.get('class_distribution', {}).get(cls, 0)
            print(f"  {cls}: {count} samples")
        
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error reading model metadata: {e}")


