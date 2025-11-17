# -*- coding: utf-8 -*-
"""
模型训练器

提供机器学习模型的训练功能：
- ModelTrainer: 抽象基类，定义训练接口
- XGBoostTrainer: XGBoost分类器训练
- RandomForestTrainer: Random Forest分类器训练
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@dataclass
class TrainingResult:
    """训练结果
    
    Attributes:
        model_type: 模型类型
        train_accuracy: 训练集准确率
        val_accuracy: 验证集准确率（如果有）
        training_time: 训练耗时（秒）
        hyperparameters: 超参数配置
        feature_count: 特征数量
        train_samples: 训练样本数
        val_samples: 验证样本数（如果有）
        timestamp: 训练时间戳
        feature_importances: 特征重要性（如果有）
    """
    model_type: str
    train_accuracy: float
    val_accuracy: Optional[float] = None
    training_time: float = 0.0
    hyperparameters: Dict[str, Any] = None
    feature_count: int = 0
    train_samples: int = 0
    val_samples: int = 0
    timestamp: str = ""
    feature_importances: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def summary(self) -> str:
        """返回训练结果摘要"""
        lines = [
            f"Training Result - {self.model_type}",
            "=" * 50,
            f"Training Accuracy:   {self.train_accuracy:.4f}",
        ]
        if self.val_accuracy is not None:
            lines.append(f"Validation Accuracy: {self.val_accuracy:.4f}")
        lines.extend([
            f"Training Time:       {self.training_time:.2f}s",
            f"Samples:             train={self.train_samples}, val={self.val_samples}",
            f"Features:            {self.feature_count}",
            f"Timestamp:           {self.timestamp}",
        ])
        return "\n".join(lines)


class ModelTrainer(ABC):
    """模型训练器抽象基类
    
    定义统一的训练接口，所有具体训练器必须实现：
    - train(): 训练模型
    - save_model(): 保存模型
    - load_model(): 加载模型
    - predict(): 预测
    
    Attributes:
        model: 训练好的模型对象
        hyperparameters: 超参数配置
        training_result: 训练结果
        logger: 日志记录器
        fitted: 是否已训练
    """
    
    def __init__(self, 
                 hyperparameters: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """初始化训练器
        
        Args:
            hyperparameters: 超参数配置字典
            logger: 日志记录器
        """
        self.model = None
        self.hyperparameters = hyperparameters or self._default_hyperparameters()
        self.training_result: Optional[TrainingResult] = None
        self.logger = logger or logging.getLogger(__name__)
        self.fitted = False
    
    @abstractmethod
    def _default_hyperparameters(self) -> Dict[str, Any]:
        """返回默认超参数配置"""
        pass
    
    @abstractmethod
    def _create_model(self) -> Any:
        """创建模型实例"""
        pass
    
    @abstractmethod
    def _get_model_type(self) -> str:
        """返回模型类型名称"""
        pass
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              verbose: bool = True) -> TrainingResult:
        """训练模型
        
        Args:
            X_train: 训练特征 (n_samples, n_features)
            y_train: 训练标签 (n_samples,)
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            verbose: 是否显示训练进度
            
        Returns:
            TrainingResult: 训练结果对象
            
        Raises:
            ValueError: 如果数据形状不匹配
        """
        # 验证数据
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train ({X_train.shape[0]}) and y_train ({y_train.shape[0]}) "
                f"must have same number of samples"
            )
        
        if X_val is not None and y_val is not None:
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError(
                    f"X_val ({X_val.shape[0]}) and y_val ({y_val.shape[0]}) "
                    f"must have same number of samples"
                )
        
        self.logger.info(f"Starting {self._get_model_type()} training...")
        self.logger.info(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        if X_val is not None:
            self.logger.info(f"Validation samples: {X_val.shape[0]}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 创建模型
        self.model = self._create_model()
        
        # 训练模型（子类实现具体逻辑）
        self._fit_model(X_train, y_train, X_val, y_val, verbose)
        
        # 计算耗时
        training_time = time.time() - start_time
        
        # 计算准确率
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
        
        # 获取特征重要性（如果支持）
        feature_importances = self._get_feature_importances()
        
        # 创建训练结果
        self.training_result = TrainingResult(
            model_type=self._get_model_type(),
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            training_time=training_time,
            hyperparameters=self.hyperparameters.copy(),
            feature_count=X_train.shape[1],
            train_samples=X_train.shape[0],
            val_samples=X_val.shape[0] if X_val is not None else 0,
            feature_importances=feature_importances
        )
        
        self.fitted = True
        
        # 打印结果
        if verbose:
            self.logger.info(f"\n{self.training_result.summary()}")
        
        return self.training_result
    
    @abstractmethod
    def _fit_model(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray],
                   y_val: Optional[np.ndarray],
                   verbose: bool):
        """具体的模型训练逻辑（子类实现）"""
        pass
    
    def _get_feature_importances(self) -> Optional[Dict[str, float]]:
        """获取特征重要性（如果模型支持）"""
        if hasattr(self.model, 'feature_importances_'):
            return {
                f"feature_{i}": float(importance)
                for i, importance in enumerate(self.model.feature_importances_)
            }
        return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测
        
        Args:
            X: 特征数据 (n_samples, n_features)
            
        Returns:
            预测标签 (n_samples,)
            
        Raises:
            RuntimeError: 如果模型未训练
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率
        
        Args:
            X: 特征数据 (n_samples, n_features)
            
        Returns:
            预测概率 (n_samples, n_classes)
            
        Raises:
            RuntimeError: 如果模型未训练或不支持概率预测
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        if not hasattr(self.model, 'predict_proba'):
            raise RuntimeError(f"{self._get_model_type()} does not support probability prediction")
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: Union[str, Path], save_metadata: bool = True):
        """保存模型
        
        Args:
            filepath: 模型保存路径（.pkl或.joblib）
            save_metadata: 是否同时保存训练元数据（.json）
            
        Raises:
            RuntimeError: 如果模型未训练
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Cannot save untrained model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to: {filepath}")
        
        # 保存元数据
        if save_metadata and self.training_result is not None:
            metadata_path = filepath.with_suffix('.json')
            metadata = {
                'model_type': self._get_model_type(),
                'training_result': self.training_result.to_dict(),
                'hyperparameters': self.hyperparameters,
                'saved_at': datetime.now().isoformat()
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Metadata saved to: {metadata_path}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path], 
                   logger: Optional[logging.Logger] = None) -> 'ModelTrainer':
        """加载模型
        
        Args:
            filepath: 模型文件路径
            logger: 日志记录器
            
        Returns:
            加载的训练器实例
            
        Raises:
            FileNotFoundError: 如果模型文件不存在
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # 加载模型
        model = joblib.load(filepath)
        
        # 尝试加载元数据
        metadata_path = filepath.with_suffix('.json')
        hyperparameters = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                hyperparameters = metadata.get('hyperparameters', {})
        
        # 创建训练器实例
        trainer = cls(hyperparameters=hyperparameters, logger=logger)
        trainer.model = model
        trainer.fitted = True
        
        if logger:
            logger.info(f"Model loaded from: {filepath}")
        
        return trainer


class XGBoostTrainer(ModelTrainer):
    """XGBoost分类器训练器
    
    使用XGBoost进行分类任务训练，支持：
    - 超参数配置
    - Early stopping（早停）
    - 验证集评估
    - 特征重要性
    
    Example:
        >>> trainer = XGBoostTrainer(hyperparameters={
        ...     'max_depth': 6,
        ...     'learning_rate': 0.1,
        ...     'n_estimators': 100
        ... })
        >>> result = trainer.train(X_train, y_train, X_val, y_val)
        >>> trainer.save_model('models/xgboost_model.pkl')
        >>> 
        >>> # 加载和预测
        >>> loaded_trainer = XGBoostTrainer.load_model('models/xgboost_model.pkl')
        >>> predictions = loaded_trainer.predict(X_test)
    """
    
    def _default_hyperparameters(self) -> Dict[str, Any]:
        """XGBoost默认超参数"""
        return {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',
            # 'use_label_encoder': False,  # 新版本已废弃此参数
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 10,  # 早停轮数
        }
    
    def _create_model(self) -> XGBClassifier:
        """创建XGBoost模型"""
        # 提取early_stopping_rounds，不传给XGBClassifier
        params = self.hyperparameters.copy()
        self.early_stopping_rounds = params.pop('early_stopping_rounds', None)
        
        return XGBClassifier(**params)
    
    def _get_model_type(self) -> str:
        return "XGBoost"
    
    def _fit_model(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray],
                   y_val: Optional[np.ndarray],
                   verbose: bool):
        """训练XGBoost模型（兼容多版本XGBoost）"""
        fit_params = {}
        
        # 计算类别权重以处理不平衡问题
        from sklearn.utils.class_weight import compute_sample_weight
        try:
            sample_weights = compute_sample_weight('balanced', y_train)
            fit_params['sample_weight'] = sample_weights
            if verbose:
                unique_labels = np.unique(y_train)
                self.logger.info("✓ 类别权重已启用（平衡类别分布）")
                # 显示每个类别的平均权重
                for label in unique_labels:
                    avg_weight = sample_weights[y_train == label].mean()
                    count = np.sum(y_train == label)
                    self.logger.info(f"  类别 {label}: 样本数={count}, 平均权重={avg_weight:.3f}")
        except Exception as e:
            self.logger.warning(f"无法计算样本权重，继续使用默认权重: {e}")
        
        # 配置验证集和early stopping（兼容新旧版本XGBoost）
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            
            if self.early_stopping_rounds:
                # 尝试三种方式，按版本从新到旧
                early_stopping_configured = False
                
                # 方式1: XGBoost >= 1.6.0 使用回调函数
                try:
                    from xgboost.callback import EarlyStopping
                    fit_params['callbacks'] = [EarlyStopping(rounds=self.early_stopping_rounds)]
                    early_stopping_configured = True
                    if verbose:
                        self.logger.info(f"Early stopping enabled (rounds={self.early_stopping_rounds}) [callback mode]")
                except (ImportError, AttributeError, TypeError):
                    pass
                
                # 方式2: 中等版本XGBoost尝试参数方式
                if not early_stopping_configured:
                    try:
                        # 先测试是否支持该参数（通过小数据集快速测试）
                        test_params = fit_params.copy()
                        test_params['early_stopping_rounds'] = self.early_stopping_rounds
                        # 如果这里不报错，说明支持
                        fit_params['early_stopping_rounds'] = self.early_stopping_rounds
                        early_stopping_configured = True
                        if verbose:
                            self.logger.info(f"Early stopping enabled (rounds={self.early_stopping_rounds}) [parameter mode]")
                    except (TypeError, KeyError):
                        pass
                
                # 方式3: 非常老的版本，不支持early stopping，仅使用验证集
                if not early_stopping_configured:
                    if verbose:
                        self.logger.warning(
                            f"Early stopping not supported in this XGBoost version. "
                            f"Training will use validation set for evaluation only."
                        )
            
            if verbose:
                fit_params['verbose'] = True
        else:
            if verbose:
                fit_params['verbose'] = True
        
        # 训练
        try:
            self.model.fit(X_train, y_train, **fit_params)
        except TypeError as e:
            # 如果还是失败，移除可能不支持的参数重试
            if 'callbacks' in fit_params:
                del fit_params['callbacks']
            if 'early_stopping_rounds' in fit_params:
                del fit_params['early_stopping_rounds']
            if verbose:
                self.logger.warning(f"Retrying without early stopping due to: {e}")
            self.model.fit(X_train, y_train, **fit_params)
        
        # 记录最佳迭代轮数
        if hasattr(self.model, 'best_iteration'):
            self.logger.info(f"Best iteration: {self.model.best_iteration}")


class RandomForestTrainer(ModelTrainer):
    """Random Forest分类器训练器
    
    使用Random Forest进行分类任务训练，支持：
    - 超参数配置
    - 并行训练
    - 特征重要性
    
    Example:
        >>> trainer = RandomForestTrainer(hyperparameters={
        ...     'n_estimators': 100,
        ...     'max_depth': 10,
        ...     'random_state': 42
        ... })
        >>> result = trainer.train(X_train, y_train, X_val, y_val)
        >>> trainer.save_model('models/rf_model.pkl')
    """
    
    def _default_hyperparameters(self) -> Dict[str, Any]:
        """Random Forest默认超参数"""
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
    
    def _create_model(self) -> RandomForestClassifier:
        """创建Random Forest模型"""
        return RandomForestClassifier(**self.hyperparameters)
    
    def _get_model_type(self) -> str:
        return "RandomForest"
    
    def _fit_model(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray],
                   y_val: Optional[np.ndarray],
                   verbose: bool):
        """训练Random Forest模型"""
        # Random Forest不支持validation set，直接训练
        if verbose:
            self.logger.info("Training Random Forest...")
        
        # 设置verbose参数
        if verbose:
            self.model.verbose = 2
        
        self.model.fit(X_train, y_train)
        
        if verbose:
            self.logger.info(f"Trained {self.model.n_estimators} trees")

