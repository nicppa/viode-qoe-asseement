# -*- coding: utf-8 -*-
"""
特征预处理器

负责训练数据的预处理：
- 缺失值处理
- 特征归一化
- 标签编码
- 数据划分（训练/验证/测试）
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


@dataclass
class DataSplit:
    """数据划分结果
    
    Attributes:
        X_train: 训练特征
        X_val: 验证特征
        X_test: 测试特征
        y_train: 训练标签
        y_val: 验证标签
        y_test: 测试标签
        feature_names: 特征名称列表
        label_names: 标签名称列表
    """
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    label_names: List[str]
    
    def summary(self) -> str:
        """返回数据划分摘要"""
        return f"""Data Split Summary:
  Training:   {self.X_train.shape[0]:5d} samples ({self.X_train.shape[0]/(self.X_train.shape[0]+self.X_val.shape[0]+self.X_test.shape[0])*100:.1f}%)
  Validation: {self.X_val.shape[0]:5d} samples ({self.X_val.shape[0]/(self.X_train.shape[0]+self.X_val.shape[0]+self.X_test.shape[0])*100:.1f}%)
  Test:       {self.X_test.shape[0]:5d} samples ({self.X_test.shape[0]/(self.X_train.shape[0]+self.X_val.shape[0]+self.X_test.shape[0])*100:.1f}%)
  Total:      {self.X_train.shape[0]+self.X_val.shape[0]+self.X_test.shape[0]:5d} samples
  Features:   {self.X_train.shape[1]} dimensions
  Labels:     {len(self.label_names)} classes ({', '.join(self.label_names)})
"""


class FeaturePreprocessor:
    """特征预处理器
    
    提供完整的特征预处理流程：
    1. 缺失值处理（中位数填充）
    2. 特征归一化（StandardScaler，零均值单位方差）
    3. 标签编码（字符串 → 整数）
    4. 数据划分（训练/验证/测试）
    
    特点：
    - 可保存和加载（保持预处理一致性）
    - 支持增量转换（transform新数据）
    - 详细的统计信息
    
    Attributes:
        scaler: StandardScaler 对象
        label_encoder: LabelEncoder 对象
        feature_names: 特征名称列表
        label_names: 标签名称列表（按编码顺序）
        median_values: 每个特征的中位数（用于填充缺失值）
        fitted: 是否已拟合
        logger: 日志记录器
        
    Example:
        >>> # 训练时
        >>> preprocessor = FeaturePreprocessor()
        >>> data_split = preprocessor.fit_transform(
        ...     df, 
        ...     target_col='actual_resolution',
        ...     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        ... )
        >>> preprocessor.save('models/preprocessor.pkl')
        >>> 
        >>> # 预测时
        >>> preprocessor = FeaturePreprocessor.load('models/preprocessor.pkl')
        >>> X_new = preprocessor.transform(df_new)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """初始化预处理器
        
        Args:
            logger: 日志记录器（可选）
        """
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.label_names: List[str] = []
        self.median_values: Optional[pd.Series] = None
        self.fitted = False
        self.logger = logger or logging.getLogger(__name__)
    
    def fit_transform(self,
                     df: pd.DataFrame,
                     target_col: str = 'actual_resolution',
                     exclude_cols: Optional[List[str]] = None,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     random_state: int = 42) -> DataSplit:
        """拟合预处理器并转换数据（训练时使用）
        
        Args:
            df: 原始数据 DataFrame
            target_col: 标签列名
            exclude_cols: 需要排除的列（不作为特征）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_state: 随机种子
            
        Returns:
            DataSplit 对象，包含划分后的训练/验证/测试数据
            
        Raises:
            ValueError: 如果比例之和不等于1或目标列不存在
        """
        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )
        
        # 验证目标列
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        self.logger.info("Starting feature preprocessing...")
        
        # 1. 分离特征和标签
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'exp_id', 'scenario', 
                           'predicted_resolution', 'confidence']
        
        # 确保目标列在排除列表中
        if target_col not in exclude_cols:
            exclude_cols.append(target_col)
        
        # 提取特征列名
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        self.logger.info(f"Selected {len(self.feature_names)} features")
        
        # 提取特征和标签
        X = df[self.feature_names].copy()
        y = df[target_col].copy()
        
        # 2. 处理缺失值（使用中位数填充）
        self.logger.info("Handling missing values...")
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            self.logger.warning(f"Found {missing_count} missing values, filling with median")
            self.median_values = X.median()
            X = X.fillna(self.median_values)
        else:
            self.logger.info("No missing values found")
            self.median_values = X.median()  # 保存以备后用
        
        # 3. 标签编码
        self.logger.info("Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(y)
        self.label_names = self.label_encoder.classes_.tolist()
        self.logger.info(f"Label mapping: {dict(zip(self.label_names, range(len(self.label_names))))}")
        
        # 显示标签分布
        unique, counts = np.unique(y_encoded, return_counts=True)
        for label_idx, count in zip(unique, counts):
            label_name = self.label_names[label_idx]
            pct = count / len(y_encoded) * 100
            self.logger.info(f"  {label_name}: {count} samples ({pct:.1f}%)")
        
        # 4. 数据划分
        self.logger.info(f"Splitting data: train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")
        
        # 第一次划分：分离出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded,
            test_size=test_ratio,
            random_state=random_state,
            stratify=y_encoded  # 保持标签分布
        )
        
        # 第二次划分：从剩余数据中分离训练集和验证集
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        # 5. 特征归一化（只在训练集上拟合）
        self.logger.info("Normalizing features (StandardScaler)...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.fitted = True
        self.logger.info("Preprocessing completed successfully")
        
        # 创建 DataSplit 对象
        data_split = DataSplit(
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=self.feature_names,
            label_names=self.label_names
        )
        
        # 打印摘要
        self.logger.info("\n" + data_split.summary())
        
        return data_split
    
    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> np.ndarray:
        """转换新数据（预测时使用）
        
        使用已拟合的预处理器转换新数据。
        
        Args:
            df: 新数据 DataFrame
            target_col: 标签列名（如果存在）
            
        Returns:
            归一化后的特征矩阵，如果提供了 target_col，返回 (X, y) 元组
            
        Raises:
            RuntimeError: 如果预处理器未拟合
            ValueError: 如果特征列不匹配
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform() first.")
        
        # 检查特征列
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        # 提取特征
        X = df[self.feature_names].copy()
        
        # 处理缺失值
        if X.isnull().sum().sum() > 0:
            X = X.fillna(self.median_values)
        
        # 归一化
        X_scaled = self.scaler.transform(X)
        
        # 如果提供了标签列，也进行编码
        if target_col and target_col in df.columns:
            y = df[target_col]
            y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        
        return X_scaled
    
    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """将编码后的标签还原为原始标签
        
        Args:
            y_encoded: 编码后的标签（整数）
            
        Returns:
            原始标签（字符串）
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted.")
        
        return self.label_encoder.inverse_transform(y_encoded)
    
    def save(self, path: Path):
        """保存预处理器到文件
        
        Args:
            path: 保存路径
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存所有状态
        state = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'label_names': self.label_names,
            'median_values': self.median_values,
            'fitted': self.fitted
        }
        
        joblib.dump(state, path)
        self.logger.info(f"Preprocessor saved to: {path}")
    
    @classmethod
    def load(cls, path: Path, logger: Optional[logging.Logger] = None) -> 'FeaturePreprocessor':
        """从文件加载预处理器
        
        Args:
            path: 文件路径
            logger: 日志记录器（可选）
            
        Returns:
            加载的 FeaturePreprocessor 对象
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {path}")
        
        # 加载状态
        state = joblib.load(path)
        
        # 创建新实例并恢复状态
        preprocessor = cls(logger=logger)
        preprocessor.scaler = state['scaler']
        preprocessor.label_encoder = state['label_encoder']
        preprocessor.feature_names = state['feature_names']
        preprocessor.label_names = state['label_names']
        preprocessor.median_values = state['median_values']
        preprocessor.fitted = state['fitted']
        
        if logger:
            logger.info(f"Preprocessor loaded from: {path}")
        
        return preprocessor
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """获取特征统计信息
        
        Returns:
            包含统计信息的字典
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted")
        
        stats = {
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'num_labels': len(self.label_names),
            'label_names': self.label_names,
            'label_mapping': dict(zip(self.label_names, range(len(self.label_names)))),
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_std': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
        }
        
        return stats


if __name__ == '__main__':
    """测试模块"""
    import sys
    import io
    
    # 修复终端编码
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("Testing FeaturePreprocessor")
    print("=" * 70)
    
    # 创建示例数据
    print("\n[Test 1] Create sample data")
    np.random.seed(42)
    n_samples = 100
    
    sample_data = {
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples) * 2 + 5,
        'feature_3': np.random.randn(n_samples) * 0.5 - 1,
        'actual_resolution': np.random.choice(['480p', '720p', '1080p'], n_samples),
        'timestamp': np.arange(n_samples),
        'exp_id': ['exp_001'] * n_samples
    }
    
    # 添加一些缺失值
    sample_data['feature_2'][10:15] = np.nan
    
    df = pd.DataFrame(sample_data)
    print(f"  Created DataFrame: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    
    # 测试预处理
    print("\n[Test 2] Fit and transform")
    preprocessor = FeaturePreprocessor()
    
    try:
        data_split = preprocessor.fit_transform(
            df,
            target_col='actual_resolution',
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        print("  ✓ Preprocessing completed")
        print(f"\n{data_split.summary()}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 测试保存和加载
    print("\n[Test 3] Save and load")
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    save_path = temp_dir / 'preprocessor.pkl'
    
    preprocessor.save(save_path)
    print(f"  ✓ Saved to: {save_path}")
    
    loaded_preprocessor = FeaturePreprocessor.load(save_path)
    print(f"  ✓ Loaded from: {save_path}")
    
    # 测试转换新数据
    print("\n[Test 4] Transform new data")
    new_df = pd.DataFrame({
        'feature_1': np.random.randn(10),
        'feature_2': np.random.randn(10) * 2 + 5,
        'feature_3': np.random.randn(10) * 0.5 - 1,
        'actual_resolution': np.random.choice(['480p', '720p', '1080p'], 10),
        'timestamp': np.arange(10),
        'exp_id': ['exp_002'] * 10
    })
    
    X_new, y_new = loaded_preprocessor.transform(new_df, target_col='actual_resolution')
    print(f"  ✓ Transformed: X={X_new.shape}, y={y_new.shape}")
    
    # 测试标签还原
    y_decoded = loaded_preprocessor.inverse_transform_labels(y_new)
    print(f"  ✓ Decoded labels: {y_decoded[:5]}")
    
    # 获取统计信息
    print("\n[Test 5] Get feature stats")
    stats = loaded_preprocessor.get_feature_stats()
    print(f"  Features: {stats['num_features']}")
    print(f"  Labels: {stats['num_labels']} - {stats['label_names']}")
    print(f"  Label mapping: {stats['label_mapping']}")
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


