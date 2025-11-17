"""
预测引擎

实现视频质量预测功能。
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
import time
import logging
from pathlib import Path


@dataclass
class Prediction:
    """预测结果
    
    Attributes:
        resolution: 预测的分辨率 ('480p', '720p', '1080p')
        confidence: 置信度 (0-1)
        probabilities: 三个分辨率的概率 [p_480, p_720, p_1080]
        timestamp: 预测时间戳
        metrics: 相关的网络指标
    """
    resolution: str
    confidence: float
    probabilities: np.ndarray
    timestamp: float
    metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'resolution': self.resolution,
            'confidence': self.confidence,
            'probabilities': self.probabilities.tolist() if isinstance(self.probabilities, np.ndarray) else self.probabilities,
            'timestamp': self.timestamp,
            'metrics': self.metrics or {}
        }


class RuleBasedPredictor:
    """基于规则的简单预测器
    
    这是一个演示用的预测器，基于简单规则判断视频分辨率。
    后续可以替换为训练好的ML模型。
    
    规则：
    - 吞吐量 > 5 Mbps && 丢包 < 1% && RTT < 100ms -> 1080p
    - 吞吐量 > 2.5 Mbps && 丢包 < 2% && RTT < 150ms -> 720p
    - 其他 -> 480p
    
    Attributes:
        logger: 日志记录器
        classes: 分辨率类别列表
    
    Example:
        >>> predictor = RuleBasedPredictor()
        >>> prediction = predictor.predict(features)
        >>> print(f"{prediction.resolution} ({prediction.confidence:.1%})")
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """初始化预测器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.classes = ['480p', '720p', '1080p']
        self.logger.info("RuleBasedPredictor initialized")
    
    def predict(self, features: np.ndarray, feature_names: Optional[List[str]] = None) -> Prediction:
        """预测视频分辨率
        
        Args:
            features: 35维特征向量
            feature_names: 特征名称列表（可选）
            
        Returns:
            Prediction对象
            
        Example:
            >>> features = calculator.compute_all_features(packets)
            >>> prediction = predictor.predict(features)
        """
        if features.shape != (35,):
            raise ValueError(f"Expected 35 features, got {features.shape}")
        
        # 提取关键特征（假设特征顺序：TCP 0-9, Traffic 10-24, Temporal 25-34）
        # traffic_avg_throughput = features[10]  # 特征11
        # tcp_retrans_rate = features[0]          # 特征1
        # traffic_total_bytes = features[20]      # 特征21
        # traffic_duration = features[22]         # 特征23
        
        # 从特征中提取网络指标
        metrics = self._extract_metrics(features)
        
        # 基于规则的简单预测
        throughput = metrics['throughput']
        loss_rate = metrics['loss_rate']
        rtt = metrics['rtt']
        
        # 决策规则
        if throughput > 5.0 and loss_rate < 1.0 and rtt < 100:
            # 高质量网络 -> 1080p
            resolution = '1080p'
            probabilities = np.array([0.05, 0.15, 0.80])
        elif throughput > 2.5 and loss_rate < 2.0 and rtt < 150:
            # 中等质量网络 -> 720p
            resolution = '720p'
            probabilities = np.array([0.10, 0.75, 0.15])
        else:
            # 低质量网络 -> 480p
            resolution = '480p'
            probabilities = np.array([0.80, 0.15, 0.05])
        
        confidence = probabilities[self.classes.index(resolution)]
        
        prediction = Prediction(
            resolution=resolution,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=time.time(),
            metrics=metrics
        )
        
        self.logger.debug(f"Prediction: {resolution} ({confidence:.2%}), "
                         f"throughput={throughput:.2f}Mbps, loss={loss_rate:.2f}%, rtt={rtt:.1f}ms")
        
        return prediction
    
    def _extract_metrics(self, features: np.ndarray) -> Dict[str, float]:
        """从特征向量中提取关键网络指标
        
        Args:
            features: 35维特征向量
            
        Returns:
            网络指标字典
        """
        # 特征索引（基于FeatureCalculator的顺序）
        # TCP: 0-9
        # Traffic: 10-24
        # Temporal: 25-34
        
        metrics = {
            'throughput': float(features[10]),      # traffic_avg_throughput (特征11)
            'loss_rate': float(features[0] * 100),  # tcp_retrans_rate * 100 (特征1)
            'rtt': float(features[1] * 1000),       # tcp_avg_rtt * 1000 (特征2, 转ms)
            'packet_size': float(features[15]),     # traffic_avg_packet_size (特征16)
            'total_bytes': float(features[20]),     # traffic_total_bytes (特征21)
            'total_packets': float(features[21]),   # traffic_total_packets (特征22)
            'duration': float(features[22]),        # traffic_duration (特征23)
        }
        
        return metrics
    
    def predict_batch(self, features_batch: np.ndarray) -> List[Prediction]:
        """批量预测
        
        Args:
            features_batch: (N, 35) 特征矩阵
            
        Returns:
            预测结果列表
        """
        if features_batch.ndim == 1:
            features_batch = features_batch.reshape(1, -1)
        
        predictions = []
        for features in features_batch:
            prediction = self.predict(features)
            predictions.append(prediction)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, str]:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_type': 'RuleBasedPredictor',
            'version': '1.0',
            'classes': ', '.join(self.classes),
            'description': 'Simple rule-based predictor for demonstration'
        }


class ModelBasedPredictor:
    """基于ML模型的预测器
    
    这是为真实ML模型预留的接口。
    可以加载训练好的XGBoost/RandomForest/LSTM模型。
    
    Attributes:
        model: 加载的模型对象
        logger: 日志记录器
        classes: 分辨率类别列表
    
    Example:
        >>> predictor = ModelBasedPredictor('models/xgboost_model.pkl')
        >>> prediction = predictor.predict(features)
    """
    
    def __init__(self, model_path: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        """初始化预测器
        
        Args:
            model_path: 模型文件路径
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.classes = ['480p', '720p', '1080p']
        self.model = None
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Path):
        """加载模型
        
        Args:
            model_path: 模型文件路径
            
        Raises:
            FileNotFoundError: 模型文件不存在
            Exception: 模型加载失败
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # TODO: 实现模型加载
            # import joblib
            # self.model = joblib.load(model_path)
            
            self.logger.info(f"Model loaded from {model_path}")
            self.logger.warning("ModelBasedPredictor is not yet implemented. "
                              "Please use RuleBasedPredictor for now.")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> Prediction:
        """预测视频分辨率
        
        Args:
            features: 35维特征向量
            
        Returns:
            Prediction对象
            
        Raises:
            RuntimeError: 模型未加载
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please load a model first or use RuleBasedPredictor.")
        
        # TODO: 实现ML模型预测
        # if features.ndim == 1:
        #     features = features.reshape(1, -1)
        #
        # probabilities = self.model.predict_proba(features)[0]
        # pred_class = np.argmax(probabilities)
        #
        # return Prediction(
        #     resolution=self.classes[pred_class],
        #     confidence=probabilities[pred_class],
        #     probabilities=probabilities,
        #     timestamp=time.time()
        # )
        
        raise NotImplementedError("ModelBasedPredictor is not yet implemented")
    
    def get_model_info(self) -> Dict[str, str]:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_type': 'ModelBasedPredictor',
            'model_path': str(self.model_path) if self.model_path else 'N/A',
            'classes': ', '.join(self.classes),
            'status': 'loaded' if self.model else 'not_loaded'
        }


# 工厂函数
def create_predictor(use_ml_model: bool = False, 
                    model_path: Optional[Path] = None,
                    logger: Optional[logging.Logger] = None):
    """创建预测器
    
    Args:
        use_ml_model: 是否使用ML模型
        model_path: ML模型路径（如果use_ml_model=True）
        logger: 日志记录器
        
    Returns:
        预测器对象（RuleBasedPredictor或ModelBasedPredictor）
        
    Example:
        >>> # 使用规则预测器（推荐用于演示）
        >>> predictor = create_predictor(use_ml_model=False)
        >>>
        >>> # 使用ML模型（需要训练好的模型）
        >>> predictor = create_predictor(use_ml_model=True, model_path='models/xgb.pkl')
    """
    if use_ml_model:
        if model_path is None:
            raise ValueError("model_path is required when use_ml_model=True")
        return ModelBasedPredictor(model_path=model_path, logger=logger)
    else:
        return RuleBasedPredictor(logger=logger)


if __name__ == '__main__':
    """测试模块"""
    print("=" * 70)
    print("Testing RuleBasedPredictor")
    print("=" * 70)
    
    # 创建预测器
    print("\n[Test 1] Create predictor")
    predictor = RuleBasedPredictor()
    print(f"  Predictor: {predictor.get_model_info()}")
    
    # 创建测试特征（35维）
    print("\n[Test 2] Create test features")
    # 模拟高质量网络特征
    high_quality_features = np.zeros(35)
    high_quality_features[10] = 8.0  # 高吞吐量 8 Mbps
    high_quality_features[0] = 0.001  # 低重传率 0.1%
    high_quality_features[1] = 0.030  # 低RTT 30ms
    
    print("  High quality features created")
    
    # 测试预测
    print("\n[Test 3] Test prediction")
    prediction = predictor.predict(high_quality_features)
    print(f"  Resolution: {prediction.resolution}")
    print(f"  Confidence: {prediction.confidence:.2%}")
    print(f"  Probabilities: {prediction.probabilities}")
    print(f"  Metrics: {prediction.metrics}")
    
    # 测试中等质量
    print("\n[Test 4] Test medium quality")
    medium_quality_features = np.zeros(35)
    medium_quality_features[10] = 3.5  # 中等吞吐量
    medium_quality_features[0] = 0.01  # 1% 重传
    medium_quality_features[1] = 0.080  # 80ms RTT
    
    prediction = predictor.predict(medium_quality_features)
    print(f"  Resolution: {prediction.resolution}")
    print(f"  Confidence: {prediction.confidence:.2%}")
    
    # 测试低质量
    print("\n[Test 5] Test low quality")
    low_quality_features = np.zeros(35)
    low_quality_features[10] = 1.5  # 低吞吐量
    low_quality_features[0] = 0.03  # 3% 重传
    low_quality_features[1] = 0.200  # 200ms RTT
    
    prediction = predictor.predict(low_quality_features)
    print(f"  Resolution: {prediction.resolution}")
    print(f"  Confidence: {prediction.confidence:.2%}")
    
    print("\n" + "=" * 70)
    print("Basic tests completed!")
    print("For full testing, run scripts/test_story_5_1.py")
    print("=" * 70)



