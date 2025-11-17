"""
实验数据加载器

负责从多个实验目录加载训练数据：
- 扫描 experiments/ 目录
- 加载 features.csv（特征数据）
- 加载 ground_truth.json（标签数据）
- 合并为完整的训练数据集
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class ExperimentInfo:
    """实验信息
    
    Attributes:
        exp_id: 实验ID
        exp_dir: 实验目录路径
        has_features: 是否有特征文件
        has_ground_truth: 是否有ground truth
        feature_count: 特征行数
        resolution: 实际分辨率（标签）
        scenario: 实验场景
    """
    exp_id: str
    exp_dir: Path
    has_features: bool
    has_ground_truth: bool
    feature_count: int = 0
    resolution: Optional[str] = None
    scenario: Optional[str] = None
    
    def is_valid(self) -> bool:
        """判断实验是否有效（可用于训练）"""
        return self.has_features and self.has_ground_truth and self.resolution is not None


class ExperimentDataLoader:
    """实验数据加载器
    
    从多个实验目录加载和合并训练数据。
    
    工作流程：
    1. 扫描 experiments/ 目录，找到所有 exp_* 子目录
    2. 对每个实验：
       - 尝试加载 features.csv（35个特征 + timestamp + predicted_resolution + confidence）
       - 尝试加载 ground_truth.json（提取 actual_resolution 作为标签）
       - 合并特征和标签
    3. 合并所有实验数据为单个 DataFrame
    4. 返回 (X_df, y_series, metadata)
    
    Attributes:
        experiments_dir: 实验根目录
        logger: 日志记录器
        experiment_infos: 实验信息列表
        
    Example:
        >>> loader = ExperimentDataLoader('experiments/')
        >>> df = loader.load_experiments()
        >>> print(f"Loaded {len(df)} samples from {loader.valid_count} experiments")
        >>> 
        >>> # 分离特征和标签
        >>> X = df.drop(['resolution', 'timestamp', 'exp_id'], axis=1)
        >>> y = df['resolution']
    """
    
    # 35个特征名称（与 DataWriter 一致）
    FEATURE_NAMES = [
        # TCP特征（10个）
        'retrans_rate', 'avg_rtt', 'rtt_std', 'max_rtt', 'avg_window', 
        'window_var', 'slow_start_count', 'congestion_events', 'ack_delay', 'conn_setup_time',
        # 流量统计特征（15个）
        'avg_throughput', 'throughput_std', 'throughput_min', 'throughput_max', 'throughput_cv',
        'avg_packet_size', 'packet_size_std', 'large_packet_ratio', 'packet_size_entropy',
        'uplink_downlink_ratio', 'total_bytes', 'total_packets', 'conn_duration', 'byte_rate_var', 'flow_count',
        # 时序特征（10个）
        'interval_mean', 'interval_std', 'interval_cv', 'periodicity_score', 'num_gaps',
        'gap_duration_avg', 'burst_count', 'burst_intensity', 'autocorrelation', 'trend_slope',
    ]
    
    def __init__(self, experiments_dir: Union[str, Path], logger: Optional[logging.Logger] = None):
        """初始化数据加载器
        
        Args:
            experiments_dir: 实验根目录路径
            logger: 日志记录器（可选）
        """
        self.experiments_dir = Path(experiments_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.experiment_infos: List[ExperimentInfo] = []
        
        if not self.experiments_dir.exists():
            raise FileNotFoundError(f"Experiments directory not found: {self.experiments_dir}")
        
        self.logger.info(f"ExperimentDataLoader initialized: {self.experiments_dir}")
    
    def scan_experiments(self) -> List[ExperimentInfo]:
        """扫描实验目录，收集实验信息
        
        Returns:
            实验信息列表
        """
        self.experiment_infos = []
        
        # 查找所有 exp_* 目录
        exp_dirs = sorted(self.experiments_dir.glob('exp_*'))
        
        if not exp_dirs:
            self.logger.warning(f"No experiment directories found in {self.experiments_dir}")
            return []
        
        self.logger.info(f"Found {len(exp_dirs)} experiment directories")
        
        for exp_dir in exp_dirs:
            if not exp_dir.is_dir():
                continue
            
            exp_id = exp_dir.name
            
            # 检查必需文件
            features_csv = exp_dir / 'features.csv'
            ground_truth_json = exp_dir / 'ground_truth.json'
            
            has_features = features_csv.exists()
            has_ground_truth = ground_truth_json.exists()
            
            # 创建实验信息
            exp_info = ExperimentInfo(
                exp_id=exp_id,
                exp_dir=exp_dir,
                has_features=has_features,
                has_ground_truth=has_ground_truth
            )
            
            # 如果有 ground_truth，提取分辨率和场景
            if has_ground_truth:
                try:
                    with open(ground_truth_json, 'r') as f:
                        gt_data = json.load(f)
                    
                    exp_info.resolution = gt_data.get('video', {}).get('actual_resolution', None)
                    exp_info.scenario = gt_data.get('scenario', None)
                except Exception as e:
                    self.logger.warning(f"{exp_id}: Failed to read ground_truth.json: {e}")
            
            # 如果有 features.csv，统计行数
            if has_features:
                try:
                    df = pd.read_csv(features_csv)
                    exp_info.feature_count = len(df)
                except Exception as e:
                    self.logger.warning(f"{exp_id}: Failed to read features.csv: {e}")
                    exp_info.has_features = False
            
            self.experiment_infos.append(exp_info)
        
        # 统计有效实验
        valid_count = sum(1 for info in self.experiment_infos if info.is_valid())
        self.logger.info(f"Valid experiments: {valid_count}/{len(self.experiment_infos)}")
        
        return self.experiment_infos
    
    def load_experiment(self, exp_info: ExperimentInfo) -> Optional[pd.DataFrame]:
        """加载单个实验的数据
        
        Args:
            exp_info: 实验信息
            
        Returns:
            DataFrame 包含特征和标签，如果加载失败返回 None
        """
        if not exp_info.is_valid():
            self.logger.debug(f"{exp_info.exp_id}: Skipping invalid experiment")
            return None
        
        try:
            # 加载特征数据
            features_csv = exp_info.exp_dir / 'features.csv'
            df = pd.read_csv(features_csv)
            
            # 添加实验ID
            df['exp_id'] = exp_info.exp_id
            
            # 添加场景信息
            if exp_info.scenario:
                df['scenario'] = exp_info.scenario
            
            # 添加实际分辨率（ground truth标签）
            df['actual_resolution'] = exp_info.resolution
            
            self.logger.debug(
                f"{exp_info.exp_id}: Loaded {len(df)} samples, "
                f"resolution={exp_info.resolution}, scenario={exp_info.scenario}"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"{exp_info.exp_id}: Failed to load data: {e}")
            return None
    
    def load_experiments(self, 
                        min_samples_per_exp: int = 1,
                        include_predicted: bool = True) -> pd.DataFrame:
        """加载所有实验数据并合并
        
        Args:
            min_samples_per_exp: 每个实验最少样本数（过滤太短的实验）
            include_predicted: 是否包含预测的分辨率和置信度列
            
        Returns:
            合并后的 DataFrame
            
        Raises:
            ValueError: 如果没有找到有效的实验数据
        """
        # 扫描实验
        self.scan_experiments()
        
        if not self.experiment_infos:
            raise ValueError("No experiments found")
        
        # 加载所有有效实验
        all_dfs = []
        
        for exp_info in self.experiment_infos:
            if not exp_info.is_valid():
                continue
            
            # 检查最少样本数
            if exp_info.feature_count < min_samples_per_exp:
                self.logger.debug(
                    f"{exp_info.exp_id}: Skipping (only {exp_info.feature_count} samples)"
                )
                continue
            
            df = self.load_experiment(exp_info)
            if df is not None:
                all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError(
                f"No valid experiment data found. "
                f"Make sure experiments have both features.csv and ground_truth.json. "
                f"Run full pipeline to generate features.csv files."
            )
        
        # 合并所有数据
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # 如果不需要预测列，删除它们
        if not include_predicted:
            cols_to_drop = ['predicted_resolution', 'confidence']
            existing_cols = [col for col in cols_to_drop if col in combined_df.columns]
            if existing_cols:
                combined_df = combined_df.drop(columns=existing_cols)
        
        self.logger.info(
            f"Loaded total {len(combined_df)} samples from {len(all_dfs)} experiments"
        )
        self.logger.info(f"Columns: {list(combined_df.columns)}")
        self.logger.info(f"Shape: {combined_df.shape}")
        
        # 显示分辨率分布
        if 'actual_resolution' in combined_df.columns:
            resolution_counts = combined_df['actual_resolution'].value_counts()
            self.logger.info(f"Resolution distribution:\n{resolution_counts}")
        
        return combined_df
    
    def get_summary(self) -> Dict[str, Any]:
        """获取数据加载摘要
        
        Returns:
            包含统计信息的字典
        """
        if not self.experiment_infos:
            self.scan_experiments()
        
        valid_exps = [info for info in self.experiment_infos if info.is_valid()]
        invalid_exps = [info for info in self.experiment_infos if not info.is_valid()]
        
        total_samples = sum(info.feature_count for info in valid_exps)
        
        # 统计分辨率分布
        resolution_dist = {}
        for info in valid_exps:
            if info.resolution:
                resolution_dist[info.resolution] = resolution_dist.get(info.resolution, 0) + info.feature_count
        
        # 统计场景分布
        scenario_dist = {}
        for info in valid_exps:
            if info.scenario:
                scenario_dist[info.scenario] = scenario_dist.get(info.scenario, 0) + 1
        
        summary = {
            'total_experiments': len(self.experiment_infos),
            'valid_experiments': len(valid_exps),
            'invalid_experiments': len(invalid_exps),
            'total_samples': total_samples,
            'resolution_distribution': resolution_dist,
            'scenario_distribution': scenario_dist,
            'invalid_exp_ids': [info.exp_id for info in invalid_exps]
        }
        
        return summary
    
    def print_summary(self):
        """打印数据加载摘要"""
        summary = self.get_summary()
        
        print("=" * 70)
        print("Experiment Data Loader Summary")
        print("=" * 70)
        print(f"Total experiments:    {summary['total_experiments']}")
        print(f"  ✓ Valid:            {summary['valid_experiments']}")
        print(f"  ✗ Invalid:          {summary['invalid_experiments']}")
        print(f"Total samples:        {summary['total_samples']}")
        print()
        
        if summary['resolution_distribution']:
            print("Resolution distribution:")
            for res, count in sorted(summary['resolution_distribution'].items()):
                pct = count / summary['total_samples'] * 100 if summary['total_samples'] > 0 else 0
                print(f"  - {res:6s}: {count:5d} samples ({pct:5.1f}%)")
            print()
        
        if summary['scenario_distribution']:
            print("Scenario distribution:")
            for scenario, count in sorted(summary['scenario_distribution'].items()):
                print(f"  - {scenario}: {count} experiments")
            print()
        
        if summary['invalid_exp_ids']:
            print("Invalid experiments (missing features.csv or labels):")
            for exp_id in summary['invalid_exp_ids'][:10]:  # 最多显示10个
                print(f"  - {exp_id}")
            if len(summary['invalid_exp_ids']) > 10:
                print(f"  ... and {len(summary['invalid_exp_ids']) - 10} more")
        
        print("=" * 70)
    
    @property
    def valid_count(self) -> int:
        """有效实验数量"""
        return sum(1 for info in self.experiment_infos if info.is_valid())
    
    @property
    def invalid_count(self) -> int:
        """无效实验数量"""
        return sum(1 for info in self.experiment_infos if not info.is_valid())


if __name__ == '__main__':
    """测试模块"""
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("Testing ExperimentDataLoader")
    print("=" * 70)
    
    # 使用项目的 experiments/ 目录
    experiments_dir = Path(__file__).parents[2] / 'experiments'
    
    if not experiments_dir.exists():
        print(f"\n✗ Experiments directory not found: {experiments_dir}")
        print("\nPlease run some experiments first to generate data.")
        sys.exit(1)
    
    print(f"\nExperiments directory: {experiments_dir}")
    
    try:
        # 测试1: 初始化
        print("\n[Test 1] Initialize loader")
        loader = ExperimentDataLoader(experiments_dir)
        print("  ✓ Loader initialized")
        
        # 测试2: 扫描实验
        print("\n[Test 2] Scan experiments")
        exp_infos = loader.scan_experiments()
        print(f"  Found {len(exp_infos)} experiment directories")
        print(f"  Valid: {loader.valid_count}, Invalid: {loader.invalid_count}")
        
        # 测试3: 显示摘要
        print("\n[Test 3] Display summary")
        loader.print_summary()
        
        # 测试4: 加载数据
        if loader.valid_count > 0:
            print("\n[Test 4] Load experiment data")
            df = loader.load_experiments()
            print(f"  ✓ Loaded {len(df)} total samples")
            print(f"  ✓ Shape: {df.shape}")
            print(f"  ✓ Columns: {list(df.columns)[:10]}...")
            
            # 显示前几行
            print("\n[Test 5] Display first few rows")
            print(df.head(3))
            
            # 分离特征和标签
            print("\n[Test 6] Separate features and labels")
            feature_cols = [col for col in df.columns 
                           if col not in ['timestamp', 'exp_id', 'scenario', 
                                         'actual_resolution', 'predicted_resolution', 'confidence']]
            X = df[feature_cols]
            y = df['actual_resolution']
            
            print(f"  Features shape: {X.shape}")
            print(f"  Labels shape: {y.shape}")
            print(f"  Label distribution:\n{y.value_counts()}")
            
            print("\n" + "=" * 70)
            print("✓ All tests passed!")
            print("=" * 70)
        else:
            print("\n⚠ No valid experiments found.")
            print("\nTo generate training data:")
            print("  1. Run full pipeline experiments with real traffic capture")
            print("  2. Ensure each experiment generates features.csv")
            print("  3. Set actual_resolution in ground_truth.json")
            print("\n" + "=" * 70)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

