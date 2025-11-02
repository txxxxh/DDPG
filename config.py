# config.py
from dataclasses import dataclass

@dataclass
class Config:
    """统一配置类 - 包含所有系统参数"""
    
    # ================== 基础资产配置 ==================
    n_assets: int = 10  # 资产数量
    n_factors: int = 24  # 因子总数量
    
    # ================== 数据处理配置 ==================
    return_col_index: int = 6  # 收益率在Excel中的列索引
    rsi_period: int = 14  # RSI计算周期
    trading_days_per_year: int = 252  # 年化交易日数
    lookback_window: int = 120  # 滚动窗口大小
    min_weight_threshold: float = 0.01  # 最小权重阈值
    
    # ================== 文件路径配置 ==================
    data_path: str = "/Users/txh/Desktop/DDPG/合并三因子后的股票数据.xlsx"
    
    # ================== 训练参数配置 ==================
    gamma: float = 0.95  # 折扣因子
    tau: float = 0.001  # 软更新参数
    lr_actor: float = 1e-4  # Actor学习率
    lr_critic: float = 1e-3  # Critic学习率
    batch_size: int = 32  # 批次大小
    grad_clip_norm: float = 0.2  # 梯度裁剪范数
    exploration_rate_start: float = 0.5  # 初始探索率
    temperature_start: float = 0.5  # 初始温度参数
    
    # ================== 探索策略配置 ==================
    min_weight: float = 0.01  # 最小权重约束
    max_weight: float = 0.2  # 最大权重约束
    exploration_noise_scale: float = 0.1  # 探索噪声缩放
    diversification_alpha: float = 0.2  # 多样化系数
    default_epsilon: float = 0.05  # 默认扰动系数
    min_probability: float = 1e-6  # 最小概率值
    
    # ================== 计算属性 ==================
    @property
    def feature_dim(self) -> int:
        """特征维度（等于因子数量）"""
        return self.n_factors
    
    @property
    def time_features_dim(self) -> int:
        """时序特征维度（除权重外）"""
        return self.feature_dim - 1

# 全局配置实例
config = Config()