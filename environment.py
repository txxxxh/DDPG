# environment.py
import numpy as np
import torch
from typing import Tuple, Dict, Any
from config import config

class PortfolioEnvironment:
    def __init__(self,config, returns_data, factors_data, window_size=config.lookback_window, initial_balance=1.0):
        self.returns_data = returns_data
        self.factors_data = factors_data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.n_assets = returns_data.shape[1]
        self.n_factors = factors_data.shape[2]  # 动态获取因子数量

        # 验证因子数量
        if self.n_factors != config.n_factors:
            raise ValueError(f"因子数量不匹配：期望{config.n_factors}，实际{self.n_factors}")

        # 特征维度配置（使用全局变量）
        self.feature_dim = config.feature_dim  # 总特征维度
        self.feature_dim = config.feature_dim  # 兼容性
        self.WEIGHT_DIM = self.n_assets  # 权重维度

        # 特征索引常量
        self.PREV_RETURNS_IDX = 0  # 收益率特征索引

        self.n_steps = returns_data.shape[0]

        # 标准化参数
        self.returns_mean = np.mean(returns_data[:-1], axis=0)
        self.factors_mean = np.mean(factors_data[:-1], axis=0)  # (n_assets, n_factors)
        self.returns_std = torch.tensor(np.std(returns_data[:-1], axis=0) + 1e-6, dtype=torch.float32)
        self.factors_std = torch.tensor(np.std(factors_data[:-1], axis=0) + 1e-6, dtype=torch.float32)

        self._validate_data()
        self.reset()

        print(f"=== 环境初始化完成 ===")
        print(f"资产数量: {self.n_assets}")
        print(f"因子数量: {self.n_factors}")
        print(f"特征维度: {self.feature_dim}")
        print(f"状态形状: ({self.n_assets}, {self.window_size}, {self.feature_dim})")

    def _validate_data(self):
        if self.returns_data.shape[0] != self.factors_data.shape[0]:
            raise ValueError("收益率数据和因子数据时间步不一致")
        if self.returns_data.shape[1] != self.factors_data.shape[1]:
            raise ValueError("收益率数据和因子数据资产数量不一致")

    def reset(self):
        """重置环境状态"""
        min_start = self.window_size
        max_start = self.n_steps - self.window_size - 1

        self.current_step = min_start + np.random.randint(0, max_start - min_start)
        print(f"current_step reset: {self.current_step}")
        self.balance = 1.0
        self.portfolio_value = 1.0

        # 初始化为等权重
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.prev_weights += np.random.normal(0, 0.01, self.n_assets)
        self.prev_weights = np.maximum(self.prev_weights, 0)
        self.prev_weights /= self.prev_weights.sum()

        self.done = False
        self.episode_start_step = self.current_step
        self.episode_start_value = self.portfolio_value

        # 重置历史记录
        self.historical_returns = []
        self.historical_weights = [self.prev_weights.copy()]

        state = self.get_state()
        print(f"Reset state shape: {state.shape}")
        assert state.shape == (self.n_assets, self.window_size, config.feature_dim), \
            f"State shape error: {state.shape}, expected ({self.n_assets}, {self.window_size}, {config.feature_dim})"
        return state


    def get_state(self) -> torch.Tensor:

        decision_day = self.current_step
        start_idx = decision_day - self.window_size
        end_idx = decision_day

        if start_idx < 0:
            raise ValueError(f"历史数据不足，需要 {self.window_size} 天历史数据")

        # 获取历史数据
        factors_window = self.factors_data[start_idx:end_idx]  # (window_size, n_assets, config.n_factors)

        # 构建状态
        state = np.zeros((self.n_assets, self.window_size, config.feature_dim), dtype=np.float32)

        for asset_idx in range(self.n_assets):

            # 因子特征
            asset_factors = factors_window[:, asset_idx, :]  # (window_size, config.n_factors)
            factors_norm = (asset_factors - self.factors_mean[asset_idx]) / (self.factors_std[asset_idx] + 1e-8)

            # 所有因子特征
            state[asset_idx, :, :] = factors_norm

        return torch.tensor(state, dtype=torch.float32)

    def _calculate_reward(self, portfolio_return: float, action: torch.Tensor) -> float:
        """增强奖励函数"""
        weights_np = action.detach().numpy()
        weights_np = np.maximum(weights_np, 1e-8)

        # 基础收益奖励
        base_reward = portfolio_return * 100

        # 交易成本
        weight_changes = torch.abs(action - torch.tensor(self.prev_weights, dtype=torch.float32))
        total_trading_volume = torch.sum(weight_changes).item()

        if total_trading_volume > 0.1:
            transaction_cost = total_trading_volume * 0.3
        else:
            transaction_cost = total_trading_volume * 0.15

        # 多元化奖励
        hhi = np.sum(weights_np ** 2)
        ideal_hhi = 1.0 / self.n_assets
        diversification_reward = (ideal_hhi / hhi - 1) * 0.1
        diversification_reward = np.clip(diversification_reward, -0.05, 0.2)

        # 风险惩罚
        risk_penalty = 0
        if len(self.historical_returns) > 10:
            returns_array = np.array(self.historical_returns[-30:])
            volatility = np.std(returns_array)
            if volatility > 0.02:
                risk_penalty = volatility * 1.5
            elif volatility > 0.01:
                risk_penalty = volatility * 1.0
            else:
                risk_penalty = volatility * 0.5

        # 集中度惩罚
        max_weight = np.max(weights_np)
        if max_weight > 0.25:
            max_weight_penalty = (max_weight - 0.25) * 0.5
        elif max_weight > 0.15:
            max_weight_penalty = (max_weight - 0.15) * 0.3
        else:
            max_weight_penalty = 0

        # 稳定性奖励
        stability_reward = 0
        if total_trading_volume < 0.02:
            stability_reward = 0.02
        elif total_trading_volume < 0.05:
            stability_reward = 0.01

        # 总奖励
        total_reward = (
                base_reward
                - transaction_cost
                + diversification_reward
                - risk_penalty
                + stability_reward
                - max_weight_penalty
        )

        return np.clip(total_reward, -2.0, 2.0)

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """执行动作"""
        if action.dim() > 1:
            action = action.squeeze(0)

        if action.shape[0] != self.n_assets:
            raise ValueError(f"Action dimension {action.shape[0]} doesn't match n_assets {self.n_assets}")

        action = torch.softmax(action, dim=0)

        if self.current_step >= self.n_steps - 1:
            self.done = True
            return self.get_state(), 0.0, self.done, {"portfolio_value": self.portfolio_value}

        # 计算投资组合收益
        next_day_returns = self.returns_data[self.current_step]
        portfolio_return = torch.sum(action * torch.tensor(next_day_returns, dtype=torch.float32)).item()

        # 计算奖励
        reward = self._calculate_reward(portfolio_return, action)

        # 更新状态
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_value = np.clip(self.portfolio_value, 0.1, 10.0)

        self.prev_weights = action.detach().numpy()
        self.historical_returns.append(portfolio_return)
        self.historical_weights.append(self.prev_weights.copy())

        if len(self.historical_returns) > 100:
            self.historical_returns = self.historical_returns[-100:]
            self.historical_weights = self.historical_weights[-100:]

        self.current_step += 1
        self.done = self.current_step >= self.n_steps - 1

        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "diversification_score": 1 - np.sum(self.prev_weights ** 2),
            "trading_volume": torch.sum(torch.abs(action - torch.tensor(
                self.historical_weights[-2] if len(self.historical_weights) > 1 else self.prev_weights))).item(),
            "max_weight": np.max(self.prev_weights),
            "effective_assets": np.sum(self.prev_weights > 0.01)
        }

        return self.get_state(), reward, self.done, info

    def get_episode_metrics(self) -> Dict[str, float]:
        """计算回测指标"""
        if len(self.historical_returns) < 2:
            return {}

        returns = np.array(self.historical_returns)
        total_return = (self.portfolio_value - self.episode_start_value) / self.episode_start_value
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1

        excess_returns = returns - 0.02 / 252
        sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)

        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        weights_history = np.array(self.historical_weights)
        avg_hhi = np.mean([np.sum(w ** 2) for w in weights_history])
        avg_diversification = 1 - avg_hhi

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": np.std(returns) * np.sqrt(252),
            "avg_diversification": avg_diversification,
            "avg_max_weight": np.mean([np.max(w) for w in weights_history]),
            "avg_effective_assets": np.mean([np.sum(w > 0.01) for w in weights_history]),
        }



    # ====================== 调试辅助函数（可选添加）======================
    def get_reward_breakdown(self, action, portfolio_return, net_return):
        """
        返回奖励函数各组成部分的详细信息，便于调试
        """
        breakdown = {}

        # 基础收益
        if net_return > -0.99:
            breakdown['log_return'] = torch.log(1 + net_return).item()
        else:
            breakdown['log_return'] = -10.0

        # 交易成本
        weight_changes = torch.abs(action - torch.tensor(self.prev_weights, dtype=torch.float32))
        breakdown['trading_volume'] = torch.sum(weight_changes).item()

        # 多元化分数
        weights_np = action.detach().numpy()
        weight_entropy = -np.sum(weights_np * np.log(weights_np + 1e-8))
        max_entropy = np.log(len(weights_np))  # 对于config.n_assets个资产，最大熵为ln(config.n_assets)
        breakdown['diversification_score'] = weight_entropy / max_entropy

        # 风险指标
        if len(self.historical_returns) > 10:
            recent_returns = np.array(self.historical_returns[-20:])
            breakdown['volatility'] = np.std(recent_returns)
            breakdown['mean_return'] = np.mean(recent_returns)

        # 新增：config.n_assets资产相关指标
        breakdown['max_weight'] = np.max(weights_np)
        breakdown['top_5_concentration'] = np.sum(np.sort(weights_np)[-5:])
        breakdown['effective_assets'] = np.sum(weights_np > 0.01)

        return breakdown

    def calculate_episode_alpha(self):
        """计算一个回合的 Alpha"""
        # 计算策略收益
        strategy_return = (self.portfolio_value - self.episode_start_value) / self.episode_start_value

        # 计算基准收益，这里使用所有资产平均收益作为示例
        episode_returns = self.returns_data[self.episode_start_step:self.current_step]
        benchmark_return = np.mean(episode_returns)

        # 计算 Alpha
        alpha = strategy_return - benchmark_return
        return alpha

    def get_weight_statistics(self):
        """获取权重分布统计信息 - config.n_assets资产专用"""
        if len(self.historical_weights) == 0:
            return {}

        current_weights = self.prev_weights

        return {
            "max_weight": np.max(current_weights),
            "min_weight": np.min(current_weights),
            "weight_std": np.std(current_weights),
            "herfindahl_index": np.sum(current_weights ** 2),
            "top_3_sum": np.sum(np.sort(current_weights)[-3:]),
            "top_5_sum": np.sum(np.sort(current_weights)[-5:]),
            "top_10_sum": np.sum(np.sort(current_weights)[-10:]),
            "effective_assets_1pct": np.sum(current_weights > 0.01),
            "effective_assets_2pct": np.sum(current_weights > 0.02),
            "effective_assets_5pct": np.sum(current_weights > 0.05),
            "entropy": -np.sum(current_weights * np.log(current_weights + 1e-8)),
            "normalized_entropy": -np.sum(current_weights * np.log(current_weights + 1e-8)) / np.log(config.n_assets)
        }

