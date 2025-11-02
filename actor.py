# models/actor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import config
from .components import NonStationaryTransformer, AssetGCN

class Actor(nn.Module):
    def __init__(self, env, num_heads=4, min_weight=0.01, max_weight=0.15, temperature_start=1.0):
        super().__init__()
        self.env = env
        self.n_assets = env.n_assets
        self.window_size = env.window_size
        self.feature_dim = env.feature_dim
        self.min_weight = min_weight
        self.max_weight = max_weight

        # 确保num_heads能被feature_dim整除
        if self.feature_dim % num_heads != 0:
            for h in [8, 6, 4, 3, 2, 1]:
                if self.feature_dim % h == 0:
                    num_heads = h
                    break
            print(f"警告：Actor调整num_heads为{num_heads}以适应feature_dim={self.feature_dim}")

        # 时间序列模块
        self.nst_modules = nn.ModuleList([
            NonStationaryTransformer(self.feature_dim, num_heads=num_heads)
            for _ in range(self.n_assets)
        ])

        # 资产依赖模块
        self.gcn = AssetGCN(self.feature_dim, hidden_dim=64, window_size=self.window_size, n_assets=self.n_assets)

        # 特征融合层
        fusion_input_dim = self.feature_dim + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # 资产注意力层
        self.asset_attention = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 决策网络 - 增加多头输出
        self.decision_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # 多头输出 - 关键改进！
        self.policy_head = nn.Linear(128, self.n_assets)  # 主策略头
        self.exploration_head = nn.Linear(128, self.n_assets)  # 探索策略头

        # 策略混合权重
        self.policy_mixer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


        # 层标准化
        self.layer_norm = nn.LayerNorm(128)

        # 动态权重偏置系统 - 关键改进！
        self.register_buffer('performance_memory', torch.zeros(self.n_assets))
        self.register_buffer('exploration_bonus', torch.ones(self.n_assets))
        self.memory_decay = 0.95
        self.exploration_decay = 0.99

        # 可学习的温度参数 - 改为自适应
        self.base_temperature = nn.Parameter(torch.tensor(temperature_start))
        self.adaptive_temp_scale = nn.Parameter(torch.tensor(0.1))

        # 集中度监控和惩罚
        self.concentration_threshold = 0.3  # HHI阈值
        self.concentration_penalty = 0.1

        # 多样性奖励机制
        self.diversity_bonus_scale = 0.05
        self.register_buffer('asset_usage_count', torch.zeros(self.n_assets))

        # 周期性重置机制
        self.reset_period = 500  # 每500步检查一次
        self.performance_window = 50  # 性能监控窗口
        self.register_buffer('recent_performance', torch.zeros(self.performance_window))
        self.performance_idx = 0

        # 初始化权重
        self._initialize_weights()

        # 监控统计
        self.training_step = 0
        self.last_hhi = 0.0

    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)  # 稍微小一点的gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 特别初始化输出层
        with torch.no_grad():
            # 主策略头：偏向均匀分布
            nn.init.xavier_uniform_(self.policy_head.weight, gain=0.1)
            nn.init.constant_(self.policy_head.bias, 0)

            # 探索策略头：更大的随机性
            nn.init.xavier_uniform_(self.exploration_head.weight, gain=0.3)
            nn.init.constant_(self.exploration_head.bias, 0)

    def forward(self, state, add_noise=True, training_mode=None, portfolio_return=None):
        """
        增强版前向传播
        portfolio_return: 用于更新性能记忆
        """
        if training_mode is not None:
            self.training = training_mode

        batch_size, n_assets, window_size, feature_dim = state.shape
        self.training_step += 1

        # 更新性能记忆
        if portfolio_return is not None and self.training:
            self._update_performance_memory(portfolio_return)

        # 检查是否需要重置
        if self.training_step % self.reset_period == 0:
            self._check_and_reset()

        # 1. 特征提取（与原始代码相同）
        temporal_features = []
        for i in range(self.n_assets):
            asset_data = state[:, i, :, :]
            asset_data = asset_data.permute(1, 0, 2)
            asset_feat = self.nst_modules[i](asset_data)
            asset_feat = asset_feat.permute(1, 0, 2)
            temporal_features.append(asset_feat.mean(dim=1))

        temporal_features = torch.stack(temporal_features, dim=1)

        # 2. 资产依赖特征提取
        x_reshaped = state.view(-1, window_size, feature_dim)
        dependency_features = self.gcn(x_reshaped)
        dependency_features = dependency_features.view(batch_size, n_assets, -1)

        # 3. 特征融合
        fused = torch.cat([temporal_features, dependency_features], dim=2)
        asset_attention = self.asset_attention(fused)
        asset_attention = torch.softmax(asset_attention, dim=1)
        global_features = torch.sum(fused * asset_attention, dim=1)

        fused = self.fusion(global_features)
        fused = self.layer_norm(fused)

        # 4. 多头决策 - 关键改进！
        shared_features = self.decision_net(fused)

        # 主策略logits
        main_logits = self.policy_head(shared_features)

        # 探索策略logits（更随机）
        explore_logits = self.exploration_head(shared_features)

        # 动态混合权重
        mix_weight = self.policy_mixer(shared_features)

        # 根据集中度动态调整混合比例
        current_concentration = self._estimate_concentration(main_logits)
        if current_concentration > self.concentration_threshold:
            # 如果过于集中，增加探索权重
            mix_weight = mix_weight * 0.3  # 降低主策略权重

        # 混合两个策略
        combined_logits = mix_weight * main_logits + (1 - mix_weight) * explore_logits

        # 5. 添加动态偏置 - 关键改进！
        dynamic_bias = self._compute_dynamic_bias()
        final_logits = combined_logits + dynamic_bias.unsqueeze(0)

        # 6. 应用约束获得权重
        weights = self._enhanced_constrained_softmax(final_logits, add_noise)

        # 7. 更新使用统计
        if self.training:
            self._update_usage_stats(weights)

        # 8. 监控和诊断
        if self.training_step % 100 == 0:
            self._enhanced_diagnose(weights, final_logits, self.training_step)

        return weights


    def _compute_dynamic_bias(self):
        """计算动态偏置 - 核心反过早收敛机制"""
        # 1. 性能记忆偏置：表现差的资产获得探索奖励
        performance_bias = -self.performance_memory * 0.1

        # 2. 探索奖励：使用少的资产获得奖励
        exploration_bias = torch.log(self.exploration_bonus + 1e-8) * self.diversity_bonus_scale

        # 3. 反集中偏置：权重过高的资产被惩罚
        usage_penalty = -torch.log(self.asset_usage_count + 1.0) * 0.02

        # 4. 随机探索偏置
        random_bias = torch.randn(self.n_assets, device=self.performance_memory.device) * 0.01

        total_bias = performance_bias + exploration_bias + usage_penalty + random_bias
        return total_bias

    def _enhanced_constrained_softmax(self, logits, add_noise=True):
        """增强版约束softmax"""
        # 1. 自适应温度
        concentration = self._estimate_concentration(logits)
        if concentration > self.concentration_threshold:
            # 如果过于集中，提高温度增加随机性
            temperature = self.base_temperature + self.adaptive_temp_scale * (
                        concentration - self.concentration_threshold) * 10
        else:
            temperature = self.base_temperature

        temperature = torch.clamp(temperature, min=0.3, max=3.0)
        scaled_logits = logits / temperature

        # 2. 数值稳定的softmax
        weights = self._stable_softmax(scaled_logits)

        # 3. 集中度惩罚
        weights = self._apply_concentration_penalty(weights)

        # 4. 多样性奖励
        if self.training:
            weights = self._apply_diversity_bonus(weights)

        # 5. 添加探索噪声
        if add_noise and self.training:
            weights = self._enhanced_exploration_noise(weights)

        return weights

    def _estimate_concentration(self, logits):
        """估计当前集中度"""
        with torch.no_grad():
            weights = F.softmax(logits, dim=1)
            hhi = (weights ** 2).sum(dim=1).mean()
            return hhi.item()

    def _apply_concentration_penalty(self, weights):
        """应用集中度惩罚"""
        hhi = (weights ** 2).sum(dim=1, keepdim=True)
        penalty_mask = (hhi > self.concentration_threshold).float()

        if penalty_mask.sum() > 0:
            # 对过于集中的组合进行软性均匀化
            uniform_weights = torch.ones_like(weights) / self.n_assets
            penalty_strength = (hhi - self.concentration_threshold) * self.concentration_penalty
            penalty_strength = torch.clamp(penalty_strength, 0, 0.3)

            weights = weights * (1 - penalty_strength) + uniform_weights * penalty_strength
            # 重新标准化
            weights = weights / weights.sum(dim=1, keepdim=True)

        return weights

    def _apply_diversity_bonus(self, weights):
        """应用多样性奖励"""
        # 给使用较少的资产额外的权重提升
        underused_bonus = (1.0 / (self.asset_usage_count + 1.0)) * self.diversity_bonus_scale
        underused_bonus = underused_bonus / underused_bonus.sum()  # 标准化

        # 软性混合
        mix_ratio = 0.95
        enhanced_weights = mix_ratio * weights + (1 - mix_ratio) * underused_bonus.unsqueeze(0)

        return enhanced_weights / enhanced_weights.sum(dim=1, keepdim=True)

    def _enhanced_exploration_noise(self, weights):
        """增强版探索噪声"""
        if not self.training:
            return weights

        # 1. 基础Dirichlet噪声
        alpha = weights * self.n_assets * 2.0 + 0.1
        alpha = torch.clamp(alpha, min=0.1, max=10.0)

        try:
            gamma_samples = torch.distributions.Gamma(alpha, 1.0).sample()
            dirichlet_noise = gamma_samples / gamma_samples.sum(dim=1, keepdim=True)
        except:
            # 如果采样失败，使用简单的均匀噪声
            dirichlet_noise = torch.ones_like(weights) / self.n_assets

        # 2. 反集中噪声：对集中度高的组合增加更多噪声
        hhi = (weights ** 2).sum(dim=1, keepdim=True)
        noise_strength = torch.clamp((hhi - 0.1) * 0.5, 0.05, 0.3)

        # 3. 混合
        noisy_weights = (1 - noise_strength) * weights + noise_strength * dirichlet_noise

        return noisy_weights

    def _update_performance_memory(self, portfolio_return):
        """更新性能记忆"""
        if portfolio_return is not None:
            # 更新最近性能记录
            self.recent_performance[self.performance_idx] = portfolio_return
            self.performance_idx = (self.performance_idx + 1) % self.performance_window

            # 简单的性能归因（这里可以改进）
            # 假设表现好时所有当前权重的资产都获得正面记忆
            current_weights = getattr(self, '_last_weights', torch.ones(self.n_assets) / self.n_assets)

            # 指数移动平均更新
            performance_signal = portfolio_return * current_weights
            self.performance_memory = (self.memory_decay * self.performance_memory +
                                       (1 - self.memory_decay) * performance_signal)

    def _update_usage_stats(self, weights):
        """更新使用统计"""
        # 更新资产使用计数
        usage_this_step = (weights.detach().cpu() > 0.02).float().mean(dim=0)  # 权重>2%算作使用
        self.asset_usage_count = (self.asset_usage_count * 0.999 +
                                  usage_this_step.to(self.asset_usage_count.device) * 0.001)

        # 更新探索奖励
        self.exploration_bonus *= self.exploration_decay
        unused_assets = (usage_this_step < 0.1)
        self.exploration_bonus[unused_assets] += 0.1  # 给未使用的资产奖励

        # 记录最后权重用于性能归因
        self._last_weights = weights[0].detach().cpu()

    def _check_and_reset(self):
        """检查是否需要重置"""
        if self.recent_performance.sum() != 0:  # 确保有数据
            recent_std = self.recent_performance.std()
            recent_mean = self.recent_performance.mean()

            # 如果最近表现停滞且集中度过高
            if recent_std < 0.001 and self.last_hhi > 0.25:
                print(f"Step {self.training_step}: 检测到过早收敛，执行软重置")
                self._soft_reset()

    def _soft_reset(self):
        """软重置：不完全重置，只调整关键参数"""
        # 1. 重置探索奖励
        self.exploration_bonus.fill_(1.0)

        # 2. 重置使用统计
        self.asset_usage_count.fill_(0.0)

        # 3. 重置性能记忆（保留一些历史）
        self.performance_memory *= 0.5

        # 4. 增加温度参数
        with torch.no_grad():
            self.base_temperature.data = torch.clamp(self.base_temperature + 0.2, 0.5, 2.0)

        # 5. 给探索头添加小的随机扰动
        with torch.no_grad():
            self.exploration_head.weight.data += torch.randn_like(self.exploration_head.weight) * 0.01

        print("软重置完成，恢复探索能力")

    def _stable_softmax(self, logits):
        """数值稳定的Softmax实现"""
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits - max_logits)
        sum_exp = torch.sum(exp_logits, dim=1, keepdim=True) + 1e-8
        weights = exp_logits / sum_exp
        return weights

    def _enhanced_diagnose(self, weights, logits, step):
        """增强版诊断"""
        with torch.no_grad():
            weights_np = weights[0].detach().cpu().numpy()
            logits_np = logits[0].detach().cpu().numpy()

            # 计算关键指标
            hhi = np.sum(weights_np ** 2)
            self.last_hhi = hhi
            entropy = -np.sum(weights_np * np.log(weights_np + 1e-8))
            max_entropy = np.log(len(weights_np))

            print(f"\n=== Step {step} Enhanced Diagnosis ===")
            print(f"HHI: {hhi:.4f} (目标: 0.05-0.15)")
            print(f"Entropy: {entropy:.4f} / {max_entropy:.4f} = {entropy / max_entropy:.4f}")
            print(f"Temperature: {self.base_temperature.item():.4f}")
            print(f"Top 3 weights: {sorted(weights_np, reverse=True)[:3]}")
            print(f"Assets > 5%: {(weights_np > 0.05).sum()}")

            # 检查集中度
            if hhi > 0.3:
                print("🚨 CRITICAL: 严重过度集中！")
            elif hhi > 0.2:
                print("⚠️ WARNING: 过度集中")
            elif hhi < 0.04:
                print("⚠️ WARNING: 过度分散")
            else:
                print("✅ GOOD: 集中度合理")

            # 显示探索状态
            unused_assets = (self.asset_usage_count < 0.01).sum()
            print(f"未充分使用的资产数: {unused_assets}")

    def get_enhanced_stats(self, weights):
        """获取增强统计信息"""
        with torch.no_grad():
            stats = {
                'max_weight': weights.max().item(),
                'min_weight': weights.min().item(),
                'std_weight': weights.std().item(),
                'effective_assets': (weights > 0.05).sum().item(),
                'hhi': (weights ** 2).sum().item(),
                'entropy': -(weights * torch.log(weights + 1e-8)).sum().item(),
                'temperature': self.base_temperature.item(),
                'unused_assets': (self.asset_usage_count < 0.01).sum().item(),
            }
            return stats

    def force_diversify(self):
        """强制多样化 - 外部调用接口"""
        print("执行强制多样化...")
        self._soft_reset()
        # 额外：临时提高探索噪声
        self.temp_high_exploration = True

    def set_exploration_mode(self, high_exploration=False):
        """设置探索模式"""
        if high_exploration:
            with torch.no_grad():
                self.base_temperature.data = torch.tensor(2.0)
            self.diversity_bonus_scale = 0.1
        else:
            with torch.no_grad():
                self.base_temperature.data = torch.tensor(1.0)
            self.diversity_bonus_scale = 0.05


# 使用示例和训练建议
class EnhancedTrainingManager:
    """增强版训练管理器"""

    def __init__(self, actor):
        self.actor = actor
        self.concentration_history = []
        self.performance_history = []

    def step(self, episode, portfolio_return=None):
        """每步调用"""
        # 记录表现
        if portfolio_return is not None:
            self.performance_history.append(portfolio_return)

        # 动态调整策略
        if episode % 100 == 0:
            self._check_and_adjust(episode)

    def _check_and_adjust(self, episode):
        """检查并调整训练策略"""
        if len(self.performance_history) < 50:
            return

        recent_perf = self.performance_history[-50:]
        perf_std = np.std(recent_perf)

        # 如果表现波动很小，可能陷入了局部最优
        if perf_std < 0.001:
            print(f"Episode {episode}: 检测到可能的过早收敛，增强探索")
            self.actor.set_exploration_mode(high_exploration=True)
        else:
            self.actor.set_exploration_mode(high_exploration=False)




