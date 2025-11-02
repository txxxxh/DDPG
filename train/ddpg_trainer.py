# training/ddpg_trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import os
import time
import matplotlib.pyplot as plt
from config import config, Config
from training.utils import *
from utils.metrics import TrainingMetrics, EpisodeMetrics
from visualization.portfolio_visualizer import PortfolioVisualizer

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




def train_ddpg_enhanced(env, actor, critic, replay_buffer, num_episodes=100, max_steps=200,
                        fund_names=None, visualization_freq=20, save_dir="./training_results",
                        config=None, resume_from_checkpoint=None):
    """
    增强版DDPG训练函数 - 支持新版ImprovedActor的所有功能

    主要改进：
    1. 集成EnhancedTrainingManager进行智能训练管理
    2. 支持动态探索策略调整
    3. 性能记忆更新和软重置机制
    4. 更详细的诊断和监控
    5. 自适应训练参数调整
    6. 支持检查点断点续训

    Args:
        resume_from_checkpoint: 检查点文件路径，如果提供则从该点继续训练
    """

    # 使用配置
    if config is None:
        config = Config()

    # 验证环境与配置的兼容性
    _validate_environment_compatibility(env, config)

    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/visualizations", exist_ok=True)
    os.makedirs(f"{save_dir}/diagnostics", exist_ok=True)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 网络初始化
    actor = actor.to(device)
    critic = critic.to(device)

    # 目标网络
    actor_target = copy.deepcopy(actor).to(device)
    critic_target = copy.deepcopy(critic).to(device)

    # 优化器
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.lr_critic)

    # ===== 检查点加载逻辑 =====
    start_episode = 0
    loaded_training_metrics = None

    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"🔄 从检查点恢复训练: {resume_from_checkpoint}")
        start_episode, loaded_training_metrics = load_checkpoint(
            resume_from_checkpoint, actor, critic, actor_optimizer, critic_optimizer
        )
        print(f"✅ 成功加载检查点，从第 {start_episode + 1} 回合继续训练")
    else:
        if resume_from_checkpoint:
            print(f"⚠️ 检查点文件不存在: {resume_from_checkpoint}")
            print("🆕 开始新的训练")

    # ===== 新增：增强训练管理器 =====
    training_manager = EnhancedTrainingManager(actor)

    # ===== 探索参数设置 =====
    exploration_params = {
        'start_rate': 0.5,
        'end_rate': 0.05,
        'decay_rate': 0.995,
        'diversity_reward_weight': 0.1,
        'concentration_threshold': 0.25,
        'force_diversify_threshold': 0.4,  # 新增：强制多样化阈值
        'performance_stagnation_threshold': 0.001  # 新增：性能停滞阈值
    }

    current_exploration_rate = exploration_params['start_rate']

    # 可视化器和数据存储
    visualizer = PortfolioVisualizer()

    # 使用加载的训练指标或创建新的
    if loaded_training_metrics is not None:
        training_metrics = loaded_training_metrics
        print(f"📊 已加载 {len(training_metrics.episodes)} 个历史回合的训练数据")
    else:
        training_metrics = TrainingMetrics()

    best_reward = -float('inf')

    # 如果从检查点恢复，计算历史最佳奖励
    if loaded_training_metrics is not None and len(loaded_training_metrics.episodes) > 0:
        historical_rewards = [ep.total_reward for ep in loaded_training_metrics.episodes]
        best_reward = max(historical_rewards)
        print(f"📈 历史最佳奖励: {best_reward:.4f}")

    # ===== 新增：增强监控变量 =====
    recent_performance_window = []
    concentration_violations = 0
    forced_diversification_count = 0
    last_step_return = None  # 用于传递给actor的portfolio_return

    # 从检查点恢复时，填充性能窗口
    if loaded_training_metrics is not None and len(loaded_training_metrics.episodes) > 0:
        # 使用最近的回合数据填充性能窗口
        recent_episodes = loaded_training_metrics.episodes[-50:] if len(
            loaded_training_metrics.episodes) >= 50 else loaded_training_metrics.episodes
        recent_performance_window = [ep.total_reward for ep in recent_episodes]
        print(f"📊 已恢复 {len(recent_performance_window)} 个回合的性能历史")

    print("开始训练增强版DDPG...")
    print(f"资产数量: {config.n_assets}")
    print(f"因子数量: {config.n_factors}")
    print(f"时间窗口: {config.lookback_window}")
    print(f"Actor特性: 多头决策 + 动态偏置 + 性能记忆")
    print(f"训练管理: 智能探索调整 + 软重置机制")
    print(f"训练范围: Episode {start_episode + 1} -> {num_episodes}")

    for episode in range(start_episode, num_episodes):
        state = env.reset()
        episode_metrics = EpisodeMetrics()

        # 验证状态维度
        _validate_state_dimensions(state, config)

        # ===== 新增：每回合开始时的策略调整 =====
        training_manager.step(episode)

        # 随机选择探索策略（保持原有逻辑）
        exploration_strategy = np.random.choice(
            ['gaussian', 'diversified', 'adaptive', 'default'],
            p=[0.3, 0.3, 0.2, 0.2]
        )

        episode_step_returns = []  # 记录本回合每步的收益

        for step in range(max_steps):
            # ===== 核心改进：使用新版Actor =====
            with torch.no_grad():
                # 传入上一步的收益更新性能记忆
                action = actor(
                    state.unsqueeze(0),
                    add_noise=True,
                    training_mode=True,
                    portfolio_return=last_step_return
                ).squeeze(0)

            # 验证动作维度
            assert len(action) == config.n_assets, f"动作维度错误: 期望{config.n_assets}, 实际{len(action)}"

            # ===== 新增：实时监控和诊断 =====
            if step % 20 == 0:  # 每20步监控一次
                stats = actor.get_enhanced_stats(action.unsqueeze(0))

                # 检查是否需要强制多样化
                if stats['hhi'] > exploration_params['force_diversify_threshold']:
                    print(f"🚨 Episode {episode + 1}, Step {step}: HHI={stats['hhi']:.4f} 超过阈值，触发强制多样化")
                    actor.force_diversify()
                    forced_diversification_count += 1

                    # 重新生成action
                    action = actor(
                        state.unsqueeze(0),
                        add_noise=True,
                        training_mode=True,
                        portfolio_return=last_step_return
                    ).squeeze(0)

                if step % 100 == 0:  # 详细诊断
                    print(f"Episode {episode + 1}, Step {step} - 增强诊断:")
                    print(f"  HHI: {stats['hhi']:.4f}, 有效资产: {stats['effective_assets']}")
                    print(f"  最大权重: {stats['max_weight']:.4f}, 温度: {stats['temperature']:.4f}")
                    print(f"  未充分使用资产: {stats['unused_assets']}")

            # 记录权重和指标
            episode_metrics.add_step(action.detach().cpu().numpy(), step)

            # 执行环境步
            next_state, reward, done, info = env.step(action)

            # ===== 新增：计算步收益率用于下一步 =====
            current_portfolio_value = info.get("portfolio_value", 0)
            if step > 0:
                # 使用 values_history 获取前一步的价值
                if len(episode_metrics.values_history) > 1:
                    previous_value = episode_metrics.values_history[-2]
                else:
                    previous_value = current_portfolio_value

                step_return = (current_portfolio_value - previous_value) / (previous_value + 1e-8)
                episode_step_returns.append(step_return)
                last_step_return = step_return
            else:
                last_step_return = 0.0
                episode_step_returns.append(0.0)

            episode_metrics.add_value(current_portfolio_value)

            # ===== 多样化奖励计算（保持原有逻辑）=====
            diversity_metrics = _calculate_diversity_metrics(action, config)
            total_reward = _calculate_total_reward(
                reward, diversity_metrics, exploration_params
            )

            # 存储经验
            replay_buffer.add(state, action.cpu(), total_reward, next_state, done)

            # 更新状态和奖励
            state = next_state
            episode_metrics.add_reward(total_reward)

            # 经验回放和网络更新
            if len(replay_buffer) >= config.batch_size:
                _update_networks(
                    actor, critic, actor_target, critic_target,
                    actor_optimizer, critic_optimizer,
                    replay_buffer, device, config
                )

            if done:
                break

        # ===== 新增：回合结束后的增强处理 =====
        episode_total_return = sum(episode_step_returns)

        # 更新训练管理器
        training_manager.step(episode, episode_total_return)

        # 更新性能监控窗口
        recent_performance_window.append(episode_total_return)
        if len(recent_performance_window) > 50:
            recent_performance_window.pop(0)

        # 参数衰减更新
        current_exploration_rate = max(
            exploration_params['end_rate'],
            current_exploration_rate * exploration_params['decay_rate']
        )

        # 记录本回合数据
        training_metrics.add_episode(episode_metrics)

        # ===== 增强版回合总结 =====
        _print_enhanced_episode_summary(
            episode, episode_metrics, current_exploration_rate,
            exploration_strategy, config, actor, forced_diversification_count
        )

        # ===== 新增：智能策略调整 =====
        if (episode + 1) % 50 == 0:
            _perform_intelligent_adjustment(
                actor, recent_performance_window, exploration_params, episode + 1
            )

        # 可视化（增强版）
        if (episode + 1) % visualization_freq == 0:
            _visualize_enhanced_episode(
                visualizer, episode_metrics, episode + 1, save_dir, actor
            )

        # 保存最佳模型
        if episode_metrics.total_reward > best_reward:
            best_reward = episode_metrics.total_reward
            _save_enhanced_models(actor, critic, best_reward, save_dir, episode + 1)

        # ===== 新增：定期保存训练状态 =====
        if (episode + 1) % 100 == 0:
            _save_training_checkpoint(
                actor, critic, actor_optimizer, critic_optimizer,
                episode, training_metrics, save_dir
            )

        # ===== 新增：自动保存最新检查点（较高频率）=====
        if (episode + 1) % 20 == 0:  # 每20回合保存一次最新状态
            _save_latest_checkpoint(
                actor, critic, actor_optimizer, critic_optimizer,
                episode, training_metrics, save_dir
            )

        print("-" * 80)

    # ===== 增强版训练结束分析 =====
    _post_training_enhanced_analysis(
        training_metrics, exploration_params, config, save_dir,
        actor, forced_diversification_count
    )

    return actor, critic



# [移动所有训练相关的辅助函数]
def load_checkpoint(checkpoint_path, actor, critic, actor_opt, critic_opt):
    """
    加载训练检查点

    Returns:
        start_episode: 继续训练的起始回合
        training_metrics: 历史训练指标
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 加载网络状态
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        actor_opt.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        critic_opt.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # 恢复Actor增强状态
        if 'actor_enhanced_state' in checkpoint:
            enhanced_state = checkpoint['actor_enhanced_state']
            actor.performance_memory.copy_(enhanced_state['performance_memory'])
            actor.exploration_bonus.copy_(enhanced_state['exploration_bonus'])
            actor.asset_usage_count.copy_(enhanced_state['asset_usage_count'])
            actor.training_step = enhanced_state['training_step']

            print(f"🧠 已恢复Actor增强状态:")
            print(f"   - 训练步数: {actor.training_step}")
            print(f"   - 性能记忆范围: [{actor.performance_memory.min():.6f}, {actor.performance_memory.max():.6f}]")
            print(f"   - 平均资产使用率: {actor.asset_usage_count.mean():.6f}")

        # 恢复训练指标
        training_metrics = checkpoint.get('training_metrics', None)
        start_episode = checkpoint['episode']

        return start_episode, training_metrics

    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        raise e



def _save_latest_checkpoint(actor, critic, actor_opt, critic_opt, episode, metrics, save_dir):
    """保存最新检查点（每20回合，用于意外中断恢复）"""
    checkpoint = {
        'episode': episode,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': actor_opt.state_dict(),
        'critic_optimizer_state_dict': critic_opt.state_dict(),
        'training_metrics': metrics,
        # Actor增强状态
        'actor_enhanced_state': {
            'performance_memory': actor.performance_memory.clone(),
            'exploration_bonus': actor.exploration_bonus.clone(),
            'asset_usage_count': actor.asset_usage_count.clone(),
            'training_step': actor.training_step
        },
        'checkpoint_info': {
            'save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint_type': 'latest'
        }
    }

    # 保存为latest，方便自动恢复
    latest_path = f"{save_dir}/latest_checkpoint.pth"
    torch.save(checkpoint, latest_path)

    # 同时保存一个带回合数的副本
    backup_path = f"{save_dir}/checkpoint_latest_episode_{episode + 1}.pth"
    torch.save(checkpoint, backup_path)

    if (episode + 1) % 100 == 0:  # 只在里程碑时打印
        print(f"💾 更新最新检查点: Episode {episode + 1}")


def auto_find_latest_checkpoint(save_dir):
    """自动查找最新的检查点文件"""
    latest_path = f"{save_dir}/latest_checkpoint.pth"

    if os.path.exists(latest_path):
        return latest_path

    # 如果latest不存在，查找最新的编号检查点
    import glob
    checkpoint_files = glob.glob(f"{save_dir}/checkpoint_episode_*.pth")

    if checkpoint_files:
        # 按文件修改时间排序，取最新的
        latest_file = max(checkpoint_files, key=os.path.getmtime)
        return latest_file

    return None


def _print_enhanced_episode_summary(episode, episode_metrics, exploration_rate,
                                    strategy, config, actor, forced_div_count):

    final_weights = episode_metrics.weights_history[-1] if episode_metrics.weights_history else None

    if final_weights is not None:
        hhi = np.sum(final_weights ** 2)
        max_weight = np.max(final_weights)
        effective_assets = np.sum(final_weights > 0.05)
        entropy = -np.sum(final_weights * np.log(final_weights + 1e-8))
        max_entropy = np.log(len(final_weights))

        print(f"回合 {episode + 1:4d} | 奖励: {episode_metrics.total_reward:8.4f} | "
              f"组合价值: {episode_metrics.final_value:10.2f}")
        print(f"         | HHI: {hhi:.4f} | 最大权重: {max_weight:.4f} | "
              f"有效资产: {effective_assets:2d}/{config.n_assets}")
        print(f"         | 熵: {entropy / max_entropy:.4f} | 探索率: {exploration_rate:.4f} | "
              f"策略: {strategy}")
        print(f"         | 强制多样化次数: {forced_div_count} | "
              f"温度: {actor.base_temperature.item():.4f}")

        # 风险提示
        if hhi > 0.3:
            print(f"         | ⚠️  严重集中风险！")
        elif effective_assets < config.n_assets * 0.3:
            print(f"         | ⚠️  多样化不足")
        else:
            print(f"         | ✅ 风险分散良好")


def _perform_intelligent_adjustment(actor, performance_window, exploration_params, episode):
    """智能策略调整"""
    if len(performance_window) < 30:
        return

    recent_performance = np.array(performance_window[-30:])
    performance_std = np.std(recent_performance)
    performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]

    print(f"\n=== Episode {episode} 智能调整分析 ===")
    print(f"最近30回合表现标准差: {performance_std:.6f}")
    print(f"表现趋势斜率: {performance_trend:.6f}")

    # 如果表现停滞且趋势向下
    if performance_std < exploration_params['performance_stagnation_threshold'] and performance_trend < 0:
        print("🔄 检测到性能停滞，启动增强探索模式")
        actor.set_exploration_mode(high_exploration=True)

        # 额外的软重置
        if performance_trend < -0.001:
            print("🔄 趋势明显下降，执行软重置")
            actor.force_diversify()

    # 如果表现稳定向好
    elif performance_std > 0.01 and performance_trend > 0:
        print("📈 表现良好，切换到精细调优模式")
        actor.set_exploration_mode(high_exploration=False)

    print("=" * 40)


def _visualize_enhanced_episode(visualizer, episode_metrics, episode, save_dir, actor):
    """增强版可视化"""
    # 原有可视化
    _visualize_episode(visualizer, episode_metrics, episode, save_dir)

    # 新增：Actor内部状态可视化
    if hasattr(actor, 'asset_usage_count'):
        plt.figure(figsize=(12, 8))

        # 子图1：资产使用频率
        plt.subplot(2, 2, 1)
        usage_counts = actor.asset_usage_count.detach().cpu().numpy()
        plt.bar(range(len(usage_counts)), usage_counts)
        plt.title('资产使用频率')
        plt.xlabel('资产ID')
        plt.ylabel('使用频率')

        # 子图2：性能记忆
        plt.subplot(2, 2, 2)
        perf_memory = actor.performance_memory.detach().cpu().numpy()
        plt.bar(range(len(perf_memory)), perf_memory)
        plt.title('资产性能记忆')
        plt.xlabel('资产ID')
        plt.ylabel('性能记忆值')

        # 子图3：探索奖励
        plt.subplot(2, 2, 3)
        exploration_bonus = actor.exploration_bonus.detach().cpu().numpy()
        plt.bar(range(len(exploration_bonus)), exploration_bonus)
        plt.title('探索奖励')
        plt.xlabel('资产ID')
        plt.ylabel('奖励值')

        # 子图4：最终权重分布
        plt.subplot(2, 2, 4)
        if episode_metrics.weights_history:
            final_weights = episode_metrics.weights_history[-1]
            plt.bar(range(len(final_weights)), final_weights)
            plt.title(f'Episode {episode} 最终权重分布')
            plt.xlabel('资产ID')
            plt.ylabel('权重')
            plt.axhline(y=1 / len(final_weights), color='r', linestyle='--', alpha=0.5, label='均匀分布')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{save_dir}/diagnostics/actor_states_episode_{episode}.png", dpi=300, bbox_inches='tight')
        plt.close()


def _save_enhanced_models(actor, critic, best_reward, save_dir, episode):
    """增强版模型保存"""
    # 原有保存逻辑
    _save_best_models(actor, critic, best_reward, save_dir)

    # 新增：保存Actor的内部状态
    actor_state = {
        'model_state_dict': actor.state_dict(),
        'performance_memory': actor.performance_memory.clone(),
        'exploration_bonus': actor.exploration_bonus.clone(),
        'asset_usage_count': actor.asset_usage_count.clone(),
        'base_temperature': actor.base_temperature.clone(),
        'training_step': actor.training_step,
        'episode': episode,
        'best_reward': best_reward
    }

    torch.save(actor_state, f"{save_dir}/best_actor_enhanced.pth")
    print(f"💾 已保存增强Actor状态 (Episode {episode}, Reward: {best_reward:.4f})")


def _save_training_checkpoint(actor, critic, actor_opt, critic_opt, episode, metrics, save_dir):
    """保存训练检查点"""
    checkpoint = {
        'episode': episode,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': actor_opt.state_dict(),
        'critic_optimizer_state_dict': critic_opt.state_dict(),
        'training_metrics': metrics,
        # Actor增强状态
        'actor_enhanced_state': {
            'performance_memory': actor.performance_memory.clone(),
            'exploration_bonus': actor.exploration_bonus.clone(),
            'asset_usage_count': actor.asset_usage_count.clone(),
            'training_step': actor.training_step
        }
    }

    torch.save(checkpoint, f"{save_dir}/checkpoint_episode_{episode}.pth")
    print(f"💾 保存训练检查点: Episode {episode}")


def _post_training_enhanced_analysis(training_metrics, exploration_params, config,
                                     save_dir, actor, forced_div_count):
    """增强版训练后分析"""
    # 原有分析
    _post_training_analysis(training_metrics, exploration_params, config, save_dir)

    # 新增：Actor性能分析报告
    print("\n" + "=" * 60)
    print("🔍 增强Actor性能分析报告")
    print("=" * 60)

    print(f"📊 训练统计:")
    print(f"  - 总训练步数: {actor.training_step}")
    print(f"  - 强制多样化次数: {forced_div_count}")
    print(f"  - 最终温度参数: {actor.base_temperature.item():.4f}")

    # 资产使用分析
    usage_counts = actor.asset_usage_count.detach().cpu().numpy()
    print(f"\n📈 资产使用分析:")
    print(f"  - 最常用资产使用率: {usage_counts.max():.4f}")
    print(f"  - 最少用资产使用率: {usage_counts.min():.4f}")
    print(f"  - 使用率标准差: {usage_counts.std():.4f}")
    print(f"  - 未充分使用资产数 (<0.01): {(usage_counts < 0.01).sum()}")

    # 性能记忆分析
    perf_memory = actor.performance_memory.detach().cpu().numpy()
    print(f"\n🧠 性能记忆分析:")
    print(f"  - 最优表现资产记忆值: {perf_memory.max():.6f}")
    print(f"  - 最差表现资产记忆值: {perf_memory.min():.6f}")
    print(f"  - 记忆值分布范围: {perf_memory.max() - perf_memory.min():.6f}")

    # 保存详细分析报告
    analysis_report = {
        'training_steps': actor.training_step,
        'forced_diversifications': forced_div_count,
        'final_temperature': actor.base_temperature.item(),
        'asset_usage_stats': {
            'usage_counts': usage_counts.tolist(),
            'max_usage': float(usage_counts.max()),
            'min_usage': float(usage_counts.min()),
            'std_usage': float(usage_counts.std()),
            'underused_assets': int((usage_counts < 0.01).sum())
        },
        'performance_memory_stats': {
            'memory_values': perf_memory.tolist(),
            'max_memory': float(perf_memory.max()),
            'min_memory': float(perf_memory.min()),
            'memory_range': float(perf_memory.max() - perf_memory.min())
        }
    }

    import json
    with open(f"{save_dir}/enhanced_analysis_report.json", 'w') as f:
        json.dump(analysis_report, f, indent=2)

    print(f"\n💾 详细分析报告已保存到: {save_dir}/enhanced_analysis_report.json")
    print("=" * 60)



# =======================================辅助函数=========================================

def _validate_environment_compatibility(env, config):
    """验证环境与配置的兼容性"""
    if hasattr(env, 'n_assets') and env.n_assets != config.n_assets:
        raise ValueError(f"环境资产数量({env.n_assets})与配置不匹配({config.n_assets})")

    if hasattr(env, 'feature_dim') and env.feature_dim != config.n_factors:
        print(f"警告: 环境特征维度({env.feature_dim})与配置因子数量({config.n_factors})不匹配")

def _validate_state_dimensions(state, config):
    """验证状态维度"""
    expected_shape = (config.n_assets, config.lookback_window, config.n_factors)
    if hasattr(state, 'shape'):
        if state.shape != expected_shape:
            print(f"警告: 状态维度{state.shape}与期望维度{expected_shape}不匹配")


def _calculate_diversity_metrics(action, config):
    """计算多样化指标"""
    weights_np = action.detach().cpu().numpy()

    # Herfindahl指数
    herfindahl_index = np.sum(weights_np ** 2)
    diversity_score_1 = 1.0 - herfindahl_index

    # 有效资产数量
    effective_assets = 1.0 / herfindahl_index
    normalized_effective_assets = effective_assets / config.n_assets

    # 综合多样化分数
    diversity_score = 0.7 * diversity_score_1 + 0.3 * normalized_effective_assets

    return {
        'herfindahl_index': herfindahl_index,
        'diversity_score': diversity_score,
        'effective_assets': effective_assets,
        'max_weight': np.max(weights_np)
    }

def _calculate_total_reward(base_reward, diversity_metrics, exploration_params):
    """计算总奖励"""
    # 多样化奖励
    diversity_reward = exploration_params['diversity_reward_weight'] * diversity_metrics['diversity_score']

    # 集中度惩罚
    concentration_penalty = 0
    max_weight = diversity_metrics['max_weight']
    if max_weight > exploration_params['concentration_threshold']:
        concentration_penalty = -0.1 * (max_weight - exploration_params['concentration_threshold']) / \
                               (1 - exploration_params['concentration_threshold'])

    return base_reward + diversity_reward + concentration_penalty

def _update_networks(actor, critic, actor_target, critic_target,
                    actor_optimizer, critic_optimizer, replay_buffer, device, config):
    """更新网络参数"""
    states, actions, rewards, next_states, dones = replay_buffer.sample(config.batch_size)

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # Critic更新
    with torch.no_grad():
        next_actions = actor_target(next_states)
        if next_actions.dim() > 2:
            next_actions = next_actions.squeeze(1)
        next_q_values = critic_target(next_states, next_actions)
        target_q_values = rewards + (1 - dones) * config.gamma * next_q_values

    current_q_values = critic(states, actions)
    critic_loss = F.mse_loss(current_q_values, target_q_values.detach())

    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), config.grad_clip_norm)
    critic_optimizer.step()

    # Actor更新
    actor_loss = -critic(states, actor(states)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip_norm)
    actor_optimizer.step()

    # 软更新目标网络
    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)

    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)


def _visualize_episode(visualizer, episode_metrics, episode_num, save_dir):
    """生成回合可视化"""
    visualizer.visualize_episode_allocation(
        episode_metrics.weights_history,
        episode_metrics.values_history,
        episode_num,
        save_path=f"{save_dir}/visualizations"
    )

def _save_best_models(actor, critic, best_reward, save_dir):
    """保存最佳模型"""
    torch.save(actor.state_dict(), f'{save_dir}/best_actor_improved.pth')
    torch.save(critic.state_dict(), f'{save_dir}/best_critic_improved.pth')
    print(f"  ✓ 保存最佳模型，奖励: {best_reward:.4f}")

def _post_training_analysis(training_metrics, exploration_params, config, save_dir):
    """训练后分析和可视化"""
    print("\n=== 训练完成分析 ===")
    final_stats = training_metrics.get_final_stats()

    print(f"最终探索率: {exploration_params['end_rate']:.3f}")
    print(f"平均多样化程度: {final_stats['avg_diversity']:.3f}")
    print(f"多样化程度标准差: {final_stats['diversity_std']:.3f}")
    print(f"实际平均有效资产数: {final_stats['avg_effective_assets']:.1f}")

    # 生成详细可视化
    _generate_training_visualizations(training_metrics, exploration_params, config, save_dir)

def _generate_training_visualizations(training_metrics, exploration_params, config, save_dir):
    """生成训练可视化图表"""
    plt.figure(figsize=(20, 12))

    # 创建6个子图
    plots_config = [
        {'data': training_metrics.diversity_history, 'title': '投资组合多样化程度变化',
         'ylabel': '多样化分数 (0-1)', 'subplot': (2, 3, 1)},

        {'data': training_metrics.effective_assets_history, 'title': '有效资产数量变化',
         'ylabel': '有效资产数', 'subplot': (2, 3, 2),
         'hline': {'y': config.n_assets, 'label': '理论最大值'}},

        {'data': training_metrics.exploration_rates, 'title': '探索率变化',
         'ylabel': '探索率', 'subplot': (2, 3, 3)},

        {'data': training_metrics.get_final_weights(), 'title': '最终权重分布',
         'ylabel': '权重', 'subplot': (2, 3, 4), 'plot_type': 'bar'},

        {'data': training_metrics.max_weights_history, 'title': '最大单资产权重变化',
         'ylabel': '最大权重', 'subplot': (2, 3, 5),
         'hline': {'y': config.max_weight, 'label': '最大权重限制'}},

        {'data': training_metrics.min_weights_history, 'title': '最小单资产权重变化',
         'ylabel': '最小权重', 'subplot': (2, 3, 6),
         'hline': {'y': config.min_weight, 'label': '最小权重限制'}}
    ]

    for plot_config in plots_config:
        plt.subplot(*plot_config['subplot'])

        if plot_config.get('plot_type') == 'bar':
            plt.bar(range(len(plot_config['data'])), plot_config['data'])
            plt.xlabel('资产编号')
        else:
            plt.plot(plot_config['data'])
            plt.xlabel('Episode')

        plt.title(plot_config['title'])
        plt.ylabel(plot_config['ylabel'])
        plt.grid(True)

        if 'hline' in plot_config:
            hline = plot_config['hline']
            plt.axhline(y=hline['y'], color='r', linestyle='--', label=hline['label'])
            plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/exploration_analysis_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
