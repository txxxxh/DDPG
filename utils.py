# training/utils.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from config import config

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




def verify_dimensions(env, expected_assets=config.n_assets, expected_factors=config.n_factors):
    """
    验证环境和网络的维度是否匹配 - 支持动态因子和资产数量
    """
    print("=== 系统维度验证 ===")
    print(f"期望配置: {expected_assets}个资产, {expected_factors}个因子")
    print(f"实际环境: {env.n_assets}个资产")

    # 检查资产数量
    if hasattr(env, 'n_assets') and env.n_assets != expected_assets:
        print(f"⚠️ 警告：资产数量不匹配！期望{expected_assets}，实际{env.n_assets}")

    # 检查因子数量（通过特征维度推断）
    if hasattr(env, 'config.feature_dim'):
        expected_feature_dim = expected_factors

        print(
            f"特征维度: {env.feature_dim} (期望: 因子{expected_factors} = {expected_feature_dim})")

        if abs(env.feature_dim - expected_feature_dim) > 0:  # 允许小幅偏差
            print(f"⚠️ 警告：特征维度可能不匹配！")
            print("请检查环境中的因子数量配置")

    print(f"时间窗口: {getattr(env, 'window_size', 'N/A')}")

    # 检查必要的属性
    required_attrs = [
        'config.feature_dim', 'feature_dim',
        'PREV_RETURNS_IDX', 'WEIGHT_IDX', 'FACTORS_START_IDX'
    ]
    for attr in required_attrs:
        if hasattr(env, attr):
            value = getattr(env, attr)
            print(f"✓ {attr}: {value}")

            # 特殊检查因子相关索引
            if attr == 'FACTORS_START_IDX' and isinstance(value, int):
                factors_end_idx = value + expected_factors
                if hasattr(env, 'config.feature_dim') and factors_end_idx > env.feature_dim:
                    print(f"⚠️ 警告：因子索引范围({value}~{factors_end_idx})超出特征维度({env.feature_dim})")
        else:
            print(f"✗ 缺少属性: {attr}")

    # 测试状态形状
    try:
        state = env.get_state()
        print(f"状态形状: {state.shape}")
        expected_shape = (
        env.n_assets, getattr(env, 'window_size', config.lookback_window), getattr(env, 'config.feature_dim', expected_factors + 3))
        print(f"期望形状: {expected_shape}")

        # 内存使用检查
        if hasattr(state, 'numel'):
            state_memory = state.numel() * 4 / (1024 ** 2)  # MB (假设float32)
            print(f"单个状态内存使用: {state_memory:.2f} MB")

            if state_memory > 20:  # 大于20MB
                print("⚠️ 警告：状态空间较大，考虑优化内存使用")

        print("✓ 状态维度验证通过！")

    except Exception as e:
        print(f"✗ 状态维度验证失败: {e}")

    print("=== 验证完成 ===\n")

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

###添加可视化

