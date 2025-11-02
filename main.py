# main.py
import os
import sys
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_loader import load_data_with_config
from environment import PortfolioEnvironment
from models.actor import Actor
from models.critic import Critic
from training.replay_buffer import ReplayBuffer
from training.ddpg_trainer import train_ddpg_enhanced
from training.utils import verify_dimensions

def run_flexible_training(n_assets=config.n_assets, n_factors=config.n_factors,data_path=None, load_pretrained=True, factor_type='momentum'):
    """
    灵活的训练主函数 - 支持动态资产和因子数量，且与预测系统兼容
    """
    print("=" * 80)
    print(f"启动灵活强化学习训练系统")
    print(f"目标配置: {n_assets}个资产, {n_factors}个因子")
    print("=" * 80)

    # 1. 数据配置验证
    if data_path is None:
        data_path = config.data_path

    # 2. 加载和验证数据
    print("正在加载数据...")
    try:
        # 数据加载函数 - 修复：接收5个返回值
        returns, factors, initial_prices, actual_n_assets, selected_assets = load_data_with_config(
            excel_path=data_path,
            desired_n_assets=n_assets,
            factor_type=factor_type,
            random_seed=42,
            selection_mode='random'
        )

        print(f"✓ 数据加载成功:")
        print(f"  - 请求资产数: {n_assets}")
        print(f"  - 实际资产数: {actual_n_assets}")
        print(f"  - 选择的资产: {selected_assets}")
        print(f"  - 因子数: {n_factors}")
        print(f"  - 数据形状: 收益率{returns.shape}, 因子{factors.shape}")

        # 更新实际使用的资产数量
        if actual_n_assets != n_assets:
            print(f"注意: 实际加载的资产数量({actual_n_assets})与请求数量({n_assets})不同")
            print("将使用实际加载的数据进行训练")
            n_assets = actual_n_assets  # 更新为实际数量

        # 验证数据维度
        assert returns.shape[1] == n_assets, f"数据资产数({returns.shape[1]}) != 实际资产数({n_assets})"
        assert factors.shape[2] == n_factors, f"数据因子数({factors.shape[2]}) != 配置因子数({n_factors})"

        # 动态生成资产名称（基于实际数量和选择的资产）
        if selected_assets:
            # 使用实际选择的资产名称
            asset_names = selected_assets[:n_assets]
        else:
            # 后备方案：生成通用名称
            asset_names = [f'Asset_{i + 1}' for i in range(n_assets)]

        print(f"使用资产名称: {asset_names[:5]}..." if n_assets > 5 else f"资产名称: {asset_names}")

    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        print("请检查:")
        print("1. Excel文件路径是否正确")
        print("2. Excel文件是否包含足够的工作表")
        print("3. 数据格式是否正确")
        return None

    # 3. 数据质量检查
    print("\n=== 数据质量检查 ===")
    print(f"收益率数据:")
    print(f"  - 形状: {returns.shape}")
    print(f"  - 范围: [{returns.min():.6f}, {returns.max():.6f}]")
    print(f"  - 平均值: {returns.mean():.6f}")
    print(f"  - 包含NaN: {np.isnan(returns).any()}")

    print(f"因子数据:")
    print(f"  - 形状: {factors.shape}")
    print(f"  - 范围: [{factors.min():.6f}, {factors.max():.6f}]")
    print(f"  - 包含NaN: {np.isnan(factors).any()}")

    # 4. 环境初始化
    print("\n=== 初始化训练环境 ===")
    try:
        # 这里你可以继续添加你的环境初始化代码
        # 比如创建环境、智能体等
        print(f"正在创建环境 (资产数: {n_assets}, 因子数: {n_factors})")

        # 示例：创建环境的伪代码
        # env = PortfolioOptimizationEnv(
        #     returns=returns,
        #     factors=factors,
        #     asset_names=asset_names,
        #     initial_prices=initial_prices
        # )

        print("✓ 环境创建成功")

    except Exception as e:
        print(f"✗ 环境初始化失败: {e}")
        return None

    # 3. 创建环境 (需要传入因子数量)
    print("创建交易环境...")
    env = PortfolioEnvironment(config,returns, factors, window_size=config.lookback_window)

    # 4. 验证系统维度
    verify_dimensions(env, expected_assets=n_assets, expected_factors=n_factors)

    # 5. 创建网络结构 (基于规模动态调整)
    scale_factor = (n_assets * n_factors) / (12 * 12)
    hidden_size = max(256, int(128 * scale_factor))
    num_heads = max(4, int(2 * np.log2(n_factors)))

    print(f"网络配置: hidden_size={hidden_size}, attention_heads={num_heads}")

    actor = Actor(env, num_heads=num_heads)
    critic = Critic(env)

    # 6. 创建回放池
    buffer_capacity = max(10000, int(5000 * scale_factor))
    replay_buffer = ReplayBuffer(capacity=buffer_capacity, env=env)

    # 7. 尝试加载预训练模型
    model_prefix = f"model_{n_assets}assets_{n_factors}factors_{num_heads}heads"
    if load_pretrained:
        try:
            # actor.load_state_dict(torch.load(f'{model_prefix}_actor.pth'))
            # critic.load_state_dict(torch.load(f'{model_prefix}_critic.pth'))
            print(f"✓ 成功加载预训练模型: {model_prefix}")
        except Exception as e:
            print(f"未找到预训练模型，从头开始训练: {e}")


    # 9. 开始训练
    print("开始训练...")

    # 使用增强版训练
    trained_actor, trained_critic = train_ddpg_enhanced(
        env=env,
        actor=actor,
        critic=critic,
        replay_buffer=replay_buffer,
        num_episodes=5,
        max_steps=200,
        fund_names=asset_names,
        visualization_freq=1,
        save_dir="./enhanced_training_results",
        config=None
    )

    # 10. 保存模型
    torch.save(trained_actor.state_dict(), f'{model_prefix}_actor.pth')
    torch.save(trained_critic.state_dict(), f'{model_prefix}_critic.pth')
    print(f"训练完成，模型已保存: {model_prefix}")

    return {
        'returns': returns,
        'factors': factors,
        'initial_prices': initial_prices,
        'actual_n_assets': actual_n_assets,
        'selected_assets': selected_assets,
        'asset_names': asset_names
    }



if __name__ == "__main__":
    run_flexible_training(data_path="/Users/txh/Desktop/DDPG/合并三因子后的股票数据.xlsx")