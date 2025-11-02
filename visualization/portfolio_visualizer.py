# visualization/portfolio_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import os
from config import config

class PortfolioVisualizer:
    def __init__(self, fund_names=None):
        """
        资产配置可视化工具 - 使用全局配置
        :param fund_names: 基金名称列表，如果为None则使用默认名称
        """
        if fund_names is None:
            self.fund_names = [f'Fund_{i + 1}' for i in range(config.n_assets)]
        else:
            # 确保基金名称数量正确
            if len(fund_names) != config.n_assets:
                print(f"警告: 提供的基金名称数量({len(fund_names)})与配置不符({config.n_assets})，使用默认名称")
                self.fund_names = [f'Fund_{i + 1}' for i in range(config.n_assets)]
            else:
                self.fund_names = fund_names

        # 设置颜色方案 - 基于config.n_assets动态调整
        if config.n_assets > 20:
            # 对于大量资产，使用更丰富的颜色方案
            self.colors = plt.cm.tab20(np.linspace(0, 1, min(20, config.n_assets)))
            if config.n_assets > 20:
                additional_colors = plt.cm.Set3(np.linspace(0, 1, config.n_assets - 20))
                self.colors = np.vstack([self.colors, additional_colors])
        else:
            self.colors = plt.cm.Set3(np.linspace(0, 1, config.n_assets))

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def visualize_episode_allocation(self, episode_weights, episode_values, episode_num, save_path=None):
        """
        可视化单个回合的资产配置变化
        """
        if not episode_weights:
            print("没有权重数据可视化")
            return

        weights_array = np.array(episode_weights)

        # 验证权重数据维度
        if weights_array.shape[1] != config.n_assets:
            print(f"警告: 权重数据资产数量({weights_array.shape[1]})与配置不符({config.n_assets})")
            return

        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Episode {episode_num} - 资产配置分析 ({config.n_assets}个资产)',
                     fontsize=14, fontweight='bold')

        # 1. 堆叠面积图 - 权重变化
        ax1.stackplot(range(len(weights_array)), *weights_array.T,
                      labels=self.fund_names, colors=self.colors, alpha=0.8)
        ax1.set_title('权重配置变化 (堆叠面积图)', fontsize=12)
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('权重比例')
        # 对于大量资产，调整图例显示
        if config.n_assets > 10:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                       fontsize=8, ncol=2 if config.n_assets <= 20 else 3)
        else:
            ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # 2. 热力图 - 权重变化
        im = ax2.imshow(weights_array.T, aspect='auto', cmap='YlOrRd',
                        interpolation='nearest')
        ax2.set_title('权重热力图', fontsize=12)
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('资产')
        ax2.set_yticks(range(len(self.fund_names)))
        ax2.set_yticklabels(self.fund_names, fontsize=6 if config.n_assets > 20 else 8)
        plt.colorbar(im, ax=ax2, shrink=0.8)

        # 3. 投资组合价值变化
        ax3.plot(episode_values, 'b-', linewidth=2, marker='o', markersize=3)
        ax3.set_title('投资组合价值变化', fontsize=12)
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('组合价值')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='初始值')
        ax3.legend()

        # 4. 最终权重饼图
        final_weights = weights_array[-1]
        # 只显示权重大于阈值的资产
        significant_mask = final_weights > config.min_weight_threshold
        significant_weights = final_weights[significant_mask]
        significant_names = [self.fund_names[i] for i in range(config.n_assets) if significant_mask[i]]
        significant_colors = [self.colors[i] for i in range(min(len(self.colors), config.n_assets)) if significant_mask[i]]

        # 如果有小权重资产，合并为"其他"
        if len(significant_weights) < config.n_assets:
            other_weight = final_weights[~significant_mask].sum()
            if other_weight > 0.001:
                significant_weights = np.append(significant_weights, other_weight)
                significant_names.append('其他')
                significant_colors.append('lightgray')

        wedges, texts, autotexts = ax4.pie(significant_weights, labels=significant_names,
                                           colors=significant_colors, autopct='%1.1f%%',
                                           startangle=90)
        ax4.set_title(f'最终权重分布 (>{config.min_weight_threshold * 100:.0f}%)', fontsize=12)

        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/episode_{episode_num}_allocation.png",
                        dpi=300, bbox_inches='tight')
            print(f"Episode {episode_num} 可视化图表已保存")

        plt.show()

    def create_allocation_animation(self, all_episode_weights, save_path=None):
        """
        创建资产配置变化的动画 - 使用全局配置
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        def animate(episode):
            ax1.clear()
            ax2.clear()

            if episode < len(all_episode_weights):
                weights = np.array(all_episode_weights[episode])

                # 验证数据维度
                if weights.shape[1] != config.n_assets:
                    print(f"Episode {episode}: 权重数据维度错误")
                    return

                # 堆叠面积图
                ax1.stackplot(range(len(weights)), *weights.T,
                              labels=self.fund_names, colors=self.colors, alpha=0.8)
                ax1.set_title(f'Episode {episode + 1} - 权重配置变化 ({config.n_assets}个资产)')
                ax1.set_xlabel('时间步')
                ax1.set_ylabel('权重比例')
                ax1.set_ylim(0, 1)
                ax1.grid(True, alpha=0.3)

                # 最终权重条形图
                final_weights = weights[-1]
                bars = ax2.bar(range(config.n_assets), final_weights,
                               color=self.colors[:config.n_assets], alpha=0.8)
                ax2.set_title(f'Episode {episode + 1} - 最终权重分布')
                ax2.set_xlabel('资产')
                ax2.set_ylabel('权重')
                ax2.set_xticks(range(config.n_assets))
                ax2.set_xticklabels([f'F{i + 1}' for i in range(config.n_assets)],
                                    rotation=45 if config.n_assets > 10 else 0)
                ax2.set_ylim(0, max(0.5, final_weights.max() * 1.1))
                ax2.grid(True, alpha=0.3)

                # 添加数值标签 - 只为显著权重添加
                for bar, weight in zip(bars, final_weights):
                    if weight > config.min_weight_threshold * 2:
                        ax2.text(bar.get_x() + bar.get_width() / 2,
                                 bar.get_height() + 0.01,
                                 f'{weight:.2f}', ha='center', va='bottom',
                                 fontsize=8 if config.n_assets <= 20 else 6)

        anim = FuncAnimation(fig, animate, frames=len(all_episode_weights),
                             interval=1000, repeat=True)

        if save_path:
            anim.save(f"{save_path}/allocation_animation.gif", writer='pillow', fps=1)
            print("动画已保存为 allocation_animation.gif")

        plt.tight_layout()
        plt.show()

    def plot_training_summary(self, all_episode_weights, all_episode_values, save_path=None):
        """
        训练过程总结图表 - 使用全局配置
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'训练过程总结分析 ({config.n_assets}个资产, {config.n_factors}个因子)',
                     fontsize=16, fontweight='bold')

        # 1. 每个回合的最终投资组合价值
        final_values = [values[-1] for values in all_episode_values]
        ax1.plot(final_values, 'bo-', linewidth=2, markersize=4)
        ax1.set_title('各回合最终投资组合价值')
        ax1.set_xlabel('回合')
        ax1.set_ylabel('最终价值')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='初始值')
        ax1.legend()

        # 2. 权重多样性变化（使用Herfindahl指数）
        diversity_scores = []
        for episode_weights in all_episode_weights:
            final_weights = np.array(episode_weights[-1])
            # 验证维度
            if len(final_weights) != config.n_assets:
                print(f"权重数据维度错误: {len(final_weights)} vs {config.n_assets}")
                continue
            hhi = np.sum(final_weights ** 2)
            diversity = 1 - hhi
            diversity_scores.append(diversity)

        ax2.plot(diversity_scores, 'go-', linewidth=2, markersize=4)
        ax2.set_title('投资组合多样性演变')
        ax2.set_xlabel('回合')
        ax2.set_ylabel('多样性指数 (1-HHI)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # 3. 平均权重热力图
        avg_weights = np.zeros((len(all_episode_weights), config.n_assets))
        for i, episode_weights in enumerate(all_episode_weights):
            weights_array = np.array(episode_weights)
            if weights_array.shape[1] != config.n_assets:
                print(f"Episode {i}: 权重维度错误")
                continue
            avg_weights[i] = np.mean(weights_array, axis=0)

        im = ax3.imshow(avg_weights.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax3.set_title('各回合平均权重热力图')
        ax3.set_xlabel('回合')
        ax3.set_ylabel('资产')
        ax3.set_yticks(range(config.n_assets))
        ax3.set_yticklabels([f'Fund_{i + 1}' for i in range(config.n_assets)],
                            fontsize=6 if config.n_assets > 20 else 8)
        plt.colorbar(im, ax=ax3, shrink=0.8)

        # 4. 权重标准差（衡量权重变化的稳定性）
        weight_stds = []
        for episode_weights in all_episode_weights:
            weights_array = np.array(episode_weights)
            if weights_array.shape[1] != config.n_assets:
                continue
            std_per_asset = np.std(weights_array, axis=0)
            avg_std = np.mean(std_per_asset)
            weight_stds.append(avg_std)

        ax4.plot(weight_stds, 'ro-', linewidth=2, markersize=4)
        ax4.set_title('权重变化稳定性')
        ax4.set_xlabel('回合')
        ax4.set_ylabel('平均权重标准差')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/training_summary.png", dpi=300, bbox_inches='tight')
            print("训练总结图表已保存")

        plt.show()


