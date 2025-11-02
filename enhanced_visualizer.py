# visualization/enhanced_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import os
from config import config

class EnhancedPortfolioVisualizer:
    def __init__(self, asset_names=None):
        self.asset_names = asset_names or [f'Asset_{i+1}' for i in range(10)]
        # 设置样式
        try:
            import seaborn as sns
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
    def create_portfolio_dashboard(self, episode_data, save_path=None):
        """创建投资组合仪表板"""
        try:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. 投资组合价值曲线
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_portfolio_value(ax1, episode_data)
            
            # 2. 实时权重分配
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_weight_stream(ax2, episode_data)
            
            # 3. 收益分布
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_return_distribution(ax3, episode_data)
            
            # 4. 风险指标
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_risk_metrics(ax4, episode_data)
            
            # 5. 资产贡献度
            ax5 = fig.add_subplot(gs[1, 2])
            self._plot_asset_contribution(ax5, episode_data)
            
            # 6. 多样化指标
            ax6 = fig.add_subplot(gs[1, 3])
            self._plot_diversification(ax6, episode_data)
            
            # 7. 最终权重饼图
            ax7 = fig.add_subplot(gs[2, :2])
            self._plot_final_allocation_pie(ax7, episode_data)
            
            # 8. 表现总结表格
            ax8 = fig.add_subplot(gs[2, 2:])
            self._plot_performance_table(ax8, episode_data)
            
            plt.suptitle(f'投资组合智能分析仪表板 - Episode {episode_data.get("episode", 1)}', 
                         fontsize=16, fontweight='bold')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 增强版仪表板已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"创建仪表板失败: {e}")
            raise e
    
    def _plot_portfolio_value(self, ax, data):
        """绘制投资组合价值变化"""
        values = data['values_history']
        times = range(len(values))
        
        ax.plot(times, values, 'b-', linewidth=2, label='投资组合价值')
        ax.fill_between(times, values, alpha=0.3)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='初始价值')
        
        # 标注关键点
        if len(values) > 1:
            max_idx = np.argmax(values)
            min_idx = np.argmin(values)
            ax.scatter([max_idx], [values[max_idx]], color='green', s=100, zorder=5)
            ax.scatter([min_idx], [values[min_idx]], color='red', s=100, zorder=5)
        
        ax.set_title('投资组合价值曲线', fontsize=12, fontweight='bold')
        ax.set_xlabel('交易日')
        ax.set_ylabel('价值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加收益率标注
        if len(values) > 1:
            final_return = (values[-1] - values[0]) / values[0] * 100
            ax.text(0.02, 0.98, f'总收益率: {final_return:.2f}%', 
                    transform=ax.transAxes, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
                    verticalalignment='top')
    
    def _plot_weight_stream(self, ax, data):
        """绘制权重流图"""
        weights_history = np.array(data['weights_history'])
        
        # 使用更美观的颜色
        n_assets = len(self.asset_names)
        colors = plt.cm.Set3(np.linspace(0, 1, n_assets))
        
        ax.stackplot(range(len(weights_history)), *weights_history.T, 
                    labels=self.asset_names, colors=colors, alpha=0.8)
        
        ax.set_title('资产权重动态分配', fontsize=12, fontweight='bold')
        ax.set_xlabel('交易日')
        ax.set_ylabel('权重比例')
        ax.set_ylim(0, 1)
        
        # 简化图例
        if len(self.asset_names) <= 8:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_return_distribution(self, ax, data):
        """绘制收益率分布"""
        values = data['values_history']
        if len(values) > 1:
            returns = np.diff(values) / np.array(values[:-1])
            
            ax.hist(returns, bins=min(20, len(returns)//2 + 1), alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(returns), color='red', linestyle='--', 
                      label=f'均值: {np.mean(returns):.4f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('日收益率分布', fontsize=10, fontweight='bold')
        ax.set_xlabel('收益率')
        ax.set_ylabel('频次')
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_metrics(self, ax, data):
        """绘制风险指标"""
        values = data['values_history']
        if len(values) > 20:
            returns = np.diff(values) / np.array(values[:-1])
            
            # 计算滚动波动率
            window = min(20, len(returns)//3)
            rolling_vol = []
            for i in range(window, len(returns)):
                vol = np.std(returns[i-window:i]) * np.sqrt(252)
                rolling_vol.append(vol)
            
            ax.plot(range(window, len(returns)), rolling_vol, 'orange', linewidth=2)
            ax.set_xlabel('交易日')
            ax.set_ylabel('年化波动率')
        else:
            ax.text(0.5, 0.5, '数据不足\n需要>20个数据点', ha='center', va='center', transform=ax.transAxes)
            
        ax.set_title('滚动波动率', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_asset_contribution(self, ax, data):
        """绘制资产贡献度"""
        final_weights = data['weights_history'][-1]
        
        # 只显示权重大于1%的资产
        significant_mask = np.array(final_weights) > 0.01
        if significant_mask.any():
            sig_weights = np.array(final_weights)[significant_mask]
            sig_names = [self.asset_names[i] for i in range(len(final_weights)) if significant_mask[i]]
            
            bars = ax.bar(range(len(sig_weights)), sig_weights, 
                         color=plt.cm.viridis(np.linspace(0, 1, len(sig_weights))))
            
            ax.set_xticks(range(len(sig_names)))
            ax.set_xticklabels(sig_names, rotation=45)
            
            # 添加数值标签
            for bar, weight in zip(bars, sig_weights):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{weight:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, '所有权重<1%', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('主要资产配置', fontsize=10, fontweight='bold')
        ax.set_xlabel('资产')
        ax.set_ylabel('权重')
        ax.grid(True, alpha=0.3)
    
    def _plot_diversification(self, ax, data):
        """绘制多样化指标"""
        weights_history = np.array(data['weights_history'])
        
        # HHI指数历史
        hhi_history = [np.sum(w**2) for w in weights_history]
        diversity_history = [1 - hhi for hhi in hhi_history]
        
        ax.plot(diversity_history, 'green', linewidth=2, label='多样化指数')
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='理想水平')
        
        ax.set_title('投资多样化程度', fontsize=10, fontweight='bold')
        ax.set_xlabel('交易日')
        ax.set_ylabel('多样化指数')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_final_allocation_pie(self, ax, data):
        """绘制最终配置饼图"""
        final_weights = data['weights_history'][-1]
        
        # 合并小权重
        threshold = 0.02
        large_weights = []
        large_names = []
        small_total = 0
        
        for i, weight in enumerate(final_weights):
            if weight > threshold:
                large_weights.append(weight)
                large_names.append(self.asset_names[i] if i < len(self.asset_names) else f'资产{i+1}')
            else:
                small_total += weight
        
        if small_total > 0.001:
            large_weights.append(small_total)
            large_names.append('其他')
        
        if large_weights:
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(large_weights)))
            wedges, texts, autotexts = ax.pie(large_weights, labels=large_names, 
                                             autopct='%1.1f%%', colors=colors,
                                             startangle=90)
        else:
            ax.text(0.5, 0.5, '无显著权重', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('最终资产配置', fontsize=12, fontweight='bold')
    
    def _plot_performance_table(self, ax, data):
        """绘制表现总结表"""
        ax.axis('tight')
        ax.axis('off')
        
        values = data['values_history']
        
        if len(values) > 1:
            returns = np.diff(values) / np.array(values[:-1])
            
            # 计算关键指标
            total_return = (values[-1] - values[0]) / values[0] * 100
            if len(values) > 252:
                annual_return = ((values[-1]/values[0])**(252/len(values)) - 1) * 100
            else:
                annual_return = total_return * (252/len(values))
            
            volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 0
            sharpe = (np.mean(returns) * 252) / (volatility/100) if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(values) * 100
            
            # 创建表格数据
            table_data = [
                ['指标', '数值'],
                ['总收益率', f'{total_return:.2f}%'],
                ['年化收益率', f'{annual_return:.2f}%'],
                ['年化波动率', f'{volatility:.2f}%'],
                ['夏普比率', f'{sharpe:.3f}'],
                ['最大回撤', f'{max_drawdown:.2f}%'],
                ['交易天数', f'{len(values)}'],
                ['最终价值', f'{values[-1]:.3f}']
            ]
        else:
            table_data = [
                ['指标', '数值'],
                ['数据不足', '-'],
            ]
        
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='center', loc='center',
                        colWidths=[0.4, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 美化表格
        for i in range(len(table_data)):
            for j in range(2):
                if (i, j) in table.get_celld():
                    cell = table[(i, j)]
                    if i == 0:  # 表头
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('投资表现总结', fontsize=12, fontweight='bold')
    
    def _calculate_max_drawdown(self, values):
        """计算最大回撤"""
        values = np.array(values)
        if len(values) <= 1:
            return 0
        
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        return np.min(drawdowns)
    
# 使用示例
def create_enhanced_visualization(episode_metrics, episode_num, save_dir):
    """创建增强版可视化 - 直接可用版本"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 检查数据是否存在
    if not hasattr(episode_metrics, 'weights_history') or not episode_metrics.weights_history:
        print("警告：没有权重历史数据，跳过可视化")
        return
    
    if not hasattr(episode_metrics, 'values_history') or not episode_metrics.values_history:
        print("警告：没有价值历史数据，跳过可视化")
        return
    
    # 创建增强版可视化器
    asset_names = [f'资产{i+1}' for i in range(len(episode_metrics.weights_history[0]))]
    visualizer = EnhancedPortfolioVisualizer(asset_names)
    
    # 准备数据
    episode_data = {
        'episode': episode_num,
        'weights_history': episode_metrics.weights_history,
        'values_history': episode_metrics.values_history,
        'returns_history': getattr(episode_metrics, 'rewards', [0] * len(episode_metrics.values_history))
    }
    
    # 创建保存路径
    import os
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/enhanced_dashboard_episode_{episode_num}.png"
    
    try:
        # 生成增强版仪表板
        visualizer.create_portfolio_dashboard(episode_data, save_path)
        print(f"✅ 增强版可视化已保存: {save_path}")
    except Exception as e:
        print(f"❌ 增强版可视化失败: {e}")
        # 回退到简单版本
        print("🔄 使用简化版可视化...")
        _create_simple_visualization(episode_data, save_path)

def _create_simple_visualization(episode_data, save_path):
    """简化版可视化作为后备方案"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 投资组合价值
    values = episode_data['values_history']
    ax1.plot(values, 'b-', linewidth=2)
    ax1.set_title('投资组合价值变化')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('价值')
    ax1.grid(True, alpha=0.3)
    
    # 2. 权重变化
    weights_history = np.array(episode_data['weights_history'])
    for i in range(weights_history.shape[1]):
        ax2.plot(weights_history[:, i], label=f'资产{i+1}')
    ax2.set_title('资产权重变化')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('权重')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. 最终权重分布
    final_weights = weights_history[-1]
    ax3.bar(range(len(final_weights)), final_weights)
    ax3.set_title('最终权重分布')
    ax3.set_xlabel('资产')
    ax3.set_ylabel('权重')
    ax3.grid(True, alpha=0.3)
    
    # 4. 收益率分布
    returns = np.diff(values) / np.array(values[:-1])
    ax4.hist(returns, bins=20, alpha=0.7, color='skyblue')
    ax4.set_title('收益率分布')
    ax4.set_xlabel('收益率')
    ax4.set_ylabel('频次')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 简化版可视化已保存: {save_path}")
