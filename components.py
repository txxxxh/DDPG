# models/components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import config

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.05):  # 减小sigma
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class NonStationaryTransformer(nn.Module):
    def __init__(self, input_dim=config.n_factors, num_heads=8, dropout=0.1):  # 更新默认input_dim
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        # 确保input_dim能被num_heads整除
        if input_dim % num_heads != 0:
            # 调整num_heads使其能整除input_dim
            for h in [8, 6, 4, 3, 2, 1]:
                if input_dim % h == 0:
                    num_heads = h
                    break
            print(f"警告：调整num_heads从{self.num_heads}到{num_heads}以适应input_dim={input_dim}")
            self.num_heads = num_heads

        self.head_dim = input_dim // num_heads

        assert self.head_dim * num_heads == input_dim, f"input_dim({input_dim}) must be divisible by num_heads({num_heads})"

        self.norm = nn.LayerNorm(input_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )

        self.tau_net = nn.Sequential(
            nn.Linear(input_dim, 64),  # 增加隐藏层大小以适应更高维度
            nn.ReLU(),
            nn.Linear(64, num_heads)
        )

        self.delta_net = nn.Sequential(
            nn.Linear(input_dim, 64),  # 增加隐藏层大小
            nn.ReLU(),
            nn.Linear(64, num_heads * self.head_dim)
        )

        # 修正：输出投影层输入应为num_heads * head_dim
        self.out_proj = nn.Linear(num_heads * self.head_dim, input_dim)

    def forward(self, x):
        """ x shape: [seq_len, batch_size, features] """
        # print(f"Transformer 输入维度:", x.shape)
        if torch.isnan(x).any():
            print("NaN detected in input!")
            x = torch.nan_to_num(x)
        seq_len, batch_size, _ = x.shape

        # 1. 序列标准化
        mu = x.mean(dim=0, keepdim=True)
        sigma = x.std(dim=0, keepdim=True) + 1e-6
        """
        x_norm = (x - mu) / sigma
        """
        x_norm = self.norm(x)

        # 2. 计算τ和Δ
        pooled = x_norm.mean(dim=0)  # [batch_size, features]
        assert pooled.shape[1] == self.input_dim, f"Input dimension mismatch: {pooled.shape[1]} != {self.input_dim}"
        tau = torch.exp(self.tau_net(pooled))  # [batch_size, num_heads]
        delta = self.delta_net(pooled)  # [batch_size, num_heads * head_dim]
        delta = delta.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        # 使用局部变量tau和delta
        # print(f"tau: {tau.mean().item():.4f}, delta: {delta.mean().item():.4f}")

        # 在计算注意力前添加查询/键标准化
        q = F.normalize(x, dim=-1)
        k = F.normalize(x, dim=-1)

        # 修改后的注意力计算
        attn_output, attn_weights = self.attention(
            query=q, key=k, value=x
        )

        # 打印注意力权重
        # print(f"注意力权重形状:", attn_weights.shape)  # 应输出 torch.Size([batch_size, seq_len, seq_len])
        # print(f"第一个时间步的注意力权重:", attn_weights[0, 0])  # 查看第一个时间步对其他时间步的注意力

        # 打印注意力权重的方差（检查是否有变化）
        # if attn_weights is not None:
        # print(f"注意力权重方差:", attn_weights.var().item())

        # 4. 分割多头输出
        attn_output = attn_output.view(seq_len, batch_size, self.num_heads, self.head_dim)

        # 5. 应用去稳定因子
        # 维度扩展：
        tau = tau.unsqueeze(0).unsqueeze(-1)  # [1, batch_size, num_heads, 1]
        delta = delta.unsqueeze(0)  # [1, batch_size, num_heads, head_dim]

        # 应用缩放和偏移
        attn_output = tau * attn_output + delta

        # 6. 合并多头输出
        # 保持维度顺序 [seq_len, batch_size, num_heads, head_dim]
        attn_output = attn_output.contiguous()
        attn_output = attn_output.view(seq_len, batch_size, -1)  # [seq_len, batch_size, num_heads * head_dim]

        # 7. 输出投影
        output = self.out_proj(attn_output)

        # 8. 反标准化
        output = output * sigma + mu

        return output

    # =======================================图卷积和资产依赖模块==============================================================================

# 图卷积层
class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # 添加自环并归一化
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = torch.diag(1 / torch.sqrt(adj.sum(dim=1)))
        norm_adj = deg @ adj @ deg
        return self.linear(norm_adj @ x)

# 资产依赖模块
class AssetGCN(nn.Module):
    def __init__(self, feature_dim=config.n_factors, hidden_dim=128, window_size=10, n_assets=config.n_assets, returns_idx=-1):
        super().__init__()
        self.window_size = window_size
        self.n_assets_in_group = n_assets
        self.feature_dim = feature_dim
        self.returns_idx = returns_idx  # 添加收益率索引参数

        # 三层GCN - 输入维度现在是window_size * feature_dim (例如: 10 * 24 = 240)
        self.graph_conv1 = GraphConv(window_size * feature_dim, hidden_dim)
        self.graph_conv2 = GraphConv(hidden_dim, hidden_dim)
        self.graph_conv3 = GraphConv(hidden_dim, hidden_dim)

        # 两步2D卷积
        self.conv2d_1 = nn.Conv2d(1, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        self.conv2d_2 = nn.Conv2d(hidden_dim, 1, kernel_size=(1, 3), padding=(0, 1))

    def build_adjacency(self, returns):
        """构建资产相关性邻接矩阵 (论文公式13)"""
        # returns shape: [n_assets, window_size]
        corr_matrix = torch.corrcoef(returns)
        return 1 - torch.abs(corr_matrix)  # 距离度量

    def forward(self, x):
        """
        输入: x [batch_size * n_assets, window_size, features]
        输出: X_dep [batch_size * n_assets, hidden_dim]
        """
        # 1. 构建图结构
        n_total_assets = x.size(0)

        # 方案1：使用构造函数中的returns_idx参数
        returns = x[:, :, self.returns_idx]  # 使用指定的收益率索引

        # 方案2：如果知道具体的索引值，可以直接写死
        # returns = x[:, :, 0]  # 假设收益率在第0个特征维度

        # 方案3：如果收益率总是最后一个特征
        # returns = x[:, :, -1]

        # 按原始资产分组计算相关性
        adj_list = []
        for i in range(0, n_total_assets, self.n_assets_in_group):
            group_returns = returns[i:i + self.n_assets_in_group]
            corr_matrix = torch.corrcoef(group_returns)
            adj = 1 - torch.abs(corr_matrix)
            adj_list.append(adj)

        adj = torch.block_diag(*adj_list)  # 创建块对角矩阵

        # 2. GCN处理
        node_features = x.flatten(start_dim=1)  # [n_total_assets, window_size*features]
        h = F.relu(self.graph_conv1(node_features, adj))
        h = F.relu(self.graph_conv2(h, adj))
        h = F.relu(self.graph_conv3(h, adj))  # [n_total_assets, hidden_dim]

        # 3. 2D卷积处理
        h = h.unsqueeze(0).unsqueeze(0)  # [1, 1, n_total_assets, hidden_dim]
        h = F.relu(self.conv2d_1(h))  # [1, hidden_dim, n_total_assets, hidden_dim]
        h = self.conv2d_2(h)  # [1, 1, n_total_assets, hidden_dim]

        return h.squeeze(0).squeeze(0)
