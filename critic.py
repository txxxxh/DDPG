# models/critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class Critic(nn.Module):
    def __init__(self, env=None, n_assets=None, window_size=None, feature_dim=None, action_dim=None):
        super().__init__()

        # 支持两种初始化方式：传入env或直接传入参数
        if env is not None:
            self.n_assets = env.n_assets
            self.window_size = env.window_size
            self.feature_dim = env.feature_dim  # 使用环境的特征维度（现在是24）
            self.action_dim = env.n_assets
        else:
            self.n_assets = n_assets if n_assets is not None else config.n_assets
            self.window_size = window_size if window_size is not None else config.lookback_window
            self.feature_dim = feature_dim if feature_dim is not None else config.n_factors  # 更新默认值
            self.action_dim = action_dim if action_dim is not None else config.n_assets

        # 输入维度: n_assets * window_size * feature_dim + action_dim
        input_dim = self.n_assets * self.window_size * self.feature_dim + self.action_dim

        print(
            f"Critic输入维度: {input_dim} (资产数={self.n_assets}, 窗口={self.window_size}, 特征={self.feature_dim}, 动作={self.action_dim})")

        # 增加网络容量以处理更高维度的输入
        self.fc1 = nn.Linear(input_dim, 1024)  # 增加第一层容量
        self.fc2 = nn.Linear(1024, 512)  # 增加第二层容量
        self.fc3 = nn.Linear(512, 256)  # 添加第三层
        self.fc4 = nn.Linear(256, 1)  # 输出层

        # 添加batch normalization和dropout防止过拟合
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, state, action):
        batch_size = state.size(0)

        # 确保state和action的维度匹配
        state_flat = state.view(batch_size, -1)  # [batch, n_assets*window*features]

        # 确保action是2维的
        if action.dim() > 2:
            action = action.squeeze(1)  # 移除多余的维度
        elif action.dim() == 1:
            action = action.unsqueeze(0)  # 添加batch维度

        # 连接前再次检查维度
        assert state_flat.dim() == 2, f"State维度错误: {state_flat.shape}"
        assert action.dim() == 2, f"Action维度错误: {action.shape}"
        assert state_flat.size(0) == action.size(
            0), f"Batch size不匹配: state {state_flat.size(0)}, action {action.size(0)}"

        x = torch.cat([state_flat, action], dim=1)

        # 更深的网络结构
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)

