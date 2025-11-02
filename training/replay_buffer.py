# training/replay_buffer.py
import random
import torch
import numpy as np
from collections import deque, namedtuple
from config import config

class ReplayBuffer:
    def __init__(self, capacity, env=None, state_dim=None, action_dim=None):
        self.capacity = capacity

        # 支持两种初始化方式：传入env或直接传入维度
        if env is not None:
            self.state_dim = env.feature_dim  # 使用环境的特征维度（现在是24）
            self.action_dim = env.n_assets
        else:
            # 更新默认维度为24
            self.state_dim = state_dim if state_dim is not None else 24  # config.n_factors
            self.action_dim = action_dim if action_dim is not None else 10  # config.n_assets

        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        # 确保存储的动作是1维的
        if isinstance(action, torch.Tensor) and action.dim() > 1:
            action = action.squeeze(0)
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        """
        修复版本：确保返回的张量是独立的，不会导致梯度计算问题
        """
        # 随机采样批次
        batch = random.sample(self.buffer, batch_size)

        # 解包样本
        states, actions, rewards, next_states, dones = zip(*batch)

        # 关键修复：使用 .clone().detach() 创建独立副本
        # 处理states
        state_list = []
        for state in states:
            if isinstance(state, torch.Tensor):
                state_list.append(state.clone().detach())
            else:
                state_list.append(torch.tensor(state, dtype=torch.float32))

        # 处理actions
        action_list = []
        for action in actions:
            if isinstance(action, torch.Tensor):
                action_list.append(action.clone().detach())
            else:
                action_list.append(torch.tensor(action, dtype=torch.float32))

        # 处理next_states
        next_state_list = []
        for next_state in next_states:
            if isinstance(next_state, torch.Tensor):
                next_state_list.append(next_state.clone().detach())
            else:
                next_state_list.append(torch.tensor(next_state, dtype=torch.float32))

        # 转换为张量并修正维度
        states = torch.stack(state_list)
        actions = torch.stack(action_list)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
        next_states = torch.stack(next_state_list)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)

        # 处理actions的维度
        if actions.dim() > 2:
            actions = actions.squeeze(1)

        # 检查并修正维度
        if states.dim() == 5:  # [batch, 1, n_assets, window, feat]
            states = states.squeeze(1)
        if next_states.dim() == 5:
            next_states = next_states.squeeze(1)
        if actions.dim() == 3:  # [batch, 1, n_assets]
            actions = actions.squeeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)



