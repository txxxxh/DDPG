# utils/metrics.py
import numpy as np
from typing import Dict, List

class EpisodeMetrics:
    """单回合指标记录"""
    def __init__(self):
        self.weights_history = []
        self.values_history = []
        self.rewards = []
        self.total_reward = 0
        self.final_value = 0

    def add_step(self, weights, step):
        self.weights_history.append(weights.copy())

    def add_value(self, value):
        self.values_history.append(value)
        self.final_value = value

    def add_reward(self, reward):
        self.rewards.append(reward)
        self.total_reward += reward

    def get_diversity_stats(self):
        if not self.weights_history:
            return {'avg_diversity': 0, 'avg_effective_assets': 0}

        diversities = [1.0 - np.sum(w ** 2) for w in self.weights_history]
        effective_assets = [1.0 / np.sum(w ** 2) for w in self.weights_history]

        return {
            'avg_diversity': np.mean(diversities),
            'avg_effective_assets': np.mean(effective_assets)
        }

    def get_weight_stats(self):
        if not self.weights_history:
            return {'min_weight': 0, 'max_weight': 0}

        all_weights = np.concatenate(self.weights_history)
        return {
            'min_weight': np.min(all_weights),
            'max_weight': np.max(all_weights)
        }

class TrainingMetrics:
    """训练过程指标记录"""
    def __init__(self):
        self.episodes = []
        self.diversity_history = []
        self.effective_assets_history = []
        self.exploration_rates = []
        self.max_weights_history = []
        self.min_weights_history = []

    def add_episode(self, episode_metrics):
        self.episodes.append(episode_metrics)

        # 计算并记录各种指标
        diversity_stats = episode_metrics.get_diversity_stats()
        self.diversity_history.append(diversity_stats['avg_diversity'])
        self.effective_assets_history.append(diversity_stats['avg_effective_assets'])

        weight_stats = episode_metrics.get_weight_stats()
        self.max_weights_history.append(weight_stats['max_weight'])
        self.min_weights_history.append(weight_stats['min_weight'])

    def get_final_stats(self):
        return {
            'avg_diversity': np.mean(self.diversity_history[-10:]),
            'diversity_std': np.std(self.diversity_history),
            'avg_effective_assets': np.mean(self.effective_assets_history[-10:])
        }

    def get_final_weights(self):
        if self.episodes:
            return self.episodes[-1].weights_history[-1]
        return []


