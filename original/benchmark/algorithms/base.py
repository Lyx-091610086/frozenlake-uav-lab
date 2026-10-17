from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
from environments.frozenlake_env import FrozenLakeEnv

class RLAlgorithm(ABC):
    """强化学习算法基类"""

    def __init__(self, env: FrozenLakeEnv, config: Optional[Dict[str, Any]] = None):
        self.env = env
        self.config = config or {}
        self.n_states = env.n_states
        self.n_actions = env.n_actions

        # 初始化值函数和策略
        self.reset()

    @abstractmethod
    def reset(self):
        """重置算法状态"""
        pass

    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        """训练算法"""
        pass

    @abstractmethod
    def get_policy(self) -> np.ndarray:
        """获取最优策略"""
        pass

    @abstractmethod
    def get_value_function(self) -> np.ndarray:
        """获取值函数"""
        pass

    def evaluate_policy(self, policy: np.ndarray, n_episodes: int = 1000,
                       max_steps: int = 100) -> Dict[str, float]:
        """评估策略性能"""
        total_rewards = []
        total_steps = []
        successes = 0

        for _ in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done and steps < max_steps:
                action = policy[state]
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                steps += 1
                state = next_state

            total_rewards.append(episode_reward)
            total_steps.append(steps)
            if episode_reward == 1.0:  # 成功到达目标
                successes += 1

        return {
            'success_rate': successes / n_episodes,
            'avg_reward': np.mean(total_rewards),
            'avg_steps': np.mean(total_steps),
            'std_reward': np.std(total_rewards),
            'std_steps': np.std(total_steps)
        }
