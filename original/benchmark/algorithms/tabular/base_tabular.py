import numpy as np
from typing import Dict, Any, Optional
from ..base import RLAlgorithm

class TabularRLAlgorithm(RLAlgorithm):
    """表格型强化学习算法基类"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

        # 学习参数
        self.alpha = self.config.get('alpha', 0.1)  # 学习率
        self.gamma = self.config.get('gamma', 0.99)  # 折扣因子
        self.epsilon = self.config.get('epsilon', 0.1)  # ε-greedy参数
        self.max_episodes = self.config.get('max_episodes', 10000)
        self.max_steps = self.config.get('max_steps', 100)

        # 评估参数
        self.eval_every = self.config.get('eval_every', 1000)
        self.eval_episodes = self.config.get('eval_episodes', 100)

    def reset(self):
        """重置算法状态"""
        self.Q = np.zeros((self.n_states, self.n_actions))  # Q值表
        self.policy = np.zeros(self.n_states, dtype=int)  # 贪心策略
        self.training_history = []  # 训练历史

    def _epsilon_greedy_action(self, state: int, epsilon: Optional[float] = None) -> int:
        """ε-greedy策略选择动作"""
        eps = epsilon if epsilon is not None else self.epsilon

        if np.random.random() < eps:
            return np.random.randint(self.n_actions)
        else:
            # 随机选择最大值动作（打破平局）
            max_q = np.max(self.Q[state])
            best_actions = np.where(np.abs(self.Q[state] - max_q) < 1e-6)[0]
            return np.random.choice(best_actions)

    def _update_policy(self):
        """根据Q值更新贪心策略"""
        for s in range(self.n_states):
            max_q = np.max(self.Q[s])
            best_actions = np.where(np.abs(self.Q[s] - max_q) < 1e-6)[0]
            self.policy[s] = np.random.choice(best_actions)

    def get_policy(self) -> np.ndarray:
        """获取最优策略"""
        return self.policy.copy()

    def get_value_function(self) -> np.ndarray:
        """获取值函数（V值）"""
        return np.max(self.Q, axis=1)

    def get_q_table(self) -> np.ndarray:
        """获取Q值表"""
        return self.Q.copy()

    def evaluate_current_policy(self, n_episodes: int = 100, max_steps: int = 100) -> Dict[str, float]:
        """评估当前策略"""
        return self.evaluate_policy(self.policy, n_episodes, max_steps)

    def _should_early_stop(self, success_rate: float) -> bool:
        """检查是否应该早停（成功率达到100%）"""
        return success_rate >= 1.0
