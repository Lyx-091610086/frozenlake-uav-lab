import numpy as np
from typing import Dict, Any, Optional
from ..base import RLAlgorithm

class ValueIteration(RLAlgorithm):
    """值迭代算法 - 动态规划方法"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

        # 默认配置
        self.theta = self.config.get('theta', 1e-6)  # 收敛阈值
        self.gamma = self.config.get('gamma', 0.99)  # 折扣因子
        self.max_iterations = self.config.get('max_iterations', 1000)

    def reset(self):
        """重置算法状态"""
        self.V = np.zeros(self.n_states)  # 状态值函数
        self.policy = np.zeros(self.n_states, dtype=int)  # 策略

    def _get_transition_model(self):
        """获取环境的状态转移模型 - 手动构建确定性模型"""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions))

        # 获取地图信息
        desc = self.env.env.unwrapped.desc
        n_rows, n_cols = len(desc), len(desc[0])

        for state in range(self.n_states):
            row, col = state // n_cols, state % n_cols

            for action in range(self.n_actions):
                # FrozenLake动作映射: 0=左, 1=下, 2=右, 3=上
                dr, dc = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
                next_row, next_col = row + dr, col + dc

                # 检查边界
                if next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols:
                    # 撞墙，停留在原地，无奖励
                    next_state = state
                    reward = 0.0
                else:
                    next_state = next_row * n_cols + next_col
                    cell = desc[next_row][next_col]

                    # 解码字节（如果需要）
                    if isinstance(cell, bytes):
                        cell = cell.decode()

                    if cell == 'G':  # 目标
                        reward = 1.0
                    elif cell == 'H':  # 洞
                        reward = 0.0
                    else:  # 冰面
                        reward = 0.0

                # 设置确定性转移
                P[state, action, next_state] = 1.0
                R[state, action] = reward

        return P, R

    def train(self, **kwargs) -> Dict[str, Any]:
        """执行值迭代"""
        print("开始值迭代...")

        # 获取转移模型
        P, R = self._get_transition_model()

        iteration = 0
        delta_history = []

        while iteration < self.max_iterations:
            delta = 0
            V_new = np.copy(self.V)

            for s in range(self.n_states):
                # 计算每个动作的值
                action_values = []
                for a in range(self.n_actions):
                    # Bellman方程: V(s) = R(s,a) + γ * Σ P(s'|s,a) * V(s')
                    value = R[s, a] + self.gamma * np.sum(P[s, a] * self.V)
                    action_values.append(value)

                # 更新值函数
                V_new[s] = max(action_values)
                delta = max(delta, abs(V_new[s] - self.V[s]))

            self.V = V_new
            delta_history.append(delta)

            if delta < self.theta:
                print(f"值迭代收敛于第{iteration+1}次迭代")
                break

            iteration += 1

        # 从值函数中提取最优策略
        self._extract_policy(P, R)

        return {
            'iterations': iteration + 1,
            'final_delta': delta,
            'delta_history': delta_history,
            'converged': delta < self.theta
        }

    def _extract_policy(self, P, R):
        """从值函数中提取最优策略"""
        for s in range(self.n_states):
            action_values = []
            for a in range(self.n_actions):
                value = R[s, a] + self.gamma * np.sum(P[s, a] * self.V)
                action_values.append(value)

            # 选择最优动作（随机打破平局）
            max_value = max(action_values)
            best_actions = [a for a, v in enumerate(action_values) if abs(v - max_value) < 1e-6]
            self.policy[s] = np.random.choice(best_actions)

    def get_policy(self) -> np.ndarray:
        """获取最优策略"""
        return self.policy.copy()

    def get_value_function(self) -> np.ndarray:
        """获取值函数"""
        return self.V.copy()
