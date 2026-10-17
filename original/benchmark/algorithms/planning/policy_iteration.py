import numpy as np
from typing import Dict, Any, Optional
from ..base import RLAlgorithm

class PolicyIteration(RLAlgorithm):
    """策略迭代算法 - 动态规划方法"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

        # 默认配置
        self.theta = self.config.get('theta', 1e-6)  # 值函数评估收敛阈值
        self.gamma = self.config.get('gamma', 0.99)  # 折扣因子
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.max_policy_iterations = self.config.get('max_policy_iterations', 100)

    def reset(self):
        """重置算法状态"""
        self.V = np.zeros(self.n_states)  # 状态值函数
        self.policy = np.random.randint(0, self.n_actions, self.n_states)  # 随机初始化策略

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

    def _policy_evaluation(self, P, R):
        """策略评估：计算当前策略下的值函数"""
        iteration = 0
        while iteration < self.max_iterations:
            delta = 0
            V_new = np.zeros(self.n_states)

            for s in range(self.n_states):
                a = self.policy[s]
                # Bellman方程 for policy evaluation
                V_new[s] = R[s, a] + self.gamma * np.sum(P[s, a] * self.V)

                delta = max(delta, abs(V_new[s] - self.V[s]))

            self.V = V_new

            if delta < self.theta:
                break

            iteration += 1

        return iteration

    def _policy_improvement(self, P, R):
        """策略改进"""
        policy_stable = True

        for s in range(self.n_states):
            old_action = self.policy[s]

            # 计算每个动作的值
            action_values = []
            for a in range(self.n_actions):
                value = R[s, a] + self.gamma * np.sum(P[s, a] * self.V)
                action_values.append(value)

            # 选择最优动作
            max_value = max(action_values)
            best_actions = [a for a, v in enumerate(action_values) if abs(v - max_value) < 1e-6]
            self.policy[s] = np.random.choice(best_actions)

            if old_action != self.policy[s]:
                policy_stable = False

        return policy_stable

    def train(self, **kwargs) -> Dict[str, Any]:
        """执行策略迭代"""
        print("开始策略迭代...")

        P, R = self._get_transition_model()

        policy_iterations = 0
        total_eval_iterations = 0

        while policy_iterations < self.max_policy_iterations:
            # 策略评估
            eval_iters = self._policy_evaluation(P, R)
            total_eval_iterations += eval_iters

            # 策略改进
            policy_stable = self._policy_improvement(P, R)

            policy_iterations += 1

            if policy_stable:
                print(f"策略迭代收敛于第{policy_iterations}次策略迭代")
                break

        return {
            'policy_iterations': policy_iterations,
            'total_eval_iterations': total_eval_iterations,
            'converged': policy_stable
        }

    def get_policy(self) -> np.ndarray:
        """获取最优策略"""
        return self.policy.copy()

    def get_value_function(self) -> np.ndarray:
        """获取值函数"""
        return self.V.copy()
