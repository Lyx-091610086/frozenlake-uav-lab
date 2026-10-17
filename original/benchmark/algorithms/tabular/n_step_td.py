import numpy as np
from typing import Dict, Any, Optional, List, Deque
from collections import deque
from .base_tabular import TabularRLAlgorithm

class NStepTD(TabularRLAlgorithm):
    """n-step TD算法 - 多步时序差分学习"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

        # n-step TD 特定参数
        self.n = self.config.get('n', 3)  # n步数
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)

        # 是否使用Expected SARSA
        self.expected_sarsa = self.config.get('expected_sarsa', False)

    def train(self, **kwargs) -> Dict[str, Any]:
        """执行n-step TD训练"""
        print(f"开始{self.n}-step TD训练...")

        episode_rewards = []
        episode_lengths = []

        for episode in range(self.max_episodes):
            # 初始化存储容器
            states = deque(maxlen=self.n + 1)
            actions = deque(maxlen=self.n + 1)
            rewards = deque(maxlen=self.n)

            # 初始化状态
            state, _ = self.env.reset()
            states.append(state)

            # 选择初始动作
            epsilon = max(self.epsilon_end, self.epsilon_start * (self.epsilon_decay ** episode))
            action = self._epsilon_greedy_action(state, epsilon)
            actions.append(action)

            T = float('inf')  # 终止时间
            t = 0  # 时间步
            tau = 0  # 更新起点

            episode_reward = 0

            while tau < T - 1:
                if t < T:
                    # 执行动作
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    states.append(next_state)
                    rewards.append(reward)
                    episode_reward += reward

                    if done:
                        T = t + 1
                    else:
                        # 选择下一个动作
                        next_action = self._epsilon_greedy_action(next_state, epsilon)
                        actions.append(next_action)
                        action = next_action

                # 更新时间步
                tau = t - self.n + 1

                if tau >= 0:
                    # 计算n步回报
                    G = self._compute_n_step_return(tau, t, T, rewards, states, actions)

                    if tau + self.n < T:
                        # 非终止状态，使用n步前向视图
                        if self.expected_sarsa:
                            # Expected SARSA
                            state_tau = states[tau - tau]  # states[0] 是tau时刻的状态
                            expected_value = self._compute_expected_value(state_tau, epsilon)
                            G += (self.gamma ** self.n) * expected_value
                        else:
                            # 普通SARSA
                            state_tau_plus_n = states[tau + self.n - tau]  # states[n] 是tau+n时刻的状态
                            action_tau_plus_n = actions[tau + self.n - tau]  # actions[n] 是tau+n时刻的动作
                            G += (self.gamma ** self.n) * self.Q[state_tau_plus_n, action_tau_plus_n]

                    # 更新Q值
                    state_tau = states[tau - tau]  # states[0] 是tau时刻的状态
                    action_tau = actions[tau - tau]  # actions[0] 是tau时刻的动作
                    self.Q[state_tau, action_tau] += self.alpha * (G - self.Q[state_tau, action_tau])

                t += 1

            # 记录回合信息
            episode_rewards.append(episode_reward)
            episode_lengths.append(t)

            # 定期评估
            if (episode + 1) % self.eval_every == 0:
                self._update_policy()
                eval_results = self.evaluate_current_policy(self.eval_episodes)

                self.training_history.append({
                    'episode': episode + 1,
                    'epsilon': epsilon,
                    'avg_reward': np.mean(episode_rewards[-100:]),
                    'success_rate': eval_results['success_rate'],
                    'avg_steps': eval_results['avg_steps']
                })

                print(f"Episode {episode + 1:6d}: ε={epsilon:.3f}, "
                      f"成功率={eval_results['success_rate']:.3f}, "
                      f"平均步数={eval_results['avg_steps']:.1f}")

                # 早停机制：如果成功率达到1.0，停止训练
                if self._should_early_stop(eval_results['success_rate']):
                    print(f"🎯 成功率达到100%，在第{episode + 1}轮提前停止训练")
                    break

        # 最终更新策略
        self._update_policy()

        print(f"{self.n}-step TD训练完成")
        return {
            'total_episodes': self.max_episodes,
            'training_history': self.training_history,
            'n_step': self.n,
            'final_success_rate': self.training_history[-1]['success_rate'] if self.training_history else 0
        }

    def _compute_n_step_return(self, tau: int, t: int, T: float,
                             rewards: deque, states: deque, actions: deque) -> float:
        """计算n步回报"""
        G = 0

        # 处理终止时间为无穷大的情况
        if T == float('inf'):
            max_steps = self.n
        else:
            # 确保T - tau是有限数值
            if T - tau < 0:
                max_steps = 0
            else:
                max_steps = min(self.n, int(T - tau))

        for i in range(max_steps):
            G += (self.gamma ** i) * rewards[tau + i - tau]  # rewards[i] 是tau+i时刻的奖励
        return G

    def _compute_expected_value(self, state: int, epsilon: float) -> float:
        """计算期望值 (Expected SARSA)"""
        greedy_action = self._get_greedy_action(state)
        expected_value = 0

        for a in range(self.n_actions):
            if a == greedy_action:
                prob = 1 - epsilon + epsilon / self.n_actions
            else:
                prob = epsilon / self.n_actions
            expected_value += prob * self.Q[state, a]

        return expected_value

    def _get_greedy_action(self, state: int) -> int:
        """获取贪心动作"""
        max_q = np.max(self.Q[state])
        best_actions = np.where(np.abs(self.Q[state] - max_q) < 1e-6)[0]
        return np.random.choice(best_actions)
