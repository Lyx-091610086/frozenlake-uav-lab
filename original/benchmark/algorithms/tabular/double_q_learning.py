import numpy as np
from typing import Dict, Any, Optional
from .base_tabular import TabularRLAlgorithm

class DoubleQLearning(TabularRLAlgorithm):
    """Double Q-Learning算法 - 缓解过估计问题"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

        # Double Q-Learning 特定参数
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)

    def reset(self):
        """重置算法状态"""
        # 使用两个Q表
        self.Q1 = np.zeros((self.n_states, self.n_actions))  # 第一个Q表
        self.Q2 = np.zeros((self.n_states, self.n_actions))  # 第二个Q表
        self.Q = self.Q1  # 为了兼容基类，指向Q1
        self.policy = np.zeros(self.n_states, dtype=int)  # 策略
        self.training_history = []  # 训练历史

    def train(self, **kwargs) -> Dict[str, Any]:
        """执行Double Q-Learning训练"""
        print("开始Double Q-Learning训练...")

        epsilon = self.epsilon_start
        episode_rewards = []
        episode_lengths = []

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done and steps < self.max_steps:
                # ε-greedy选择动作（基于两个Q表的平均）
                action = self._double_epsilon_greedy_action(state, epsilon)

                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # 随机选择更新哪个Q表 (50%概率)
                if np.random.random() < 0.5:
                    # 更新Q1，使用Q2选择动作
                    if done:
                        target = reward
                    else:
                        # 使用Q2选择最优动作，然后用Q2计算值
                        best_action_q2 = np.argmax(self.Q2[next_state])
                        target = reward + self.gamma * self.Q2[next_state, best_action_q2]

                    self.Q1[state, action] += self.alpha * (target - self.Q1[state, action])
                else:
                    # 更新Q2，使用Q1选择动作
                    if done:
                        target = reward
                    else:
                        # 使用Q1选择最优动作，然后用Q1计算值
                        best_action_q1 = np.argmax(self.Q1[next_state])
                        target = reward + self.gamma * self.Q1[next_state, best_action_q1]

                    self.Q2[state, action] += self.alpha * (target - self.Q2[state, action])

                state = next_state
                steps += 1

            # 记录回合信息
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

            # 衰减ε
            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)

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

        print("Double Q-Learning训练完成")
        return {
            'total_episodes': self.max_episodes,
            'training_history': self.training_history,
            'final_epsilon': epsilon,
            'final_success_rate': self.training_history[-1]['success_rate'] if self.training_history else 0
        }

    def _double_epsilon_greedy_action(self, state: int, epsilon: float) -> int:
        """基于两个Q表平均的ε-greedy动作选择"""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            # 使用两个Q表的平均来选择动作
            avg_q = (self.Q1[state] + self.Q2[state]) / 2
            max_avg_q = np.max(avg_q)
            best_actions = np.where(np.abs(avg_q - max_avg_q) < 1e-6)[0]
            return np.random.choice(best_actions)

    def _update_policy(self):
        """根据两个Q表的平均更新贪心策略"""
        for s in range(self.n_states):
            avg_q = (self.Q1[s] + self.Q2[s]) / 2
            max_avg_q = np.max(avg_q)
            best_actions = np.where(np.abs(avg_q - max_avg_q) < 1e-6)[0]
            self.policy[s] = np.random.choice(best_actions)

    def get_q_table(self) -> np.ndarray:
        """返回两个Q表的平均"""
        return (self.Q1 + self.Q2) / 2

    def get_value_function(self) -> np.ndarray:
        """获取值函数（两个Q表的平均最大值）"""
        return np.maximum(np.max(self.Q1, axis=1), np.max(self.Q2, axis=1))
