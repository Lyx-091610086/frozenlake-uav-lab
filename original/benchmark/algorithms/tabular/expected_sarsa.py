import numpy as np
from typing import Dict, Any, Optional
from .base_tabular import TabularRLAlgorithm

class ExpectedSARSA(TabularRLAlgorithm):
    """Expected SARSA算法 - 使用期望值的SARSA"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

        # Expected SARSA 特定参数
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)

    def train(self, **kwargs) -> Dict[str, Any]:
        """执行Expected SARSA训练"""
        print("开始Expected SARSA训练...")

        epsilon = self.epsilon_start
        episode_rewards = []
        episode_lengths = []

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False

            # 选择初始动作
            action = self._epsilon_greedy_action(state, epsilon)

            while not done and steps < self.max_steps:
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward

                if done:
                    # 终止状态的目标值为奖励
                    target = reward
                else:
                    # Expected SARSA: 使用所有动作的期望值
                    expected_value = self._compute_expected_value(next_state, epsilon)
                    target = reward + self.gamma * expected_value

                # 更新Q值
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])

                # 移动到下一个状态
                state = next_state

                if not done:
                    # 选择下一个动作
                    action = self._epsilon_greedy_action(state, epsilon)

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

        print("Expected SARSA训练完成")
        return {
            'total_episodes': self.max_episodes,
            'training_history': self.training_history,
            'final_epsilon': epsilon,
            'final_success_rate': self.training_history[-1]['success_rate'] if self.training_history else 0
        }

    def _compute_expected_value(self, state: int, epsilon: float) -> float:
        """计算期望值 E[Q(s',a') | s']"""
        expected_value = 0

        # 获取贪心动作
        greedy_action = self._get_greedy_action(state)

        for a in range(self.n_actions):
            if a == greedy_action:
                # 贪心动作的概率
                prob = 1 - epsilon + epsilon / self.n_actions
            else:
                # 非贪心动作的概率
                prob = epsilon / self.n_actions

            expected_value += prob * self.Q[state, a]

        return expected_value

    def _get_greedy_action(self, state: int) -> int:
        """获取贪心动作"""
        max_q = np.max(self.Q[state])
        best_actions = np.where(np.abs(self.Q[state] - max_q) < 1e-6)[0]
        return np.random.choice(best_actions)
