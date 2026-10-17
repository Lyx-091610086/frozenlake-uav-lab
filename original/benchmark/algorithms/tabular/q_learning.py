import numpy as np
from typing import Dict, Any, Optional
from .base_tabular import TabularRLAlgorithm

class QLearning(TabularRLAlgorithm):
    """Q-Learning算法 - off-policy TD学习"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

        # Q-learning特定参数
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)

    def train(self, **kwargs) -> Dict[str, Any]:
        """执行Q-learning训练"""
        print("开始Q-Learning训练...")

        epsilon = self.epsilon_start
        episode_rewards = []
        episode_lengths = []

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done and steps < self.max_steps:
                # ε-greedy选择动作
                action = self._epsilon_greedy_action(state, epsilon)

                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Q-learning更新
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * np.max(self.Q[next_state])

                self.Q[state, action] += self.alpha * (target - self.Q[state, action])

                state = next_state
                episode_reward += reward
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

                # 计算Q值的统计信息
                q_values = self.Q.flatten()
                q_stats = {
                    'q_mean': float(np.mean(q_values)),
                    'q_std': float(np.std(q_values)),
                    'q_max': float(np.max(q_values)),
                    'q_min': float(np.min(q_values)),
                    'q_range': float(np.max(q_values) - np.min(q_values))
                }

                # 计算贪心策略的确定性
                policy_determinacy = np.mean([len(np.where(np.abs(self.Q[s] - np.max(self.Q[s])) < 1e-6)[0]) == 1
                                             for s in range(self.n_states)])

                self.training_history.append({
                    'episode': episode + 1,
                    'epsilon': epsilon,
                    'avg_reward': np.mean(episode_rewards[-100:]),  # 最近100回合平均奖励
                    'success_rate': eval_results['success_rate'],
                    'avg_steps': eval_results['avg_steps'],
                    'total_reward': sum(episode_rewards[-100:]),   # 最近100回合总奖励
                    'policy_determinacy': policy_determinacy,      # 策略确定性
                    'q_stats': q_stats,                            # Q值统计
                    'exploration_rate': epsilon,                   # 探索率
                    'learning_progress': len(self.training_history) + 1  # 学习进度
                })

                print(f"Episode {episode + 1:6d}: ε={epsilon:.3f}, "
                      f"成功率={eval_results['success_rate']:.3f}, "
                      f"平均步数={eval_results['avg_steps']:.1f}, "
                      f"Q均值={q_stats['q_mean']:.3f}, "
                      f"策略确定性={policy_determinacy:.3f}")

                # 早停机制：如果成功率达到1.0，停止训练
                if self._should_early_stop(eval_results['success_rate']):
                    print(f"🎯 成功率达到100%，在第{episode + 1}轮提前停止训练")
                    break

        # 最终更新策略
        self._update_policy()

        print("Q-Learning训练完成")
        return {
            'total_episodes': self.max_episodes,
            'training_history': self.training_history,
            'final_epsilon': epsilon,
            'final_success_rate': self.training_history[-1]['success_rate'] if self.training_history else 0
        }
