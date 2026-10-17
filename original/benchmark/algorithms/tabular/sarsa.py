import numpy as np
from typing import Dict, Any, Optional
from .base_tabular import TabularRLAlgorithm

class SARSA(TabularRLAlgorithm):
    """SARSA算法 - on-policy TD学习"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

        # SARSA特定参数
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)

    def train(self, **kwargs) -> Dict[str, Any]:
        """执行SARSA训练"""
        print("开始SARSA训练...")

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

                # 选择下一个动作（SARSA是on-policy）
                if done:
                    next_action = None  # 终止状态没有下一个动作
                    target = reward
                else:
                    next_action = self._epsilon_greedy_action(next_state, epsilon)
                    target = reward + self.gamma * self.Q[next_state, next_action]

                # SARSA更新
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])

                state = next_state
                action = next_action
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

                self.training_history.append({
                    'episode': episode + 1,
                    'epsilon': epsilon,
                    'avg_reward': np.mean(episode_rewards[-100:]),  # 最近100回合平均奖励
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

        print("SARSA训练完成")
        return {
            'total_episodes': self.max_episodes,
            'training_history': self.training_history,
            'final_epsilon': epsilon,
            'final_success_rate': self.training_history[-1]['success_rate'] if self.training_history else 0
        }
