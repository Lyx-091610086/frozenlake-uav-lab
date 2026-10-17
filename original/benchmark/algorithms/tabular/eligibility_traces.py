import numpy as np
from typing import Dict, Any, Optional
from .base_tabular import TabularRLAlgorithm

class EligibilityTraces(TabularRLAlgorithm):
    """Eligibility Traces算法 - SARSA(λ) 和 Q(λ)"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

        # Eligibility Traces 特定参数
        self.lambda_ = self.config.get('lambda', 0.9)  # λ参数
        self.replacing_traces = self.config.get('replacing_traces', False)  # 是否使用替换迹

        # 算法类型: 'sarsa_lambda' 或 'q_lambda'
        self.algorithm = self.config.get('algorithm', 'sarsa_lambda')

        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)

    def reset(self):
        """重置算法状态"""
        super().reset()
        self.E = np.zeros((self.n_states, self.n_actions))  # 资格迹

    def train(self, **kwargs) -> Dict[str, Any]:
        """执行Eligibility Traces训练"""
        print(f"开始{self.algorithm} (λ={self.lambda_}) 训练...")

        epsilon = self.epsilon_start
        episode_rewards = []
        episode_lengths = []

        for episode in range(self.max_episodes):
            # 重置资格迹
            self.E.fill(0)

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
                    # 终止状态的目标值
                    target = reward
                    next_action = None
                else:
                    # 选择下一个动作
                    next_action = self._epsilon_greedy_action(next_state, epsilon)

                    if self.algorithm == 'sarsa_lambda':
                        # SARSA(λ): 使用下一个动作的Q值
                        target = reward + self.gamma * self.Q[next_state, next_action]
                    else:  # q_lambda (Watkins' Q(λ))
                        # Q(λ): 使用最大Q值，但只在非贪心动作时截断
                        target = reward + self.gamma * np.max(self.Q[next_state])

                # 计算TD误差
                td_error = target - self.Q[state, action]

                # 更新资格迹
                if self.replacing_traces:
                    # 替换迹: 设置为1
                    self.E[state, action] = 1
                else:
                    # 累积迹: 增加当前值
                    self.E[state, action] += 1

                # 对于Q(λ)，在非贪心动作时截断资格迹
                if self.algorithm == 'q_lambda' and not done:
                    greedy_action = np.argmax(self.Q[next_state])
                    if next_action != greedy_action:
                        # 截断非贪心动作的资格迹
                        self.E[next_state, :] = 0
                        self.E[next_state, greedy_action] = 0

                # 更新所有状态-动作对的Q值
                self.Q += self.alpha * td_error * self.E

                # 衰减资格迹
                self.E *= self.gamma * self.lambda_

                # 移动到下一个状态
                state = next_state
                action = next_action
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

        print(f"{self.algorithm} (λ={self.lambda_}) 训练完成")
        return {
            'total_episodes': self.max_episodes,
            'training_history': self.training_history,
            'lambda': self.lambda_,
            'algorithm': self.algorithm,
            'final_success_rate': self.training_history[-1]['success_rate'] if self.training_history else 0
        }
