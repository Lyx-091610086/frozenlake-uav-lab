import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .base_tabular import TabularRLAlgorithm

class MonteCarloControl(TabularRLAlgorithm):
    """Monte Carlo Control 算法 - 基于完整回合的强化学习"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

        # Monte Carlo 特定参数
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)

        # 是否使用off-policy (重要性采样)
        self.off_policy = self.config.get('off_policy', False)

        # 首次访问或每次访问
        self.first_visit = self.config.get('first_visit', True)

    def train(self, **kwargs) -> Dict[str, Any]:
        """执行Monte Carlo Control训练"""
        print("开始Monte Carlo Control训练...")

        # 用于off-policy的重要性采样
        if self.off_policy:
            C = np.zeros((self.n_states, self.n_actions))  # 累积重要性采样权重

        episode_rewards = []
        episode_lengths = []

        for episode in range(self.max_episodes):
            # 生成一个回合
            episode_data = self._generate_episode(episode)

            # 计算回报
            G = 0
            W = 1.0  # 重要性采样权重 (off-policy用)

            # 从后往前遍历 (backward view)
            visited_states_actions = set() if self.first_visit else None

            for t in reversed(range(len(episode_data['states']))):
                state = episode_data['states'][t]
                action = episode_data['actions'][t]
                reward = episode_data['rewards'][t]

                G = self.gamma * G + reward

                if self.first_visit:
                    state_action = (state, action)
                    if state_action not in visited_states_actions:
                        visited_states_actions.add(state_action)
                    else:
                        continue

                if self.off_policy:
                    # 计算重要性采样比率
                    target_policy_prob = self._epsilon_greedy_prob(state, action, self._get_greedy_action(state))
                    behavior_policy_prob = self._epsilon_greedy_prob(state, action, action)
                    W *= target_policy_prob / behavior_policy_prob

                    if W == 0:
                        break

                    # 更新Q值和累积权重
                    C[state, action] += W
                    self.Q[state, action] += (W / C[state, action]) * (G - self.Q[state, action])
                else:
                    # on-policy: 直接更新
                    self.Q[state, action] += self.alpha * (G - self.Q[state, action])

            # 更新策略
            self._update_policy()

            # 衰减ε
            epsilon = max(self.epsilon_end, self.epsilon_start * (self.epsilon_decay ** episode))

            # 记录回合信息
            episode_rewards.append(sum(episode_data['rewards']))
            episode_lengths.append(len(episode_data['actions']))

            # 定期评估
            if (episode + 1) % self.eval_every == 0:
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

        print("Monte Carlo Control训练完成")
        return {
            'total_episodes': self.max_episodes,
            'training_history': self.training_history,
            'final_success_rate': self.training_history[-1]['success_rate'] if self.training_history else 0
        }

    def _generate_episode(self, episode: int) -> Dict[str, List]:
        """生成一个完整的回合"""
        epsilon = max(self.epsilon_end, self.epsilon_start * (self.epsilon_decay ** episode))

        states = []
        actions = []
        rewards = []

        state, _ = self.env.reset()
        done = False

        while not done and len(actions) < self.max_steps:
            states.append(state)

            # 选择动作
            if self.off_policy:
                # 行为策略: ε-greedy
                action = self._epsilon_greedy_action(state, epsilon)
            else:
                # on-policy: ε-greedy
                action = self._epsilon_greedy_action(state, epsilon)

            actions.append(action)

            # 执行动作
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            rewards.append(reward)
            state = next_state

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }

    def _epsilon_greedy_prob(self, state: int, action: int, greedy_action: int) -> float:
        """计算ε-greedy策略下的动作概率"""
        if action == greedy_action:
            return 1 - self.epsilon + self.epsilon / self.n_actions
        else:
            return self.epsilon / self.n_actions

    def _get_greedy_action(self, state: int) -> int:
        """获取贪心动作"""
        max_q = np.max(self.Q[state])
        best_actions = np.where(np.abs(self.Q[state] - max_q) < 1e-6)[0]
        return np.random.choice(best_actions)
