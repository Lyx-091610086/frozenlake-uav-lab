import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from ..tabular.base_tabular import TabularRLAlgorithm

class DynaQ(TabularRLAlgorithm):
    """Dyna-Q算法 - 结合直接RL和模型学习的样本高效算法"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

        # Dyna-Q 特定参数
        self.n_planning = self.config.get('n_planning', 5)  # 每次规划步数
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)

        # 模型参数 (简单的表格模型)
        self.learned_model = {}  # 状态-动作 -> (奖励, 下一个状态)
        self.visited = set()  # 已访问的状态-动作对

    def reset(self):
        """重置算法状态"""
        super().reset()
        self.learned_model = {}
        self.visited = set()

    def train(self, **kwargs) -> Dict[str, Any]:
        """执行Dyna-Q训练"""
        print(f"开始Dyna-Q训练 (规划步数: {self.n_planning})...")

        epsilon = self.epsilon_start
        episode_rewards = []
        episode_lengths = []

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done and steps < self.max_steps:
                # 直接RL: 与环境交互
                action = self._epsilon_greedy_action(state, epsilon)

                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward

                # 更新模型 (学习环境动态)
                self._update_model(state, action, reward, next_state)

                # 标记已访问
                self.visited.add((state, action))

                # 直接RL更新
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * np.max(self.Q[next_state])

                self.Q[state, action] += self.alpha * (target - self.Q[state, action])

                # 规划: 使用学到的模型进行额外更新
                self._planning(n_planning=self.n_planning)

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
                    'avg_steps': eval_results['avg_steps'],
                    'model_size': len(self.learned_model)
                })

                print(f"Episode {episode + 1:6d}: ε={epsilon:.3f}, "
                      f"成功率={eval_results['success_rate']:.3f}, "
                      f"模型大小={len(self.learned_model):4d}")

        # 最终更新策略
        self._update_policy()

        print("Dyna-Q训练完成")
        return {
            'total_episodes': self.max_episodes,
            'training_history': self.training_history,
            'final_model_size': len(self.learned_model),
            'n_planning': self.n_planning,
            'final_success_rate': self.training_history[-1]['success_rate'] if self.training_history else 0
        }

    def _update_model(self, state: int, action: int, reward: float, next_state: int):
        """更新环境模型"""
        key = (state, action)
        self.learned_model[key] = (reward, next_state)

    def _planning(self, n_planning: int):
        """使用学到的模型进行规划"""
        if not self.visited:
            return

        # 从已访问的状态-动作对中随机选择进行规划
        visited_list = list(self.visited)

        for _ in range(n_planning):
            # 随机选择一个已访问的状态-动作对
            idx = np.random.randint(len(visited_list))
            state, action = visited_list[idx]

            # 从模型中获取模拟经验
            if (state, action) in self.learned_model:
                reward, next_state = self.learned_model[(state, action)]

                # 使用模拟经验更新Q值
                target = reward + self.gamma * np.max(self.Q[next_state])
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])
