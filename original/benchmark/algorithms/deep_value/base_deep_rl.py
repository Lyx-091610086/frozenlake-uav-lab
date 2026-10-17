import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Deque
from collections import deque
import random
from ..base import RLAlgorithm

class BaseDeepRL(RLAlgorithm):
    """深度强化学习算法基类"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        # 先设置必要的参数，避免在reset()中出现AttributeError
        self.config = config or {}

        # 深度学习参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.config.get('batch_size', 64)
        self.gamma = self.config.get('gamma', 0.99)
        self.learning_rate = self.config.get('learning_rate', 1e-3)
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 1000)
        self.target_update_freq = self.config.get('target_update_freq', 100)

        # 经验回放参数
        self.memory_size = self.config.get('memory_size', 10000)

        # 训练参数
        self.max_episodes = self.config.get('max_episodes', 1000)
        self.max_steps = self.config.get('max_steps', 500)
        self.eval_every = self.config.get('eval_every', 50)
        self.eval_episodes = self.config.get('eval_episodes', 10)

        # 现在调用父类初始化，它会调用reset()
        super().__init__(env, self.config)

        # 构建网络
        self._build_network()

    def _build_network(self):
        """构建神经网络 - 由子类实现"""
        raise NotImplementedError

    def reset(self):
        """重置算法状态"""
        # 初始化或清空经验回放缓冲区
        memory_size = getattr(self, 'memory_size', 10000)  # 使用默认值如果还未初始化
        if not hasattr(self, 'memory'):
            self.memory = deque(maxlen=memory_size)
        else:
            # 如果memory已存在但maxlen不同，重新创建
            if self.memory.maxlen != memory_size:
                self.memory = deque(self.memory, maxlen=memory_size)
            else:
                self.memory.clear()

        self.training_history = []
        self.steps_done = 0

    def _state_to_tensor(self, state: int) -> torch.Tensor:
        """将状态转换为张量"""
        # 对于表格状态，创建一个one-hot向量
        state_tensor = torch.zeros(self.n_states, dtype=torch.float32)
        state_tensor[state] = 1.0
        return state_tensor.unsqueeze(0).to(self.device)

    def _select_action(self, state: int, epsilon: float = None) -> int:
        """选择动作"""
        if epsilon is None:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                     np.exp(-self.steps_done / self.epsilon_decay)

        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)

    def _store_transition(self, state: int, action: int, reward: float,
                         next_state: int, done: bool):
        """存储转换到经验回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))

    def _sample_batch(self) -> Tuple[torch.Tensor, ...]:
        """从经验回放中采样批次"""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        state_batch = torch.stack([self._state_to_tensor(s) for s in states]).squeeze(1)
        action_batch = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_state_batch = torch.stack([self._state_to_tensor(s) for s in next_states]).squeeze(1)
        done_batch = torch.tensor(dones, dtype=torch.float32).to(self.device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def train(self, **kwargs) -> Dict[str, Any]:
        """训练深度RL算法"""
        print(f"开始{self.__class__.__name__}训练...")

        episode_rewards = []
        episode_lengths = []

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0

            while steps < self.max_steps:
                # 选择动作
                action = self._select_action(state)
                self.steps_done += 1

                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward

                # 存储经验
                self._store_transition(state, action, reward, next_state, done)

                # 训练网络
                if len(self.memory) >= self.batch_size:
                    self._train_step()

                # 更新目标网络
                if self.steps_done % self.target_update_freq == 0:
                    self._update_target_network()

                if done:
                    break

                state = next_state
                steps += 1

            # 记录回合信息
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

            # 定期评估 (使用确定性策略，epsilon=0)
            if (episode + 1) % self.eval_every == 0:
                eval_results = self.evaluate_policy(self.get_policy(), n_episodes=self.eval_episodes)

                self.training_history.append({
                    'episode': episode + 1,
                    'avg_reward': np.mean(episode_rewards[-20:]),
                    'success_rate': eval_results['success_rate'],
                    'avg_steps': eval_results['avg_steps'],
                    'epsilon': self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                              np.exp(-self.steps_done / self.epsilon_decay)
                })

                print(f"Episode {episode + 1:4d}: "
                      f"成功率={eval_results['success_rate']:.3f}, "
                      f"平均奖励={np.mean(episode_rewards[-20:]):.1f}")

                # 早停机制：如果成功率达到1.0，停止训练
                if eval_results['success_rate'] >= 1.0:
                    print(f"🎯 成功率达到100%，在第{episode + 1}轮提前停止训练")
                    break

        print(f"{self.__class__.__name__}训练完成")
        return {
            'total_episodes': episode + 1,  # 实际训练的轮数
            'training_history': self.training_history,
            'final_success_rate': self.training_history[-1]['success_rate'] if self.training_history else 0
        }

    def _train_step(self):
        """训练一步 - 由子类实现"""
        raise NotImplementedError

    def _update_target_network(self):
        """更新目标网络 - 由子类实现"""
        pass

    def get_policy(self) -> np.ndarray:
        """获取策略"""
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            with torch.no_grad():
                state_tensor = self._state_to_tensor(s)
                q_values = self.policy_net(state_tensor)
                policy[s] = q_values.max(1)[1].item()
        return policy

    def get_value_function(self) -> np.ndarray:
        """获取值函数"""
        values = np.zeros(self.n_states)
        for s in range(self.n_states):
            with torch.no_grad():
                state_tensor = self._state_to_tensor(s)
                q_values = self.policy_net(state_tensor)
                values[s] = q_values.max(1)[0].item()
        return values
