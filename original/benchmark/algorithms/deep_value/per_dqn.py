import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import random
from .dqn import QNetwork
from .base_deep_rl import BaseDeepRL

class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state: int, action: int, reward: float,
             next_state: int, done: bool):
        """添加经验"""
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """优先级采样"""
        if len(self.buffer) < batch_size:
            return None

        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化

        # 获取样本
        samples = [self.buffer[idx] for idx in indices]

        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

class PER_DQN(BaseDeepRL):
    """带优先级经验回放的DQN"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        # 先设置PER参数，避免在reset()中出现AttributeError
        self.config = config or {}
        self.alpha = self.config.get('alpha', 0.6)  # 优先级指数
        self.beta_start = self.config.get('beta_start', 0.4)  # IS权重指数初始值
        self.beta_end = self.config.get('beta_end', 1.0)  # IS权重指数最终值
        self.beta_decay = self.config.get('beta_decay', 10000)  # β衰减步数

        super().__init__(env, self.config)

    def reset(self):
        """重置算法状态"""
        super().reset()
        # 使用优先级回放缓冲区替代普通deque
        self.memory = PrioritizedReplayBuffer(self.memory_size, self.alpha)

    def _store_transition(self, state: int, action: int, reward: float,
                         next_state: int, done: bool):
        """存储转换 - PER会自动设置初始优先级"""
        self.memory.push(state, action, reward, next_state, done)

    def _sample_batch(self):
        """从优先级缓冲区采样"""
        beta = self.beta_end + (self.beta_start - self.beta_end) * \
               np.exp(-self.steps_done / self.beta_decay)

        result = self.memory.sample(self.batch_size, beta)
        if result is None:
            return None

        samples, indices, weights = result

        states, actions, rewards, next_states, dones = zip(*samples)

        # 转换为张量
        state_batch = torch.stack([self._state_to_tensor(s) for s in states]).squeeze(1)
        action_batch = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_state_batch = torch.stack([self._state_to_tensor(s) for s in next_states]).squeeze(1)
        done_batch = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights_batch = torch.tensor(weights, dtype=torch.float32).to(self.device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, weights_batch

    def _build_network(self):
        """构建PER-DQN网络"""
        self.policy_net = QNetwork(self.n_states, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.n_states, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def _train_step(self):
        """PER-DQN训练步骤"""
        batch = self._sample_batch()
        if batch is None:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, weights = batch

        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # 计算TD误差 (用于更新优先级)
        td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach().cpu().numpy()

        # 更新优先级
        self.memory.update_priorities(indices, td_errors + 1e-6)  # 加小常数避免优先级为0

        # 计算加权损失
        loss = (weights * (current_q_values.squeeze() - target_q_values) ** 2).mean()

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
