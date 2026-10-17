import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
from .base_deep_rl import BaseDeepRL

class DuelingQNetwork(nn.Module):
    """Dueling DQN网络 - 分离Value和Advantage"""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super(DuelingQNetwork, self).__init__()

        # 特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        # Value流：估计状态值
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # 输出单个值
        )

        # Advantage流：估计优势函数
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)  # 输出每个动作的优势
        )

    def forward(self, x):
        features = self.feature_layer(x)

        # 计算Value和Advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # 组合Q值: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        # 这样可以保证优势函数的平均值为0
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

class DuelingDQN(BaseDeepRL):
    """Dueling DQN算法 - 分离状态值和动作优势"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

    def _build_network(self):
        """构建Dueling DQN网络"""
        self.policy_net = DuelingQNetwork(self.n_states, self.n_actions).to(self.device)
        self.target_net = DuelingQNetwork(self.n_states, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _train_step(self):
        """Dueling DQN训练步骤"""
        batch = self._sample_batch()
        if batch is None:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # 计算目标Q值 (使用Double DQN的方式)
        with torch.no_grad():
            # 使用policy_net选择下一个动作
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)

            # 使用target_net计算选中动作的值
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze()

            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # 计算损失
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
