import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
from .base_deep_rl import BaseDeepRL

class QNetwork(nn.Module):
    """Q网络"""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

class DQN(BaseDeepRL):
    """深度Q网络 (DQN) 算法"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

    def _build_network(self):
        """构建DQN网络"""
        self.policy_net = QNetwork(self.n_states, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.n_states, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _train_step(self):
        """DQN训练步骤"""
        batch = self._sample_batch()
        if batch is None:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
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
