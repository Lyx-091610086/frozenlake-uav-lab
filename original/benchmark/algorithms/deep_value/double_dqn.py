import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
from .dqn import QNetwork
from .base_deep_rl import BaseDeepRL

class DoubleDQN(BaseDeepRL):
    """Double DQN算法 - 缓解过估计问题"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

    def _build_network(self):
        """构建Double DQN网络"""
        self.policy_net = QNetwork(self.n_states, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.n_states, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _train_step(self):
        """Double DQN训练步骤"""
        batch = self._sample_batch()
        if batch is None:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Double DQN: 使用policy_net选择动作，target_net计算值
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
