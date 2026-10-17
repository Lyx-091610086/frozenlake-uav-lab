import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LargeDQN(nn.Module):
    def __init__(self, input_dim, output_dim=4):
        super(LargeDQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def add(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# 训练主逻辑
def run_dqn():
    # 基础地图定义
    map_data = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])

    #定义路径权重
    weight_map = np.ones_like(map_data, dtype=np.float32)
    weight_map[1:4, 4:6] = 8.0  # 在地图上方中央设置一块高权重
    weight_map[6:8, 4:5] = 5.0  # 设置另一条较难通过的路径

    m, n = map_data.shape
    start = (1, 1)
    goal = (m - 2, n - 2)
    move = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    MAP_DIM = m * n

    #超参数
    total_episodes = 600
    max_steps_per_episode = 400
    batch_size = 128
    gamma = 0.98
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.998
    target_update_freq = 200

    policy_net = LargeDQN(input_dim=MAP_DIM, output_dim=4).to(device)
    target_net = LargeDQN(input_dim=MAP_DIM, output_dim=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.0003)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer()

    # 可视化增强：使用不同的颜色深度表示权重
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    def get_state(pos, map_static):
        temp_map = map_static.copy().astype(np.float32)
        # 将权重信息融合进状态输入
        state_view = temp_map + (weight_map * 0.1)
        state_view[pos] = 2.0
        state_view[goal] = 3.0
        return state_view.flatten().astype(np.float32)

    total_steps = 0

    for episode in range(total_episodes):
        pos = start
        current_state = get_state(pos, map_data)
        done = False
        steps_in_episode = 0
        episode_reward = 0

        render = (episode % 50 == 0)

        while not done and steps_in_episode < max_steps_per_episode:
            if random.random() < epsilon:
                action_idx = random.randint(0, 3)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(current_state).to(device)
                    q_values = policy_net(state_tensor)
                    action_idx = torch.argmax(q_values).item()

            move_x, move_y = move[action_idx]
            next_pos = (pos[0] + move_x, pos[1] + move_y)

            #基于权重的奖励逻辑
            if map_data[next_pos] == 1: # 撞墙/危险区
                reward = -10.0
                next_pos = pos
            elif next_pos == goal:      # 终点
                reward = 150.0
                done = True
            else:
                # 核心公式：奖励 = - (基础消耗 + 路径权重惩罚)
                reward = -0.1 * weight_map[next_pos]

            next_state = get_state(next_pos, map_data)
            episode_reward += reward

            buffer.add(current_state, action_idx, reward, next_state, done)

            if len(buffer) > batch_size:
                samples = buffer.sample(batch_size)
                batch_s = torch.FloatTensor(np.array([x[0] for x in samples])).to(device)
                batch_a = torch.LongTensor(np.array([x[1] for x in samples])).unsqueeze(1).to(device)
                batch_r = torch.FloatTensor(np.array([x[2] for x in samples])).unsqueeze(1).to(device)
                batch_s_ = torch.FloatTensor(np.array([x[3] for x in samples])).to(device)
                batch_done = torch.FloatTensor(np.array([x[4] for x in samples])).unsqueeze(1).to(device)

                current_q = policy_net(batch_s).gather(1, batch_a)
                with torch.no_grad():
                    next_q = target_net(batch_s_).max(1)[0].unsqueeze(1)
                    target_q = batch_r + gamma * next_q * (1 - batch_done)

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if total_steps % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            pos = next_pos
            current_state = next_state
            steps_in_episode += 1
            total_steps += 1

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            if render:
                # 可视化地图
                vis_map = map_data.copy().astype(np.float32)
                vis_map = vis_map + (weight_map * 0.05)
                vis_map[pos] = 2.0
                vis_map[goal] = 3.0
                ax.clear()
                ax.imshow(vis_map, cmap='YlGnBu')
                ax.set_title(f"Ep: {episode} (Weight Path Finding)")
                ax.axis('off')
                plt.pause(0.001)

        if episode % 10 == 0:
            print(f"Episode {episode}: Steps={steps_in_episode}, Reward={episode_reward:.2f}, Epsilon={epsilon:.4f}")

    policy_net.to('cpu')
    torch.save(policy_net.state_dict(), 'dqn_weighted_path.pth')
    print("训练结束。")
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    run_dqn()