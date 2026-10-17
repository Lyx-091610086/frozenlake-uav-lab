import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 检查是否有可用的 CUDA 设备，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")


# 网络结构
# 输入: 展平的地图 (m*n 维)
# 输出: [Q_up, Q_down, Q_left, Q_right] (4维)
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


# 经验池
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
    #地图与环境设置

    '''Small MAP
    map_data = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],

    ]) '''


    '''Mid MAP'''

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
    '''
    #Large MAP    
    map_data = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])'''



    m, n = map_data.shape
    start = (1, 1)
    goal = (m - 2, n - 2)
    # 动作定义: 上, 下, 左, 右
    move = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    MAP_DIM = m * n # 地图输入维度

    # 超参数
    total_episodes = 500
    max_steps_per_episode = 400
    batch_size = 128
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.999
    target_update_freq = 100

    #初始化
    policy_net = LargeDQN(input_dim=MAP_DIM, output_dim=4).to(device)
    target_net = LargeDQN(input_dim=MAP_DIM, output_dim=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer()

    # 可视化设置
    cmap = ListedColormap(((1, 1, 1), (0, 0, 0), (0, 0, 1), (1, 0.5, 0)))
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    #高维状态获取
    def get_state(pos, map_static):
        temp_map = map_static.copy()
        temp_map[pos] = 2.0  # 智能体标记
        temp_map[goal] = 3.0 # 目标标记
        # 展平并返回
        return temp_map.flatten().astype(np.float32)

    total_steps = 0


    for episode in range(total_episodes):
        pos = start
        #初始状态 S
        current_state = get_state(pos, map_data)
        done = False
        steps_in_episode = 0
        episode_reward = 0

        render = (episode % 50 == 0)

        while not done and steps_in_episode < max_steps_per_episode:
            #动作选择
            if random.random() < epsilon:
                action_idx = random.randint(0, 3)
            else:
                with torch.no_grad():
                    # 状态数据迁移
                    state_tensor = torch.FloatTensor(current_state).to(device)
                    q_values = policy_net(state_tensor)
                    action_idx = torch.argmax(q_values).item()

            # 执行动作
            move_x, move_y = move[action_idx]
            next_pos = (pos[0] + move_x, pos[1] + move_y)

            reward = -0.1

            if map_data[next_pos] == 1:
                reward = -5
                next_pos = pos
            elif next_pos == goal:
                reward = 100
                done = True

            # 下一状态 S'
            next_state = get_state(next_pos, map_data)
            episode_reward += reward

            # 存储经验
            buffer.add(current_state, action_idx, reward, next_state, done)

            # 训练模型
            if len(buffer) > batch_size:
                samples = buffer.sample(batch_size)

                # 数据迁移到 GPU
                batch_s = torch.FloatTensor(np.array([x[0] for x in samples])).to(device)
                batch_a = torch.LongTensor(np.array([x[1] for x in samples])).unsqueeze(1).to(device)
                batch_r = torch.FloatTensor(np.array([x[2] for x in samples])).unsqueeze(1).to(device)
                batch_s_ = torch.FloatTensor(np.array([x[3] for x in samples])).to(device)
                batch_done = torch.FloatTensor(np.array([x[4] for x in samples])).unsqueeze(1).to(device)

                # Q(s, a)
                current_q = policy_net(batch_s).gather(1, batch_a)

                # r + gamma * max(Q(s', a'))
                with torch.no_grad():
                    next_q = target_net(batch_s_).max(1)[0].unsqueeze(1)
                    target_q = batch_r + gamma * next_q * (1 - batch_done)

                loss = loss_fn(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if total_steps % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            # 更新状态和参数
            pos = next_pos
            current_state = next_state
            steps_in_episode += 1
            total_steps += 1

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # 可视化渲染
            if render:
                map_show = map_data.copy()
                map_show[pos] = 2
                map_show[goal] = 3
                ax.clear()
                ax.imshow(map_show, cmap=cmap)
                ax.set_title(f"Ep: {episode}, Step: {steps_in_episode}")
                ax.axis('off')
                plt.pause(0.01)

        # 回合结束打印日志
        print(f"Episode {episode}: Steps={steps_in_episode}, Reward={episode_reward:.2f}, Epsilon={epsilon:.4f}")

    # 保存模型
    policy_net.to('cpu')
    torch.save(policy_net.state_dict(), 'ddqn_m.pth')
    print("训练结束，模型已保存。")
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run_dqn()