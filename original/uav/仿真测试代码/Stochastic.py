import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from random import choice  # 随机选择函数

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# 网络结构
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


# 离线寻路与可视化函数
def load_and_visualize_path(model_path='ddqn_m.pth'):
    # 示例地图 (Small MAP)
    '''Small Map'''
    map_data = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ])

    '''Mid MAP
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

    '''
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
    ])
    '''
    m, n = map_data.shape
    MAP_DIM = m * n
    start = (1, 1)
    goal = (m - 2, n - 2)

    # 动作定义: 0:上, 1:下, 2:左, 3:右
    move = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 高维状态获取函数
    def get_state(pos, map_static):
        temp_map = map_static.copy()
        temp_map[pos] = 2.0
        temp_map[goal] = 3.0
        return temp_map.flatten().astype(np.float32)

    # 颜色映射
    cmap = ListedColormap((
        (1.0, 1.0, 1.0),  # 0: 空地
        (0.0, 0.0, 0.0),  # 1: 障碍
        (0.0, 0.0, 1.0),  # 2: 智能体
        (1.0, 0.5, 0.0)  # 3: 目标
    ))

    #模型加载
    policy_net = LargeDQN(input_dim=MAP_DIM, output_dim=4)
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        policy_net.eval()
        print(f"成功加载模型参数: {model_path}")
    except FileNotFoundError:
        print(f" 错误: 未找到模型文件 {model_path}。")
        return

    # 可视化初始化
    pos = start
    map_display = map_data.copy()
    map_display[start[0], start[1]] = 2
    map_display[goal[0], goal[1]] = 3

    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(map_display, cmap=cmap)
    ax.set_title("DQN 离线寻路 (加入随机干扰)")
    ax.axis('off')
    plt.ion()
    plt.show()

    max_steps = 500

    # 寻路循环
    for step in range(1, max_steps + 1):
        if pos == goal:
            break

        current_state = get_state(pos, map_data)

        # 模型决策
        with torch.no_grad():
            q_values_tensor = policy_net(torch.FloatTensor(current_state).unsqueeze(0))
            best_action_idx = torch.argmax(q_values_tensor).item()

        # 加入不稳定因素
        actual_action_idx = best_action_idx  # 默认为模型选择的动作

        # 动作映射回顾: 0:上, 1:下, 2:左, 3:右
        # 逻辑：尝试向某方向飞时，有概率偏离到左右两侧，但不会反向飞
        if best_action_idx == 0:  # 计划向上
            actual_action_idx = choice([0, 2, 3])  # 实际可能：上、左、右
        elif best_action_idx == 1:  # 计划向下
            actual_action_idx = choice([1, 2, 3])  # 实际可能：下、左、右
        elif best_action_idx == 2:  # 计划向左
            actual_action_idx = choice([2, 0, 1])  # 实际可能：左、上、下
        elif best_action_idx == 3:  # 计划向右
            actual_action_idx = choice([3, 0, 1])  # 实际可能：右、上、下

        if actual_action_idx != best_action_idx:
            direction_name = ["上", "下", "左", "右"]
            print(
                f"步数 {step}: 发生气流干扰! 原计划: {direction_name[best_action_idx]} -> 实际: {direction_name[actual_action_idx]}")
        # -----------------------------------------------------

        #计算新位置
        if actual_action_idx != -1:
            move_x, move_y = move[actual_action_idx]
            new_pos = (pos[0] + move_x, pos[1] + move_y)
        else:
            new_pos = pos

        # 检查撞墙
        if map_data[new_pos] == 1:
            new_pos = pos

        #更新可视化
        map_display[pos] = 0
        map_display[new_pos] = 2
        map_display[goal] = 3

        img.set_data(map_display)
        ax.set_title(f"DQN 寻路 (干扰模式) - 步数: {step}")
        plt.pause(0.2)

        pos = new_pos

    #结果输出
    plt.ioff()
    if pos == goal:
        print(f"\n 目标达成! 总步数: {step}")
    else:
        print(f"\n 达到最大步数限制，未能找到路径。")
    plt.show()


if __name__ == '__main__':
    load_and_visualize_path(model_path='ddqn_s.pth')