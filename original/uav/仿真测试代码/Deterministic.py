import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

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


#离线寻路与可视化函数
def load_and_visualize_path(model_path='ddqn_m.pth'):

    '''Small MAP
    map_data = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],

    ])'''

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
    # Large MAP
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
    MAP_DIM = m * n
    start = (1, 1)
    goal = (m - 2, n - 2)
    move = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 动作: 上, 下, 左, 右

    def get_state(pos, map_static):
        temp_map = map_static.copy()
        temp_map[pos] = 2.0  # 智能体标记
        temp_map[goal] = 3.0 # 目标标记
        #展平并返回
        return temp_map.flatten().astype(np.float32)

    # 颜色映射：0:空地(白) 1:障碍(黑) 2:智能体(蓝) 3:目标(橙)
    cmap = ListedColormap((
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.5, 0.0)
    ))

    #模型加载
    policy_net = LargeDQN(input_dim=MAP_DIM, output_dim=4)
    try:
        policy_net.load_state_dict(torch.load(model_path, weights_only=True))
        policy_net.eval()  # 设置为评估模式
        print(f"成功加载模型参数: {model_path}")
    except FileNotFoundError:
        print(f"错误: 未找到模型文件 {model_path}。")
        return

    # 寻路模拟与可视化设置
    pos = start
    map_display = map_data.copy()
    map_display[start[0], start[1]] = 2  # 智能体
    map_display[goal[0], goal[1]] = 3  # 目标

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(map_display, cmap=cmap)
    ax.set_title("DQN 离线寻路可视化 (步数: 0)")
    ax.axis('off')
    plt.ion()  # 开启交互模式，用于动画更新
    plt.show()

    max_steps = 500  # 最大步数限制

    #离线寻路循环
    for step in range(1, max_steps + 1):
        if pos == goal:
            break

        current_state = get_state(pos, map_data)
        # 模型预测所有动作的 Q 值
        with torch.no_grad():
            q_values_tensor = policy_net(torch.FloatTensor(current_state).unsqueeze(0))

            #从 4 个 Q 值中选择最大的那个，即最优动作
            best_action_idx = torch.argmax(q_values_tensor).item()

        # 执行最优动作
        if best_action_idx != -1:
            move_x, move_y = move[best_action_idx]
            new_pos = (pos[0] + move_x, pos[1] + move_y)
        else:
            new_pos = pos

        # 检查是否撞墙
        if map_data[new_pos] == 1:
            new_pos = pos

        #更新可视化
        map_display[pos] = 0  # 原位置变为空地
        map_display[new_pos] = 2  # 新位置变为智能体
        map_display[goal] = 3  # 确保目标位置一直是目标

        img.set_data(map_display)
        ax.set_title(f"DQN 离线寻路可视化 (步数: {step})")
        plt.pause(0.1)

        pos = new_pos

    #结果输出
    plt.ioff()
    if pos == goal:
        ax.set_title(f" 寻路成功! 总步数: {step}")
        print(f"\n 目标达成! 总步数: {step}")
    else:
        ax.set_title(f" 未找到路径，达到最大步数: {max_steps}")
        print(f"\n 达到最大步数限制，未能找到路径。")

    plt.show()


if __name__ == '__main__':
    load_and_visualize_path(model_path='ddqn_m.pth')
