import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm  # 引入必要的颜色工具
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


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
def load_and_visualize_path(model_path='dqn_weighted_path.pth'):
    # 基础地图设置
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

    # 权重设置
    weight_map = np.ones_like(map_data, dtype=np.float32)
    weight_map[1:4, 4:6] = 8.0  # 高代价区域 1
    weight_map[6:8, 4:5] = 5.0  # 高代价区域 2

    m, n = map_data.shape
    MAP_DIM = m * n
    start = (1, 1)
    goal = (m - 2, n - 2)
    move = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右

    # 同步状态获取函数 (用于输入给神经网络)
    def get_state(pos, map_static):
        temp_map = map_static.copy().astype(np.float32)
        state_view = temp_map + (weight_map * 0.1)
        state_view[pos] = 2.0
        state_view[goal] = 3.0
        return state_view.flatten().astype(np.float32)

    # 模型加载
    policy_net = LargeDQN(input_dim=MAP_DIM, output_dim=4)
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        policy_net.eval()
        print(f" 成功加载加权寻路模型: {model_path}")
    except FileNotFoundError:
        print(f" 错误: 未找到文件 {model_path}")
        return

    # --- 可视化样式定义 (移植部分) ---
    # 定义特定的颜色映射
    colors = [
        '#FFFFFF',  # 0: 空地 (白)
        '#000000',  # 1: 墙 (黑)
        '#0000FF',  # 2: 智能体 (蓝)
        '#FF8C00',  # 3: 终点 (橙)
        '#FFB6C1',  # 4: 权重=5 (浅红)
        '#8B0000',  # 5: 权重=8 (深红)
    ]
    cmap = ListedColormap(colors)
    # 定义边界，确保整数值对应正确的颜色
    norm = BoundaryNorm([0, 1, 2, 3, 4, 5, 6], cmap.N)
    # ----------------------------------

    # 寻路模拟可视化
    pos = start
    fig, ax = plt.subplots(figsize=(8, 8))
    max_steps = 100

    plt.ion()
    plt.show()

    for step in range(1, max_steps + 1):
        if pos == goal:
            break

        # 1. 神经网络决策
        current_state = get_state(pos, map_data)
        with torch.no_grad():
            q_values_tensor = policy_net(torch.FloatTensor(current_state).unsqueeze(0))
            best_action_idx = torch.argmax(q_values_tensor).item()

        # 2. 物理位置更新逻辑
        move_x, move_y = move[best_action_idx]
        new_pos = (pos[0] + move_x, pos[1] + move_y)

        if map_data[new_pos] == 1:
            new_pos = pos  # 撞墙停留原地

        # 3. 构造可视化矩阵
        display_grid = np.zeros_like(map_data, dtype=int)

        # (1) 填充基础地形
        display_grid[map_data == 1] = 1  # 墙

        # (2) 填充权重区域 (注意不覆盖墙)
        mask_w5 = (weight_map == 5.0) & (map_data == 0)
        mask_w8 = (weight_map == 8.0) & (map_data == 0)
        display_grid[mask_w5] = 4
        display_grid[mask_w8] = 5

        # (3) 填充动态物体
        display_grid[goal] = 3
        display_grid[new_pos] = 2

        # 4. 绘图更新
        ax.clear()
        ax.imshow(display_grid, cmap=cmap, norm=norm)

        # 添加网格线
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, m, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)

        # 标题与文本
        curr_weight = weight_map[new_pos]
        ax.set_title(f"离线模拟中... | 步数: {step} | 当前地块代价: {curr_weight:.1f}")

        # 标注特殊区域文字
        ax.text(4.5, 2, "高危区(W=8)", color='white', ha='center', fontsize=8, fontweight='bold')
        ax.text(4.0, 7, "中危区(W=5)", color='black', ha='center', fontsize=8)

        plt.pause(0.2)
        pos = new_pos

    plt.ioff()
    if pos == goal:
        print(f" 成功找到最佳安全路径! 总步数: {step}")
    else:
        print(" 未能到达目标。")
    plt.show()


if __name__ == '__main__':
    load_and_visualize_path()