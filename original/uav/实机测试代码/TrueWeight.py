import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import serial.tools.list_ports
from struct import pack_into

from SerialThread import SerialThread
from CommandConstructor import CmdType

MODEL_PATH = 'dqn_weighted_path.pth'
STEP_DISTANCE = 50
FLIGHT_HEIGHT = 60
COM_PORT = 'COM5'

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

#网络结构
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


# -辅助函数
def get_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]
def manual_hover(driver):
    cc = driver.send()
    params = cc._base_params(0xFE)
    pack_into('<B', params, 3, 4)
    cc.sendCommand(cc.join_cmd(CmdType.SINGLE_CONTROL, params))


# 主控制程序
def run_weighted_drone_control():
    #连接串口
    available_ports = get_serial_ports()
    print(f"当前可用串口: {available_ports}")
    port_name = input(f"请输入串口号 (默认为 {COM_PORT}): ").strip()
    if not port_name: port_name = COM_PORT

    print(f"正在打开串口 {port_name}...")
    try:
        driver = SerialThread(port_name)
        time.sleep(2)
        print(" 串口连接成功！")
    except Exception as e:
        print(f" 串口失败: {e}")
        return

    # 地图与权重初始化
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

    # 权重地图
    # 默认权重 1.0, 高权重区域设为 8.0 或 5.0
    weight_map = np.ones_like(map_data, dtype=np.float32)
    weight_map[1:4, 4:6] = 8.0  # 高代价区域 1
    weight_map[6:8, 4:5] = 5.0  # 高代价区域 2

    m, n = map_data.shape
    MAP_DIM = m * n
    start = (1, 1)
    goal = (m - 2, n - 2)

    # 动作映射: 0上, 1下, 2左, 3右
    move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    action_names = ["前 (上)", "后 (下)", "左", "右"]

    #状态获取函数
    def get_state(pos, map_static):
        temp_map = map_static.copy().astype(np.float32)
        # 融合权重信息到输入向量
        state_view = temp_map + (weight_map * 0.1)
        state_view[pos] = 2.0  # 智能体标记
        state_view[goal] = 3.0  # 目标标记
        return state_view.flatten().astype(np.float32)

    #加载模型
    policy_net = LargeDQN(input_dim=MAP_DIM, output_dim=4)
    try:
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
        policy_net.eval()
        print(f" 成功加载加权模型: {MODEL_PATH}")
    except FileNotFoundError:
        print(f" 错误: 未找到模型文件 {MODEL_PATH}")
        driver.shutdown()
        return

    #可视化初始化
    pos = start
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))


    colors = [
        '#FFFFFF',  # 0: 空地 (白)
        '#000000',  # 1: 墙 (黑)
        '#0000FF',  # 2: 飞机 (蓝)
        '#FF8C00',  # 3: 终点 (橙)
        '#FFB6C1',  # 4: 权重=5 (浅红)
        '#8B0000',  # 5: 权重=8 (深红)
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0, 1, 2, 3, 4, 5, 6], cmap.N)

    def update_plot(curr_pos, step_count):
        # 构造显示矩阵
        display_grid = np.zeros_like(map_data, dtype=int)

        # 填充地形
        display_grid[map_data == 1] = 1  # 墙

        # 填充权重区域
        mask_w5 = (weight_map == 5.0) & (map_data == 0)
        mask_w8 = (weight_map == 8.0) & (map_data == 0)
        display_grid[mask_w5] = 4
        display_grid[mask_w8] = 5

        # 填充动态物体
        display_grid[goal] = 3
        display_grid[curr_pos] = 2

        ax.clear()
        ax.imshow(display_grid, cmap=cmap, norm=norm)

        # 添加网格线辅助
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, m, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)

        ax.set_title(f"加权路径控制 | 步数: {step_count} | 当前权重: {weight_map[curr_pos]:.1f}")

        # 标注特殊区域
        ax.text(4.5, 2, "高危区(W=8)", color='white', ha='center', fontsize=8, fontweight='bold')
        ax.text(4.5, 7, "中危区(W=5)", color='black', ha='center', fontsize=8)

        plt.pause(0.1)

    # 初始显示
    update_plot(pos, 0)

    # 起飞序列
    print(f" 起飞指令已发送 (高度 {FLIGHT_HEIGHT})")
    driver.send().takeoff(FLIGHT_HEIGHT)

    for i in range(5, 0, -1):
        print(f"等待起飞稳定... {i}s")
        if i < 3: manual_hover(driver)
        time.sleep(1)

    #主循环 (确定性动作 + 加权逻辑)
    max_steps = 100
    for step in range(1, max_steps + 1):
        if pos == goal:
            print("\n 到达目的地！")
            break

        # DQN 决策
        current_state = get_state(pos, map_data)
        with torch.no_grad():
            q_vals = policy_net(torch.FloatTensor(current_state).unsqueeze(0))
            best_action = torch.argmax(q_vals).item()

        action_name = action_names[best_action]
        curr_weight = weight_map[pos]
        print(f"Step {step}: 动作->{action_name} | 当前地块代价: {curr_weight}")

        #计算新位置
        move_x, move_y = move_map[best_action]
        new_pos = (pos[0] + move_x, pos[1] + move_y)

        # 发送指令
        if map_data[new_pos] == 1:
            print(" 前方是墙，强制悬停！")
            manual_hover(driver)
            new_pos = pos
        else:
            # 发送物理指令 (1前, 2后, 3左, 4右)
            if best_action == 0:
                driver.send().move(1, STEP_DISTANCE)
            elif best_action == 1:
                driver.send().move(2, STEP_DISTANCE)
            elif best_action == 2:
                driver.send().move(3, STEP_DISTANCE)
            elif best_action == 3:
                driver.send().move(4, STEP_DISTANCE)

            # 物理动作等待
            time.sleep(1.5)
            manual_hover(driver)
            time.sleep(0.5)

        # 更新可视化
        pos = new_pos
        update_plot(pos, step)

    #  降落
    print(" 发送降落指令...")
    driver.send().land()
    time.sleep(3)

    plt.ioff()
    plt.show()
    driver.shutdown()
    print("程序结束")

if __name__ == '__main__':
    run_weighted_drone_control()