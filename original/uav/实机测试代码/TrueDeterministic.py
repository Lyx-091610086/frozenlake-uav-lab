import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import serial.tools.list_ports
from struct import pack_into

#导入串口通信模块
from SerialThread import SerialThread
from CommandConstructor import CmdType

# 配置部分
MODEL_PATH = 'ddqn_m.pth'
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
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x): return self.model(x)


# 辅助函数
def get_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

def manual_hover(driver):
    cc = driver.send()
    params = cc._base_params(0xFE)
    pack_into('<B', params, 3, 4)  # 强制 Mode=4
    cc.sendCommand(cc.join_cmd(CmdType.SINGLE_CONTROL, params))


#主程序
def run_drone_control():
    # 连接串口
    available_ports = get_serial_ports()
    print(f"当前可用串口: {available_ports}")
    port_name = input(f"请输入串口号 (默认为 {COM_PORT}): ").strip()
    if not port_name: port_name = COM_PORT

    print(f"正在打开串口 {port_name}...")
    try:
        driver = SerialThread(port_name)
        time.sleep(2)
        print("串口连接成功！")
    except Exception as e:
        print(f"串口失败: {e}")
        return

    #  地图初始化
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

    m, n = map_data.shape
    MAP_DIM = m * n
    start = (1, 1)
    goal = (m - 2, n - 2)

    # 动作映射: 0上, 1下, 2左, 3右
    move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    action_names = ["前 (上)", "后 (下)", "左", "右"]

    def get_state(pos, map_static):
        temp_map = map_static.copy()
        temp_map[pos] = 2.0
        temp_map[goal] = 3.0
        return temp_map.flatten().astype(np.float32)

    #加载模型
    policy_net = LargeDQN(input_dim=MAP_DIM, output_dim=4)
    try:
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
        policy_net.eval()
        print(f" 模型加载成功: {MODEL_PATH}")
    except FileNotFoundError:
        print(f" 模型未找到")
        driver.shutdown()
        return

    # 可视化初始化
    cmap = ListedColormap(((1, 1, 1), (0, 0, 0), (0, 0, 1), (1, 0.5, 0)))
    pos = start
    map_display = map_data.copy()
    map_display[start] = 2
    map_display[goal] = 3

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(map_display, cmap=cmap)
    ax.set_title("DQN 无人机控制 (确定性模式)")
    plt.show()
    plt.pause(0.1)

    print(f" 起飞指令已发送 (高度 {FLIGHT_HEIGHT})")
    driver.send().takeoff(FLIGHT_HEIGHT)

    for i in range(5, 0, -1):
        print(f"等待起飞稳定... {i}s")
        if i < 3: manual_hover(driver)  # 确保悬停
        time.sleep(1)

    # 主循环 (确定性动作)
    max_steps = 100
    for step in range(1, max_steps + 1):
        if pos == goal:
            print("\n 到达目的地！")
            break

        # DQN 决策 (确定性)
        current_state = get_state(pos, map_data)
        with torch.no_grad():
            q_vals = policy_net(torch.FloatTensor(current_state).unsqueeze(0))
            # 不加随机干扰
            best_action = torch.argmax(q_vals).item()

        print(f"Step {step}: 执行动作 -> {action_names[best_action]}")

        # 计算新位置
        move_x, move_y = move_map[best_action]
        new_pos = (pos[0] + move_x, pos[1] + move_y)

        # 发送指令
        if map_data[new_pos] == 1:
            print(" 前方是墙，强制悬停！")
            manual_hover(driver)
            new_pos = pos
        else:
            if best_action == 0:  # 上 -> 前
                driver.send().move(1, STEP_DISTANCE)
            elif best_action == 1:  # 下 -> 后
                driver.send().move(2, STEP_DISTANCE)
            elif best_action == 2:  # 左 -> 左
                driver.send().move(3, STEP_DISTANCE)
            elif best_action == 3:  # 右 -> 右
                driver.send().move(4, STEP_DISTANCE)

            # 等待动作完成并悬停
            time.sleep(1.5)
            manual_hover(driver)
            time.sleep(0.5)

        #更新可视化
        map_display[pos] = 0
        map_display[new_pos] = 2
        map_display[goal] = 3
        img.set_data(map_display)
        ax.set_title(f"Step: {step} | 动作: {action_names[best_action]}")
        plt.pause(0.1)

        pos = new_pos

    print(" 发送降落指令...")
    driver.send().land()
    time.sleep(3)

    plt.ioff()
    plt.show()
    driver.shutdown()
    print("程序结束")


if __name__ == '__main__':
    run_drone_control()