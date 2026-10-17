import gym
import torch
import torch.nn as nn
import numpy as np
import time
from random import choice


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

def map_to_gym_desc(full_map, start_global, goal_global):

    # 切去四周的一圈
    inner_map = full_map[1:-1, 1:-1]

    # 转换 Start 和 Goal 到局部坐标
    start_local = (start_global[0] - 1, start_global[1] - 1)
    goal_local = (goal_global[0] - 1, goal_global[1] - 1)

    desc = []
    rows, cols = inner_map.shape
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            if (r, c) == start_local:
                row_str += "S"
            elif (r, c) == goal_local:
                row_str += "G"
            elif inner_map[r, c] == 1:
                row_str += "H"
            else:
                row_str += "F"
        desc.append(row_str)
    return desc


def get_state(pos, map_static, goal):
    temp_map = map_static.copy().astype(np.float32)
    temp_map[pos] = 2.0  # 智能体
    temp_map[goal] = 3.0  # 目标
    return temp_map.flatten()


def run_gym_simulation(model_path='ddqn_s.pth'):
    map_data = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ])
    m, n = map_data.shape
    MAP_DIM = m * n

    start_pos = (1, 1)
    goal_pos = (m - 2, n - 2)

    # 动作定义 (dx, dy): 0:上, 1:下, 2:左, 3:右
    move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    action_names = ["上", "下", "左", "右"]

    gym_desc = map_to_gym_desc(map_data, start_pos, goal_pos)

    # 创建环境
    env = gym.make("FrozenLake-v1", desc=gym_desc, is_slippery=False, render_mode="human")
    env.reset()
    env.render()

    policy_net = LargeDQN(input_dim=MAP_DIM, output_dim=4)
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        policy_net.eval()
        print(f" 成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f" 模型文件未找到: {model_path}")
        return

    # 0:上, 1:下, 2:左, 3:右
    action_converter = {
        0: 3,  # 用户上 -> Gym上(3)
        1: 1,  # 用户下 -> Gym下(1)
        2: 0,  # 用户左 -> Gym左(0)
        3: 2  # 用户右 -> Gym右(2)
    }

    pos = start_pos  # 当前位于 (1, 1)
    max_steps = 100

    print("\n--- 开始仿真 ---")
    time.sleep(1)

    for step in range(1, max_steps + 1):
        if pos == goal_pos:
            print(f"\n 目标达成： 总步数: {step}")
            break

        current_state = get_state(pos, map_data, goal_pos)

        # DQN 决策
        with torch.no_grad():
            q_values = policy_net(torch.FloatTensor(current_state).unsqueeze(0))
            best_action = torch.argmax(q_values).item()

        # Stochasticity
        actual_action = best_action
        if best_action == 0:
            actual_action = choice([0, 2, 3])
        elif best_action == 1:
            actual_action = choice([1, 2, 3])
        elif best_action == 2:
            actual_action = choice([2, 0, 1])
        elif best_action == 3:
            actual_action = choice([3, 0, 1])

        # 打印信息
        if actual_action != best_action:
            print(f"Step {step}: ️ 气流干扰：计划:{action_names[best_action]} -> 实际:{action_names[actual_action]}")

        # 计算物理位置
        dx, dy = move_map[actual_action]
        new_pos = (pos[0] + dx, pos[1] + dy)

        # 撞墙逻辑：如果撞到 1，保持原地
        if map_data[new_pos] == 1:
            new_pos = pos

        # 同步 Gym 画面
        gym_r = new_pos[0] - 1
        gym_c = new_pos[1] - 1

        gym_state_index = gym_r * 4 + gym_c
        env.unwrapped.s = gym_state_index

        # 刷新画面
        env.render()
        pos = new_pos
        time.sleep(0.5)  # 慢放

    time.sleep(2)
    env.close()
    print("演示结束")


if __name__ == '__main__':
    run_gym_simulation(model_path='ddqn_s.pth')