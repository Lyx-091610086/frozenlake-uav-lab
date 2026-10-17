import numpy as np
import heapq
from typing import Dict, Any, Optional, List, Tuple
from ..base import RLAlgorithm

class AStar(RLAlgorithm):
    """A*算法 - 最短路径搜索"""

    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(env, config)

        # 获取起点和终点位置
        self.start_pos = None
        self.goal_pos = None
        self._find_special_positions()

    def _find_special_positions(self):
        """找到起点和终点位置"""
        desc = self.env.desc
        for i in range(len(desc)):
            for j in range(len(desc[i])):
                cell = desc[i][j]
                # 处理字节字符串或普通字符串
                if isinstance(cell, bytes):
                    cell = cell.decode('utf-8')
                if cell == 'S':
                    self.start_pos = (i, j)
                elif cell == 'G':
                    self.goal_pos = (i, j)

    def reset(self):
        """重置算法状态"""
        self.policy = np.zeros(self.n_states, dtype=int)
        self.V = np.zeros(self.n_states)

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """曼哈顿距离作为启发函数"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取有效邻居位置"""
        row, col = pos
        neighbors = []

        # 四个可能的方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # 检查边界
            if 0 <= new_row < self.env.nrow and 0 <= new_col < self.env.ncol:
                # 检查是否是洞（H）
                cell = self.env.desc[new_row][new_col]
                if isinstance(cell, bytes):
                    cell = cell.decode('utf-8')
                if cell != 'H':
                    neighbors.append((new_row, new_col))

        return neighbors

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """位置转换为状态"""
        return pos[0] * self.env.ncol + pos[1]

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """状态转换为位置"""
        return divmod(state, self.env.ncol)

    def train(self, **kwargs) -> Dict[str, Any]:
        """执行A*搜索"""
        print("开始A*搜索...")

        if self.start_pos is None or self.goal_pos is None:
            raise ValueError("找不到起点或终点")

        # A*搜索
        frontier = []  # 优先队列
        heapq.heappush(frontier, (0, self.start_pos))  # (优先级, 位置)

        came_from = {self.start_pos: None}  # 记录路径
        cost_so_far = {self.start_pos: 0}  # 到当前位置的代价

        found = False

        while frontier and not found:
            current_priority, current = heapq.heappop(frontier)

            if current == self.goal_pos:
                found = True
                break

            for neighbor in self._get_neighbors(current):
                # 计算新代价（每个移动代价为1）
                new_cost = cost_so_far[current] + 1

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost

                    # 优先级 = 实际代价 + 启发函数
                    priority = new_cost + self._manhattan_distance(neighbor, self.goal_pos)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if not found:
            print("A*搜索失败：找不到路径")
            return {'success': False, 'path_length': 0}

        # 重建路径
        path = []
        current = self.goal_pos
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()

        # 将路径转换为策略
        self._path_to_policy(path)

        # 计算值函数（到目标的距离）
        self._compute_value_function(cost_so_far)

        path_length = len(path) - 1  # 不包括起点
        print(f"A*搜索成功，路径长度: {path_length}")

        return {
            'success': True,
            'path_length': path_length,
            'path': path,
            'nodes_explored': len(came_from)
        }

    def _path_to_policy(self, path: List[Tuple[int, int]]):
        """将路径转换为策略"""
        # 为路径上的每个状态设置动作
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]

            current_state = self._pos_to_state(current_pos)

            # 计算动作方向
            dr = next_pos[0] - current_pos[0]
            dc = next_pos[1] - current_pos[1]

            # 转换为动作 (0:LEFT, 1:DOWN, 2:RIGHT, 3:UP)
            if dr == 0 and dc == -1:  # 左
                action = 0
            elif dr == 1 and dc == 0:  # 下
                action = 1
            elif dr == 0 and dc == 1:  # 右
                action = 2
            elif dr == -1 and dc == 0:  # 上
                action = 3
            else:
                continue

            self.policy[current_state] = action

    def _compute_value_function(self, cost_so_far: Dict[Tuple[int, int], float]):
        """计算值函数（到目标的距离的负值）"""
        for pos, cost in cost_so_far.items():
            state = self._pos_to_state(pos)
            self.V[state] = -cost  # 距离越小，价值越高

    def get_policy(self) -> np.ndarray:
        """获取策略"""
        return self.policy.copy()

    def get_value_function(self) -> np.ndarray:
        """获取值函数"""
        return self.V.copy()
