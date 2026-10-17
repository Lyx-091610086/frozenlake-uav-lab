import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import List, Dict, Any, Optional
import os

class ExperimentVisualizer:
    """实验结果可视化器"""

    def __init__(self):
        # 设置matplotlib参数
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

        # 定义颜色映射
        self.cmap = LinearSegmentedColormap.from_list("ice_grid",
            ["#87CEEB", "#F0F8FF", "#FF6347", "#FFD700"])  # 冰面、起点、洞、终点

    def plot_training_curve(self, training_history: List[Dict[str, Any]],
                           save_path: str, algorithm_name: str):
        """绘制训练曲线"""
        if not training_history:
            return

        episodes = [h['episode'] for h in training_history]
        success_rates = [h.get('success_rate', 0) for h in training_history]
        epsilons = [h.get('epsilon', 0) for h in training_history]
        avg_rewards = [h.get('avg_reward', 0) for h in training_history]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 成功率曲线
        ax1.plot(episodes, success_rates, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_title('Success Rate vs Episodes')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Success Rate')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.1])

        # ε衰减曲线
        ax2.plot(episodes, epsilons, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.set_title('Epsilon Decay')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Epsilon')
        ax2.grid(True, alpha=0.3)

        # 平均奖励曲线
        ax3.plot(episodes, avg_rewards, 'g-', linewidth=2, marker='^', markersize=4)
        ax3.set_title('Average Reward (Last 100 Episodes)')
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Average Reward')
        ax3.grid(True, alpha=0.3)

        # 平均步数曲线（如果有的话）
        if 'avg_steps' in training_history[0]:
            avg_steps = [h.get('avg_steps', 0) for h in training_history]
            ax4.plot(episodes, avg_steps, 'm-', linewidth=2, marker='d', markersize=4)
            ax4.set_title('Average Steps to Goal')
            ax4.set_xlabel('Episodes')
            ax4.set_ylabel('Steps')
            ax4.grid(True, alpha=0.3)

        plt.suptitle(f'{algorithm_name.upper()} Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training curve saved to: {save_path}")

    def plot_q_heatmap(self, q_table: np.ndarray, env, save_path: str):
        """绘制Q值热力图"""
        n_states, n_actions = q_table.shape

        # 将Q表转换为网格形式
        if hasattr(env, 'nrow') and hasattr(env, 'ncol'):
            nrow, ncol = env.nrow, env.ncol
            q_grid = np.zeros((nrow, ncol, n_actions))

            for s in range(n_states):
                row, col = divmod(s, ncol)
                q_grid[row, col] = q_table[s]

            # 为每个动作创建子图
            action_names = ['Left', 'Down', 'Right', 'Up']
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            for i in range(n_actions):
                ax = axes[i // 2, i % 2]
                im = ax.imshow(q_grid[:, :, i], cmap='RdYlBu_r', aspect='equal')

                # 添加网格线
                for r in range(nrow + 1):
                    ax.axhline(r - 0.5, color='black', linewidth=1)
                for c in range(ncol + 1):
                    ax.axvline(c - 0.5, color='black', linewidth=1)

                # 添加状态标签
                for r in range(nrow):
                    for c in range(ncol):
                        state = r * ncol + c
                        ax.text(c, r, f'S{state}',
                               ha='center', va='bottom', fontsize=8, color='black')
                        ax.text(c, r, f'{q_grid[r, c, i]:.2f}',
                               ha='center', va='top', fontsize=7, color='blue')

                ax.set_title(f'Q-values for {action_names[i]}')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, shrink=0.8)

            plt.suptitle('Q-Value Heatmaps by Action', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Q-value heatmap saved to: {save_path}")

    def plot_policy(self, policy: np.ndarray, env, save_path: str):
        """绘制策略可视化"""
        if not hasattr(env, 'nrow') or not hasattr(env, 'ncol'):
            return

        nrow, ncol = env.nrow, env.ncol
        policy_grid = np.zeros((nrow, ncol), dtype=int)

        for s in range(len(policy)):
            row, col = divmod(s, ncol)
            policy_grid[row, col] = policy[s]

        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制网格
        for r in range(nrow):
            for c in range(ncol):
                # 绘制单元格背景
                cell_color = self._get_cell_color(env, r, c)
                rect = patches.Rectangle((c, r), 1, 1, facecolor=cell_color,
                                       edgecolor='black', linewidth=2)
                ax.add_patch(rect)

                # 绘制策略箭头
                action = policy_grid[r, c]
                self._draw_action_arrow(ax, c, r, action)

                # 添加状态标签
                state = r * ncol + c
                ax.text(c + 0.5, r + 0.8, f'S{state}',
                       ha='center', va='center', fontsize=10, fontweight='bold')

        ax.set_xlim(0, ncol)
        ax.set_ylim(0, nrow)
        ax.set_aspect('equal')
        ax.set_title('Learned Policy Visualization', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.axis('off')

        # 添加图例
        self._add_policy_legend(ax)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Policy visualization saved to: {save_path}")

    def plot_value_function(self, values: np.ndarray, env, save_path: str):
        """绘制值函数可视化"""
        if not hasattr(env, 'nrow') or not hasattr(env, 'ncol'):
            return

        nrow, ncol = env.nrow, env.ncol
        value_grid = np.zeros((nrow, ncol))

        for s in range(len(values)):
            row, col = divmod(s, ncol)
            value_grid[row, col] = values[s]

        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制值函数热力图
        im = ax.imshow(value_grid, cmap='viridis', aspect='equal')

        # 添加网格线
        for r in range(nrow + 1):
            ax.axhline(r - 0.5, color='white', linewidth=2)
        for c in range(ncol + 1):
            ax.axvline(c - 0.5, color='white', linewidth=2)

        # 添加状态标签和值
        for r in range(nrow):
            for c in range(ncol):
                state = r * ncol + c
                # 状态标签
                ax.text(c, r - 0.3, f'S{state}',
                       ha='center', va='center', fontsize=10, color='white', fontweight='bold')
                # 值
                ax.text(c, r + 0.3, f'{value_grid[r, c]:.3f}',
                       ha='center', va='center', fontsize=9, color='yellow')

        ax.set_title('State Value Function', fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8, label='Value')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Value function visualization saved to: {save_path}")

    def _get_cell_color(self, env, row: int, col: int) -> str:
        """获取单元格颜色"""
        try:
            desc = env.desc
            if hasattr(desc[row][col], 'decode'):
                cell = desc[row][col].decode('utf-8')
            else:
                cell = desc[row][col]

            if cell == 'S':
                return '#98FB98'  # 浅绿色 - 起点
            elif cell == 'G':
                return '#FFD700'  # 金色 - 终点
            elif cell == 'H':
                return '#FF6347'  # 红色 - 洞
            else:
                return '#87CEEB'  # 浅蓝色 - 冰面
        except:
            return '#87CEEB'  # 默认颜色

    def _draw_action_arrow(self, ax, col: float, row: float, action: int):
        """绘制动作箭头"""
        center_x, center_y = col + 0.5, row + 0.5

        # 箭头颜色和方向
        if action == 0:  # Left
            dx, dy = -0.3, 0
            color = 'red'
        elif action == 1:  # Down
            dx, dy = 0, 0.3
            color = 'blue'
        elif action == 2:  # Right
            dx, dy = 0.3, 0
            color = 'green'
        elif action == 3:  # Up
            dx, dy = 0, -0.3
            color = 'purple'
        else:
            return

        # 绘制箭头
        ax.arrow(center_x - dx, center_y - dy, dx, dy,
                head_width=0.1, head_length=0.1, fc=color, ec=color,
                linewidth=2, alpha=0.8)

    def _add_policy_legend(self, ax):
        """添加策略图例"""
        # 创建图例元素
        legend_elements = [
            patches.Patch(facecolor='#98FB98', edgecolor='black', label='Start (S)'),
            patches.Patch(facecolor='#FFD700', edgecolor='black', label='Goal (G)'),
            patches.Patch(facecolor='#FF6347', edgecolor='black', label='Hole (H)'),
            patches.Patch(facecolor='#87CEEB', edgecolor='black', label='Frozen (F)'),
            plt.arrow(0, 0, 0.1, 0, head_width=0.05, head_length=0.05, fc='red', ec='red', label='Left (0)'),
            plt.arrow(0, 0, 0, 0.1, head_width=0.05, head_length=0.05, fc='blue', ec='blue', label='Down (1)'),
            plt.arrow(0, 0, 0.1, 0, head_width=0.05, head_length=0.05, fc='green', ec='green', label='Right (2)'),
            plt.arrow(0, 0, 0, -0.1, head_width=0.05, head_length=0.05, fc='purple', ec='purple', label='Up (3)')
        ]

        # 添加图例
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',
                 borderaxespad=0., fontsize=10)
