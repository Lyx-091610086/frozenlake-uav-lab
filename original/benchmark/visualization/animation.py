#!/usr/bin/env python3
"""
强化学习路径动画可视化模块
支持FrozenLake环境中的策略执行动画
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
from typing import List, Tuple, Dict, Any, Optional

class RLAnimation:
    """强化学习路径动画生成器"""

    def __init__(self):
        # 设置matplotlib参数
        plt.rcParams['figure.figsize'] = (8, 8)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        # 定义颜色映射
        self.cmap = LinearSegmentedColormap.from_list("ice_grid",
            ["#87CEEB", "#F0F8FF", "#FF6347", "#FFD700"])  # 冰面、起点、洞、终点

    def create_policy_animation(self, env, policy: np.ndarray, save_path: str,
                               episode_length: int = 50, fps: int = 2):
        """
        创建策略执行动画

        Args:
            env: FrozenLake环境实例
            policy: 策略数组
            save_path: 保存路径
            episode_length: 每集最大步数
            fps: 动画帧率
        """
        print("🎬 生成策略执行动画...")

        # 运行一个episode来收集轨迹
        trajectory = self._run_episode(env, policy, episode_length)

        if not trajectory:
            print("❌ 未能生成有效轨迹")
            return

        # 创建动画
        fig, ax = plt.subplots(figsize=(10, 10))

        def animate(frame):
            ax.clear()
            self._draw_environment(ax, env, trajectory, frame)

        # 创建动画
        anim = animation.FuncAnimation(
            fig, animate, frames=len(trajectory),
            interval=1000//fps, blit=False
        )

        # 保存动画
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            # 保存为GIF
            anim.save(save_path, writer='pillow', fps=fps, dpi=150)
            print(f"✅ 动画已保存: {save_path}")
        except Exception as e:
            print(f"❌ 保存动画失败: {e}")
            # 尝试保存为MP4
            try:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='RL Animation'), bitrate=1800)
                anim.save(save_path.replace('.gif', '.mp4'), writer=writer, dpi=150)
                print(f"✅ 动画已保存为MP4: {save_path.replace('.gif', '.mp4')}")
            except Exception as e2:
                print(f"❌ MP4保存也失败: {e2}")

        plt.close()

    def create_learning_progress_animation(self, training_history: List[Dict],
                                        env, save_path: str, fps: int = 3,
                                        policy_file: str = None, q_table_file: str = None,
                                        value_file: str = None):
        """
        创建学习进度动画 - 显示训练过程的关键指标

        Args:
            training_history: 训练历史记录
            env: 环境实例
            save_path: 保存路径
            fps: 帧率
            policy_file: 策略文件路径（可选）
            q_table_file: Q表文件路径（可选）
            value_file: 值函数文件路径（可选）
        """
        print("📈 生成学习进度动画...")

        if not training_history:
            print("❌ 无训练历史数据")
            return

        # 尝试加载最终策略和Q表
        final_policy = None
        final_q_table = None
        final_values = None

        if policy_file and os.path.exists(policy_file):
            try:
                final_policy = np.load(policy_file)
            except:
                pass

        if q_table_file and os.path.exists(q_table_file):
            try:
                final_q_table = np.load(q_table_file)
            except:
                pass

        if value_file and os.path.exists(value_file):
            try:
                final_values = np.load(value_file)
            except:
                pass

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        def animate(frame):
            # 清除之前的绘制
            for ax in axes.flat:
                ax.clear()

            # 获取当前帧的数据
            current_data = training_history[min(frame, len(training_history)-1)]
            episode = current_data.get("episode", frame + 1)

            # 左上：Q值统计信息（如果有）
            if final_q_table is not None:
                self._plot_q_stats_animation(axes[0, 0], training_history, frame, final_q_table)
            else:
                axes[0, 0].text(0.5, 0.5, 'Q值数据不可用', ha='center', va='center',
                               fontsize=14, transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Q值统计', fontsize=12)

            # 右上：成功率趋势
            self._plot_success_rate_trend(axes[0, 1], training_history, frame)

            # 左下：策略可视化（如果有最终策略）
            if final_policy is not None:
                self._plot_policy_animation(axes[1, 0], final_policy, env, episode)
            else:
                axes[1, 0].text(0.5, 0.5, '策略数据不可用', ha='center', va='center',
                               fontsize=14, transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('学习策略', fontsize=12)

            # 右下：学习曲线
            self._plot_learning_curve_animation(axes[1, 1], training_history, frame)

            # 设置总标题
            fig.suptitle(f'强化学习训练进度 - Episode {episode}',
                        fontsize=16, fontweight='bold')

        # 创建动画
        frames = min(len(training_history), 50)  # 最多50帧
        if frames > 1:
            frame_indices = np.linspace(0, len(training_history)-1, frames, dtype=int)
        else:
            frame_indices = [0]

        anim = animation.FuncAnimation(
            fig, lambda f: animate(f), frames=frame_indices,
            interval=1000//fps, blit=False, repeat=True
        )

        # 保存动画
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            anim.save(save_path, writer='pillow', fps=fps, dpi=120)
            print(f"✅ 学习进度动画已保存: {save_path}")
        except Exception as e:
            print(f"❌ 保存动画失败: {e}")

        plt.close()

    def create_comparison_animation(self, experiments_data: List[Dict],
                                  save_path: str, fps: int = 2):
        """
        创建算法对比动画

        Args:
            experiments_data: 多个实验的数据
            save_path: 保存路径
            fps: 帧率
        """
        print("🔄 生成算法对比动画...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        def animate(frame):
            for ax in axes.flat:
                ax.clear()

            # 动态显示实验结果
            self._plot_comparison_animation(axes, experiments_data, frame)

            fig.suptitle(f'算法性能对比 - 实验进度 {frame+1}/{len(experiments_data)}',
                        fontsize=16, fontweight='bold')

        anim = animation.FuncAnimation(
            fig, animate, frames=len(experiments_data),
            interval=1000//fps, blit=False
        )

        # 保存动画
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            anim.save(save_path, writer='pillow', fps=fps, dpi=120)
            print(f"✅ 对比动画已保存: {save_path}")
        except Exception as e:
            print(f"❌ 保存动画失败: {e}")

        plt.close()

    def _run_episode(self, env, policy: np.ndarray, max_steps: int) -> List[Tuple]:
        """运行一个episode并返回轨迹"""
        trajectory = []

        try:
            obs, info = env.reset()
            trajectory.append((obs, None, 0, False))  # (state, action, reward, done)

            for _ in range(max_steps):
                action = int(policy[obs])

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                trajectory.append((next_obs, action, reward, done))

                if done:
                    break

                obs = next_obs

        except Exception as e:
            print(f"运行episode时出错: {e}")
            return []

        return trajectory

    def _draw_environment(self, ax, env, trajectory: List, current_step: int):
        """绘制环境状态"""
        if not hasattr(env, 'nrow') or not hasattr(env, 'ncol'):
            return

        nrow, ncol = env.nrow, env.ncol

        # 绘制网格
        for r in range(nrow):
            for c in range(ncol):
                # 绘制单元格背景
                cell_color = self._get_cell_color(env, r, c)
                rect = patches.Rectangle((c, r), 1, 1, facecolor=cell_color,
                                       edgecolor='black', linewidth=2)
                ax.add_patch(rect)

        # 绘制轨迹
        for step, (state, action, reward, done) in enumerate(trajectory[:current_step+1]):
            row, col = divmod(state, ncol)

            # 绘制智能体位置
            if step == current_step:
                # 当前位置 - 高亮
                agent = patches.Circle((col + 0.5, row + 0.5), 0.3,
                                     facecolor='red', edgecolor='white', linewidth=3, alpha=0.9)
                ax.add_patch(agent)

                # 显示步骤信息
                ax.text(col + 0.5, row + 0.2, f'Step {step}',
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8))

            else:
                # 历史位置 - 半透明
                agent = patches.Circle((col + 0.5, row + 0.5), 0.2,
                                     facecolor='blue', edgecolor='white', linewidth=2, alpha=0.6)
                ax.add_patch(agent)

        # 设置坐标轴
        ax.set_xlim(0, ncol)
        ax.set_ylim(0, nrow)
        ax.set_aspect('equal')
        ax.set_title(f'策略执行动画 - Step {current_step}',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.axis('off')

        # 添加图例
        self._add_animation_legend(ax)

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

    def _add_animation_legend(self, ax):
        """添加动画图例"""
        legend_elements = [
            patches.Patch(facecolor='#98FB98', edgecolor='black', label='起点 (S)'),
            patches.Patch(facecolor='#FFD700', edgecolor='black', label='终点 (G)'),
            patches.Patch(facecolor='#FF6347', edgecolor='black', label='洞 (H)'),
            patches.Patch(facecolor='#87CEEB', edgecolor='black', label='冰面 (F)'),
            patches.Circle((0, 0), 0.2, facecolor='red', edgecolor='white',
                          linewidth=3, label='当前智能体'),
            patches.Circle((0, 0), 0.2, facecolor='blue', edgecolor='white',
                          linewidth=2, alpha=0.6, label='历史路径')
        ]

        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',
                 borderaxespad=0., fontsize=10)

    def _plot_q_heatmap_animation(self, ax, q_table: np.ndarray, env, frame: int):
        """绘制Q值热力图动画"""
        n_states, n_actions = q_table.shape

        if not hasattr(env, 'nrow') or not hasattr(env, 'ncol'):
            return

        nrow, ncol = env.nrow, env.ncol
        q_grid = np.zeros((nrow, ncol, n_actions))

        for s in range(n_states):
            row, col = divmod(s, ncol)
            q_grid[row, col] = q_table[s]

        # 显示第一个动作的Q值热力图
        im = ax.imshow(q_grid[:, :, 0], cmap='RdYlBu_r', aspect='equal')

        # 添加数值标签
        for r in range(nrow):
            for c in range(ncol):
                ax.text(c, r, f'{q_grid[r, c, 0]:.2f}',
                       ha='center', va='center', fontsize=8, color='black')

        ax.set_title(f'Q值热力图 (Action 0) - Frame {frame}')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)

    def _plot_value_function_animation(self, ax, values: np.ndarray, env, frame: int):
        """绘制值函数动画"""
        if not hasattr(env, 'nrow') or not hasattr(env, 'ncol'):
            return

        nrow, ncol = env.nrow, env.ncol
        value_grid = np.zeros((nrow, ncol))

        for s in range(len(values)):
            row, col = divmod(s, ncol)
            value_grid[row, col] = values[s]

        im = ax.imshow(value_grid, cmap='viridis', aspect='equal')

        # 添加数值标签
        for r in range(nrow):
            for c in range(ncol):
                ax.text(c, r, f'{value_grid[r, c]:.3f}',
                       ha='center', va='center', fontsize=8, color='white')

        ax.set_title(f'状态值函数 - Frame {frame}')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)

    def _plot_policy_animation(self, ax, policy: np.ndarray, env, episode: int = None):
        """绘制策略动画"""
        if not hasattr(env, 'nrow') or not hasattr(env, 'ncol'):
            return

        nrow, ncol = env.nrow, env.ncol
        policy_grid = np.zeros((nrow, ncol), dtype=int)

        for s in range(len(policy)):
            row, col = divmod(s, ncol)
            policy_grid[row, col] = policy[s]

        # 绘制网格
        for r in range(nrow):
            for c in range(ncol):
                cell_color = self._get_cell_color(env, r, c)
                rect = patches.Rectangle((c, r), 1, 1, facecolor=cell_color,
                                       edgecolor='black', linewidth=2)
                ax.add_patch(rect)

                # 绘制策略箭头
                action = policy_grid[r, c]
                self._draw_action_arrow(ax, c, r, action)

        ax.set_xlim(0, ncol)
        ax.set_ylim(0, nrow)
        ax.set_aspect('equal')
        title = f'学习策略'
        if episode:
            title += f' - Episode {episode}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.axis('off')

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

    def _plot_learning_curve_animation(self, ax, training_history: List[Dict], current_frame: int):
        """绘制学习曲线动画"""
        if not training_history[:current_frame+1]:
            return

        episodes = [h.get('episode', i+1) for i, h in enumerate(training_history[:current_frame+1])]
        success_rates = [h.get('success_rate', 0) for h in training_history[:current_frame+1]]
        epsilons = [h.get('epsilon', 0) for h in training_history[:current_frame+1]]

        ax.plot(episodes, success_rates, 'b-', linewidth=2, marker='o', markersize=4, label='成功率')
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('成功率', color='b', fontsize=11)
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_ylim(0, 1.1)

        ax2 = ax.twinx()
        ax2.plot(episodes, epsilons, 'r-', linewidth=2, marker='s', markersize=4, label='探索率')
        ax2.set_ylabel('探索率', color='r', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(0, 1.1)

        ax.set_title(f'学习曲线 - Episode {episodes[-1]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_q_stats_animation(self, ax, training_history: List[Dict], current_frame: int, q_table: np.ndarray):
        """绘制Q值统计信息动画"""
        if not training_history[:current_frame+1]:
            return

        episodes = [h.get('episode', i+1) for i, h in enumerate(training_history[:current_frame+1])]
        
        # 提取Q值统计
        q_means = []
        q_stds = []
        q_maxs = []
        q_mins = []

        for h in training_history[:current_frame+1]:
            if 'q_stats' in h:
                q_stats = h['q_stats']
                q_means.append(q_stats.get('q_mean', 0))
                q_stds.append(q_stats.get('q_std', 0))
                q_maxs.append(q_stats.get('q_max', 0))
                q_mins.append(q_stats.get('q_min', 0))
            else:
                # 如果没有统计信息，从Q表计算
                q_means.append(np.mean(q_table))
                q_stds.append(np.std(q_table))
                q_maxs.append(np.max(q_table))
                q_mins.append(np.min(q_table))

        # 绘制Q值统计
        ax.plot(episodes, q_means, 'g-', linewidth=2, marker='o', markersize=3, label='Q均值')
        ax.fill_between(episodes, 
                       [m - s for m, s in zip(q_means, q_stds)],
                       [m + s for m, s in zip(q_means, q_stds)],
                       alpha=0.3, color='green', label='±1标准差')

        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Q值', fontsize=11)
        ax.set_title('Q值统计趋势', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_success_rate_trend(self, ax, training_history: List[Dict], current_frame: int):
        """绘制成功率趋势"""
        if not training_history[:current_frame+1]:
            return

        episodes = [h.get('episode', i+1) for i, h in enumerate(training_history[:current_frame+1])]
        success_rates = [h.get('success_rate', 0) for h in training_history[:current_frame+1]]
        avg_steps = [h.get('avg_steps', 0) for h in training_history[:current_frame+1]]

        # 绘制成功率
        ax.plot(episodes, success_rates, 'b-', linewidth=2.5, marker='o', markersize=4, label='成功率')
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('成功率', color='b', fontsize=11)
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_ylim(0, 1.1)

        # 添加平均步数（如果有）
        if avg_steps and any(avg_steps):
            ax2 = ax.twinx()
            ax2.plot(episodes, avg_steps, 'r--', linewidth=2, marker='s', markersize=3, label='平均步数', alpha=0.7)
            ax2.set_ylabel('平均步数', color='r', fontsize=11)
            ax2.tick_params(axis='y', labelcolor='r')

        ax.set_title('成功率趋势', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_comparison_animation(self, axes, experiments_data: List[Dict], frame: int):
        """绘制对比动画"""
        current_data = experiments_data[:frame+1]

        if not current_data:
            return

        # 收集数据
        algorithms = []
        success_rates = []
        training_times = []

        for exp in current_data:
            algorithms.append(exp.get('algorithm', 'Unknown'))
            success_rates.append(exp.get('success_rate', 0))
            training_times.append(exp.get('training_time', 0))

        # 去重算法
        unique_algorithms = list(set(algorithms))

        # 计算平均值
        algo_success = {}
        algo_times = {}
        for algo in unique_algorithms:
            algo_data = [success_rates[i] for i in range(len(algorithms)) if algorithms[i] == algo]
            algo_success[algo] = sum(algo_data) / len(algo_data) if algo_data else 0

            time_data = [training_times[i] for i in range(len(algorithms)) if algorithms[i] == algo]
            algo_times[algo] = sum(time_data) / len(time_data) if time_data else 0

        # 绘制成功率对比
        axes[0, 0].bar(unique_algorithms, [algo_success[algo] for algo in unique_algorithms],
                       color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[0, 0].set_title('算法成功率对比')
        axes[0, 0].set_ylabel('成功率')
        axes[0, 0].set_ylim(0, 1.1)

        # 绘制训练时间对比
        axes[0, 1].bar(unique_algorithms, [algo_times[algo] for algo in unique_algorithms],
                       color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[0, 1].set_title('算法训练时间对比')
        axes[0, 1].set_ylabel('时间 (秒)')

        # 绘制实验数量统计
        algo_counts = [algorithms.count(algo) for algo in unique_algorithms]
        axes[1, 0].bar(unique_algorithms, algo_counts, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[1, 0].set_title('实验数量统计')
        axes[1, 0].set_ylabel('实验数')

        # 绘制实时效率图
        if len(unique_algorithms) >= 2:
            efficiency = [algo_success[algo] / max(algo_times[algo], 0.01) for algo in unique_algorithms]
            axes[1, 1].bar(unique_algorithms, efficiency, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
            axes[1, 1].set_title('算法效率对比')
            axes[1, 1].set_ylabel('效率 (成功率/时间)')

        # 旋转x轴标签
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
