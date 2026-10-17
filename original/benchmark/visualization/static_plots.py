#!/usr/bin/env python3
"""
现代化可视化 - 生成美观的静态图表
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib import cm
import seaborn as sns

matplotlib.use('Agg')  # 使用非交互式后端

# 配置现代化样式
plt.style.use('seaborn-v0_8-whitegrid')

# 配置中文字体 - 按优先级尝试多种字体
font_candidates = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
available_fonts = []
for font in font_candidates:
    try:
        # 测试字体是否可用
        plt.figure(figsize=(1, 1))
        plt.text(0.5, 0.5, '测试', fontdict={'family': font, 'size': 10})
        plt.close()
        available_fonts.append(font)
        break  # 找到第一个可用的就停止
    except:
        continue

if available_fonts:
    plt.rcParams['font.sans-serif'] = available_fonts
else:
    # 如果都没有中文字体，回退到英文模式
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 静默处理字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 现代化配色方案
COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
]

# 算法名称映射（英文显示名称，避免字体问题）
ALGO_NAME_MAP = {
    'astar': 'A*',
    'double_q_learning': 'Double\nQ-Learning',
    'dqn': 'DQN',
    'double_dqn': 'Double\nDQN',
    'dueling_dqn': 'Dueling\nDQN',
    'dyna_q': 'Dyna-Q',
    'eligibility_traces': 'Eligibility\nTraces',
    'expected_sarsa': 'Expected\nSARSA',
    'monte_carlo_control': 'MC\nControl',
    'n_step_td': 'n-step\nTD',
    'per_dqn': 'PER-DQN',
    'policy_iteration': 'Policy\nIteration',
    'q_learning': 'Q-Learning',
    'sarsa': 'SARSA',
    'value_iteration': 'Value\nIteration'
}

# 英文标题映射
TITLE_MAP = {
    'success_rate': 'Algorithm Success Rate Comparison',
    'avg_steps': 'Algorithm Average Steps Comparison',
    'training_time': 'Algorithm Training Time Comparison',
    'efficiency': 'Algorithm Efficiency Analysis'
}

def create_simple_plots(all_experiments=None, output_base_dir=None):
    """创建简单的可视化图表 - 按地图和模式整合所有算法"""
    print("🎨 生成简单可视化图表...")

    # 读取数据
    if all_experiments is None:
        with open('results/experiment_summary.json', 'r', encoding='utf-8') as f:
            all_experiments = json.load(f)
    
    if output_base_dir is None:
        output_base_dir = "results/simple_plots"

    # 按地图和模式分组（整合所有算法）
    grouped_experiments = {}
    for exp in all_experiments:
        map_name = exp['map']
        mode = 'stochastic' if exp['slippery'] else 'deterministic'
        
        key = (map_name, mode)
        if key not in grouped_experiments:
            grouped_experiments[key] = []
        grouped_experiments[key].append(exp)

    # 为每个地图+模式组合生成整合的图表
    for (map_name, mode), exp_list in grouped_experiments.items():
        # 创建输出目录：results/simple_plots/{map}/{mode}/
        output_dir = os.path.join(output_base_dir, map_name, mode)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 生成 {map_name}/{mode} 的整合可视化（包含 {len(set(e['algorithm'] for e in exp_list))} 种算法）...")
        _create_plots_for_group(exp_list, output_dir, map_name, mode)

    # 在根目录生成全局汇总
    print("📊 生成全局汇总图表...")
    _create_global_summary(all_experiments, output_base_dir)

def _create_plots_for_group(experiments, output_dir, map_name, mode):
    """为单个地图+模式组合创建整合图表（包含所有算法）"""
    if not experiments:
        return

    # 提取数据
    algorithms = []
    success_rates = []
    training_times = []
    avg_steps_list = []
    seeds = []

    for exp in experiments:
        algorithms.append(exp['algorithm'])
        success_rates.append(exp['success_rate'])
        training_times.append(exp['training_time'])
        avg_steps_list.append(exp.get('avg_steps', 0))
        seeds.append(exp.get('seed', 42))

    # 去重并计算每个算法的平均值（如果有多个种子）
    unique_algorithms = sorted(list(set(algorithms)))

    # 按算法聚合数据（处理多个种子的情况）
    algo_data = {}
    for algo in unique_algorithms:
        algo_exps = [exp for exp in experiments if exp['algorithm'] == algo]
        algo_data[algo] = {
            'success_rates': [e['success_rate'] for e in algo_exps],
            'training_times': [e['training_time'] for e in algo_exps],
            'avg_steps': [e.get('avg_steps', 0) for e in algo_exps]
        }

    # 计算平均值
    algo_success = {algo: sum(data['success_rates']) / len(data['success_rates'])
                    for algo, data in algo_data.items()}
    algo_times = {algo: sum(data['training_times']) / len(data['training_times'])
                  for algo, data in algo_data.items()}
    algo_avg_steps = {algo: sum(data['avg_steps']) / len(data['avg_steps'])
                      for algo, data in algo_data.items()}

    # 创建现代化图表
    _create_success_rate_plot(unique_algorithms, algo_success, output_dir, map_name, mode)
    _create_avg_steps_plot(unique_algorithms, algo_avg_steps, output_dir, map_name, mode)
    _create_training_time_plot(unique_algorithms, algo_times, output_dir, map_name, mode)
    _create_efficiency_plot(unique_algorithms, algo_success, algo_times, output_dir, map_name, mode)

    # 5. 创建实验汇总表格图片（整合所有算法）
    if len(experiments) > 0:
        create_summary_table(experiments, output_dir)
        create_text_report(experiments, output_dir)

    print(f"✅ {map_name}/{mode} 的整合可视化已保存到: {output_dir}")

def _create_success_rate_plot(algorithms, success_rates, output_dir, map_name, mode):
    """创建现代化成功率对比图"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # 获取显示名称
    display_names = [ALGO_NAME_MAP.get(algo, algo.replace('_', '\n')) for algo in algorithms]
    values = [success_rates[algo] for algo in algorithms]

    # 创建渐变色
    colors = COLOR_PALETTE[:len(algorithms)]

    # 绘制柱状图
    bars = ax.bar(range(len(display_names)), values, color=colors, alpha=0.85,
                  width=0.7, edgecolor='white', linewidth=1.5)

    # 设置标题和标签（英文）
    mode_name = 'Stochastic' if mode == 'stochastic' else 'Deterministic'
    ax.set_title(f'{TITLE_MAP["success_rate"]} - {map_name} ({mode_name})',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Success Rate', fontsize=14, fontweight='medium')
    ax.set_xlabel('Algorithm', fontsize=14, fontweight='medium')

    # 设置x轴标签（避免重叠）
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=0, ha='center', fontsize=10)

    # 设置y轴范围
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10, color='#2c3e50')

    # 美化边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_success_rate.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 生成: algorithm_success_rate.png")

def _create_avg_steps_plot(algorithms, avg_steps, output_dir, map_name, mode):
    """创建现代化平均步数对比图"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # 获取显示名称
    display_names = [ALGO_NAME_MAP.get(algo, algo.replace('_', '\n')) for algo in algorithms]
    values = [avg_steps[algo] for algo in algorithms]

    # 创建颜色（根据值大小排序）
    sorted_indices = np.argsort(values)
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(algorithms))]

    # 绘制柱状图
    bars = ax.bar(range(len(display_names)), values, color=colors, alpha=0.85,
                  width=0.7, edgecolor='white', linewidth=1.5)

    # 设置标题和标签（英文）
    mode_name = 'Stochastic' if mode == 'stochastic' else 'Deterministic'
    ax.set_title(f'{TITLE_MAP["avg_steps"]} - {map_name} ({mode_name})',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Average Steps', fontsize=14, fontweight='medium')
    ax.set_xlabel('Algorithm', fontsize=14, fontweight='medium')

    # 设置x轴标签
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=0, ha='center', fontsize=10)

    # 设置y轴
    max_value = max(values)
    ax.set_ylim(0, max_value * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_value * 0.02,
                f'{value:.1f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10, color='#2c3e50')

    # 美化边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_avg_steps.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 生成: algorithm_avg_steps.png")

def _create_training_time_plot(algorithms, training_times, output_dir, map_name, mode):
    """创建现代化训练时间对比图"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # 获取显示名称
    display_names = [ALGO_NAME_MAP.get(algo, algo.replace('_', '\n')) for algo in algorithms]
    values = [training_times[algo] for algo in algorithms]

    # 创建颜色
    colors = COLOR_PALETTE[:len(algorithms)]

    # 绘制柱状图
    bars = ax.bar(range(len(display_names)), values, color=colors, alpha=0.85,
                  width=0.7, edgecolor='white', linewidth=1.5)

    # 设置标题和标签（英文）
    mode_name = 'Stochastic' if mode == 'stochastic' else 'Deterministic'
    ax.set_title(f'{TITLE_MAP["training_time"]} - {map_name} ({mode_name})',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Training Time (seconds)', fontsize=14, fontweight='medium')
    ax.set_xlabel('Algorithm', fontsize=14, fontweight='medium')

    # 设置x轴标签
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=0, ha='center', fontsize=10)

    # 设置y轴
    max_value = max(values)
    ax.set_ylim(0, max_value * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_value * 0.02,
                f'{value:.1f}s', ha='center', va='bottom',
                fontweight='bold', fontsize=9, color='#2c3e50')

    # 美化边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 生成: training_time.png")

def _create_efficiency_plot(algorithms, success_rates, training_times, output_dir, map_name, mode):
    """创建现代化效率分析散点图 - 使用对数坐标和改进布局"""
    fig, ax = plt.subplots(figsize=(18, 12))

    # 计算效率值
    x_values = [training_times[algo] for algo in algorithms]
    y_values = [success_rates[algo] for algo in algorithms]
    efficiency_values = [y / max(x, 0.01) for x, y in zip(x_values, y_values)]

    # 颜色和标记方案 - 扩展以支持更多算法
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F',
              '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA', '#F1948A', '#AED6F1',
              '#D7BDE2', '#FADBD8', '#FDEDEC']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'P', 'X', 'd', '8']

    # 为每个算法分配颜色和标记
    algo_colors = {}
    algo_markers = {}
    for i, algo in enumerate(algorithms):
        algo_colors[algo] = colors[i % len(colors)]
        algo_markers[algo] = markers[i % len(markers)]

    # 分离快速和慢速算法用于不同处理
    fast_algorithms = [algo for algo in algorithms if training_times[algo] < 1.0]
    slow_algorithms = [algo for algo in algorithms if training_times[algo] >= 1.0]

    # 绘制所有算法
    legend_elements = []
    for algo in algorithms:
        color = algo_colors[algo]
        marker = algo_markers[algo]
        x_val = training_times[algo]
        y_val = success_rates[algo]

        # 根据算法类型调整点的大小和样式
        if algo in fast_algorithms:
            size = max(y_val * 400, 120)  # 快速算法点更大，更突出
            alpha = 0.9  # 更高的透明度
        else:
            size = max(y_val * 300, 100)  # 慢速算法点正常大小
            alpha = 0.8

        # 对于未填充标记，不使用edgecolors
        if marker in ['+', 'x']:
            scatter = ax.scatter(x_val, y_val, s=size, c=color, marker=marker, alpha=alpha)
        else:
            scatter = ax.scatter(x_val, y_val, s=size, c=color, marker=marker,
                               alpha=alpha, edgecolors='white', linewidth=1.5)

        # 创建详细的图例标签，包含时间和成功率信息
        legend_elements.append(
            plt.Line2D([0], [0], marker=marker, color='w',
                      markerfacecolor=color, markersize=14,
                      label=f'{ALGO_NAME_MAP.get(algo, algo)}\n({x_val:.2f}s, {y_val:.3f})')
        )

    # 设置标题和标签
    mode_name = 'Stochastic' if mode == 'stochastic' else 'Deterministic'
    ax.set_title(f'Algorithm Efficiency Analysis - {map_name} ({mode_name})',
                fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Training Time (seconds) - Log Scale', fontsize=14, fontweight='medium')
    ax.set_ylabel('Success Rate', fontsize=14, fontweight='medium')

    # 使用对数坐标轴来分散数据点，让快速算法也能清晰显示
    ax.set_xscale('log')
    ax.set_xlim(0.01, max(x_values) * 2 if x_values else 10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, which='both')

    # 设置对数刻度标签格式
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}' if x < 1 else f'{x:.1f}'))

    # 创建分栏图例 - 放在底部以节省空间，显示详细信息
    n_cols = min(5, len(algorithms))  # 根据算法数量动态调整列数
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
             fontsize=10, frameon=True, framealpha=0.95, ncol=n_cols,
             title='Algorithms (Time, Success Rate)', title_fontsize=13, borderpad=1.5,
             labelspacing=1.2, handletextpad=1.0, columnspacing=1.5)

    # 美化边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.8)

    # 添加次要网格线，帮助查看对数坐标
    ax.grid(True, alpha=0.2, which='minor', linestyle=':')

    # 可选：添加效率等高线背景
    if len(algorithms) > 3:
        try:
            # 创建效率等高线背景，帮助理解效率分布
            x_range = np.logspace(-2, 2, 100)  # 0.01 到 100秒
            y_range = np.linspace(0.1, 1.0, 50)

            X, Y = np.meshgrid(x_range, y_range)
            Z = Y / X  # 效率 = 成功率/时间

            # 只绘制几个关键等高线
            levels = [0.1, 0.5, 1.0, 2.0, 5.0]
            cs = ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.15, linestyles='--', linewidths=0.8)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f', alpha=0.7)
        except:
            # 如果等高线绘制失败，跳过
            pass

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_analysis.png'),
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
    plt.close()
    print("✅ 生成现代化效率分析图: efficiency_analysis.png")

def _create_global_summary(all_experiments, output_base_dir=None):
    """在根目录创建全局汇总图表 - 现代化设计，避免文本重叠"""
    if not all_experiments:
        return

    if output_base_dir is None:
        output_base_dir = "results/simple_plots"
    os.makedirs(output_base_dir, exist_ok=True)

    # 提取所有数据
    algorithms = []
    success_rates = []
    training_times = []
    maps = []

    for exp in all_experiments:
        algorithms.append(exp['algorithm'])
        success_rates.append(exp['success_rate'])
        training_times.append(exp['training_time'])
        maps.append(exp['map'])

    # 去重
    unique_algorithms = list(set(algorithms))
    unique_maps = list(set(maps))

    # 计算平均值
    algo_success = {}
    algo_times = {}
    for algo in unique_algorithms:
        algo_data = [success_rates[i] for i in range(len(algorithms)) if algorithms[i] == algo]
        algo_success[algo] = sum(algo_data) / len(algo_data) if algo_data else 0
        time_data = [training_times[i] for i in range(len(algorithms)) if algorithms[i] == algo]
        algo_times[algo] = sum(time_data) / len(time_data) if time_data else 0

    map_success = {}
    for map_name in unique_maps:
        map_data = [success_rates[i] for i in range(len(maps)) if maps[i] == map_name]
        map_success[map_name] = sum(map_data) / len(map_data) if map_data else 0

    # 生成全局对比图表 - 使用更大的图和更宽松的布局，进一步增高子图
    fig = plt.figure(figsize=(28, 46))

    # 创建子图布局：上方两个横向图，下方一个大图，大幅增加高度比例
    gs = fig.add_gridspec(3, 2, hspace=0.7, wspace=0.4,
                         height_ratios=[1.8, 1.8, 2.2])

    ax1 = fig.add_subplot(gs[0, 0])  # 算法成功率
    ax2 = fig.add_subplot(gs[0, 1])  # 地图成功率
    ax3 = fig.add_subplot(gs[1, :])  # 训练时间
    ax4 = fig.add_subplot(gs[2, :])  # 效率分析

    # 颜色方案
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F',
              '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA', '#F1948A', '#85C1E9']

    # 1. 算法成功率对比 - 横向条形图
    algo_names = [ALGO_NAME_MAP.get(algo, algo) for algo in unique_algorithms]
    y_pos = range(len(unique_algorithms))
    bars = ax1.barh(y_pos, [algo_success[algo] for algo in unique_algorithms],
                   color=colors[:len(unique_algorithms)], alpha=0.85, height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(algo_names, fontsize=13, fontweight='medium')
    ax1.set_title('Algorithm Success Rate Comparison', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Success Rate', fontsize=14)
    ax1.set_xlim(0, 1.1)
    ax1.grid(axis='x', alpha=0.3)

    # 添加数值标签
    for i, v in enumerate([algo_success[algo] for algo in unique_algorithms]):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

    # 2. 地图成功率对比 - 保持纵向但优化标签
    bars = ax2.bar(range(len(unique_maps)), [map_success[map_name] for map_name in unique_maps],
                  color=colors[:len(unique_maps)], alpha=0.85, width=0.6)
    ax2.set_title('Map Success Rate Comparison', fontsize=18, fontweight='bold', pad=20)
    ax2.set_ylabel('Success Rate', fontsize=14)
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(range(len(unique_maps)))
    ax2.set_xticklabels(unique_maps, rotation=45, ha='right', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for i, v in enumerate([map_success[map_name] for map_name in unique_maps]):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    # 3. 训练时间对比 - 横向条形图，占用全宽度
    bars = ax3.barh(y_pos, [algo_times[algo] for algo in unique_algorithms],
                   color=colors[:len(unique_algorithms)], alpha=0.85, height=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(algo_names, fontsize=13, fontweight='medium')
    ax3.set_title('Algorithm Training Time Comparison', fontsize=18, fontweight='bold', pad=20)
    ax3.set_xlabel('Training Time (seconds)', fontsize=14)
    ax3.grid(axis='x', alpha=0.3)

    # 添加数值标签
    for i, v in enumerate([algo_times[algo] for algo in unique_algorithms]):
        ax3.text(v + 0.5, i, f'{v:.1f}s', va='center', fontsize=10, fontweight='bold')

    # 4. 效率分析 - 改进版，按算法+地图聚合数据
    # 使用可靠的marker符号，确保所有都可见（15个算法需要15个不同的marker）
    # 使用填充和未填充的组合，确保每个都清晰可见
    # 使用matplotlib明确支持的、可见的marker符号
    # 15个不同的、可靠的marker符号，确保所有都可见
    # 使用matplotlib标准支持的符号，全部使用填充标记确保可见性
    # 通过颜色+marker组合确保每个算法都唯一可识别
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', 'P', 'X', '8']
    algo_colors = {}
    algo_markers = {}

    for i, algo in enumerate(unique_algorithms):
        algo_colors[algo] = colors[i % len(colors)]
        # 确保每个算法都有唯一的marker，循环使用以确保覆盖所有算法
        algo_markers[algo] = markers[i % len(markers)]

    # 聚合数据：计算每个算法在每个地图上的平均表现
    algo_map_data = {}
    for exp in all_experiments:
        key = (exp['algorithm'], exp['map'])
        if key not in algo_map_data:
            algo_map_data[key] = {'success_rates': [], 'training_times': []}
        algo_map_data[key]['success_rates'].append(exp['success_rate'])
        algo_map_data[key]['training_times'].append(exp['training_time'])

    # 计算平均值
    aggregated_data = []
    for (algo, map_name), data in algo_map_data.items():
        avg_success = sum(data['success_rates']) / len(data['success_rates'])
        avg_time = sum(data['training_times']) / len(data['training_times'])
        aggregated_data.append({
            'algorithm': algo,
            'map': map_name,
            'success_rate': avg_success,
            'training_time': avg_time
        })

    # 分离快速和慢速算法
    fast_data = [d for d in aggregated_data if d['training_time'] < 1.0]
    slow_data = [d for d in aggregated_data if d['training_time'] >= 1.0]

    # 绘制慢速算法（正常坐标）
    for data in slow_data:
        algo = data['algorithm']
        color = algo_colors[algo]
        marker = algo_markers[algo]

        # 对于未填充标记，不使用edgecolors
        if marker in ['+', 'x']:
            ax4.scatter(data['training_time'], data['success_rate'],
                       s=max(data['success_rate'] * 400, 120),
                       c=color, marker=marker, alpha=0.85,
                       label=algo if data == slow_data[0] else "")
        else:
            ax4.scatter(data['training_time'], data['success_rate'],
                       s=max(data['success_rate'] * 400, 120),
                       c=color, marker=marker, alpha=0.85,
                       edgecolors='white', linewidth=1.5, label=algo if data == slow_data[0] else "")

    # 绘制快速算法（在左侧单独区域）
    if fast_data:
        ax4_twin = ax4.twinx()
        ax4_twin.set_ylim(0, 1.1)
        ax4_twin.set_ylabel('Fast Algorithms\nSuccess Rate', fontsize=12, color='gray')

        for data in fast_data:
            algo = data['algorithm']
            color = algo_colors[algo]
            marker = algo_markers[algo]

            # 对于未填充标记，不使用edgecolors
            if marker in ['+', 'x', '1', '2', '3', '4']:
                ax4_twin.scatter(data['training_time'], data['success_rate'],
                               s=max(data['success_rate'] * 400, 120),
                               c=color, marker=marker, alpha=0.85)
            else:
                ax4_twin.scatter(data['training_time'], data['success_rate'],
                               s=max(data['success_rate'] * 400, 120),
                               c=color, marker=marker, alpha=0.85,
                               edgecolors='white', linewidth=1.5)

    ax4.set_xlabel('Training Time (seconds)', fontsize=14)
    ax4.set_ylabel('Success Rate', fontsize=14)
    ax4.set_title('Algorithm Efficiency Analysis (Log Scale)', fontsize=18, fontweight='bold', pad=20)
    ax4.set_xscale('log')  # 使用对数坐标轴
    ax4.set_xlim(0.01, max(training_times) * 2 if training_times else 10)
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3, which='both')

    # 创建图例 - 分组显示
    legend_elements = []
    for i, algo in enumerate(unique_algorithms):
        legend_elements.append(
            plt.Line2D([0], [0], marker=algo_markers[algo], color='w',
                      markerfacecolor=algo_colors[algo], markersize=12,
                      label=ALGO_NAME_MAP.get(algo, algo))
        )

    # 分5栏显示图例，调整位置确保不超出图表范围且不遮挡图表
    # 将图例放在图表外部底部，避免与图表内容重合
    ax4.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fontsize=10, frameon=True, framealpha=0.95, ncol=5,
              title='Algorithms', title_fontsize=13, borderpad=1.2,
              labelspacing=1.0, handletextpad=0.8, columnspacing=1.5)

    plt.suptitle('Global Experiment Summary (All Algorithms & Maps)',
                 fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.12, 1, 0.99])  # 为底部图例留出更多空间，避免重合
    plt.savefig(os.path.join(output_base_dir, 'global_summary.png'), dpi=400, bbox_inches='tight',
                facecolor='white', pad_inches=2.0)
    plt.close()
    print("✅ 生成现代化全局汇总: global_summary.png")

def create_summary_table(experiments, output_dir):
    """创建实验汇总表格图片"""
    print("📋 生成实验汇总表格...")

    fig, ax = plt.subplots(figsize=(16, max(8, len(experiments) * 1.0)))
    ax.axis('off')

    # 准备表格数据
    headers = ['Algorithm', 'Map', 'Environment', 'Success Rate', 'Avg Steps', 'Training Time']
    table_data = []

    for exp in experiments:
        mode = '随机' if exp['slippery'] else '确定'
        table_data.append([
            exp['algorithm'].upper(),
            exp['map'],
            mode,
            f"{exp['success_rate']:.3f}",
            f"{exp['avg_steps']:.1f}",
            f"{exp['training_time']:.1f}s"
        ])

    # 创建表格
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    colColours=['#4A90E2'] * len(headers))

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)

    # 设置单元格颜色
    for i in range(len(table_data)):
        for j in range(len(headers)):
            cell = table[(i+1, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F8F9FA')
            else:
                cell.set_facecolor('#FFFFFF')
            cell.set_edgecolor('#DEE2E6')
            cell.set_linewidth(0.5)

    # 设置标题样式
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor('#4A90E2')
        cell.set_text_props(color='white', weight='bold')

    ax.set_title('实验结果汇总表', fontsize=18, fontweight='bold', pad=30,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#4A90E2", edgecolor="none"),
                color='white')
    plt.savefig(os.path.join(output_dir, 'experiment_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 生成: experiment_summary_table.png")

def create_text_report(experiments, output_dir):
    """创建文本报告"""
    print("📝 生成训练报告...")

    report = "强化学习实验训练报告\n"
    report += "=" * 50 + "\n\n"

    report += f"总实验数量: {len(experiments)}\n"
    report += f"涉及算法: {', '.join(set(exp['algorithm'] for exp in experiments))}\n"
    report += f"涉及地图: {', '.join(set(exp['map'] for exp in experiments))}\n\n"

    # 计算统计信息
    success_rates = [exp['success_rate'] for exp in experiments]
    avg_success = sum(success_rates) / len(success_rates)
    avg_steps = sum(exp['avg_steps'] for exp in experiments) / len(experiments)
    total_time = sum(exp['training_time'] for exp in experiments)

    report += "性能统计:\n"
    report += f"- 平均成功率: {avg_success:.3f}\n"
    report += f"- 平均步数: {avg_steps:.1f}\n"
    report += f"- 总训练时间: {total_time:.1f} 秒\n\n"

    # 算法排名
    report += "\n算法排名 (按成功率):\n"
    sorted_experiments = sorted(experiments, key=lambda x: x['success_rate'], reverse=True)
    for i, exp in enumerate(sorted_experiments):
        mode = '随机性' if exp['slippery'] else '确定性'
        report += f"{i+1}. {exp['algorithm'].upper()} on {exp['map']} ({mode}): {exp['success_rate']:.3f} ({exp['avg_steps']:.1f} 步)\n"

    # 保存报告
    with open(os.path.join(output_dir, 'training_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    print("✅ 生成: training_report.txt")

if __name__ == '__main__':
    try:
        create_simple_plots()
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
