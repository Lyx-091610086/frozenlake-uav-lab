import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Optional
import json
import os

class TrainingProgressVisualizer:
    """训练进度可视化器"""

    def __init__(self):
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 12

    def visualize_experiment_results(self, results_dir: str, save_dir: str = None):
        """可视化实验结果"""
        if save_dir is None:
            save_dir = results_dir

        os.makedirs(save_dir, exist_ok=True)

        # 读取实验汇总
        summary_file = os.path.join(results_dir, 'experiment_summary.json')
        if not os.path.exists(summary_file):
            print(f"找不到实验汇总文件: {summary_file}")
            return

        with open(summary_file, 'r', encoding='utf-8') as f:
            experiments = json.load(f)

        if not experiments:
            print("没有实验数据")
            return

        # 转换为DataFrame便于分析
        df = pd.DataFrame(experiments)

        # 生成各种图表
        self._plot_algorithm_comparison(df, save_dir)
        self._plot_environment_comparison(df, save_dir)
        self._plot_performance_heatmap(df, save_dir)
        self._plot_training_efficiency(df, save_dir)
        self._plot_success_rate_distribution(df, save_dir)

        print(f"可视化结果已保存到: {save_dir}")

    def _plot_algorithm_comparison(self, df: pd.DataFrame, save_dir: str):
        """绘制算法对比图"""
        algorithms = df['algorithm'].unique()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for algo in algorithms:
            algo_data = df[df['algorithm'] == algo]

            # 成功率
            axes[0, 0].bar(algo, algo_data['success_rate'].mean(),
                          yerr=algo_data['success_rate'].std(), capsize=5,
                          alpha=0.7, label=algo)

            # 平均步数
            axes[0, 1].bar(algo, algo_data['avg_steps'].mean(),
                          yerr=algo_data['avg_steps'].std(), capsize=5,
                          alpha=0.7, label=algo)

            # 训练时间
            axes[1, 0].bar(algo, algo_data['training_time'].mean(),
                          yerr=algo_data['training_time'].std(), capsize=5,
                          alpha=0.7, label=algo)

        axes[0, 0].set_title('Success Rate by Algorithm')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].set_title('Average Steps by Algorithm')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].tick_params(axis='x', rotation=45)

        axes[1, 0].set_title('Training Time by Algorithm')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 移除空的子图
        axes[1, 1].remove()

        plt.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_environment_comparison(self, df: pd.DataFrame, save_dir: str):
        """绘制环境对比图"""
        # 按环境分组
        env_groups = df.groupby(['map', 'slippery'])

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        maps = df['map'].unique()
        modes = ['deterministic', 'stochastic']
        colors = ['lightblue', 'lightcoral']

        for i, map_name in enumerate(maps[:4]):  # 最多显示4个地图
            ax = axes[i // 2, i % 2]

            for j, mode in enumerate([False, True]):
                mode_data = df[(df['map'] == map_name) & (df['slippery'] == mode)]
                if len(mode_data) > 0:
                    ax.bar(f"{mode_data['algorithm'].iloc[0]}",
                          mode_data['success_rate'].mean(),
                          alpha=0.7, color=colors[j],
                          label=f"{modes[j]} (n={len(mode_data)})")

            ax.set_title(f'{map_name}')
            ax.set_ylabel('Success Rate')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle('Environment Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'environment_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_heatmap(self, df: pd.DataFrame, save_dir: str):
        """绘制性能热力图"""
        # 创建透视表
        pivot_success = df.pivot_table(values='success_rate',
                                     index='algorithm',
                                     columns=['map', 'slippery'],
                                     aggfunc='mean')

        fig, ax = plt.subplots(figsize=(12, 8))

        if not pivot_success.empty:
            im = ax.imshow(pivot_success, cmap='RdYlGn', aspect='auto')

            # 添加数值标签
            for i in range(len(pivot_success.index)):
                for j in range(len(pivot_success.columns)):
                    value = pivot_success.iloc[i, j]
                    if not np.isnan(value):
                        ax.text(j, i, f'{value:.2f}',
                               ha='center', va='center',
                               color='black' if value > 0.7 else 'white',
                               fontweight='bold')

            ax.set_title('Success Rate Heatmap')
            ax.set_xlabel('Environment (Map + Mode)')
            ax.set_ylabel('Algorithm')

            # 设置刻度标签
            ax.set_xticks(range(len(pivot_success.columns)))
            ax.set_xticklabels([f"{col[0]}\n({col[1]})" for col in pivot_success.columns],
                             rotation=45, ha='right')
            ax.set_yticks(range(len(pivot_success.index)))
            ax.set_yticklabels(pivot_success.index)

            plt.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_training_efficiency(self, df: pd.DataFrame, save_dir: str):
        """绘制训练效率图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 成功率 vs 训练时间
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            axes[0].scatter(algo_data['training_time'], algo_data['success_rate'],
                          alpha=0.7, label=algo, s=50)

        axes[0].set_xlabel('Training Time (seconds)')
        axes[0].set_ylabel('Success Rate')
        axes[0].set_title('Efficiency: Success Rate vs Training Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 成功率 vs 平均步数
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            axes[1].scatter(algo_data['avg_steps'], algo_data['success_rate'],
                          alpha=0.7, label=algo, s=50)

        axes[1].set_xlabel('Average Steps')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_title('Efficiency: Success Rate vs Steps')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Training Efficiency Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_efficiency.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_success_rate_distribution(self, df: pd.DataFrame, save_dir: str):
        """绘制成功率分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 按算法分组的成功率分布
        algorithms = df['algorithm'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))

        for i, algo in enumerate(algorithms):
            algo_data = df[df['algorithm'] == algo]['success_rate']
            axes[0, 0].hist(algo_data, alpha=0.7, bins=10, label=algo, color=colors[i])

        axes[0, 0].set_xlabel('Success Rate')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Success Rate Distribution by Algorithm')
        axes[0, 0].legend()

        # 按环境分组的成功率分布
        maps = df['map'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(maps)))

        for i, map_name in enumerate(maps[:8]):  # 最多显示8个地图
            map_data = df[df['map'] == map_name]['success_rate']
            axes[0, 1].hist(map_data, alpha=0.7, bins=10, label=map_name, color=colors[i])

        axes[0, 1].set_xlabel('Success Rate')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Success Rate Distribution by Map')
        axes[0, 1].legend()

        # 箱线图：算法 vs 成功率
        algo_success = [df[df['algorithm'] == algo]['success_rate'].values
                       for algo in algorithms]
        axes[1, 0].boxplot(algo_success, labels=algorithms)
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title('Success Rate Box Plot by Algorithm')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 箱线图：环境 vs 成功率
        map_success = [df[df['map'] == map_name]['success_rate'].values
                      for map_name in maps[:8]]
        axes[1, 1].boxplot(map_success, labels=maps[:8])
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Success Rate Box Plot by Map')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.suptitle('Success Rate Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'success_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_training_report(self, results_dir: str, save_path: str):
        """生成训练报告"""
        summary_file = os.path.join(results_dir, 'experiment_summary.json')

        if not os.path.exists(summary_file):
            print(f"找不到实验汇总文件: {summary_file}")
            return

        with open(summary_file, 'r', encoding='utf-8') as f:
            experiments = json.load(f)

        # 生成报告
        report = f"""# 强化学习实验报告

## 实验概览
- 总实验数量: {len(experiments)}
- 涉及算法: {len(set(e['algorithm'] for e in experiments))}
- 涉及地图: {len(set(e['map'] for e in experiments))}

## 性能统计

### 全局统计
- 平均成功率: {np.mean([e['success_rate'] for e in experiments]):.3f}
- 平均步数: {np.mean([e['avg_steps'] for e in experiments]):.1f}
- 平均训练时间: {np.mean([e['training_time'] for e in experiments]):.2f}s

### 最佳性能
- 最高成功率: {max(experiments, key=lambda x: x['success_rate'])['success_rate']:.3f}
- 最少平均步数: {min(experiments, key=lambda x: x['avg_steps'])['avg_steps']:.1f}
- 最短训练时间: {min(experiments, key=lambda x: x['training_time'])['training_time']:.2f}s

## 算法排名

### 按成功率排名
"""

        # 按成功率排序
        by_success = sorted(experiments, key=lambda x: x['success_rate'], reverse=True)
        for i, exp in enumerate(by_success[:10]):  # Top 10
            report += f"{i+1}. {exp['algorithm']} ({exp['map']}, {'slippery' if exp['slippery'] else 'deterministic'}): {exp['success_rate']:.3f}\n"

        report += "\n### 按效率排名 (成功率/训练时间)\n"
        for exp in experiments:
            exp['efficiency'] = exp['success_rate'] / max(exp['training_time'], 0.01)

        by_efficiency = sorted(experiments, key=lambda x: x['efficiency'], reverse=True)
        for i, exp in enumerate(by_efficiency[:10]):
            report += f"{i+1}. {exp['algorithm']} ({exp['map']}): {exp['efficiency']:.4f}\n"

        # 保存报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"训练报告已保存到: {save_path}")
