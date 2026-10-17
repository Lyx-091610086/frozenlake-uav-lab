#!/usr/bin/env python3
"""
完整的强化学习基准测试脚本
自动运行实验并生成可视化报告
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='运行完整的强化学习基准测试')
    parser.add_argument('--preset', type=str, default='full_benchmark',
                       choices=['quick_test', 'planning_baseline', 'tabular_comparison',
                               'advanced_tabular', 'model_based', 'deep_value_comparison',
                               'map_scalability', 'full_benchmark', 'stochastic_challenge',
                               'planning_vs_learning'],
                       help='实验预设配置')
    parser.add_argument('--run-experiments', action='store_true',
                       help='运行实验')
    parser.add_argument('--generate-plots', action='store_true',
                       help='生成静态图表')
    parser.add_argument('--generate-animations', action='store_true',
                       help='生成动画')
    parser.add_argument('--all', action='store_true',
                       help='运行完整流程（实验+所有可视化）')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='结果目录')

    args = parser.parse_args()

    # 如果指定--all，运行完整流程
    if args.all:
        args.run_experiments = True
        args.generate_plots = True
        args.generate_animations = True

    # 如果没有指定任何操作，默认运行完整流程
    if not any([args.run_experiments, args.generate_plots, args.generate_animations]):
        args.run_experiments = True
        args.generate_plots = True
        args.generate_animations = True

    print("🚀 强化学习完整基准测试")
    print("=" * 50)
    print(f"实验预设: {args.preset}")
    print(f"结果目录: {args.results_dir}")

    # 1. 运行实验
    if args.run_experiments:
        print("\n📊 第一阶段: 运行实验")
        print("-" * 30)

        # 导入实验运行器
        from experiments.experiment_runner import ExperimentRunner

        runner = ExperimentRunner(save_detailed_results=True)
        config = runner.get_experiment_config(args.preset)

        print(f"实验配置:")
        print(f"  算法数量: {len(config['algorithms'])}")
        print(f"  地图数量: {len(config['maps'])}")
        print(f"  环境模式: {len(config['slippery_modes'])}")
        print(f"  种子数量: {len(config['seeds'])}")

        total_experiments = (len(config['algorithms']) *
                           len(config['maps']) *
                           len(config['slippery_modes']) *
                           len(config['seeds']))
        print(f"  总实验数: {total_experiments}")

        # 运行实验
        results = runner.run_batch_experiments(
            algorithms=config['algorithms'],
            maps=config['maps'],
            slippery_modes=config['slippery_modes'],
            seeds=config['seeds']
        )

        print(f"\n✅ 实验阶段完成!")

    # 2. 生成静态图表
    if args.generate_plots:
        print("\n📊 第二阶段: 生成静态图表")
        print("-" * 30)

        # 运行可视化脚本
        os.system(f'python generate_visualizations.py --type static --results-dir {args.results_dir}')

        print(f"\n✅ 静态图表生成完成!")

    # 3. 生成动画
    if args.generate_animations:
        print("\n🎬 第三阶段: 生成动画")
        print("-" * 30)

        # 运行动画生成
        os.system(f'python generate_visualizations.py --type all --results-dir {args.results_dir}')

        print(f"\n✅ 动画生成完成!")

    print("\n🎉 完整基准测试流程执行完毕!")
    print("=" * 50)
    print(f"📁 结果保存在: {args.results_dir}")
    print("📊 查看结果: python -m http.server 8000 (然后访问 results/simple_plots)")

if __name__ == '__main__':
    main()
