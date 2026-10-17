#!/usr/bin/env python3
# experiments/run_experiments.py
# 主实验运行脚本

import argparse
import sys
import os
import numpy as np
import warnings

# 全局抑制警告
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*gym.*")
warnings.filterwarnings("ignore", message=".*Gym.*")
warnings.filterwarnings("ignore", message=".*numpy.*")
warnings.filterwarnings("ignore", message=".*bool8.*")

# 设置环境变量
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.append('.')

from experiment_runner import ExperimentRunner
from configs import EXPERIMENT_PRESETS, ALGORITHM_CONFIGS

def main():
    parser = argparse.ArgumentParser(description='FrozenLake强化学习实验平台')
    parser.add_argument('--preset', type=str, choices=EXPERIMENT_PRESETS.keys(),
                       help='使用预设实验配置')
    parser.add_argument('--algorithms', nargs='+',
                       help='指定算法列表')
    parser.add_argument('--maps', nargs='+',
                       help='指定地图列表')
    parser.add_argument('--slippery', action='store_true', default=None,
                       help='启用随机打滑模式')
    parser.add_argument('--deterministic', action='store_true', default=None,
                       help='使用确定性模式')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                       help='随机种子列表')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='结果保存目录')
    parser.add_argument('--save-detailed', action='store_true',
                       help='保存每个实验的详细结果（默认只保存汇总）')
    parser.add_argument('--list-maps', action='store_true',
                       help='列出所有可用地图')
    parser.add_argument('--list-algorithms', action='store_true',
                       help='列出所有可用算法')

    args = parser.parse_args()

    # 创建实验运行器
    runner = ExperimentRunner(args.results_dir, save_detailed_results=args.save_detailed)

    # 列出可用选项
    if args.list_maps:
        from environments.frozenlake_env import FrozenLakeEnv
        print("可用地图:")
        for map_name in FrozenLakeEnv.list_available_maps():
            print(f"  - {map_name}")
        return

    if args.list_algorithms:
        print("可用算法:")
        for algo_name in runner.algorithms.keys():
            print(f"  - {algo_name}")
        return

    # 确定实验配置
    if args.preset:
        config = EXPERIMENT_PRESETS[args.preset].copy()
        print(f"使用预设配置: {args.preset}")
    else:
        # 手动配置
        config = {}

        if args.algorithms:
            config['algorithms'] = args.algorithms
        else:
            config['algorithms'] = ['q_learning']  # 默认算法

        if args.maps:
            config['maps'] = args.maps
        else:
            config['maps'] = ['4x4_default']  # 默认地图

        # 处理slippery模式
        if args.slippery and args.deterministic:
            print("错误: 不能同时指定 --slippery 和 --deterministic")
            return
        elif args.slippery:
            config['slippery_modes'] = [True]
        elif args.deterministic:
            config['slippery_modes'] = [False]
        else:
            config['slippery_modes'] = [True]  # 默认随机模式

        config['seeds'] = args.seeds

    # 运行实验
    try:
        results = runner.run_batch_experiments(**config)

        print(f"\n🎉 实验完成！结果保存在 {args.results_dir} 目录")
        print(f"运行了 {len(results)} 个实验")

        # 显示简要统计
        success_rates = [r['final_evaluation']['success_rate'] for r in results]
        avg_steps = [r['final_evaluation']['avg_steps'] for r in results]

        print(f"平均成功率: {np.mean(success_rates):.3f} ± {np.std(success_rates):.3f}")
        print(f"平均步数: {np.mean(avg_steps):.1f} ± {np.std(avg_steps):.1f}")
    except Exception as e:
        print(f"实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
