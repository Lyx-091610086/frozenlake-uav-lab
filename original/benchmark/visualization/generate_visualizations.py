#!/usr/bin/env python3
"""
强化学习实验可视化主脚本
统一调用所有可视化功能
"""

import argparse
import sys
import os
import json
import numpy as np

def collect_experiment_data(results_dir: str) -> list:
    """从results目录结构中收集所有实验数据"""
    experiments = []

    # 遍历results目录结构
    # 结构: results/{map_name}/{mode}/{algorithm}/{seed}/results.json
    for map_name in os.listdir(results_dir):
        map_path = os.path.join(results_dir, map_name)
        if not os.path.isdir(map_path) or map_name.startswith('.'):
            continue

        for mode in ['deterministic', 'stochastic']:
            mode_path = os.path.join(map_path, mode)
            if not os.path.isdir(mode_path):
                continue

            for algorithm in os.listdir(mode_path):
                algo_path = os.path.join(mode_path, algorithm)
                if not os.path.isdir(algo_path):
                    continue

                for seed in os.listdir(algo_path):
                    seed_path = os.path.join(algo_path, seed)
                    if not os.path.isdir(seed_path):
                        continue

                    results_file = os.path.join(seed_path, 'results.json')
                    if not os.path.exists(results_file):
                        continue

                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # 提取用于可视化的关键信息
                        best_perf = data.get('best_performance', {})
                        experiment = {
                            'experiment_id': data.get('experiment_id', f"{algorithm}_{map_name}_{mode}_{seed}"),
                            'algorithm': algorithm,
                            'map': map_name,
                            'slippery': data.get('is_slippery', mode == 'stochastic'),
                            'success_rate': best_perf.get('success_rate', 0.0),
                            'avg_steps': best_perf.get('avg_steps', 100.0),
                            'training_time': data.get('training_time_seconds', 0.0),
                            'best_episode': best_perf.get('episode', 0),
                            'seed': seed,
                            'results_file': results_file
                        }
                        experiments.append(experiment)

                    except (json.JSONDecodeError, IOError) as e:
                        print(f"⚠️  跳过损坏的文件 {results_file}: {e}")
                        continue

    return experiments

def main():
    parser = argparse.ArgumentParser(description='生成强化学习实验的可视化结果')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='实验结果目录 (默认: results)')
    parser.add_argument('--type', choices=['static', 'animation', 'policy', 'learning', 'all'],
                       default='all', help='可视化类型')
    parser.add_argument('--experiment-id', type=str,
                       help='指定实验ID用于动画生成')
    parser.add_argument('--fps', type=int, default=2,
                       help='动画帧率 (默认: 2)')

    args = parser.parse_args()

    results_dir = args.results_dir

    print("🎨 强化学习可视化生成器")
    print(f"结果目录: {results_dir}")
    print(f"可视化类型: {args.type}")

    try:
        # 从results目录结构中收集所有实验数据
        experiments = collect_experiment_data(results_dir)

        if not experiments:
            print("❌ 没有实验数据")
            print("请先运行实验生成结果文件")
            return 1

        print(f"📊 找到 {len(experiments)} 个实验记录")

        # 生成静态图表
        if args.type in ['static', 'all']:
            print("📊 生成静态图表（整合所有算法）...")
            from .static_plots import create_simple_plots
            output_base_dir = os.path.join(results_dir, 'simple_plots')
            create_simple_plots(experiments, output_base_dir)

        # 生成动画
        if args.type in ['animation', 'policy', 'learning', 'all']:
            print("🎬 生成动画...")
            from .animation import RLAnimation
            animator = RLAnimation()

            # 生成所有实验的动画
            if args.type in ['policy', 'learning', 'all']:
                print("🎯 生成策略和学习动画...")

                # 按地图和模式分组处理所有实验
                map_mode_groups = {}
                for exp in experiments:
                    map_name = exp['map']
                    mode = 'stochastic' if exp['slippery'] else 'deterministic'
                    key = f"{map_name}_{mode}"

                    if key not in map_mode_groups:
                        map_mode_groups[key] = []
                    map_mode_groups[key].append(exp)

                # 为每个地图+模式组合生成动画
                for group_key, exp_list in map_mode_groups.items():
                    map_name, mode = group_key.split('_', 1)
                    print(f"  📍 处理 {map_name}/{mode} ({len(exp_list)} 个实验)...")

                    # 为每个实验生成动画
                    for exp in exp_list:
                        algorithm = exp['algorithm']
                        # 从experiment_id中提取种子信息
                        experiment_id = exp['experiment_id']
                        seed = experiment_id.split('_')[-1]  # 最后一个部分是种子
                        exp_dir = os.path.join(results_dir, map_name, mode, algorithm, str(seed))

                        # 确定输出目录（按地图+模式组织）
                        gif_dir = os.path.join(results_dir, 'simple_plots', map_name, mode)
                        # 只在真正需要生成动画时创建目录
                        if not os.path.exists(gif_dir):
                            os.makedirs(gif_dir, exist_ok=True)

                        # 生成策略执行动画
                        if args.type in ['policy', 'all']:
                            policy_file = os.path.join(exp_dir, 'policy.npy')
                            if os.path.exists(policy_file):
                                # 创建环境
                                import sys
                                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                                from environments.frozenlake_env import FrozenLakeEnv
                                env = FrozenLakeEnv(map_name, exp['slippery'])

                                policy = np.load(policy_file)
                                gif_path = os.path.join(gif_dir, f'{algorithm}_policy.gif')
                                animator.create_policy_animation(env.env, policy, gif_path, fps=args.fps)
                                print(f"    ✅ {algorithm} 策略动画已生成")

                        # 生成学习进度动画
                        if args.type in ['learning', 'all']:
                            history_file = os.path.join(exp_dir, 'training_history.json')
                            policy_file = os.path.join(exp_dir, 'policy.npy')
                            q_table_file = os.path.join(exp_dir, 'q_table.npy')
                            value_file = os.path.join(exp_dir, 'value_function.npy')

                            if os.path.exists(history_file):
                                with open(history_file, 'r', encoding='utf-8') as f:
                                    training_history = json.load(f)

                                if training_history:
                                    # 创建环境
                                    import sys
                                    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                                    from environments.frozenlake_env import FrozenLakeEnv
                                    env = FrozenLakeEnv(map_name, exp['slippery'])

                                    gif_path = os.path.join(gif_dir, f'{algorithm}_learning.gif')
                                    animator.create_learning_progress_animation(
                                        training_history, env.env, gif_path, fps=args.fps,
                                        policy_file=policy_file if os.path.exists(policy_file) else None,
                                        q_table_file=q_table_file if os.path.exists(q_table_file) else None,
                                        value_file=value_file if os.path.exists(value_file) else None)
                                    print(f"    ✅ {algorithm} 学习动画已生成")

        # 生成实验对比动画（保存在根目录）
        if args.type == 'animation' or args.type == 'all':
            print("🔄 生成实验对比动画...")
            gif_dir = os.path.join(results_dir, 'simple_plots')
            os.makedirs(gif_dir, exist_ok=True)
            gif_path = os.path.join(gif_dir, 'experiments_comparison.gif')
            animator.create_comparison_animation(experiments, gif_path, fps=args.fps)

        print("✅ 所有可视化生成完成！")
        plots_dir = os.path.join(results_dir, 'simple_plots')
        print(f"📁 文件保存位置: {plots_dir}")

        # 列出生成的文件
        if os.path.exists(plots_dir):
            files = os.listdir(plots_dir)
            if files:
                print("📊 生成的文件:")
                for file in sorted(files):
                    print(f"   - {file}")

        return 0

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有依赖包")
        return 1
    except Exception as e:
        print(f"❌ 生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
