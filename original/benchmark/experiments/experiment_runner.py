import os
import json
import numpy as np
import warnings
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

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

from environments.frozenlake_env import FrozenLakeEnv
# 基础算法导入
from algorithms.planning.value_iteration import ValueIteration
from algorithms.planning.policy_iteration import PolicyIteration
from algorithms.planning.astar import AStar
from algorithms.tabular.q_learning import QLearning
from algorithms.tabular.sarsa import SARSA
from algorithms.tabular.monte_carlo_control import MonteCarloControl
from algorithms.tabular.n_step_td import NStepTD
from algorithms.tabular.expected_sarsa import ExpectedSARSA
from algorithms.tabular.double_q_learning import DoubleQLearning
from algorithms.tabular.eligibility_traces import EligibilityTraces
from algorithms.model_based.dyna_q import DynaQ

# 深度学习算法导入（如果可用）
try:
    import torch
    TORCH_AVAILABLE = True
    from algorithms.deep_value.dqn import DQN
    from algorithms.deep_value.double_dqn import DoubleDQN
    from algorithms.deep_value.dueling_dqn import DuelingDQN
    from algorithms.deep_value.per_dqn import PER_DQN
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch 未安装，深度学习算法将不可用")
    print("安装 PyTorch: pip install torch torchvision torchaudio")

class ExperimentRunner:
    """实验运行器 - 统一管理算法和地图的组合实验"""

    def __init__(self, results_dir: str = "results", save_detailed_results: bool = False):
        self.results_dir = results_dir
        self.save_detailed_results = save_detailed_results  # 是否保存详细结果
        os.makedirs(results_dir, exist_ok=True)

        # 实验日志
        self.experiment_log = []

        # 算法映射
        self.algorithms = {
            # 规划算法
            'value_iteration': ValueIteration,
            'policy_iteration': PolicyIteration,
            'astar': AStar,

            # 经典表格RL
            'q_learning': QLearning,
            'sarsa': SARSA,
            'monte_carlo_control': MonteCarloControl,
            'n_step_td': NStepTD,
            'expected_sarsa': ExpectedSARSA,
            'double_q_learning': DoubleQLearning,
            'eligibility_traces': EligibilityTraces,

            # 带模型的RL
            'dyna_q': DynaQ,
        }

        # 深度学习算法（如果PyTorch可用）
        if TORCH_AVAILABLE:
            self.algorithms.update({
                'dqn': DQN,
                'double_dqn': DoubleDQN,
                'dueling_dqn': DuelingDQN,
                'per_dqn': PER_DQN,
            })

        # 实验记录
        self.experiment_log = []

    def get_experiment_config(self, preset_name: str) -> Dict[str, Any]:
        """获取实验预设配置"""
        from .configs import EXPERIMENT_PRESETS
        if preset_name not in EXPERIMENT_PRESETS:
            raise ValueError(f"未知的实验预设: {preset_name}")
        return EXPERIMENT_PRESETS[preset_name]

    def run_single_experiment(self,
                            algorithm_name: str,
                            map_name: str,
                            is_slippery: bool = True,
                            config: Optional[Dict[str, Any]] = None,
                            seed: int = 42) -> Dict[str, Any]:
        """运行单个实验"""

        print(f"\n=== 运行实验: {algorithm_name} on {map_name} (slippery={is_slippery}) ===")

        # 设置随机种子
        np.random.seed(seed)

        # 创建环境
        env = FrozenLakeEnv(map_name=map_name, is_slippery=is_slippery)
        print(f"环境信息: {env.get_map_info()}")

        # 创建算法实例
        if algorithm_name not in self.algorithms:
            raise ValueError(f"未知算法: {algorithm_name}")

        algorithm_class = self.algorithms[algorithm_name]
        algorithm = algorithm_class(env, config)

        # 训练
        train_start_time = datetime.now()
        train_results = algorithm.train()
        train_end_time = datetime.now()

        # 评估最终策略
        final_policy = algorithm.get_policy()
        eval_results = algorithm.evaluate_policy(final_policy, n_episodes=1000)

        # 从训练历史中找到最佳表现
        best_performance = self._find_best_performance(train_results, eval_results, algorithm_name)

        # 收集结果
        experiment_result = {
            'experiment_id': f"{algorithm_name}_{map_name}_{'slippery' if is_slippery else 'deterministic'}_{seed}",
            'timestamp': datetime.now().isoformat(),
            'algorithm': algorithm_name,
            'map_name': map_name,
            'is_slippery': is_slippery,
            'config': config or {},
            'seed': seed,
            'environment_info': {
                k: int(v) if isinstance(v, (np.integer, np.int64)) else v
                for k, v in env.get_map_info().items()
            },
            'training_time_seconds': (train_end_time - train_start_time).total_seconds(),
            'training_results': train_results,
            'best_performance': best_performance,
            'final_evaluation': eval_results,
            'final_policy': list(final_policy),
            'final_value_function': list(algorithm.get_value_function()) if hasattr(algorithm, 'get_value_function') else None,
        }

        # 如果是表格型算法，保存Q表
        if hasattr(algorithm, 'get_q_table'):
            experiment_result['q_table'] = algorithm.get_q_table().tolist()

        # 保存结果
        if self.save_detailed_results:
            self._save_experiment_result(experiment_result)

        # 记录到日志 (使用最佳表现而非最终表现)
        self.experiment_log.append({
            'experiment_id': experiment_result['experiment_id'],
            'algorithm': algorithm_name,
            'map': map_name,
            'slippery': is_slippery,
            'success_rate': best_performance['success_rate'],
            'avg_steps': best_performance['avg_steps'],
            'training_time': experiment_result['training_time_seconds'],
            'best_episode': best_performance['episode']
        })

        print(".3f")
        print(".1f")
        env.close()

        return experiment_result

    def run_batch_experiments(self,
                            algorithms: List[str],
                            maps: List[str],
                            slippery_modes: List[bool] = [True, False],
                            configs: Optional[Dict[str, Dict[str, Any]]] = None,
                            seeds: List[int] = [42]) -> List[Dict[str, Any]]:
        """批量运行实验（带检查点机制）"""

        results = []
        configs = configs or {}

        total_experiments = len(algorithms) * len(maps) * len(slippery_modes) * len(seeds)
        experiment_count = 0
        skipped_count = 0

        print(f"\n🚀 开始批量实验，共 {total_experiments} 个实验")
        print("🔍 检查已存在的实验结果...")

        # 加载已有的实验日志（如果存在）
        self._load_existing_experiment_log()

        for algorithm in algorithms:
            for map_name in maps:
                for is_slippery in slippery_modes:
                    for seed in seeds:
                        experiment_count += 1

                        # 检查实验结果是否已存在
                        mode = 'stochastic' if is_slippery else 'deterministic'
                        algorithm_dir = os.path.join(self.results_dir, map_name, mode, algorithm, str(seed))
                        result_file = os.path.join(algorithm_dir, 'results.json')

                        if os.path.exists(result_file):
                            # 简单检查点机制：只要result.json存在且包含基本信息就跳过
                            try:
                                with open(result_file, 'r', encoding='utf-8') as f:
                                    existing_result = json.load(f)

                                # 检查是否包含必要的基本信息
                                basic_checks = (
                                    existing_result.get('algorithm') == algorithm and
                                    existing_result.get('map_name') == map_name and
                                    existing_result.get('is_slippery') == is_slippery and
                                    existing_result.get('seed') == seed
                                )

                                if basic_checks:
                                    print(f"⏭️  跳过已存在的实验: {experiment_count}/{total_experiments} - {algorithm} on {map_name} ({mode}, seed={seed})")
                                    results.append(existing_result)
                                    skipped_count += 1
                                    continue
                                else:
                                    print(f"⚠️  实验结果信息不匹配，重新运行: {experiment_count}/{total_experiments} - {algorithm} on {map_name} ({mode}, seed={seed})")

                            except (json.JSONDecodeError, IOError) as e:
                                print(f"⚠️  无法读取实验结果，重新运行: {experiment_count}/{total_experiments} - {algorithm} on {map_name} ({mode}, seed={seed}) - 错误: {e}")
                            except Exception as e:
                                print(f"⚠️  实验结果检查出错，重新运行: {experiment_count}/{total_experiments} - {algorithm} on {map_name} ({mode}, seed={seed}) - 错误: {e}")

                        print(f"\n📊 实验进度: {experiment_count}/{total_experiments} (已跳过: {skipped_count})")

                        config = configs.get(algorithm, {})
                        result = self.run_single_experiment(
                            algorithm, map_name, is_slippery, config, seed
                        )
                        results.append(result)

        # 保存实验汇总
        self._save_experiment_summary()

        completed_count = len(results) - skipped_count
        print(f"\n✅ 批量实验完成！")
        print(f"   📊 总实验数: {total_experiments}")
        print(f"   ⏭️  已跳过: {skipped_count}")
        print(f"   ✅ 新运行: {completed_count}")
        return results

    def _load_existing_experiment_log(self):
        """加载已有的实验日志"""
        summary_file = os.path.join(self.results_dir, 'experiment_summary.json')
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    existing_log = json.load(f)
                # 将已有的实验日志合并到当前日志中
                for exp in existing_log:
                    # 检查是否已存在（避免重复）
                    if not any(e['experiment_id'] == exp['experiment_id'] for e in self.experiment_log):
                        self.experiment_log.append(exp)
                print(f"📋 已加载 {len(existing_log)} 个已有的实验记录")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  无法加载现有实验日志: {e}")
        else:
            print("📋 没有找到现有实验日志，将创建新的")

    def _find_best_performance(self, train_results: Dict[str, Any], eval_results: Dict[str, Any],
                             algorithm_name: str) -> Dict[str, Any]:
        """从训练历史中找到最佳表现"""
        training_history = train_results.get('training_history', [])

        # 规划算法（没有训练历史的算法）直接使用评估结果
        planning_algorithms = {'value_iteration', 'policy_iteration', 'astar'}
        if algorithm_name in planning_algorithms or not training_history:
            return {
                'episode': 0,  # 规划算法没有episode概念
                'success_rate': eval_results['success_rate'],
                'avg_reward': eval_results['avg_reward'],
                'avg_steps': eval_results['avg_steps'],
                'epsilon': 0.0  # 规划算法没有epsilon
            }

        # 学习算法：找到训练历史中的最佳表现
        best_entry = max(training_history, key=lambda x: x['success_rate'])

        # 如果最终评估结果更好，使用最终评估结果
        if eval_results['success_rate'] > best_entry['success_rate']:
            return {
                'episode': len(training_history) * (train_results.get('eval_every', 1000)),  # 最后一次评估的episode
                'success_rate': eval_results['success_rate'],
                'avg_reward': eval_results['avg_reward'],
                'avg_steps': eval_results['avg_steps'],
                'epsilon': best_entry.get('epsilon', 0.0)  # 使用最后一次的epsilon
            }

        return {
            'episode': best_entry['episode'],
            'success_rate': best_entry['success_rate'],
            'avg_reward': best_entry['avg_reward'],
            'avg_steps': best_entry['avg_steps'],
            'epsilon': best_entry.get('epsilon', 0.0)
        }

    def _save_experiment_result(self, result: Dict[str, Any]):
        """保存单个实验结果 - 按地图+模式+算法+种子组织"""
        # 按地图+模式+算法+种子组织目录结构
        mode = 'stochastic' if result['is_slippery'] else 'deterministic'
        algorithm = result['algorithm']
        seed = result['seed']

        # 创建层级目录结构
        algorithm_dir = os.path.join(self.results_dir, result['map_name'], mode, algorithm, str(seed))
        os.makedirs(algorithm_dir, exist_ok=True)

        # 保存完整结果 - 使用最安全的方式避免循环引用
        result_file = os.path.join(algorithm_dir, 'results.json')

        # 只保存基本类型数据
        safe_result = {
            'experiment_id': str(result['experiment_id']),
            'timestamp': str(result['timestamp']),
            'algorithm': str(result['algorithm']),
            'map_name': str(result['map_name']),
            'is_slippery': bool(result['is_slippery']),
            'config': dict(result['config']) if isinstance(result['config'], dict) else {},
            'seed': int(result['seed']),
            'environment_info': {
                'name': str(result['environment_info']['name']),
                'size': tuple(result['environment_info']['size']),
                'n_states': int(result['environment_info']['n_states']),
                'n_actions': int(result['environment_info']['n_actions']),
                'is_slippery': bool(result['environment_info']['is_slippery']),
                'difficulty': str(result['environment_info']['difficulty']),
                'description': str(result['environment_info']['description']),
            },
            'training_time_seconds': float(result['training_time_seconds']),
            'best_performance': dict(result['best_performance']),
            'final_evaluation': dict(result['final_evaluation']),
        }

        # 安全地添加训练结果
        if isinstance(result['training_results'], dict):
            safe_result['training_results'] = {}
            for key, value in result['training_results'].items():
                if key == 'training_history' and isinstance(value, list):
                    safe_result['training_results'][key] = [
                        {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in entry.items()}
                        for entry in value
                    ]
                elif isinstance(value, (int, float, str, bool)):
                    safe_result['training_results'][key] = value
                else:
                    safe_result['training_results'][key] = str(value)

        # 安全地添加策略和值函数
        safe_result['final_policy'] = [int(x) for x in result['final_policy']]
        if result['final_value_function'] is not None:
            safe_result['final_value_function'] = [float(x) for x in result['final_value_function']]
        else:
            safe_result['final_value_function'] = None

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(safe_result, f, indent=2, ensure_ascii=False)

        # 保存策略
        policy_file = os.path.join(algorithm_dir, 'policy.npy')
        np.save(policy_file, np.array(result['final_policy']))

        # 保存值函数
        if result['final_value_function'] is not None:
            value_file = os.path.join(algorithm_dir, 'value_function.npy')
            np.save(value_file, np.array(result['final_value_function']))

        # 保存Q表（如果存在）
        if 'q_table' in result and result['q_table'] is not None:
            q_file = os.path.join(algorithm_dir, 'q_table.npy')
            np.save(q_file, np.array(result['q_table']))

        # 保存训练历史（用于动画）
        if 'training_results' in result and 'training_history' in result['training_results']:
            history_file = os.path.join(algorithm_dir, 'training_history.json')
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(result['training_results']['training_history'], f, indent=2, ensure_ascii=False)

        # 生成可视化（传递正确的目录）
        self._generate_visualizations(result, algorithm_dir)

    def _generate_visualizations(self, result: Dict[str, Any], algorithm_dir: str):
        """生成实验可视化 - 保存到算法目录"""
        try:
            from visualization.experiment_viz import ExperimentVisualizer
            from environments.frozenlake_env import FrozenLakeEnv

            visualizer = ExperimentVisualizer()

            # 创建环境用于可视化
            env = FrozenLakeEnv(result['map_name'], result['is_slippery'])

            # 生成训练曲线（如果有训练历史）
            if 'training_history' in result['training_results']:
                training_history = result['training_results']['training_history']
                if training_history:
                    curve_file = os.path.join(algorithm_dir, 'training_curve.png')
                    visualizer.plot_training_curve(training_history, curve_file, result['algorithm'])

            # 生成Q值热力图（如果有Q表）
            if 'q_table' in result and result['q_table'] is not None:
                q_table = np.array(result['q_table'])
                heatmap_file = os.path.join(algorithm_dir, 'q_heatmap.png')
                visualizer.plot_q_heatmap(q_table, env.env, heatmap_file)

            # 生成策略可视化
            if result['final_policy'] is not None:
                policy = np.array(result['final_policy'])
                policy_viz_file = os.path.join(algorithm_dir, 'policy.png')
                visualizer.plot_policy(policy, env.env, policy_viz_file)

            # 生成值函数可视化
            if result['final_value_function'] is not None:
                values = np.array(result['final_value_function'])
                value_viz_file = os.path.join(algorithm_dir, 'value_function.png')
                visualizer.plot_value_function(values, env.env, value_viz_file)

        except ImportError:
            # 如果可视化模块不存在，跳过
            pass
        except Exception as e:
            print(f"可视化生成失败: {e}")

    def _save_experiment_summary(self):
        """保存实验汇总"""
        summary_file = os.path.join(self.results_dir, 'experiment_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, indent=2, ensure_ascii=False)

        # 按地图+模式统计
        map_mode_stats = {}
        for exp in self.experiment_log:
            map_name = exp['map']
            mode = 'stochastic' if exp['slippery'] else 'deterministic'
            key = f"{map_name}_{mode}"

            if key not in map_mode_stats:
                map_mode_stats[key] = []
            map_mode_stats[key].append(exp)

        # 打印汇总统计
        print("\n📈 实验汇总统计 (按地图+模式):")
        print("=" * 80)

        for map_mode, exps in sorted(map_mode_stats.items()):
            print(f"\n🔍 {map_mode}:")
            print(f"   实验数量: {len(exps)}")

            # 按算法统计
            algo_stats = {}
            for exp in exps:
                algo = exp['algorithm']
                if algo not in algo_stats:
                    algo_stats[algo] = []
                algo_stats[algo].append(exp)

            for algo, algo_exps in sorted(algo_stats.items()):
                success_rates = [e['success_rate'] for e in algo_exps]
                avg_times = [e['training_time'] for e in algo_exps]
                avg_steps = [e['avg_steps'] for e in algo_exps]

                print(f"   {algo.upper():<20} "
                      f"成功率: {np.mean(success_rates):.3f} ± {np.std(success_rates):.3f}, "
                      f"平均步数: {np.mean(avg_steps):.1f}, "
                      f"平均时间: {np.mean(avg_times):.1f}s")

        print("\n" + "=" * 80)

    def load_experiment_result(self, experiment_id: str) -> Dict[str, Any]:
        """加载实验结果"""
        result_file = os.path.join(self.results_dir, experiment_id, 'results.json')
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
