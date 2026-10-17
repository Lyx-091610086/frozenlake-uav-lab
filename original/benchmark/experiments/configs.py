# experiments/configs.py
# 实验配置文件

# 算法配置
ALGORITHM_CONFIGS = {
    # 规划算法
    'value_iteration': {
        'theta': 1e-6,          # 收敛阈值
        'gamma': 0.99,          # 折扣因子
        'max_iterations': 1000  # 最大迭代次数
    },

    'policy_iteration': {
        'theta': 1e-6,          # 值函数评估收敛阈值
        'gamma': 0.99,          # 折扣因子
        'max_iterations': 1000, # 值函数评估最大迭代
        'max_policy_iterations': 100  # 策略迭代最大次数
    },

    'astar': {},  # A*不需要额外配置

    # 经典表格型RL算法
    'q_learning': {
        'alpha': 0.1,           # 学习率
        'gamma': 0.99,          # 折扣因子
        'epsilon_start': 1.0,   # 初始ε
        'epsilon_end': 0.1,     # 最终ε (提高最终探索率)
        'epsilon_decay': 0.995, # ε衰减率
        'max_episodes': 10000,  # 最大训练回合
        'max_steps': 100,       # 每回合最大步数
        'eval_every': 1000,     # 每多少回合评估一次
        'eval_episodes': 100    # 每次评估的回合数
    },

    'sarsa': {
        'alpha': 0.1,           # 学习率
        'gamma': 0.99,          # 折扣因子
        'epsilon_start': 1.0,   # 初始ε
        'epsilon_end': 0.1,     # 最终ε (提高最终探索率)
        'epsilon_decay': 0.995, # ε衰减率
        'max_episodes': 10000,  # 最大训练回合
        'max_steps': 100,       # 每回合最大步数
        'eval_every': 1000,     # 每多少回合评估一次
        'eval_episodes': 100    # 每次评估的回合数
    },

    'monte_carlo_control': {
        'alpha': 0.1,           # 学习率 (on-policy时)
        'gamma': 0.99,          # 折扣因子
        'epsilon_start': 1.0,   # 初始ε
        'epsilon_end': 0.1,     # 最终ε (提高最终探索率)
        'epsilon_decay': 0.995, # ε衰减率
        'max_episodes': 10000,  # 最大训练回合
        'max_steps': 100,       # 每回合最大步数
        'eval_every': 1000,     # 每多少回合评估一次
        'eval_episodes': 100,   # 每次评估的回合数
        'off_policy': False,    # 是否使用off-policy MC
        'first_visit': True     # 是否使用首次访问MC
    },

    'n_step_td': {
        'n': 3,                 # n步数
        'alpha': 0.1,           # 学习率
        'gamma': 0.99,          # 折扣因子
        'epsilon_start': 1.0,   # 初始ε
        'epsilon_end': 0.1,     # 最终ε (提高最终探索率)
        'epsilon_decay': 0.995, # ε衰减率
        'max_episodes': 10000,  # 最大训练回合
        'max_steps': 100,       # 每回合最大步数
        'eval_every': 1000,     # 每多少回合评估一次
        'eval_episodes': 100,   # 每次评估的回合数
        'expected_sarsa': False # 是否使用Expected SARSA
    },

    'expected_sarsa': {
        'alpha': 0.1,           # 学习率
        'gamma': 0.99,          # 折扣因子
        'epsilon_start': 1.0,   # 初始ε
        'epsilon_end': 0.1,     # 最终ε (提高最终探索率)
        'epsilon_decay': 0.995, # ε衰减率
        'max_episodes': 10000,  # 最大训练回合
        'max_steps': 100,       # 每回合最大步数
        'eval_every': 1000,     # 每多少回合评估一次
        'eval_episodes': 100    # 每次评估的回合数
    },

    'double_q_learning': {
        'alpha': 0.1,           # 学习率
        'gamma': 0.99,          # 折扣因子
        'epsilon_start': 1.0,   # 初始ε
        'epsilon_end': 0.1,     # 最终ε (提高最终探索率)
        'epsilon_decay': 0.995, # ε衰减率
        'max_episodes': 10000,  # 最大训练回合
        'max_steps': 100,       # 每回合最大步数
        'eval_every': 1000,     # 每多少回合评估一次
        'eval_episodes': 100    # 每次评估的回合数
    },

    'eligibility_traces': {
        'lambda': 0.9,          # λ参数
        'alpha': 0.1,           # 学习率
        'gamma': 0.99,          # 折扣因子
        'epsilon_start': 1.0,   # 初始ε
        'epsilon_end': 0.1,     # 最终ε (提高最终探索率)
        'epsilon_decay': 0.995, # ε衰减率
        'max_episodes': 10000,  # 最大训练回合
        'max_steps': 100,       # 每回合最大步数
        'eval_every': 1000,     # 每多少回合评估一次
        'eval_episodes': 100,   # 每次评估的回合数
        'algorithm': 'sarsa_lambda',  # 'sarsa_lambda' 或 'q_lambda'
        'replacing_traces': False  # 是否使用替换迹
    },

    # 带模型的RL
    'dyna_q': {
        'n_planning': 5,        # 每次规划步数
        'alpha': 0.1,           # 学习率
        'gamma': 0.99,          # 折扣因子
        'epsilon_start': 1.0,   # 初始ε
        'epsilon_end': 0.1,     # 最终ε (提高最终探索率)
        'epsilon_decay': 0.995, # ε衰减率
        'max_episodes': 10000,  # 最大训练回合
        'max_steps': 100,       # 每回合最大步数
        'eval_every': 1000,     # 每多少回合评估一次
        'eval_episodes': 100    # 每次评估的回合数
    },


    # 深度价值学习
    'dqn': {
        'batch_size': 64,       # 批次大小
        'gamma': 0.99,          # 折扣因子
        'learning_rate': 1e-3,  # 学习率
        'epsilon_start': 1.0,   # 初始ε
        'epsilon_end': 0.1,     # 最终ε (提高最终探索率)
        'epsilon_decay': 2000,  # ε衰减步数 (延长衰减时间)
        'target_update_freq': 100,  # 目标网络更新频率
        'memory_size': 10000,   # 经验回放缓冲区大小
        'max_episodes': 10000,  # 最大训练回合
        'max_steps': 500,       # 每回合最大步数
        'eval_every': 50,       # 每多少回合评估一次
        'eval_episodes': 10     # 每次评估的回合数
    },

    'double_dqn': {
        'batch_size': 64,       # 批次大小
        'gamma': 0.99,          # 折扣因子
        'learning_rate': 1e-3,  # 学习率
        'epsilon_start': 1.0,   # 初始ε
        'epsilon_end': 0.1,     # 最终ε (提高最终探索率)
        'epsilon_decay': 2000,  # ε衰减步数 (延长衰减时间)
        'target_update_freq': 100,  # 目标网络更新频率
        'memory_size': 10000,   # 经验回放缓冲区大小
        'max_episodes': 10000,  # 最大训练回合
        'max_steps': 500,       # 每回合最大步数
        'eval_every': 50,       # 每多少回合评估一次
        'eval_episodes': 10     # 每次评估的回合数
    },

    'dueling_dqn': {
        'batch_size': 64,       # 批次大小
        'gamma': 0.99,          # 折扣因子
        'learning_rate': 1e-3,  # 学习率
        'epsilon_start': 1.0,   # 初始ε
        'epsilon_end': 0.1,     # 最终ε (提高最终探索率)
        'epsilon_decay': 2000,  # ε衰减步数 (延长衰减时间)
        'target_update_freq': 100,  # 目标网络更新频率
        'memory_size': 10000,   # 经验回放缓冲区大小
        'max_episodes': 10000,  # 最大训练回合
        'max_steps': 500,       # 每回合最大步数
        'eval_every': 50,       # 每多少回合评估一次
        'eval_episodes': 10     # 每次评估的回合数
    },

    'per_dqn': {
        'batch_size': 64,       # 批次大小
        'gamma': 0.99,          # 折扣因子
        'learning_rate': 1e-3,  # 学习率
        'epsilon_start': 1.0,   # 初始ε
        'epsilon_end': 0.1,     # 最终ε (提高最终探索率)
        'epsilon_decay': 2000,  # ε衰减步数 (延长衰减时间)
        'target_update_freq': 100,  # 目标网络更新频率
        'memory_size': 10000,   # 经验回放缓冲区大小
        'alpha': 0.6,           # PER优先级指数
        'beta_start': 0.4,      # IS权重初始值
        'beta_end': 1.0,        # IS权重最终值
        'beta_decay': 10000,    # β衰减步数
        'max_episodes': 10000,  # 最大训练回合
        'max_steps': 500,       # 每回合最大步数
        'eval_every': 50,       # 每多少回合评估一次
        'eval_episodes': 10     # 每次评估的回合数
    },
}

# 地图难度分组
MAP_DIFFICULTY_GROUPS = {
    'easy': [
        '4x4_easy_open', '4x4_default',
        '5x5_easy_open',
        '6x6_easy_open',
        '8x8_open_sparse',
        '10x10_easy_open'
    ],

    'medium': [
        '4x4_bridge', '4x4_maze', '4x4_right_edge',
        '5x5_medium_cross', '5x5_right_edge', '5x5_ring',
        '6x6_right_edge', '6x6_snake_corridor',
        '8x8_official', '8x8_left_edge_safe',
        '10x10_checker_safe_bottom'
    ],

    'hard': [
        '6x6_central_bridge', '6x6_hard_mixed',
        '8x8_snake_corridor', '8x8_risky_shortcut',
        '10x10_snake_corridor', '10x10_hard_mixed'
    ]
}

# 地图尺寸分组
MAP_SIZE_GROUPS = {
    '4x4': [name for name in MAP_DIFFICULTY_GROUPS['easy'] + MAP_DIFFICULTY_GROUPS['medium'] + MAP_DIFFICULTY_GROUPS['hard'] if name.startswith('4x4')],
    '5x5': [name for name in MAP_DIFFICULTY_GROUPS['easy'] + MAP_DIFFICULTY_GROUPS['medium'] + MAP_DIFFICULTY_GROUPS['hard'] if name.startswith('5x5')],
    '6x6': [name for name in MAP_DIFFICULTY_GROUPS['easy'] + MAP_DIFFICULTY_GROUPS['medium'] + MAP_DIFFICULTY_GROUPS['hard'] if name.startswith('6x6')],
    '8x8': [name for name in MAP_DIFFICULTY_GROUPS['easy'] + MAP_DIFFICULTY_GROUPS['medium'] + MAP_DIFFICULTY_GROUPS['hard'] if name.startswith('8x8')],
    '10x10': [name for name in MAP_DIFFICULTY_GROUPS['easy'] + MAP_DIFFICULTY_GROUPS['medium'] + MAP_DIFFICULTY_GROUPS['hard'] if name.startswith('10x10')]
}

# 实验预设
EXPERIMENT_PRESETS = {
    'quick_test': {
        'algorithms': ['q_learning', 'sarsa'],
        'maps': ['4x4_default', '5x5_easy_open'],
        'slippery_modes': [False, True],
        'seeds': [42]
    },

    'planning_baseline': {
        'algorithms': ['value_iteration', 'policy_iteration', 'astar'],
        'maps': ['4x4_default', '8x8_official'],
        'slippery_modes': [False],  # 规划算法通常用于确定性环境
        'seeds': [42]
    },

    'tabular_comparison': {
        'algorithms': ['q_learning', 'sarsa', 'expected_sarsa', 'double_q_learning'],
        'maps': ['4x4_default', '5x5_medium_cross', '6x6_hard_mixed'],
        'slippery_modes': [False, True],
        'seeds': [42, 123, 456]  # 多种子运行以获得统计意义
    },

    'advanced_tabular': {
        'algorithms': ['monte_carlo_control', 'n_step_td', 'eligibility_traces'],
        'maps': ['4x4_default', '5x5_medium_cross'],
        'slippery_modes': [False, True],
        'seeds': [42, 123]
    },

    'model_based': {
        'algorithms': ['dyna_q'],
        'maps': ['4x4_default', '8x8_official'],
        'slippery_modes': [True],
        'seeds': [42, 123]
    },

    'deep_value_comparison': {
        'algorithms': ['dqn', 'double_dqn', 'dueling_dqn', 'per_dqn'],
        'maps': ['4x4_default', '8x8_official'],
        'slippery_modes': [True],
        'seeds': [42]
    },

    'map_scalability': {
        'algorithms': ['q_learning'],
        'maps': ['4x4_default', '5x5_medium_cross', '6x6_hard_mixed',
                '8x8_official', '10x10_hard_mixed'],
        'slippery_modes': [True],
        'seeds': [42]
    },

    'full_benchmark': {
        'algorithms': ['value_iteration', 'policy_iteration', 'astar',
                      'q_learning', 'sarsa', 'expected_sarsa', 'double_q_learning',
                      'monte_carlo_control', 'n_step_td', 'eligibility_traces',
                      'dyna_q',
                      'dqn', 'double_dqn', 'dueling_dqn', 'per_dqn'],
        'maps': MAP_DIFFICULTY_GROUPS['easy'] + MAP_DIFFICULTY_GROUPS['medium'] + MAP_DIFFICULTY_GROUPS['hard'],
        'slippery_modes': [False, True],
        'seeds': [42, 123, 456, 789, 101112]  # 5种子以获得可靠统计
    },

    'stochastic_challenge': {
        'algorithms': ['q_learning', 'sarsa', 'expected_sarsa', 'double_q_learning',
                      'dyna_q', 'dqn', 'double_dqn', 'per_dqn'],
        'maps': ['4x4_default', '6x6_hard_mixed', '8x8_official'],
        'slippery_modes': [True],  # 只测试随机性环境
        'seeds': [42, 123, 456]
    },

    'planning_vs_learning': {
        'algorithms': ['value_iteration', 'policy_iteration', 'astar',
                      'q_learning', 'dqn'],
        'maps': ['4x4_default', '8x8_official'],
        'slippery_modes': [False, True],
        'seeds': [42, 123]
    }
}
