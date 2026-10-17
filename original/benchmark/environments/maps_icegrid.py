# maps_icegrid.py
# 24张FrozenLake地图集合，支持不同尺寸和难度
# 所有地图都保证至少存在一条可行路径

MAPS = {
    # ---------- 4x4 (5张) ----------

    "4x4_default": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ],

    "4x4_easy_open": [
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG",
    ],

    "4x4_bridge": [
        "SFFF",
        "HHFH",
        "FFFH",
        "HFFG",
    ],

    "4x4_maze": [
        "SFFH",
        "HFHF",
        "FFFH",
        "HFFG",
    ],

    "4x4_right_edge": [
        "SFFF",
        "FHHF",
        "FFFF",
        "FHHG",
    ],

    # ---------- 5x5 (5张) ----------

    "5x5_easy_open": [
        "SFFFF",
        "FFFFF",
        "FFFFF",
        "FFFFF",
        "FFFFG",
    ],

    "5x5_medium_cross": [
        "SFFFF",
        "FHFHF",
        "FFHFF",
        "FHFHF",
        "FFFFG",
    ],

    "5x5_snake_corridor": [
        "SFFFF",
        "HHHHF",
        "FFFFF",
        "FHHHH",
        "FFFFG",
    ],

    "5x5_right_edge": [
        "SFFFF",
        "HHHHF",
        "FFFHF",
        "FHHHF",
        "FFFFG",
    ],

    "5x5_ring": [
        "SFFFF",
        "FHHHF",
        "FHFHF",
        "FHHHF",
        "FFFFG",
    ],

    # ---------- 6x6 (5张) ----------

    "6x6_easy_open": [
        "SFFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFFG",
    ],

    "6x6_right_edge": [
        "SFFFFF",
        "FHHHHF",
        "FFFFHF",
        "FHHHHF",
        "FFFFHF",
        "FFFFFG",
    ],

    "6x6_snake_corridor": [
        "SFFFFF",
        "HHHHHF",
        "FFFFFF",
        "FHHHHH",
        "FFFFFF",
        "FFFFFG",
    ],

    "6x6_central_bridge": [
        "SFFFFF",
        "HHFHHH",
        "FFFHFF",
        "HHFHHH",
        "FFFFHF",
        "FFFFFG",
    ],

    "6x6_hard_mixed": [
        "SFFHFF",
        "FHFHFF",
        "FFHFFF",
        "FHFHHF",
        "FFFHFF",
        "FFFFFG",
    ],

    # ---------- 8x8 (5张) ----------

    # 官方 8x8（用于对照）
    "8x8_official": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],

    "8x8_open_sparse": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFHFFF",
        "FFFFFFFF",
        "FFHFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFFFG",
    ],

    "8x8_snake_corridor": [
        "SFFFFFFF",
        "HHHHHHHF",
        "FFFFFFFF",
        "FHHHHHHH",
        "FFFFFFFF",
        "HHHHHHHF",
        "FFFFFFFF",
        "FFFFFFFG",
    ],

    "8x8_left_edge_safe": [
        "SFFFFFFF",
        "FHHHHHHF",
        "FHFHFFFF",
        "FHFHHHHF",
        "FHFHFFFF",
        "FHFHHHHF",
        "FFFFFFHF",
        "FFFFFFFG",
    ],

    "8x8_risky_shortcut": [
        "SFFFFFHF",
        "FHHHHFHF",
        "FFFFHFHF",
        "FHFHFFFF",
        "FHFHHHHF",
        "FFFFFHHF",
        "FHHFFFHF",
        "FFFFFFFG",
    ],

    # ---------- 10x10 (4张) ----------

    "10x10_easy_open": [
        "SFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFHFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFG",
    ],

    "10x10_snake_corridor": [
        "SFFFFFFFFF",
        "HHHHHHHHHF",
        "FFFFFFFFFF",
        "FHHHHHHHHH",
        "FFFFFFFFFF",
        "HHHHHHHHHF",
        "FFFFFFFFFF",
        "FHHHHHHHHH",
        "FFFFFFFFFF",
        "FFFFFFFFFG",
    ],

    "10x10_checker_safe_bottom": [
        "SFFFFFFFFF",
        "FHFHFHFHFF",
        "FFFFFFFFFF",
        "FFHFHFHFHF",
        "FFFFFFFFFF",
        "FHFHFHFHFF",
        "FFFFFFFFFF",
        "FFHFHFHFHF",
        "FFFFFFFFFF",
        "FFFFFFFFFG",
    ],

    "10x10_hard_mixed": [
        "SFFHFFFFHF",
        "FHFHHFHFHF",
        "FFFFHFFFFF",
        "HFHFHFHHFF",
        "FFFFFHFFFF",
        "FHHFHFHFHF",
        "FFFFHFFFFF",
        "FFHFHHFHFH",
        "FFFFFHFFFF",
        "FFFFFFFFFG",
    ],
}

# 地图元数据
MAP_METADATA = {
    "4x4_default": {"size": (4, 4), "difficulty": "medium", "description": "标准4x4地图"},
    "4x4_easy_open": {"size": (4, 4), "difficulty": "easy", "description": "简单开放4x4地图"},
    "4x4_bridge": {"size": (4, 4), "difficulty": "hard", "description": "桥梁挑战4x4地图"},
    "4x4_maze": {"size": (4, 4), "difficulty": "hard", "description": "迷宫4x4地图"},
    "4x4_right_edge": {"size": (4, 4), "difficulty": "medium", "description": "右侧边缘4x4地图"},

    "5x5_easy_open": {"size": (5, 5), "difficulty": "easy", "description": "简单开放5x5地图"},
    "5x5_medium_cross": {"size": (5, 5), "difficulty": "medium", "description": "十字中级5x5地图"},
    "5x5_snake_corridor": {"size": (5, 5), "difficulty": "hard", "description": "蛇形走廊5x5地图"},
    "5x5_right_edge": {"size": (5, 5), "difficulty": "medium", "description": "右侧边缘5x5地图"},
    "5x5_ring": {"size": (5, 5), "difficulty": "hard", "description": "环形5x5地图"},

    "6x6_easy_open": {"size": (6, 6), "difficulty": "easy", "description": "简单开放6x6地图"},
    "6x6_right_edge": {"size": (6, 6), "difficulty": "medium", "description": "右侧边缘6x6地图"},
    "6x6_snake_corridor": {"size": (6, 6), "difficulty": "hard", "description": "蛇形走廊6x6地图"},
    "6x6_central_bridge": {"size": (6, 6), "difficulty": "hard", "description": "中央桥梁6x6地图"},
    "6x6_hard_mixed": {"size": (6, 6), "difficulty": "hard", "description": "混合困难6x6地图"},

    "8x8_official": {"size": (8, 8), "difficulty": "medium", "description": "官方8x8地图"},
    "8x8_open_sparse": {"size": (8, 8), "difficulty": "medium", "description": "开放稀疏8x8地图"},
    "8x8_snake_corridor": {"size": (8, 8), "difficulty": "hard", "description": "蛇形走廊8x8地图"},
    "8x8_left_edge_safe": {"size": (8, 8), "difficulty": "medium", "description": "左侧安全边缘8x8地图"},
    "8x8_risky_shortcut": {"size": (8, 8), "difficulty": "hard", "description": "风险捷径8x8地图"},

    "10x10_easy_open": {"size": (10, 10), "difficulty": "easy", "description": "简单开放10x10地图"},
    "10x10_snake_corridor": {"size": (10, 10), "difficulty": "hard", "description": "蛇形走廊10x10地图"},
    "10x10_checker_safe_bottom": {"size": (10, 10), "difficulty": "medium", "description": "棋盘安全底部10x10地图"},
    "10x10_hard_mixed": {"size": (10, 10), "difficulty": "hard", "description": "混合困难10x10地图"},
}

def get_map_names_by_size(size):
    """根据尺寸获取地图名称列表"""
    return [name for name, meta in MAP_METADATA.items() if meta["size"] == size]

def get_map_names_by_difficulty(difficulty):
    """根据难度获取地图名称列表"""
    return [name for name, meta in MAP_METADATA.items() if meta["difficulty"] == difficulty]

def create_frozenlake_env(map_name, is_slippery=True):
    """创建指定地图的FrozenLake环境"""
    import gymnasium as gym
    import warnings

    # 抑制gym警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if map_name not in MAPS:
            raise ValueError(f"Unknown map: {map_name}")

        desc = MAPS[map_name]
        env = gym.make('FrozenLake-v1', desc=desc, is_slippery=is_slippery, render_mode=None)
        return env
