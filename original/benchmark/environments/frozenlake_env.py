import warnings

# 抑制gymnasium相关的警告
warnings.filterwarnings("ignore", message=".*gymnasium.*")
warnings.filterwarnings("ignore", message=".*numpy.*")
warnings.filterwarnings("ignore", message=".*bool8.*")
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
        print("Warning: Using legacy gym instead of gymnasium")
    except ImportError:
        raise ImportError("Neither gymnasium nor gym is available. Please install gymnasium: pip install gymnasium")
from typing import Optional, Tuple, Dict, Any
from .maps_icegrid import MAPS, MAP_METADATA, create_frozenlake_env

class FrozenLakeEnv:
    """统一的FrozenLake环境包装器，支持自定义地图"""

    def __init__(self, map_name: str = "4x4_default", is_slippery: bool = True, render_mode: Optional[str] = None):
        """
        初始化FrozenLake环境

        Args:
            map_name: 地图名称，从maps_icegrid.py中选择
            is_slippery: 是否启用随机打滑
            render_mode: 渲染模式
        """
        self.map_name = map_name
        self.is_slippery = is_slippery
        self.render_mode = render_mode

        # 创建环境
        self.env = create_frozenlake_env(map_name, is_slippery)

        # 获取地图信息
        self.desc = self.env.unwrapped.desc
        self.nrow, self.ncol = self.env.unwrapped.nrow, self.env.unwrapped.ncol
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict]:
        """重置环境"""
        return self.env.reset(seed=seed)

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """执行动作"""
        return self.env.step(action)

    def render(self):
        """渲染环境"""
        return self.env.render()

    def close(self):
        """关闭环境"""
        self.env.close()

    def get_map_info(self) -> Dict[str, Any]:
        """获取地图信息"""
        return {
            "name": self.map_name,
            "size": (self.nrow, self.ncol),
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "is_slippery": self.is_slippery,
            **MAP_METADATA.get(self.map_name, {})
        }

    def state_to_coord(self, state: int) -> Tuple[int, int]:
        """状态转换为坐标"""
        return divmod(state, self.ncol)

    def coord_to_state(self, row: int, col: int) -> int:
        """坐标转换为状态"""
        return row * self.ncol + col

    @staticmethod
    def list_available_maps():
        """列出所有可用地图"""
        return list(MAPS.keys())

    @staticmethod
    def get_maps_by_size(size: Tuple[int, int]):
        """根据尺寸获取地图列表"""
        from .maps_icegrid import get_map_names_by_size
        return get_map_names_by_size(size)

    @staticmethod
    def get_maps_by_difficulty(difficulty: str):
        """根据难度获取地图列表"""
        from .maps_icegrid import get_map_names_by_difficulty
        return get_map_names_by_difficulty(difficulty)
