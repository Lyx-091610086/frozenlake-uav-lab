# 实验配置说明

## 📋 full_benchmark 配置详情

### ✅ 确认：包含两种环境模式

`full_benchmark` 预设配置**确实包含**确定性（Deterministic）和随机性（Stochastic）两种模式：

```python
'full_benchmark': {
    'algorithms': [...],  # 16种算法
    'maps': [...],        # 5张地图
    'slippery_modes': [False, True],  # ✅ 两种模式
    'seeds': [42, 123, 456, 789, 101112]  # 5个种子
}
```

### 配置验证

- **确定性模式**: `is_slippery=False` ✅
- **随机性模式**: `is_slippery=True` ✅
- **实验总数**: 16 × 5 × 2 × 5 = **800个实验** ✅

### 实验执行逻辑

在 `experiment_runner.py` 的 `run_batch_experiments` 方法中：

```python
for algorithm in algorithms:
    for map_name in maps:
        for is_slippery in slippery_modes:  # 遍历 [False, True]
            for seed in seeds:
                # 运行实验
```

这意味着每个算法-地图组合都会在两种模式下运行，确保完整的对比分析。

## 📊 其他预设配置

### planning_baseline
- **模式**: 仅确定性 (`slippery_modes: [False]`)
- **原因**: 规划算法通常用于确定性环境

### tabular_comparison
- **模式**: 两种模式 (`slippery_modes: [False, True]`) ✅

### deep_comparison
- **模式**: 仅随机性 (`slippery_modes: [True]`)
- **原因**: 深度学习算法更适合随机性环境

### stochastic_challenge
- **模式**: 仅随机性 (`slippery_modes: [True]`)
- **目的**: 专门测试随机性环境下的算法性能

## 🔍 如何验证配置

### 方法1: 查看配置文件
```bash
# 查看配置
python -c "from experiments.configs import EXPERIMENT_PRESETS; print(EXPERIMENT_PRESETS['full_benchmark'])"
```

### 方法2: 运行实验时查看输出
运行 `full_benchmark` 时，会显示：
```
Modes: Deterministic + Stochastic (both)
Total experiments: 800 (16 × 5 × 2 × 5)
```

### 方法3: 检查实验结果
运行后检查 `results/experiment_summary.json`，应该看到：
- `"slippery": false` 的实验（确定性）
- `"slippery": true` 的实验（随机性）

## 📝 注意事项

1. **规划算法在随机性环境**: 
   - Value Iteration 和 Policy Iteration 在随机性环境下仍可运行
   - 但性能可能不如在确定性环境下

2. **深度学习算法在确定性环境**:
   - DQN系列算法在确定性环境下也可以运行
   - 但可能收敛较慢或性能不如表格算法

3. **实验时间**:
   - 确定性环境通常训练更快
   - 随机性环境需要更多探索，训练时间更长

## 🎯 推荐使用场景

- **完整对比**: 使用 `full_benchmark`（包含两种模式）
- **快速验证**: 使用 `quick_test`（包含两种模式）
- **规划算法**: 使用 `planning_baseline`（仅确定性）
- **深度学习**: 使用 `deep_comparison`（仅随机性）
