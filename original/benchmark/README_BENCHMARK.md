# 完整基准测试指南

## ⚠️ 重要提示：PowerShell PATH 环境变量问题

如果遇到 **"环境变量名或值太长"** 错误，这是Windows PowerShell的常见问题。解决方案：

### 推荐解决方案：使用简化脚本
```powershell
# 快速测试
.\run_benchmark_simple.ps1 -Quick

# 完整benchmark
.\run_benchmark_simple.ps1 -Full
```

### 辅助工具：PATH修复
```powershell
# 分析并修复PATH环境变量
.\fix_path.ps1
```

### 传统方法（如果PATH正常）
```powershell
conda activate frozenlake
.\run_full_benchmark.ps1
```

## 🎯 快速开始

### 方式1: 一键运行完整基准测试（推荐）

```powershell
# 运行完整基准测试 + 自动生成所有可视化
.\run_full_benchmark.ps1
```

**包含内容**:
- ✅ 16种算法 × 5张地图 × 2种模式 × 5个种子 = **800个实验**
- ✅ 自动保存详细结果（训练历史等，用于动画生成）
- ✅ 自动生成整合的静态图表（所有算法结果整合在一起）
- ✅ 自动生成所有动画（策略执行 + 学习进度，按算法命名）
- ✅ 自动生成全局汇总和对比图表

**预计时间**: 10-20小时（取决于硬件配置）

### 方式2: 快速测试（用于验证）

```powershell
# 运行快速基准测试（约10-30分钟）
.\run_quick_benchmark.ps1
```

**包含内容**:
- ✅ 3种算法 × 2张地图 × 2种模式 = **12个实验**
- ✅ 自动生成所有可视化

### 方式3: 分阶段运行

如果不想一次性运行800个实验，可以分阶段进行：

```powershell
# 阶段1: 规划算法基准测试
.\run_experiments.ps1 -Command planning-baseline -SaveDetailed

# 阶段2: 表格算法对比
.\run_experiments.ps1 -Command tabular-comparison -SaveDetailed

# 阶段3: 深度学习算法对比
.\run_experiments.ps1 -Command deep-comparison -SaveDetailed

# 阶段4: 生成所有可视化
python generate_visualizations.py --type all
```

## 📊 完整基准测试配置

### 算法列表（16种）

**规划算法** (3种):
- Value Iteration
- Policy Iteration
- A* Search

**表格型RL算法** (7种):
- Q-Learning
- SARSA
- Expected SARSA
- Double Q-Learning
- Monte Carlo Control
- n-step TD
- Eligibility Traces

**模型-based算法** (2种):
- Dyna-Q
- Prioritized Sweeping

**深度学习算法** (4种):
- DQN
- Double DQN
- Dueling DQN
- PER-DQN

### 地图列表（5张）

- `4x4_default` - 标准4x4地图
- `5x5_medium_cross` - 中等难度5x5地图
- `6x6_hard_mixed` - 困难6x6地图
- `8x8_official` - 官方8x8地图
- `10x10_hard_mixed` - 困难10x10地图

### 环境模式（2种）

**✅ full_benchmark包含两种模式**:
- **确定性** (Deterministic): `is_slippery=False`
- **随机性** (Stochastic): `is_slippery=True`

**配置确认**: 
```python
'slippery_modes': [False, True]  # ✅ 两种模式都包含
```

**验证方法**:
- 查看 `experiments/configs.py` 第266行
- 运行实验时会显示 "Modes: Deterministic + Stochastic (both)"
- 实验结果中会同时包含 `slippery: false` 和 `slippery: true` 的实验

### 随机种子（5个）

- 种子列表: `[42, 123, 456, 789, 101112]`
- 用于获得可靠的统计结果

### 实验总数计算

- 算法: 16种
- 地图: 5张
- 模式: 2种（确定性 + 随机性）
- 种子: 5个
- **总计**: 16 × 5 × 2 × 5 = **800个实验** ✅

## 📈 结果组织

### 目录结构

```
results/
├── experiment_summary.json          # 全局实验汇总
├── {map_name}/                      # 地图名称目录
│   ├── deterministic/               # 确定性环境目录
│   │   ├── {algorithm}/             # 算法名称目录
│   │   │   ├── {seed}/              # 随机种子目录
│   │   │   │   ├── results.json     # 完整实验结果
│   │   │   │   ├── policy.npy/png   # 策略文件和可视化
│   │   │   │   ├── value_function.npy/png # 值函数
│   │   │   │   ├── q_table.npy      # Q表 (表格算法)
│   │   │   │   ├── q_heatmap.png    # Q值热力图
│   │   │   │   ├── training_curve.png # 训练曲线
│   │   │   │   └── training_history.json # 训练历史
│   │   │   └── [其他种子]/
│   │   └── [其他算法]/
│   └── stochastic/                  # 随机性环境目录
│       └── [相同结构]
└── simple_plots/                    # 全局可视化汇总
    ├── global_summary.png
    └── experiments_comparison.gif
```

**目录结构优势**：
- **层级清晰**：地图 → 模式 → 算法 → 种子
- **文件隔离**：每个实验都有独立的目录，避免文件冲突
- **便于分析**：可以轻松比较同一算法在不同种子下的结果
- **扩展性好**：支持任意数量的算法、地图、模式和种子组合

## ⚙️ 高级选项

### 自定义基准测试

```powershell
# 自定义算法和地图组合
.\run_experiments.ps1 -Command custom `
    -Algorithms q_learning,sarsa,dqn `
    -Maps 4x4_default,8x8_official `
    -Slippery -Deterministic `
    -Seeds 42,123,456 `
    -SaveDetailed
```

### 只生成可视化（不运行实验）

```powershell
# 如果已有实验结果，只生成可视化
python generate_visualizations.py --type all
```

### 选择性生成可视化

```powershell
# 只生成静态图表
python generate_visualizations.py --type static

# 只生成策略动画
python generate_visualizations.py --type policy

# 只生成学习进度动画
python generate_visualizations.py --type learning

# 为特定实验生成动画
python generate_visualizations.py --type policy --experiment-id q_learning_4x4_default_deterministic_42
```

## 💡 性能优化建议

### 1. 并行运行（如果支持）

可以修改脚本以支持并行运行多个实验（需要额外实现）。

### 2. 分批次运行

将800个实验分成多个批次，分别运行：

```powershell
# 批次1: 规划算法
.\run_experiments.ps1 -Command planning-baseline -SaveDetailed

# 批次2: 表格算法（部分）
.\run_experiments.ps1 -Command custom -Algorithms q_learning,sarsa -Maps 4x4_default,5x5_medium_cross -Slippery -Deterministic -SaveDetailed

# ... 继续其他批次
```

### 3. 跳过深度学习算法（如果不需要）

如果不需要深度学习算法，可以修改预设配置，只运行表格算法：

```powershell
.\run_experiments.ps1 -Command tabular-comparison -SaveDetailed
```

## 📝 注意事项

1. **磁盘空间**: 800个实验的详细结果可能需要几GB空间
2. **运行时间**: 完整基准测试需要很长时间，建议在服务器或高性能机器上运行
3. **内存使用**: 深度学习算法需要较多内存
4. **中断恢复**: 如果中断，可以继续运行，已完成的实验不会重复

## 🎉 完成后的检查清单

- [ ] 检查 `results/experiment_summary.json` 是否包含所有实验
- [ ] 检查 `results/simple_plots/` 目录结构是否正确
- [ ] 检查每个实验组是否都有完整的可视化文件
- [ ] 检查全局汇总图表是否生成
- [ ] 检查动画文件是否正常播放

## 📞 问题排查

如果遇到问题：

1. **实验运行失败**: 检查 `results/` 目录中的错误日志
2. **可视化生成失败**: 确保已安装所有依赖 (`pip install -r requirements.txt`)
3. **动画生成失败**: 确保有足够的磁盘空间和内存
4. **中文显示问题**: 确保系统安装了中文字体

## 🚀 开始运行

准备好后，运行：

```powershell
.\run_full_benchmark.ps1
```

祝实验顺利！🎊
