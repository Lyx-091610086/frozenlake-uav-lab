# FrozenLake强化学习实验平台

一个全面的FrozenLake强化学习实验平台，支持多种算法、地图和环境配置。

## 📁 项目结构

```
code/
├── README.md                    # 项目说明文档
├── README_BENCHMARK.md         # 完整基准测试指南
├── CONFIGURATION.md            # 实验配置详细说明
├── requirements.txt            # 依赖包列表
├── run_experiments.ps1         # 主实验脚本
├── run_full_benchmark.ps1      # 完整基准测试一键脚本
├── run_quick_benchmark.ps1     # 快速基准测试脚本
├── generate_visualizations.py  # 可视化生成入口脚本
├── algorithms/                 # 算法实现
│   ├── planning/              # 规划算法
│   ├── tabular/               # 表格型RL算法
│   ├── model_based/           # 模型-based算法
│   └── deep_value/            # 深度学习算法
├── environments/              # 环境管理
├── experiments/               # 实验管理
├── visualization/              # 可视化工具包
│   ├── generate_visualizations.py  # 可视化生成主脚本
│   ├── static_plots.py        # 静态图表生成
│   ├── animation.py           # 动画生成
│   ├── experiment_viz.py     # 实验可视化
│   └── training_progress.py   # 训练进度分析
└── results/                   # 实验结果
```

## 🚀 快速开始

### 环境配置
```bash
# 安装依赖
pip install -r requirements.txt
```

### 运行完整基准测试（推荐）


#### 方式1: 标准运行（如果PATH正常）
```powershell
# 激活环境并运行完整基准测试 + 自动生成所有可视化
conda activate frozenlake
.\run_full_benchmark.ps1
```

**包含内容**:
- ✅ **16种算法** × **5张地图** × **2种模式** × **5个种子** = **800个实验**
  - 算法: 规划(3) + 表格(7) + 模型-based(2) + 深度学习(4)
  - 地图: 4x4_default, 5x5_medium_cross, 6x6_hard_mixed, 8x8_official, 10x10_hard_mixed
  - **模式**: ✅ 确定性 (Deterministic) + ✅ 随机性 (Stochastic)
  - 种子: 5个 (42, 123, 456, 789, 101112)
- ✅ 自动保存详细结果（训练历史等，用于动画生成）
- ✅ 自动生成整合的静态图表（所有算法结果整合在一起，按地图+模式组织）
- ✅ 自动生成所有动画（策略执行 + 学习进度，按算法命名）
- ✅ 自动生成全局汇总和对比图表

**预计时间**: 10-20小时（取决于硬件配置）

#### 方式2: 快速测试（用于验证）
```powershell
# 运行快速基准测试（约10-30分钟）
.\run_quick_benchmark.ps1
```

#### 方式3: 手动运行
```powershell
# 完整基准测试（保存详细结果）
.\run_experiments.ps1 -Command full-benchmark -SaveDetailed

# 生成所有可视化
python generate_visualizations.py --type all
```

> 📖 **详细指南**: 
> - [README_BENCHMARK.md](README_BENCHMARK.md) - 完整基准测试的详细说明和分阶段运行方法
> - [CONFIGURATION.md](CONFIGURATION.md) - 实验配置详细说明（包含模式确认）

## 📋 支持的算法

### 规划算法 (Planning)
- **值迭代 (Value Iteration)**
- **策略迭代 (Policy Iteration)**
- **A* 搜索 (A* Search)**

### 表格算法 (Tabular RL)
- **Q-Learning**
- **SARSA**
- **期望SARSA (Expected SARSA)**
- **双Q学习 (Double Q-Learning)**
- **蒙特卡洛控制 (Monte Carlo Control)**
- **n步TD学习 (n-step TD)**
- **资格迹 (Eligibility Traces)**

### 模型-based算法 (Model-based)
- **Dyna-Q**
- **优先级扫描 (Prioritized Sweeping)**

### 深度学习算法 (Deep RL)
- **DQN**
- **双DQN (Double DQN)**
- **决斗DQN (Dueling DQN)**
- **优先经验回放DQN (PER-DQN)**

## 🗺️ 支持的地图

平台包含24张精心设计的FrozenLake地图：

### 标准地图
- `4x4_default`: 经典4x4地图
- `8x8_official`: 官方8x8地图

### 难度分级
- **简单 (Easy)**: 5x5_easy_*, 6x6_easy_*
- **中等 (Medium)**: 4x4_default, 6x6_medium_*
- **困难 (Hard)**: 8x8_hard_*, 10x10_hard_*

## 🎯 实验配置

### 环境模式
- **确定性 (Deterministic)**: `is_slippery=False`
- **随机性 (Stochastic)**: `is_slippery=True`

### 预设配置
- `quick_test`: 快速验证 (2算法 × 2地图 × 2模式)
- `planning_baseline`: 规划算法基准测试
- `tabular_comparison`: 表格算法对比
- `deep_comparison`: 深度学习算法对比
- `full_benchmark`: 完整基准测试 (16算法 × 5地图 × 2模式 × 5种子)

## 📊 可视化结果分析

运行实验后，使用统一的可视化系统生成静态图表和动画：

```bash
# 生成所有可视化（静态图表 + 动画）
python generate_visualizations.py

# 只生成静态图表
python generate_visualizations.py --type static

# 生成策略执行动画
python generate_visualizations.py --type policy

# 生成学习进度动画
python generate_visualizations.py --type learning
```

### 📈 实验结果组织结构

实验结果按照 **地图/模式/算法/种子** 的层级结构组织：

```
results/
├── experiment_summary.json               # 全局实验汇总
├── {map_name}/                           # 地图名称 (如: 4x4_default)
│   ├── deterministic/                    # 确定性环境
│   │   ├── {algorithm}/                  # 算法名称
│   │   │   ├── {seed}/                   # 随机种子
│   │   │   │   ├── results.json          # 完整实验结果
│   │   │   │   ├── policy.npy/png        # 策略文件和可视化
│   │   │   │   ├── value_function.npy/png # 值函数文件和可视化
│   │   │   │   ├── q_table.npy           # Q表 (表格算法)
│   │   │   │   ├── q_heatmap.png         # Q值热力图
│   │   │   │   ├── training_curve.png    # 训练曲线
│   │   │   │   └── training_history.json # 训练历史 (用于动画)
│   │   │   └── [其他种子]/
│   │   └── [其他算法]/
│   └── stochastic/                       # 随机性环境
│       └── [相同结构]
└── simple_plots/                         # 全局可视化汇总
    ├── global_summary.png                # 全局汇总图表
    └── experiments_comparison.gif         # 实验对比动画
```

**目录结构优势**：
- ✅ **清晰分层**：地图 → 模式 → 算法 → 种子
- ✅ **避免冲突**：不同种子和算法的文件相互隔离
- ✅ **易于查找**：按需定位特定实验的结果
- ✅ **可扩展**：支持任意数量的算法、地图、模式和种子

**每个实验目录包含**：
- **`results.json`** - 完整的实验配置和结果数据
- **`policy.npy/png`** - 学习的策略和可视化
- **`value_function.npy/png`** - 值函数和可视化
- **`q_table.npy`** - Q-learning等算法的Q表
- **`q_heatmap.png`** - Q值的热力图可视化
- **`training_curve.png`** - 训练过程中的性能曲线
- **`training_history.json`** - 详细的训练历史数据（用于生成动画）

**全局汇总文件**（根目录）：
- **`global_summary.png`** - 所有实验的全局对比图表
- **`experiments_comparison.gif`** - 实验对比动画

### 🎮 动画功能

#### 策略执行动画
- 显示智能体在FrozenLake环境中的完整路径
- 实时显示当前步骤和状态
- 支持起点(S)、终点(G)、洞(H)的视觉标识

#### 学习进度动画
- Q值统计趋势随时间变化
- 成功率趋势动态展示
- 最终学习策略可视化
- 学习曲线动态展示

#### 实验对比动画
- 多算法性能对比
- 训练效率动态展示
- 实验进度实时更新

### 📊 训练日志

训练过程中显示详细的中间指标：

```
Episode 1000: ε=0.010, 成功率=1.000, 平均步数=6.0, Q均值=0.314, 策略确定性=0.625
```

**📊 指标含义**:
- **ε**: 探索率 - 控制随机探索的程度
- **成功率**: 最近100回合的成功率 - 学习效果指标
- **平均步数**: 成功路径的平均步数 - 最优性指标
- **Q均值**: Q值表的平均值 - 学习进度指标
- **策略确定性**: 贪心策略的确定性 (0-1) - 收敛程度指标

### 🎨 可视化特性

- ✅ **中英双语支持**: 完美显示中文标签
- ✅ **学术级质量**: 高分辨率专业图表
- ✅ **动画演示**: 直观展示学习过程
- ✅ **交互友好**: 命令行参数控制生成类型
- ✅ **自动保存**: 智能文件管理和命名
- ✅ **层级组织**: 按地图/算法/模式清晰分类

## 💡 使用建议

### 完整基准测试流程

1. **准备阶段**
   ```powershell
   # 确保已安装所有依赖
   pip install -r requirements.txt
   ```

2. **运行基准测试**
   ```powershell
   # 一键运行（推荐）
   .\run_full_benchmark.ps1
   ```

3. **查看结果**
   - 实验结果: `results/experiment_summary.json`
   - 可视化文件: `results/simple_plots/`
   - 按地图/算法/模式浏览可视化结果

### 分阶段运行（如果时间有限）

```powershell
# 阶段1: 规划算法
.\run_experiments.ps1 -Command planning-baseline -SaveDetailed

# 阶段2: 表格算法
.\run_experiments.ps1 -Command tabular-comparison -SaveDetailed

# 阶段3: 深度学习算法
.\run_experiments.ps1 -Command deep-comparison -SaveDetailed

# 阶段4: 生成可视化
python generate_visualizations.py --type all
```

## 📚 更多信息

- **完整基准测试指南**: [README_BENCHMARK.md](README_BENCHMARK.md)
- **实验配置详细说明**: [CONFIGURATION.md](CONFIGURATION.md) - 包含full_benchmark模式确认
- **实验配置代码**: `experiments/configs.py`
- **算法实现**: `algorithms/` 目录
- **地图定义**: `environments/maps_icegrid.py`

## ✅ 配置确认

**full_benchmark 包含两种环境模式**:
- ✅ **确定性** (Deterministic): `is_slippery=False`
- ✅ **随机性** (Stochastic): `is_slippery=True`

配置位置: `experiments/configs.py` 第266行  
验证方法: 查看 [CONFIGURATION.md](CONFIGURATION.md)