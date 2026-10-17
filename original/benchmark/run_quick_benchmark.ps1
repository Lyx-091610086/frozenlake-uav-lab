# run_quick_benchmark.ps1
# 快速基准测试 + 可视化生成（用于测试）

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  快速基准测试 + 可视化生成" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否已经在frozenlake环境中
$inFrozenlakeEnv = $false
if ($env:CONDA_DEFAULT_ENV -eq "frozenlake") {
    Write-Host "已经在frozenlake conda环境中" -ForegroundColor Green
    $inFrozenlakeEnv = $true
} elseif ($env:CONDA_PREFIX -and $env:CONDA_PREFIX.Contains("frozenlake")) {
    Write-Host "已经在frozenlake conda环境中 (通过CONDA_PREFIX检测)" -ForegroundColor Green
    $inFrozenlakeEnv = $true
} else {
    # 检查PATH长度并尝试缓解
    $pathLength = $env:PATH.Length
    if ($pathLength -gt 2000) {
        Write-Host "警告: PATH 环境变量过长 ($pathLength 个字符)，可能导致问题" -ForegroundColor Yellow
    }

    Write-Host "不在frozenlake环境中，尝试激活..." -ForegroundColor Yellow

    try {
        conda activate frozenlake
        Write-Host "Conda 环境激活成功" -ForegroundColor Green
        $inFrozenlakeEnv = $true
    } catch {
        Write-Host "Conda 激活失败，尝试替代方法..." -ForegroundColor Yellow

        # 方法1: 尝试直接激活
        try {
            & conda activate frozenlake
            Write-Host "Conda 环境激活成功 (方法1)" -ForegroundColor Green
            $inFrozenlakeEnv = $true
        } catch {
            # 方法2: 使用conda完整路径
            $condaPaths = @(
                "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
                "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
                "C:\ProgramData\miniconda3\Scripts\conda.exe",
                "C:\ProgramData\anaconda3\Scripts\conda.exe"
            )

            $activated = $false
            foreach ($condaPath in $condaPaths) {
                if (Test-Path $condaPath) {
                    try {
                        & $condaPath activate frozenlake
                        Write-Host "Conda 环境激活成功 (方法2: $condaPath)" -ForegroundColor Green
                        $activated = $true
                        $inFrozenlakeEnv = $true
                        break
                    } catch {
                        continue
                    }
                }
            }

            if (-not $activated) {
                Write-Host "无法激活conda环境，将在没有激活的情况下继续" -ForegroundColor Red
                Write-Host "请确保Python和所需包可用" -ForegroundColor Yellow
            }
        }
    }
}

# 验证Python可用性
try {
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Python可用: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "警告: 当前环境可能没有Python可用" -ForegroundColor Red
    Write-Host "请确保已激活正确的conda环境" -ForegroundColor Yellow
}

# 显示配置信息
Write-Host "📊 快速测试配置:" -ForegroundColor Yellow
Write-Host "  - 算法: Q-Learning, SARSA, Value Iteration" -ForegroundColor White
Write-Host "  - 地图: 4x4_default, 5x5_medium_cross" -ForegroundColor White
Write-Host "  - 环境模式: 确定性 + 随机性" -ForegroundColor White
Write-Host "  - 随机种子: 1个" -ForegroundColor White
Write-Host "  - 总实验数: ~12个" -ForegroundColor White
Write-Host "  - 预计时间: ~10-30分钟" -ForegroundColor White
Write-Host ""

# 记录开始时间
$startTime = Get-Date

# 运行快速测试
Write-Host "🚀 开始运行快速基准测试..." -ForegroundColor Green
Write-Host ""

# 运行自定义实验
$algorithms = @("q_learning", "sarsa", "value_iteration")
$maps = @("4x4_default", "5x5_medium_cross")

.\run_experiments.ps1 -Command custom -Algorithms $algorithms -Maps $maps -Slippery -Deterministic -SaveDetailed

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ 实验运行失败" -ForegroundColor Red
    exit 1
}

# 计算运行时间
$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host ""
Write-Host "✅ 实验完成！耗时: $($duration.Minutes)分钟" -ForegroundColor Green
Write-Host ""

# 生成可视化
Write-Host "🎨 生成所有可视化..." -ForegroundColor Cyan

# Find Python for visualization
$pythonCmd = $null
$pythonCommands = @("python", "python3")
foreach ($cmd in $pythonCommands) {
    try {
        $testResult = & $cmd --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonCmd) {
    $pythonPaths = @(
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:USERPROFILE\anaconda3\python.exe",
        "C:\ProgramData\miniconda3\python.exe",
        "C:\ProgramData\anaconda3\python.exe",
        "$env:CONDA_PREFIX\python.exe"
    )

    foreach ($pythonPath in $pythonPaths) {
        if (Test-Path $pythonPath) {
            try {
                $testResult = & $pythonPath --version 2>$null
                if ($LASTEXITCODE -eq 0) {
                    $pythonCmd = $pythonPath
                    break
                }
            } catch {
                continue
            }
        }
    }
}

if ($pythonCmd) {
    & $pythonCmd generate_visualizations.py --type all
} else {
    Write-Host "❌ 找不到Python，无法生成可视化" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ✅ 快速基准测试完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📁 结果位置: results\simple_plots\" -ForegroundColor Yellow
