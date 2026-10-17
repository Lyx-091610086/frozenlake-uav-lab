# FrozenLake Reinforcement Learning Experiment Platform
# run_experiments.ps1

param(
    [string]$Command,
    [string[]]$Algorithms,
    [string[]]$Maps,
    [switch]$Slippery,
    [switch]$Deterministic,
    [string[]]$Seeds = @("42"),
    [string]$ResultsDir = "results",
    [switch]$SaveDetailed,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Check if already in frozenlake environment
$inFrozenlakeEnv = $false
if ($env:CONDA_DEFAULT_ENV -eq "frozenlake") {
    Write-Host "Already in frozenlake conda environment" -ForegroundColor Green
    $inFrozenlakeEnv = $true
} elseif ($env:CONDA_PREFIX -and $env:CONDA_PREFIX.Contains("frozenlake")) {
    Write-Host "Already in frozenlake conda environment (detected by CONDA_PREFIX)" -ForegroundColor Green
    $inFrozenlakeEnv = $true
} else {
    # Check PATH length and try to mitigate
    $pathLength = $env:PATH.Length
    if ($pathLength -gt 2000) {
        Write-Host "Warning: PATH is very long ($pathLength chars), this may cause issues" -ForegroundColor Yellow
    }

    Write-Host "Not in frozenlake environment, attempting activation..." -ForegroundColor Yellow

    try {
        conda activate frozenlake
        Write-Host "Conda environment activated successfully" -ForegroundColor Green
        $inFrozenlakeEnv = $true
    } catch {
        Write-Host "Conda activation failed, trying alternative methods..." -ForegroundColor Yellow

        # Method 1: Try direct activation
        try {
            & conda activate frozenlake
            Write-Host "Conda environment activated (method 1)" -ForegroundColor Green
            $inFrozenlakeEnv = $true
        } catch {
            # Method 2: Use full path to conda
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
                        Write-Host "Conda environment activated (method 2: $condaPath)" -ForegroundColor Green
                        $activated = $true
                        $inFrozenlakeEnv = $true
                        break
                    } catch {
                        continue
                    }
                }
            }

            if (-not $activated) {
                Write-Host "Could not activate conda environment, proceeding without it" -ForegroundColor Red
                Write-Host "Make sure Python and required packages are available" -ForegroundColor Yellow
            }
        }
    }
}

# Verify Python availability
try {
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Python is available: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "Warning: Python may not be available in current environment" -ForegroundColor Red
    Write-Host "Please ensure you have activated the correct conda environment" -ForegroundColor Yellow
}

Write-Host "=== FrozenLake Reinforcement Learning Platform ===" -ForegroundColor Green
Write-Host "Supporting 24 maps + 16 algorithms for comprehensive experiments" -ForegroundColor Cyan
Write-Host ""

# Show help information
function Show-Help {
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\run_experiments.ps1 -Command <command> [parameters...]" -ForegroundColor White
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Yellow
    Write-Host "  list-algorithms    Show all available algorithms" -ForegroundColor White
    Write-Host "  list-maps         Show all available maps" -ForegroundColor White
    Write-Host "  list-presets      Show all preset experiments" -ForegroundColor White
    Write-Host "  quick-test        Quick test (Q-Learning vs SARSA)" -ForegroundColor White
    Write-Host "  planning-baseline Planning algorithms baseline" -ForegroundColor White
    Write-Host "  tabular-comparison Tabular algorithms comparison" -ForegroundColor White
    Write-Host "  advanced-tabular  Advanced tabular algorithms" -ForegroundColor White
    Write-Host "  model-based       Model-based RL algorithms" -ForegroundColor White
    Write-Host "  deep-comparison   Deep learning algorithms comparison" -ForegroundColor White
    Write-Host "  scalability       Map scalability test" -ForegroundColor White
    Write-Host "  stochastic-challenge Stochastic environment challenge" -ForegroundColor White
    Write-Host "  planning-vs-learning Planning vs Learning comparison" -ForegroundColor White
    Write-Host "  full-benchmark    Full benchmark (16 algorithms)" -ForegroundColor White
    Write-Host "  custom            Custom experiment" -ForegroundColor White
    Write-Host ""
    Write-Host "Custom experiment parameters:" -ForegroundColor Yellow
    Write-Host "  -Algorithms <list>  Specify algorithms (comma-separated)" -ForegroundColor White
    Write-Host "  -Maps <list>       Specify maps (comma-separated)" -ForegroundColor White
    Write-Host "  -Slippery          Use slippery (stochastic) mode" -ForegroundColor White
    Write-Host "  -Deterministic     Use deterministic mode" -ForegroundColor White
    Write-Host "  -Seeds <list>      Specify random seeds (default: 42)" -ForegroundColor White
    Write-Host "  -SaveDetailed      Save detailed results for visualization" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run_experiments.ps1 -Command list-algorithms" -ForegroundColor White
    Write-Host "  .\run_experiments.ps1 -Command full-benchmark" -ForegroundColor White
    Write-Host "  .\run_experiments.ps1 -Command custom -Algorithms q_learning,sarsa -Maps 4x4_default,8x8_official -Slippery" -ForegroundColor White
    Write-Host "  .\run_experiments.ps1 -Command custom -Algorithms dqn,value_iteration -Maps 4x4_default -Deterministic" -ForegroundColor White
}

# Execute Python command helper function
function Run-PythonCommand {
    param([string]$Arguments)

    # Find Python executable
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

    # If not found via command, try specific paths
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

    if (-not $pythonCmd) {
        Write-Host "ERROR: Python executable not found!" -ForegroundColor Red
        Write-Host "Please ensure Python is installed and available in PATH" -ForegroundColor Red
        exit 1
    }

    try {
        $argArray = $Arguments -split ' '
        & $pythonCmd experiments/run_experiments.py $argArray
    } catch {
        Write-Host "Execution failed: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# Process commands
if ($Help) {
    Show-Help
    exit 0
}

switch ($Command.ToLower()) {
    "list-algorithms" {
        Write-Host "Listing all available algorithms..." -ForegroundColor Yellow
        Run-PythonCommand "--list-algorithms"
    }

    "list-maps" {
        Write-Host "Listing all available maps..." -ForegroundColor Yellow
        Run-PythonCommand "--list-maps"
    }

    "list-presets" {
        Write-Host "Available experiment presets:" -ForegroundColor Yellow
        Write-Host "  quick_test           - Quick test (Q-Learning vs SARSA)" -ForegroundColor White
        Write-Host "  planning_baseline    - Planning algorithms baseline" -ForegroundColor White
        Write-Host "  tabular_comparison   - Tabular algorithms comparison" -ForegroundColor White
        Write-Host "  advanced_tabular     - Advanced tabular algorithms" -ForegroundColor White
        Write-Host "  model_based          - Model-based RL algorithms" -ForegroundColor White
        Write-Host "  deep_value_comparison - Deep learning algorithms comparison" -ForegroundColor White
        Write-Host "  map_scalability      - Map scalability test" -ForegroundColor White
        Write-Host "  stochastic_challenge - Stochastic environment challenge" -ForegroundColor White
        Write-Host "  planning_vs_learning - Planning vs Learning comparison" -ForegroundColor White
        Write-Host "  full_benchmark       - Full benchmark (16 algorithms)" -ForegroundColor White
    }

    "quick-test" {
        Write-Host "Running quick test..." -ForegroundColor Yellow
        Write-Host "Algorithms: Q-Learning, SARSA" -ForegroundColor Cyan
        Write-Host "Maps: 4x4_default, 5x5_easy_open" -ForegroundColor Cyan
        Write-Host "Modes: Deterministic + Stochastic" -ForegroundColor Cyan
        Run-PythonCommand "--preset quick_test"
    }

    "planning-baseline" {
        Write-Host "Running planning algorithms baseline..." -ForegroundColor Yellow
        Write-Host "Algorithms: Value Iteration, Policy Iteration, A*" -ForegroundColor Cyan
        Write-Host "Maps: 4x4_default, 8x8_official" -ForegroundColor Cyan
        Write-Host "Mode: Deterministic" -ForegroundColor Cyan
        Run-PythonCommand "--preset planning_baseline"
    }

    "tabular-comparison" {
        Write-Host "Running tabular algorithms comparison..." -ForegroundColor Yellow
        Write-Host "Algorithms: Q-Learning, SARSA, Expected SARSA, Double Q-Learning" -ForegroundColor Cyan
        Write-Host "Maps: 4x4_default, 5x5_medium_cross, 6x6_hard_mixed" -ForegroundColor Cyan
        Write-Host "Modes: Deterministic + Stochastic" -ForegroundColor Cyan
        Run-PythonCommand "--preset tabular_comparison"
    }

    "advanced-tabular" {
        Write-Host "Running advanced tabular algorithms..." -ForegroundColor Yellow
        Write-Host "Algorithms: Monte Carlo Control, n-step TD, Eligibility Traces" -ForegroundColor Cyan
        Write-Host "Maps: 4x4_default, 5x5_medium_cross" -ForegroundColor Cyan
        Write-Host "Modes: Deterministic + Stochastic" -ForegroundColor Cyan
        Run-PythonCommand "--preset advanced_tabular"
    }

    "model-based" {
        Write-Host "Running model-based RL algorithms..." -ForegroundColor Yellow
        Write-Host "Algorithms: Dyna-Q, Prioritized Sweeping" -ForegroundColor Cyan
        Write-Host "Maps: 4x4_default, 8x8_official" -ForegroundColor Cyan
        Write-Host "Mode: Stochastic" -ForegroundColor Cyan
        Run-PythonCommand "--preset model_based"
    }

    "deep-comparison" {
        Write-Host "Running deep learning algorithms comparison..." -ForegroundColor Yellow
        Write-Host "Algorithms: DQN, Double DQN, Dueling DQN, PER-DQN" -ForegroundColor Cyan
        Write-Host "Maps: 4x4_default, 8x8_official" -ForegroundColor Cyan
        Write-Host "Mode: Stochastic" -ForegroundColor Cyan
        Run-PythonCommand "--preset deep_value_comparison"
    }

    "scalability" {
        Write-Host "Running map scalability test..." -ForegroundColor Yellow
        Write-Host "Algorithm: Q-Learning" -ForegroundColor Cyan
        Write-Host "Maps: 4x4 to 10x10" -ForegroundColor Cyan
        Write-Host "Mode: Stochastic" -ForegroundColor Cyan
        Run-PythonCommand "--preset map_scalability"
    }

    "stochastic-challenge" {
        Write-Host "Running stochastic environment challenge..." -ForegroundColor Yellow
        Write-Host "Algorithms: Multiple algorithms in stochastic environments" -ForegroundColor Cyan
        Write-Host "Maps: 4x4_default, 6x6_hard_mixed, 8x8_official" -ForegroundColor Cyan
        Write-Host "Mode: Stochastic" -ForegroundColor Cyan
        Run-PythonCommand "--preset stochastic_challenge"
    }

    "planning-vs-learning" {
        Write-Host "Running planning vs learning comparison..." -ForegroundColor Yellow
        Write-Host "Algorithms: Planning + Learning methods" -ForegroundColor Cyan
        Write-Host "Maps: 4x4_default, 8x8_official" -ForegroundColor Cyan
        Write-Host "Modes: Deterministic + Stochastic" -ForegroundColor Cyan
        Run-PythonCommand "--preset planning_vs_learning"
    }

    "full-benchmark" {
        Write-Host "Running full benchmark..." -ForegroundColor Yellow
        Write-Host "Algorithms: All 16 algorithms" -ForegroundColor Cyan
        Write-Host "Maps: 5 difficulty levels" -ForegroundColor Cyan
        Write-Host "Modes: Deterministic + Stochastic (both)" -ForegroundColor Cyan
        Write-Host "Seeds: 5 random seeds" -ForegroundColor Cyan
        Write-Host "Total experiments: 800 (16 × 5 × 2 × 5)" -ForegroundColor Cyan

        $args_list = @("--preset", "full_benchmark")
        if ($SaveDetailed) {
            $args_list += "--save-detailed"
            Write-Host "Save detailed results: Yes (for visualization)" -ForegroundColor Cyan
        } else {
            Write-Host "Save detailed results: No (summary only)" -ForegroundColor Cyan
        }

        # Run directly without additional confirmation (confirmation handled by caller)
        Run-PythonCommand ($args_list -join " ")
        Write-Host "WARNING: This will run many experiments and may take a long time!" -ForegroundColor Red
        $confirmation = Read-Host "Confirm running full benchmark? (y/N)"
        if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
            Run-PythonCommand "--preset full_benchmark"
        } else {
            Write-Host "Full benchmark cancelled" -ForegroundColor Yellow
        }
    }

    "custom" {
        Write-Host "Running custom experiment..." -ForegroundColor Yellow

        # Validate parameters
        if (-not $Algorithms) {
            Write-Host "Error: Must specify algorithms (-Algorithms parameter)" -ForegroundColor Red
            Write-Host "Example: -Algorithms q_learning,sarsa,dqn" -ForegroundColor Yellow
            exit 1
        }

        if (-not $Maps) {
            Write-Host "Error: Must specify maps (-Maps parameter)" -ForegroundColor Red
            Write-Host "Example: -Maps 4x4_default,8x8_official" -ForegroundColor Yellow
            exit 1
        }

        # Determine mode
        $mode = "stochastic"
        if ($Deterministic) { $mode = "deterministic" }

        Write-Host "Custom experiment configuration:" -ForegroundColor Cyan
        Write-Host "  Algorithms: $($Algorithms -join ', ')" -ForegroundColor White
        Write-Host "  Maps: $($Maps -join ', ')" -ForegroundColor White
        Write-Host "  Mode: $mode" -ForegroundColor White
        Write-Host "  Seeds: $($Seeds -join ', ')" -ForegroundColor White
        $saveMode = if ($SaveDetailed) { "Yes" } else { "No (summary only)" }
        Write-Host "  Save detailed results: $saveMode" -ForegroundColor White

        # Build command arguments
        $args_list = @("--algorithms") + $Algorithms + @("--maps") + $Maps

        if ($Slippery) {
            $args_list += "--slippery"
        } elseif ($Deterministic) {
            $args_list += "--deterministic"
        }

        if ($Seeds -and $Seeds[0] -ne "42") {
            $args_list += @("--seeds") + $Seeds
        }

        if ($SaveDetailed) {
            $args_list += "--save-detailed"
        }

        Run-PythonCommand ($args_list -join " ")
    }

    default {
        if (-not $Command) {
            Write-Host "Error: No command specified" -ForegroundColor Red
            Write-Host ""
            Show-Help
        } else {
            Write-Host "Error: Unknown command '$Command'" -ForegroundColor Red
            Write-Host ""
            Show-Help
        }
        exit 1
    }
}

Write-Host ""
Write-Host "=== Experiment completed! Results saved to results/ directory ===" -ForegroundColor Green
Write-Host "Use the following command for more options:" -ForegroundColor Yellow
Write-Host "  .\run_experiments.ps1 -Help" -ForegroundColor White
