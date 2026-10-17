# run_full_benchmark.ps1
# Complete benchmark test + visualization generation script

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Complete Benchmark Test + Visualization System" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

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
        Write-Host "Consider cleaning up PATH or using shorter paths" -ForegroundColor Yellow
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

# Display configuration information
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  - Algorithms: 16 types" -ForegroundColor White
Write-Host "  - Maps: 5 maps" -ForegroundColor White
Write-Host "  - Environment modes: Deterministic + Stochastic" -ForegroundColor White
Write-Host "    - Deterministic: is_slippery=False" -ForegroundColor Green
Write-Host "    - Stochastic: is_slippery=True" -ForegroundColor Green
Write-Host "  - Random seeds: 5 seeds (42, 123, 456, 789, 101112)" -ForegroundColor White
Write-Host "  - Total experiments: 16 x 5 x 2 x 5 = 800" -ForegroundColor White
Write-Host ""

# Estimated time
Write-Host "Estimated time:" -ForegroundColor Yellow
Write-Host "  - Planning algorithms: ~1-2 hours" -ForegroundColor White
Write-Host "  - Tabular algorithms: ~3-5 hours" -ForegroundColor White
Write-Host "  - Deep learning algorithms: ~5-10 hours" -ForegroundColor White
Write-Host "  - Total: ~10-20 hours (depending on hardware)" -ForegroundColor White
Write-Host ""

# Confirm continuation
$confirmation = Read-Host "Continue with full benchmark? (y/N)"
if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
    Write-Host "Cancelled" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Starting complete benchmark test..." -ForegroundColor Green
Write-Host ""

# Record start time
$startTime = Get-Date

# Run complete benchmark test (save detailed results for visualization)
Write-Host "Running experiments (saving detailed results)..." -ForegroundColor Cyan
Write-Host ""
.\run_experiments.ps1 -Command full-benchmark -SaveDetailed

# Check if successful
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Experiment execution failed, exit code: $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

# Calculate runtime
$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host ""
Write-Host "Experiments completed! Duration: $($duration.Hours) hours $($duration.Minutes) minutes" -ForegroundColor Green
Write-Host ""

# Generate visualizations
Write-Host "Starting visualization generation..." -ForegroundColor Cyan
Write-Host ""

# Find Python executable for visualization
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

if (-not $pythonCmd) {
    Write-Host "ERROR: Python executable not found for visualization!" -ForegroundColor Red
    Write-Host "Please ensure Python is installed and available" -ForegroundColor Yellow
    exit 1
}

# Generate static charts
Write-Host "Generating static charts..." -ForegroundColor Yellow
& $pythonCmd generate_visualizations.py --type static

if ($LASTEXITCODE -ne 0) {
    Write-Host "Static chart generation failed, but continuing with animations..." -ForegroundColor Yellow
}

# Generate animations
Write-Host ""
Write-Host "Generating animations..." -ForegroundColor Yellow
Write-Host "  Note: Generating animations for 800 experiments will be very time-consuming" -ForegroundColor White
Write-Host "  Recommendation: You can run animation generation later" -ForegroundColor White
Write-Host ""

$generateAnims = Read-Host "Generate all animations now? (y/N)"
if ($generateAnims -eq 'y' -or $generateAnims -eq 'Y') {
    Write-Host "  This may take a long time, please be patient..." -ForegroundColor White

    # Read experiment summary, group by map and mode for animation generation
    $summaryFile = "results\experiment_summary.json"
    if (Test-Path $summaryFile) {
        $experiments = Get-Content $summaryFile | ConvertFrom-Json

        # Group by map and mode (integrate all algorithms)
        $groups = @{}
        foreach ($exp in $experiments) {
            $map = $exp.map
            $mode = if ($exp.slippery) { "stochastic" } else { "deterministic" }
            $key = "$map-$mode"

            if (-not $groups.ContainsKey($key)) {
                $groups[$key] = @()
            }
            $groups[$key] += $exp
        }

        $totalGroups = $groups.Count
        $currentGroup = 0

        foreach ($key in $groups.Keys) {
            $currentGroup++
            $expList = $groups[$key]
            $firstExp = $expList[0]
            $mapName = $firstExp.map
            $modeName = if ($firstExp.slippery) { "stochastic" } else { "deterministic" }

            $progressMsg = '  [' + $currentGroup + '/' + $totalGroups + '] Generating animations for ' + $mapName + '/' + $modeName + ' (' + $expList.Count + ' experiments)...'
            Write-Host $progressMsg -ForegroundColor Cyan

            # Generate animations for all experiments in this group (animations will be saved in simple_plots with algorithm names)
            & $pythonCmd generate_visualizations.py --type policy --results-dir results
            & $pythonCmd generate_visualizations.py --type learning --results-dir results
        }
    } else {
        Write-Host "Experiment summary file not found, skipping animation generation" -ForegroundColor Yellow
    }

    # Generate global comparison animation
    Write-Host ""
    Write-Host "Generating global comparison animation..." -ForegroundColor Yellow
    & $pythonCmd generate_visualizations.py --type animation
} else {
    Write-Host "Skipping animation generation, you can run it later:" -ForegroundColor Yellow
    Write-Host "   Use run_benchmark_simple.ps1 -Full or run_full_benchmark.ps1" -ForegroundColor White
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Complete benchmark test and visualization generation completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Display result statistics
Write-Host "Result statistics:" -ForegroundColor Yellow
$totalTime = (Get-Date) - $startTime
Write-Host "  - Total time: $($totalTime.Hours) hours $($totalTime.Minutes) minutes" -ForegroundColor White
Write-Host "  - Experiment results: results\experiment_summary.json" -ForegroundColor White
Write-Host "  - Visualization files: results\simple_plots\" -ForegroundColor White
Write-Host ""

# Display directory structure
Write-Host "Directory structure:" -ForegroundColor Yellow
Write-Host "  results\" -ForegroundColor White
Write-Host "    ├── experiment_summary.json" -ForegroundColor White
Write-Host "    ├── {map_name}/" -ForegroundColor White
Write-Host "    │   ├── deterministic/" -ForegroundColor White
Write-Host "    │   │   ├── {algorithm}/" -ForegroundColor White
Write-Host "    │   │   │   └── {seed}/" -ForegroundColor White
Write-Host "    │   │   │       ├── results.json" -ForegroundColor White
Write-Host "    │   │   │       ├── policy.npy/png" -ForegroundColor White
Write-Host "    │   │   │       ├── training_curve.png" -ForegroundColor White
Write-Host "    │   │   │       └── training_history.json" -ForegroundColor White
Write-Host "    │   └── stochastic/" -ForegroundColor White
Write-Host "    └── simple_plots/" -ForegroundColor White
Write-Host "        ├── global_summary.png" -ForegroundColor White
Write-Host "        └── {map_name}/" -ForegroundColor White
Write-Host "            ├── deterministic/" -ForegroundColor White
Write-Host "            │   ├── algorithm_policy.gif" -ForegroundColor White
Write-Host "            │   └── algorithm_learning.gif" -ForegroundColor White
Write-Host "            └── stochastic/" -ForegroundColor White
Write-Host ""

Write-Host "All tasks completed!" -ForegroundColor Green
