# fix_path.ps1
# Fix PowerShell PATH environment variable length issues

Write-Host "=== PowerShell PATH Fix Tool ===" -ForegroundColor Cyan
Write-Host "Checking PATH environment variable..." -ForegroundColor Yellow
Write-Host ""

# Check current PATH length
$pathLength = $env:PATH.Length
Write-Host "Current PATH length: $pathLength characters" -ForegroundColor White

if ($pathLength -gt 2000) {
    Write-Host "WARNING: PATH is too long! This can cause PowerShell errors." -ForegroundColor Red
    Write-Host ""

    # Split PATH and analyze
    $pathEntries = $env:PATH -split ';' | Where-Object { $_ -and $_.Trim() -ne '' }
    Write-Host "PATH contains $($pathEntries.Count) entries:" -ForegroundColor Yellow

    # Show duplicate entries
    $duplicates = $pathEntries | Group-Object | Where-Object { $_.Count -gt 1 }
    if ($duplicates) {
        Write-Host "Found $($duplicates.Count) duplicate entries:" -ForegroundColor Red
        foreach ($dup in $duplicates) {
            Write-Host "  $($dup.Count)x : $($dup.Name)" -ForegroundColor Red
        }
        Write-Host ""
    }

    # Show potentially unnecessary entries
    $unnecessaryPatterns = @(
        ".*\\Chocolatey\\.*",
        ".*\\Microsoft SQL Server\\.*",
        ".*\\Windows Kits\\.*",
        ".*\\dotnet\\.*",
        ".*\\VS.*\\.*",
        ".*\\Microsoft Visual Studio\\.*",
        ".*\\Android\\.*",
        ".*\\Java\\.*",
        ".*\\nodejs\\.*",
        ".*\\yarn\\.*",
        ".*\\npm\\.*"
    )

    Write-Host "Potentially unnecessary PATH entries:" -ForegroundColor Yellow
    $unnecessary = @()
    foreach ($entry in $pathEntries) {
        foreach ($pattern in $unnecessaryPatterns) {
            if ($entry -match $pattern) {
                $unnecessary += $entry
                Write-Host "  $entry" -ForegroundColor Gray
                break
            }
        }
    }

    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "1. Create a cleaned PATH (remove duplicates and long entries)" -ForegroundColor White
    Write-Host "2. Show detailed PATH analysis" -ForegroundColor White
    Write-Host "3. Reset to minimal PATH" -ForegroundColor White
    Write-Host "4. Exit" -ForegroundColor White
    Write-Host ""

    $choice = Read-Host "Choose an option (1-4)"

    switch ($choice) {
        "1" {
            # Remove duplicates and very long entries
            $cleanedEntries = $pathEntries | Select-Object -Unique | Where-Object { $_.Length -lt 200 }

            # Always keep essential entries
            $essentialEntries = @(
                "$env:SystemRoot\System32",
                "$env:SystemRoot",
                "$env:SystemRoot\System32\Wbem",
                "$env:SystemRoot\System32\WindowsPowerShell\v1.0"
            )

            $finalEntries = @()
            $finalEntries += $essentialEntries
            $finalEntries += $cleanedEntries | Where-Object { $essentialEntries -notcontains $_ }

            $newPath = $finalEntries -join ';'
            Write-Host "Original PATH length: $pathLength" -ForegroundColor Yellow
            Write-Host "Cleaned PATH length: $($newPath.Length)" -ForegroundColor Green
            Write-Host "Removed $($pathEntries.Count - $finalEntries.Count) entries" -ForegroundColor Green

            # Apply the cleaned PATH
            $env:PATH = $newPath
            Write-Host ""
            Write-Host "PATH has been cleaned for this session!" -ForegroundColor Green
            Write-Host "To make this permanent, you need to update system environment variables." -ForegroundColor Yellow
        }

        "2" {
            Write-Host "Detailed PATH analysis:" -ForegroundColor Yellow
            for ($i = 0; $i -lt $pathEntries.Count; $i++) {
                $entry = $pathEntries[$i]
                $exists = if (Test-Path $entry -ErrorAction SilentlyContinue) { "EXISTS" } else { "MISSING" }
                Write-Host ("{0,3}: [{1}] {2}" -f $i, $exists, $entry) -ForegroundColor $(if ($exists -eq "MISSING") { "Red" } else { "White" })
            }
        }

        "3" {
            # Reset to minimal PATH
            $minimalPath = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\Wbem;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
            $env:PATH = $minimalPath
            Write-Host "PATH reset to minimal configuration" -ForegroundColor Green
            Write-Host "You may need to manually add conda and other tool paths" -ForegroundColor Yellow
        }

        default {
            Write-Host "No changes made." -ForegroundColor Yellow
        }
    }

} else {
    Write-Host "PATH length is acceptable." -ForegroundColor Green
}

Write-Host ""
Write-Host "Current PATH length: $($env:PATH.Length) characters" -ForegroundColor White
Write-Host ""
Write-Host "To make PATH changes permanent:" -ForegroundColor Yellow
Write-Host "1. Press Win+R, type 'sysdm.cpl', press Enter" -ForegroundColor White
Write-Host "2. Go to 'Advanced' tab -> 'Environment Variables'" -ForegroundColor White
Write-Host "3. Edit the 'PATH' variable in System/User variables" -ForegroundColor White
Write-Host ""
Write-Host "Done!" -ForegroundColor Green
