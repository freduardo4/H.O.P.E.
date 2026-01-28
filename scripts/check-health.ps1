<#
.SYNOPSIS
    Unified health check script for the H.O.P.E. project.
    Runs builds and tests for Desktop (.NET), Backend (Node.js), and AI (Python).

.EXAMPLE
    .\scripts\check-health.ps1
#>

$root = Get-Location
$results = @{}

Write-Host "--- H.O.P.E. Project Health Check ---" -ForegroundColor Cyan

# 1. Desktop (.NET)
Write-Host "`n[1/3] Checking Desktop (.NET)..." -ForegroundColor Yellow
try {
    dotnet build src/desktop/HOPE.Desktop/HOPE.Desktop.csproj --nologo -v q
    dotnet test src/desktop/HOPE.Desktop.Tests/HOPE.Desktop.Tests.csproj --nologo -v q
    $results["Desktop"] = "PASS"
}
catch {
    $results["Desktop"] = "FAIL"
}

# 2. Backend (Node.js)
Write-Host "`n[2/3] Checking Backend (NestJS)..." -ForegroundColor Yellow
Push-Location src/backend
try {
    # Check if node_modules exists
    if (-not (Test-Path "node_modules")) {
        npm install --silent
    }
    npm run build --silent
    npm test -- --watchAll=false --silent
    $results["Backend"] = "PASS"
}
catch {
    $results["Backend"] = "FAIL"
}
Pop-Location

# 3. AI (Python)
Write-Host "`n[3/3] Checking AI (Python)..." -ForegroundColor Yellow
Push-Location src/ai-training
try {
    if (-not (Test-Path "venv")) {
        python -m venv venv
        .\venv\Scripts\pip install -r requirements.txt --quiet
    }
    .\venv\Scripts\pytest --quiet
    $results["AI"] = "PASS"
}
catch {
    $results["AI"] = "FAIL"
}
Pop-Location

Write-Host "`n--- Health Summary ---" -ForegroundColor Cyan
foreach ($key in $results.Keys) {
    $color = if ($results[$key] -eq "PASS") { "Green" } else { "Red" }
    Write-Host "$key: $($results[$key])" -ForegroundColor $color
}

if ($results.Values -contains "FAIL") {
    exit 1
}
exit 0
