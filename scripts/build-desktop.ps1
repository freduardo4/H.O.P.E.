# Helper script to build the Desktop application reliably
# Uses absolute path to dotnet to avoid PATH issues in some environments

$DotNetPath = "C:\Program Files\dotnet\dotnet.exe"
$DesktopPath = Join-Path $PSScriptRoot "..\src\desktop"

Write-Host "Building HOPE Desktop..." -ForegroundColor Cyan
Write-Host "Using .NET at: $DotNetPath" -ForegroundColor Gray

if (-not (Test-Path $DotNetPath)) {
    Write-Error ".NET SDK not found at $DotNetPath"
    exit 1
}

Push-Location $DesktopPath

try {
    Write-Host "Restoring packages..." -ForegroundColor Yellow
    & $DotNetPath restore
    if ($LASTEXITCODE -ne 0) { throw "Restore failed" }

    Write-Host "Building solution..." -ForegroundColor Yellow
    & $DotNetPath build --no-restore
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }

    Write-Host "Build Success!" -ForegroundColor Green
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
