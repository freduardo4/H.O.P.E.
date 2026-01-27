<#
.SYNOPSIS
    HOPE Developer CLI - Unified entry point for development tasks.

.DESCRIPTION
    Wraps common tasks for Backend (Node.js), Desktop (.NET), and AI (Python) components.

.EXAMPLE
    .\hope.ps1 setup
    .\hope.ps1 build all
    .\hope.ps1 test desktop
#>

param (
    [Parameter(Mandatory=$false, Position=0)]
    [ValidateSet("setup", "build", "test", "lint", "start", "help", "clean")]
    [string]$Command = "help",

    [Parameter(Mandatory=$false, Position=1)]
    [ValidateSet("backend", "desktop", "ai", "all")]
    [string]$Module = "all"
)

$ErrorActionPreference = "Stop"
$ScriptRoot = $PSScriptRoot
$ProjectRoot = Split-Path $ScriptRoot -Parent

function Write-Header {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Show-Help {
    Write-Host "HOPE Developer CLI" -ForegroundColor Green
    Write-Host "Usage: .\hope.ps1 <command> [module]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  setup       Install dependencies for all modules"
    Write-Host "  build       Build specified module(s) (backend, desktop, all)"
    Write-Host "  test        Run tests for specified module(s)"
    Write-Host "  lint        Lint code for specified module(s)"
    Write-Host "  start       Start application/dev server (backend, desktop)"
    Write-Host "  clean       Remove build artifacts"
    Write-Host ""
    Write-Host "Modules:"
    Write-Host "  backend     NestJS API (src/backend)"
    Write-Host "  desktop     .NET WPF App (src/desktop)"
    Write-Host "  ai          Python Training Scripts (src/ai-training)"
    Write-Host "  all         All available modules"
}

function Run-Setup {
    Write-Header "Setting up Development Environment"
    
    # Check if setup-dev.ps1 exists, otherwise do manual setup
    $SetupScript = Join-Path $ScriptRoot "setup-dev.ps1"
    if (Test-Path $SetupScript) {
        & $SetupScript
    } else {
        Write-Warning "scripts/setup-dev.ps1 not found. Performing manual setup..."
        
        # Backend
        Write-Host "Installing Backend dependencies..." -ForegroundColor Yellow
        Set-Location "$ProjectRoot/src/backend"
        npm install

        # AI
        Write-Host "Setting up Python environment..." -ForegroundColor Yellow
        Set-Location "$ProjectRoot/src/ai-training"
        if (!(Test-Path "venv")) { python -m venv venv }
        ./venv/Scripts/pip install -r requirements.txt

        # Desktop
        Write-Host "Restoring .NET packages..." -ForegroundColor Yellow
        Set-Location "$ProjectRoot/src/desktop"
        dotnet restore
    }
}

function Run-Build {
    param([string]$Target)
    
    if ($Target -in @("backend", "all")) {
        Write-Header "Building Backend"
        Set-Location "$ProjectRoot/src/backend"
        npm run build
    }

    if ($Target -in @("desktop", "all")) {
        Write-Header "Building Desktop"
        Set-Location "$ProjectRoot/src/desktop"
        dotnet build
    }
}

function Run-Test {
    param([string]$Target)

    if ($Target -in @("backend", "all")) {
        Write-Header "Testing Backend"
        Set-Location "$ProjectRoot/src/backend"
        npm test
    }

    if ($Target -in @("desktop", "all")) {
        Write-Header "Testing Desktop"
        Set-Location "$ProjectRoot/src/desktop"
        dotnet test --logger "console;verbosity=normal"
    }

    if ($Target -in @("ai", "all")) {
        Write-Header "Testing AI/Python"
        Set-Location "$ProjectRoot/src/ai-training"
        # Use venv python if available
        if (Test-Path "venv/Scripts/pytest.exe") {
            ./venv/Scripts/pytest -v
        } else {
            pytest -v
        }
    }
}

function Run-Lint {
    param([string]$Target)

    if ($Target -in @("backend", "all")) {
        Write-Header "Linting Backend"
        Set-Location "$ProjectRoot/src/backend"
        npm run lint
    }
    
    if ($Target -in @("ai", "all")) {
        Write-Header "Linting AI"
        # Placeholder for pylint/flake8
        Write-Host "Python linting not yet configured in package. Skipping." -ForegroundColor DarkGray
    }
}

function Run-Start {
    param([string]$Target)

    if ($Target -eq "backend") {
        Write-Header "Starting Backend (Dev)"
        Set-Location "$ProjectRoot/src/backend"
        npm run start:dev
    } elseif ($Target -eq "desktop") {
        Write-Header "Starting Desktop App"
        Set-Location "$ProjectRoot/src/desktop"
        dotnet run --project HOPE.Desktop
    } elseif ($Target -eq "all") {
        Write-Error "Cannot start 'all' simultaneously in this console. Start them in separate terminals."
    }
}

function Run-Clean {
    Write-Header "Cleaning Build Artifacts"
    
    Remove-Item "$ProjectRoot/src/backend/dist" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item "$ProjectRoot/src/desktop/*/bin" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item "$ProjectRoot/src/desktop/*/obj" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item "$ProjectRoot/src/ai-training/__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
    
    Write-Host "Clean complete." -ForegroundColor Green
}

# Main Dispatch
try {
    switch ($Command) {
        "setup" { Run-Setup }
        "build" { Run-Build -Target $Module }
        "test"  { Run-Test -Target $Module }
        "lint"  { Run-Lint -Target $Module }
        "start" { Run-Start -Target $Module }
        "clean" { Run-Clean }
        "help"  { Show-Help }
        Default { Show-Help }
    }
} catch {
    Write-Error "Command failed: $_"
    exit 1
} finally {
    Set-Location $ScriptRoot
}
