# H.O.P.E Development Environment Setup Script
# High-Output Performance Engineering - AI-Driven Vehicle Diagnostics & Tuning

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "HOPE Development Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Warning: Not running as Administrator. Some installations may fail." -ForegroundColor Yellow
    Write-Host ""
}

# Function to check if a command exists
function Test-CommandExists {
    param($command)
    $null = Get-Command $command -ErrorAction SilentlyContinue
    return $?
}

Write-Host "Checking prerequisites..." -ForegroundColor Yellow
Write-Host ""

# Check .NET SDK
Write-Host "[1/8] Checking .NET 8 SDK..." -ForegroundColor Cyan
if (Test-CommandExists dotnet) {
    $dotnetVersion = dotnet --version
    Write-Host "  ??? .NET SDK found: $dotnetVersion" -ForegroundColor Green

    # Initialize .NET projects
    Write-Host "  ??? Initializing .NET solution..." -ForegroundColor White
    Push-Location src/desktop

    # Create solution
    if (-not (Test-Path "HOPE.Desktop.sln")) {
        dotnet new sln -n HOPE.Desktop
        Write-Host "    Created solution: HOPE.Desktop.sln" -ForegroundColor Green
    }

    # Create HOPE.Core class library
    if (-not (Test-Path "HOPE.Core")) {
        dotnet new classlib -n HOPE.Core -f net8.0
        dotnet sln add HOPE.Core/HOPE.Core.csproj
        Write-Host "    Created project: HOPE.Core (Class Library)" -ForegroundColor Green
    }

    # Create HOPE.Desktop WPF application
    if (-not (Test-Path "HOPE.Desktop")) {
        dotnet new wpf -n HOPE.Desktop -f net8.0
        dotnet sln add HOPE.Desktop/HOPE.Desktop.csproj
        dotnet add HOPE.Desktop/HOPE.Desktop.csproj reference HOPE.Core/HOPE.Core.csproj
        Write-Host "    Created project: HOPE.Desktop (WPF App)" -ForegroundColor Green
    }

    # Create HOPE.Desktop.Tests unit test project
    if (-not (Test-Path "HOPE.Desktop.Tests")) {
        dotnet new xunit -n HOPE.Desktop.Tests -f net8.0
        dotnet sln add HOPE.Desktop.Tests/HOPE.Desktop.Tests.csproj
        dotnet add HOPE.Desktop.Tests/HOPE.Desktop.Tests.csproj reference HOPE.Core/HOPE.Core.csproj
        Write-Host "    Created project: HOPE.Desktop.Tests (xUnit Tests)" -ForegroundColor Green
    }

    # Install NuGet packages for desktop
    Write-Host "  ??? Installing NuGet packages..." -ForegroundColor White
    dotnet add HOPE.Core/HOPE.Core.csproj package Prism.DryIoc --version 8.1.97
    dotnet add HOPE.Core/HOPE.Core.csproj package Microsoft.ML.OnnxRuntime --version 1.17.0
    dotnet add HOPE.Core/HOPE.Core.csproj package CommunityToolkit.Mvvm --version 8.2.2
    dotnet add HOPE.Core/HOPE.Core.csproj package Microsoft.Data.Sqlite --version 8.0.0

    dotnet add HOPE.Desktop/HOPE.Desktop.csproj package Prism.Wpf --version 8.1.97
    dotnet add HOPE.Desktop/HOPE.Desktop.csproj package LiveChartsCore.SkiaSharpView.WPF --version 2.0.0-rc2
    dotnet add HOPE.Desktop/HOPE.Desktop.csproj package QuestPDF --version 2024.1.3

    Write-Host "    ??? NuGet packages installed" -ForegroundColor Green

    # Restore packages
    dotnet restore

    Pop-Location
} else {
    Write-Host "  ??? .NET 8 SDK not found!" -ForegroundColor Red
    Write-Host "    Please install from: https://dotnet.microsoft.com/download/dotnet/8.0" -ForegroundColor Yellow
}
Write-Host ""

# Check Node.js
Write-Host "[2/8] Checking Node.js..." -ForegroundColor Cyan
if (Test-CommandExists node) {
    $nodeVersion = node --version
    Write-Host "  ??? Node.js found: $nodeVersion" -ForegroundColor Green

    # Initialize NestJS project
    Write-Host "  ??? Initializing NestJS backend..." -ForegroundColor White
    Push-Location src/backend

    if (-not (Test-Path "package.json")) {
        # Create package.json
        @"
{
  `"name`": `"hope-backend`",
  `"version`": `"1.0.0`",
  `"description`": `"HOPE Backend API - Multi-tenant vehicle diagnostics platform`",
  `"main`": `"dist/main.js`",
  `"scripts`": {
    `"build`": `"nest build`",
    `"start`": `"nest start`",
    `"start:dev`": `"nest start --watch`",
    `"start:prod`": `"node dist/main`",
    `"test`": `"jest`",
    `"test:watch`": `"jest --watch`",
    `"lint`": `"eslint \`"{src,apps,libs,test}/**/*.ts\`" --fix`",
    `"migrate`": `"typeorm migration:run`",
    `"migrate:revert`": `"typeorm migration:revert`"
  },
  `"dependencies`": {
    `"@nestjs/common`": `"^10.3.0`",
    `"@nestjs/core`": `"^10.3.0`",
    `"@nestjs/graphql`": `"^12.1.0`",
    `"@nestjs/typeorm`": `"^10.0.1`",
    `"@nestjs/jwt`": `"^10.2.0`",
    `"@nestjs/platform-express`": `"^10.3.0`",
    `"@apollo/server`": `"^4.10.0`",
    `"graphql`": `"^16.8.1`",
    `"typeorm`": `"^0.3.19`",
    `"pg`": `"^8.11.3`",
    `"@aws-sdk/client-s3`": `"^3.500.0`",
    `"bcrypt`": `"^5.1.1`",
    `"class-validator`": `"^0.14.1`",
    `"class-transformer`": `"^0.5.1`",
    `"reflect-metadata`": `"^0.2.1`",
    `"rxjs`": `"^7.8.1`"
  },
  `"devDependencies`": {
    `"@nestjs/cli`": `"^10.3.0`",
    `"@nestjs/schematics`": `"^10.1.0`",
    `"@nestjs/testing`": `"^10.3.0`",
    `"@types/node`": `"^20.11.0`",
    `"@types/bcrypt`": `"^5.0.2`",
    `"@typescript-eslint/eslint-plugin`": `"^6.19.0`",
    `"@typescript-eslint/parser`": `"^6.19.0`",
    `"eslint`": `"^8.56.0`",
    `"jest`": `"^29.7.0`",
    `"ts-jest`": `"^29.1.2`",
    `"typescript`": `"^5.3.3`"
  }
}
"@ | Out-File -FilePath "package.json" -Encoding UTF8
        Write-Host "    Created package.json" -ForegroundColor Green
    }

    if (-not (Test-Path "node_modules")) {
        Write-Host "  ??? Installing npm packages (this may take a few minutes)..." -ForegroundColor White
        npm install --silent
        Write-Host "    ??? npm packages installed" -ForegroundColor Green
    }

    Pop-Location
} else {
    Write-Host "  ??? Node.js not found!" -ForegroundColor Red
    Write-Host "    Please install from: https://nodejs.org/ (LTS version 20.x)" -ForegroundColor Yellow
}
Write-Host ""

# Check Python
Write-Host "[3/8] Checking Python..." -ForegroundColor Cyan
if (Test-CommandExists python) {
    $pythonVersion = python --version
    Write-Host "  ??? Python found: $pythonVersion" -ForegroundColor Green

    # Initialize Python project
    Write-Host "  ??? Initializing Python AI training project..." -ForegroundColor White
    Push-Location src/ai-training

    if (-not (Test-Path "requirements.txt")) {
        @"
# HOPE AI Training Dependencies
# Machine Learning and Data Processing

# Deep Learning Frameworks (choose one)
tensorflow==2.15.0
# pytorch==2.2.0  # Alternative to TensorFlow

# ONNX Export
onnx==1.15.0
tf2onnx==1.16.0

# Data Processing
pandas==2.2.0
numpy==1.26.3
scikit-learn==1.4.0

# Visualization
matplotlib==3.8.2
seaborn==0.13.1

# Utilities
tqdm==4.66.1
jupyter==1.0.0
"@ | Out-File -FilePath "requirements.txt" -Encoding UTF8
        Write-Host "    Created requirements.txt" -ForegroundColor Green
    }

    # Create virtual environment
    if (-not (Test-Path "venv")) {
        Write-Host "  ??? Creating Python virtual environment..." -ForegroundColor White
        python -m venv venv
        Write-Host "    ??? Virtual environment created" -ForegroundColor Green
    }

    Pop-Location

    Write-Host "    Note: Activate venv with: src\ai-training\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
} else {
    Write-Host "  ??? Python not found!" -ForegroundColor Red
    Write-Host "    Please install from: https://www.python.org/downloads/ (version 3.11.x)" -ForegroundColor Yellow
}
Write-Host ""

# Check PostgreSQL
Write-Host "[4/8] Checking PostgreSQL..." -ForegroundColor Cyan
if (Test-CommandExists psql) {
    $pgVersion = psql --version
    Write-Host "  ??? PostgreSQL found: $pgVersion" -ForegroundColor Green
} else {
    Write-Host "  ??? PostgreSQL not found!" -ForegroundColor Red
    Write-Host "    Please install from: https://www.postgresql.org/download/windows/" -ForegroundColor Yellow
    Write-Host "    Recommended: PostgreSQL 16 with TimescaleDB extension" -ForegroundColor Yellow
}
Write-Host ""

# Check Docker
Write-Host "[5/8] Checking Docker..." -ForegroundColor Cyan
if (Test-CommandExists docker) {
    $dockerVersion = docker --version
    Write-Host "  ??? Docker found: $dockerVersion" -ForegroundColor Green
} else {
    Write-Host "  ??? Docker not found!" -ForegroundColor Red
    Write-Host "    Please install Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
}
Write-Host ""

# Check Git
Write-Host "[6/8] Checking Git..." -ForegroundColor Cyan
if (Test-CommandExists git) {
    $gitVersion = git --version
    Write-Host "  ??? Git found: $gitVersion" -ForegroundColor Green

    # Initialize Git repository if not already initialized
    if (-not (Test-Path ".git")) {
        Write-Host "  ??? Initializing Git repository..." -ForegroundColor White
        git init
        git branch -M main
        Write-Host "    ??? Git repository initialized" -ForegroundColor Green

        # Create .gitignore
        if (-not (Test-Path ".gitignore")) {
            @"
# Build outputs
**/bin/
**/obj/
**/dist/
**/build/
**/out/

# Dependencies
**/node_modules/
**/venv/
**/.venv/
**/packages/

# IDE
.vs/
.vscode/
.idea/
*.suo
*.user
*.userosscache
*.sln.docstates

# Environment
.env
.env.local
*.env

# Logs
*.log
npm-debug.log*

# Database
*.db
*.sqlite
*.sqlite3

# Python
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/

# AI Models (large files)
**/models/*.h5
**/models/*.pb
**/data/raw/*.csv
**/data/raw/*.json

# Keep .onnx models (small enough)
# !**/data/onnx/*.onnx

# ECU files (sensitive)
**/ecu-files/*.bin
**/ecu-files/*.hex

# OS
.DS_Store
Thumbs.db
"@ | Out-File -FilePath ".gitignore" -Encoding UTF8
            Write-Host "    Created .gitignore" -ForegroundColor Green
        }
    } else {
        Write-Host "  ??? Git repository already initialized" -ForegroundColor Green
    }
} else {
    Write-Host "  ??? Git not found!" -ForegroundColor Red
    Write-Host "    Please install from: https://git-scm.com/download/win" -ForegroundColor Yellow
}
Write-Host ""

# Check Visual Studio
Write-Host "[7/8] Checking Visual Studio..." -ForegroundColor Cyan
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022"
if (Test-Path $vsPath) {
    Write-Host "  ??? Visual Studio 2022 found" -ForegroundColor Green
} else {
    Write-Host "  ??? Visual Studio 2022 not found in default location" -ForegroundColor Yellow
    Write-Host "    Recommended: VS 2022 Community with .NET Desktop Development workload" -ForegroundColor Yellow
    Write-Host "    Download from: https://visualstudio.microsoft.com/" -ForegroundColor Yellow
}
Write-Host ""

# Create additional directory structure
Write-Host "[8/8] Creating project structure..." -ForegroundColor Cyan
$dirs = @(
    "src/desktop/HOPE.Core/Models",
    "src/desktop/HOPE.Core/Services/OBD",
    "src/desktop/HOPE.Core/Services/ECU",
    "src/desktop/HOPE.Core/Services/AI",
    "src/desktop/HOPE.Core/Services/Cloud",
    "src/desktop/HOPE.Core/Services/Reports",
    "src/desktop/HOPE.Core/Protocols",
    "src/desktop/HOPE.Desktop/Views",
    "src/desktop/HOPE.Desktop/ViewModels",
    "src/desktop/HOPE.Desktop/Controls",
    "src/desktop/HOPE.Desktop/Converters",
    "src/desktop/HOPE.Desktop/Themes",
    "src/backend/src/modules/auth",
    "src/backend/src/modules/tenant",
    "src/backend/src/modules/vehicles",
    "src/backend/src/modules/diagnostics",
    "src/backend/src/modules/ecu-calibrations",
    "src/backend/src/modules/customers",
    "src/backend/src/modules/reports",
    "src/backend/src/database/migrations",
    "src/backend/src/common/decorators",
    "src/backend/src/common/filters",
    "src/backend/src/common/interceptors",
    "src/backend/test",
    "src/ai-training/scripts",
    "src/ai-training/models",
    "src/ai-training/data/raw",
    "src/ai-training/data/processed",
    "src/ai-training/data/onnx",
    "src/ai-training/notebooks",
    "src/shared/graphql-schema",
    "src/shared/types"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "  ??? Project directory structure created" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Install any missing prerequisites listed above" -ForegroundColor White
Write-Host "  2. Open src/desktop/HOPE.Desktop.sln in Visual Studio 2022" -ForegroundColor White
Write-Host "  3. Activate Python venv: src\ai-training\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  4. Install Python packages: pip install -r src\ai-training\requirements.txt" -ForegroundColor White
Write-Host "  5. Start backend dev server: cd src\backend && npm run start:dev" -ForegroundColor White
Write-Host ""
Write-Host "??? Development environment setup complete!" -ForegroundColor Green

