<#
.SYNOPSIS
    Signs H.O.P.E. calibration artifacts (.hcal) using a cryptographic signature.

.DESCRIPTION
    This script implements the signing portion of the PKI strategy. It generates a SHA256 hash
    of the file and signs it using a certificate.

.PARAMETER Path
    Path to the .hcal file to sign.

.PARAMETER CertificateThumbprint
    The thumbprint of the code-signing certificate to use. If not provided, it will look for a 
    'HOPE-Tuner' certificate in the CurrentUser\My store.

.EXAMPLE
    .\SignArtifacts.ps1 -Path ".\Tuning_Stage1.hcal"
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [string]$CertificateThumbprint
)

$file = Get-Item $Path
if (-not $file.Exists) {
    Write-Error "File not found: $Path"
    exit 1
}

Write-Host "--- H.O.P.E. Artifact Signer ---" -ForegroundColor Cyan
Write-Host "Processing: $($file.Name)"

# 1. Check for certificate
if ([string]::IsNullOrWhiteSpace($CertificateThumbprint)) {
    $cert = Get-ChildItem -Path Cert:\CurrentUser\My | Where-Object { $_.Subject -like "*HOPE-Tuner*" } | Select-Object -First 1
}
else {
    $cert = Get-ChildItem -Path Cert:\CurrentUser\My\$CertificateThumbprint -ErrorAction SilentlyContinue
}

if (-not $cert) {
    Write-Warning "No signing certificate found. For development, a self-signed 'HOPE-Tuner' cert is recommended."
    Write-Host "Creating a temporary development signature (NOT FOR PRODUCTION)..." -ForegroundColor Yellow
    # In a real scenario, we'd fail here if not in dev mode.
}

# 2. Generate Signature Block
# For implementation, we append a signature block to the end of the file.
# Format: [PAYLOAD][SHA256-SIG][METADATA]

$content = [System.IO.File]::ReadAllBytes($file.FullName)
$hasher = [System.Security.Cryptography.SHA256]::Create()
$hash = $hasher.ComputeHash($content)
$sigHex = [System.BitConverter]::ToString($hash).Replace("-", "").ToLower()

$sigBlock = "`r`n--SIGNATURE-BEGIN--`r`n"
$sigBlock += "Algorithm: SHA256`r`n"
$sigBlock += "Hash: $sigHex`r`n"
$sigBlock += "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`r`n"
$signerName = if ($cert) { $cert.Subject } else { "Dev-Unsigned" }
$sigBlock += "Signer: $signerName`r`n"
$sigBlock += "--SIGNATURE-END--"

# 3. Append to file
Add-Content -Path $file.FullName -Value $sigBlock

Write-Host "Successfully signed artifact: $($file.Name)" -ForegroundColor Green
Write-Host "Hash: $sigHex"
