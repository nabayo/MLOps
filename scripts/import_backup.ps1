#!/usr/bin/env pwsh
# Import MLflow data from a backup zip file using Docker (PowerShell version)

param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$BackupFile,
    
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

# Check if Docker is running
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Error: Docker is not running. Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

# Check if file exists (try direct path first, then relative to backups/)
$InputFile = $BackupFile
if (-not (Test-Path $InputFile)) {
    if (Test-Path "backups\$InputFile") {
        $InputFile = "backups\$InputFile"
    } else {
        Write-Host "[!] Error: Backup file not found: $InputFile" -ForegroundColor Red
        Write-Host ""
        Write-Host "Available backups:"
        
        $backups = Get-ChildItem -Path "backups" -Filter "*.zip" -ErrorAction SilentlyContinue
        if ($backups) {
            $backups | ForEach-Object { Write-Host "  $($_.Name)" }
        } else {
            Write-Host "  (none)"
        }
        exit 1
    }
}

# Extract just the filename for the docker container context if needed, 
# but here we seem to need the path relative to the workspace or just the filename if mapped.
# The original script assumed the file is in 'backups/' and mounted just that?
# Let's look at how docker-compose mounts it.
# Usually volumes are mounted. If 'import' service mounts ./backups:/app/backups, 
# then we should pass the filename relative to /app/backups.

# If the user provided a full path that IS inside backups folder, we can extract the name.
if ($InputFile -like "*backups\*") {
    $BackupFilename = Split-Path $InputFile -Leaf
} else {
    # If the file is NOT in backups/, the docker mount might not see it depending on docker-compose.yml.
    # Assuming the standard flow requires it to be in 'backups/'.
    Write-Host "[!] Warning: File '$InputFile' might not be visible to Docker if it's not in the 'backups' directory." -ForegroundColor Yellow
    $BackupFilename = Split-Path $InputFile -Leaf
}

Write-Host "[*] Importing MLflow data from: $BackupFilename" -ForegroundColor Cyan
Write-Host ""

# Build arguments string
$argsString = ""
if ($ExtraArgs) {
    $argsString = $ExtraArgs -join " "
}

# Run import
# We pass the filename assuming the volume mount handles the directory
$env:BACKUP_FILE = $BackupFilename
if ($argsString) {
    docker-compose run --rm -e BACKUP_FILE=$BackupFilename import python scripts/import_mlflow.py /app/backups/$BackupFilename $argsString
} else {
    docker-compose run --rm -e BACKUP_FILE=$BackupFilename import python scripts/import_mlflow.py /app/backups/$BackupFilename
}

Write-Host ""
Write-Host "[+] Import complete!" -ForegroundColor Green
