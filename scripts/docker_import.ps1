#!/usr/bin/env pwsh
# Import MLflow data from a backup zip file using Docker (PowerShell version)

param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$BackupFile,
    
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

# Check if file exists
if (-not (Test-Path "backups\$BackupFile")) {
    Write-Host "[!] Error: Backup file not found: backups\$BackupFile" -ForegroundColor Red
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

Write-Host "[*] Importing MLflow data from: $BackupFile" -ForegroundColor Cyan
Write-Host ""

# Build arguments string
$argsString = ""
if ($ExtraArgs) {
    $argsString = $ExtraArgs -join " "
}

# Run import
$env:BACKUP_FILE = $BackupFile
if ($argsString) {
    docker-compose run --rm -e BACKUP_FILE=$BackupFile import python scripts/import_mlflow.py /app/backups/$BackupFile $argsString
} else {
    docker-compose run --rm -e BACKUP_FILE=$BackupFile import python scripts/import_mlflow.py /app/backups/$BackupFile
}

Write-Host ""
Write-Host "[+] Import complete!" -ForegroundColor Green
