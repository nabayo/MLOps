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
# Handle path input (e.g. "backups\file.zip" vs "file.zip")
$BackupFilename = Split-Path $BackupFile -Leaf
$TargetFile = Join-Path "backups" $BackupFilename

if (Test-Path $TargetFile) {
    # File exists in the correct backups/ directory
    $FinalFilename = $BackupFilename
} elseif (Test-Path $BackupFile) {
    # File exists at the provided path, check if it's strictly in backups/
    $FileItem = Get-Item $BackupFile
    $BackupsDir = Get-Item "backups"
    
    # Simple check if parent directory matches
    if ($FileItem.DirectoryName -eq $BackupsDir.FullName) {
        $FinalFilename = $FileItem.Name
    } else {
        Write-Host "[!] Error: Backup file must be inside the 'backups' directory." -ForegroundColor Red
        Write-Host "    Docker container cannot verify files outside this directory."
        Write-Host "    Please move '$BackupFile' to 'backups\' and try again." -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "[!] Error: Backup file not found in 'backups' folder: $BackupFilename" -ForegroundColor Red
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

Write-Host "[*] Importing MLflow data from: $FinalFilename" -ForegroundColor Cyan
Write-Host ""

# Build arguments string
$argsString = ""
if ($ExtraArgs) {
    $argsString = $ExtraArgs -join " "
}

# Run import
# We pass the filename assuming the volume mount handles the directory
# Run import
# We pass the filename assuming the volume mount handles the directory
$env:BACKUP_FILE = $FinalFilename
if ($argsString) {
    docker-compose run --rm -e BACKUP_FILE=$FinalFilename import python scripts/import_mlflow.py /app/backups/$FinalFilename $argsString
} else {
    docker-compose run --rm -e BACKUP_FILE=$FinalFilename import python scripts/import_mlflow.py /app/backups/$FinalFilename
}

Write-Host ""
Write-Host "[+] Import complete!" -ForegroundColor Green
