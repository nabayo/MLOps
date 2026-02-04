#!/usr/bin/env pwsh
# Export MLflow data to a backup zip file using Docker (PowerShell version)

param(
    [Parameter(Position = 0)]
    [string]$BackupName = ""
)

Write-Host "[*] Exporting MLflow data via Docker..." -ForegroundColor Cyan
Write-Host ""

# Run export
if ($BackupName -eq "") {
    docker-compose run --rm export
} else {
    $env:BACKUP_NAME = $BackupName
    docker-compose run --rm -e BACKUP_NAME=$BackupName export python scripts/export_mlflow.py --output-dir /app/backups --name $BackupName
}

Write-Host ""
Write-Host "[+] Export complete! Backup saved in .\backups\" -ForegroundColor Green

# Show the latest backup
$latestBackup = Get-ChildItem -Path "backups" -Filter "*.zip" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latestBackup) {
    $size = "{0:N2} MB" -f ($latestBackup.Length / 1MB)
    Write-Host "$($latestBackup.Name) - $size" -ForegroundColor Gray
}
