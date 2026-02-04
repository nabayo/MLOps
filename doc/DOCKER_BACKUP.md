# Quick Reference: Docker Export/Import

## 🚀 Quick Commands

### Export (Backup)

```bash
# Auto-generated name (recommended)
docker-compose run --rm export

# OR use helper script
./scripts/docker_export.sh

# Custom name
./scripts/docker_export.sh my_backup
```

### Import (Restore)

```bash
# Basic import (skips existing data)
./scripts/docker_import.sh my_backup.zip

# Dry run (preview without changes)
./scripts/docker_import.sh my_backup.zip --dry-run

# Overwrite existing data
./scripts/docker_import.sh my_backup.zip --overwrite
```

## 📋 Direct Docker Compose Commands

### Export

```bash
# Default export
docker-compose run --rm export

# Custom output name
docker-compose run --rm export \
    python scripts/export_mlflow.py --output-dir /app/backups --name my_custom_backup
```

### Import

```bash
# Import with filename
BACKUP_FILE=mlflow_backup_20260204_120000.zip docker-compose run --rm import

# Dry run
docker-compose run --rm import \
    python scripts/import_mlflow.py /app/backups/my_backup.zip --dry-run

# Overwrite mode
docker-compose run --rm import \
    python scripts/import_mlflow.py /app/backups/my_backup.zip --overwrite
```

## 📁 File Locations

- **Backups directory:** `./backups/`
- **Format:** `mlflow_backup_YYYYMMDD_HHMMSS.zip`
- **Access:** Backups are saved on your host machine, not inside containers

## 💡 Common Workflows

### Before Making Major Changes

```bash
# Create backup
./scripts/docker_export.sh before_experiment_v2

# Make your changes...

# Restore if needed
./scripts/docker_import.sh before_experiment_v2.zip
```

### Regular Backups

```bash
# Add to crontab for daily backups
0 2 * * * cd /path/to/MLOps && docker-compose run --rm export
```

### Share with Team

```bash
# Export
./scripts/docker_export.sh team_share_$(date +%Y%m%d)

# Send backups/team_share_20260204.zip to teammates

# Teammates import
./scripts/docker_import.sh team_share_20260204.zip
```

## 🔍 List Available Backups

```bash
ls -lh backups/
```

## 🗑️ Clean Old Backups

```bash
# Delete backups older than 30 days
find backups/ -name "*.zip" -mtime +30 -delete

# Keep only last 5 backups
ls -t backups/*.zip | tail -n +6 | xargs rm -f
```

## ⚙️ Advanced Options

### Export to Different Location

```bash
# Export to external USB drive
docker-compose run --rm \
    -v /media/usb/mlflow_backups:/app/backups \
    export
```

### Import from Different Location

```bash
# Import from external source
docker-compose run --rm \
    -v /media/usb/mlflow_backups:/app/external \
    import python scripts/import_mlflow.py /app/external/backup.zip
```

## 🐛 Troubleshooting

### "Container not found" error

Make sure MLflow services are running:
```bash
docker-compose up -d
```

### "Backup file not found"

Check the backups directory:
```bash
ls backups/
```

### "Permission denied"

Ensure scripts are executable:
```bash
chmod +x scripts/docker_*.sh
```

## 📖 Full Documentation

See `scripts/BACKUP_README.md` for detailed documentation, examples, and advanced usage.
