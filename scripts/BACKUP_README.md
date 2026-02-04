# MLflow Backup & Restore Scripts

Utility scripts for exporting and importing MLflow data (experiments, runs, models, and artifacts).

## 📤 Export (Backup)

Export all MLflow data to a timestamped zip file.

### Usage

```bash
# Basic export (creates backups/mlflow_backup_YYYYMMDD_HHMMSS.zip)
python scripts/export_mlflow.py

# Custom output directory
python scripts/export_mlflow.py --output-dir /path/to/backups

# Custom backup name
python scripts/export_mlflow.py --name my_backup

# Specify MLflow tracking URI
python scripts/export_mlflow.py --tracking-uri http://localhost:5000
```

### What Gets Exported

- ✅ All experiments with metadata and tags
- ✅ All runs with parameters, metrics, and tags
- ✅ All model artifacts
- ✅ All run artifacts
- ✅ All registered models and versions
- ✅ Model staging information

### Docker Usage

```bash
# Export from running MLflow container
docker-compose run --rm training python scripts/export_mlflow.py

# Mount local directory for backup output
docker-compose run --rm -v $(pwd)/backups:/app/backups training \
    python scripts/export_mlflow.py --output-dir /app/backups
```

## 📥 Import (Restore)

Restore MLflow data from a backup zip file.

### Usage

```bash
# Basic import (skips existing data by default)
python scripts/import_mlflow.py backups/mlflow_backup_20260204_120000.zip

# Dry run (see what would be imported without making changes)
python scripts/import_mlflow.py --dry-run backup.zip

# Overwrite existing data
python scripts/import_mlflow.py --overwrite backup.zip

# Specify MLflow tracking URI
python scripts/import_mlflow.py --tracking-uri http://localhost:5000 backup.zip
```

### Collision Handling

**Default behavior (--skip-existing, default):**
- ⊘ Skips existing experiments (by name)
- ⊘ Skips existing runs (by run_id)
- ⊘ Skips existing model versions
- ⊘ Skips existing artifact files

**With --overwrite flag:**
- ⚠️ Overwrites existing data (use with caution!)

### Docker Usage

```bash
# Import into running MLflow container
docker-compose run --rm \
    -v $(pwd)/backups:/app/backups \
    training python scripts/import_mlflow.py /app/backups/my_backup.zip

# Dry run to preview
docker-compose run --rm \
    -v $(pwd)/backups:/app/backups \
    training python scripts/import_mlflow.py --dry-run /app/backups/my_backup.zip
```

## 🔄 Common Workflows

### Backup Before Major Changes

```bash
# Create backup before experimenting
python scripts/export_mlflow.py --name before_v2_experiments

# ... do your work ...

# Restore if needed
python scripts/import_mlflow.py backups/before_v2_experiments.zip
```

### Migrate Between Environments

```bash
# On development machine:
python scripts/export_mlflow.py --name dev_models

# Transfer dev_models.zip to production

# On production machine:
python scripts/import_mlflow.py dev_models.zip
```

### Share Experiments with Team

```bash
# Export your experiments
python scripts/export_mlflow.py --name team_share_$(date +%Y%m%d)

# Share the zip file

# Teammates can import:
python scripts/import_mlflow.py team_share_20260204.zip
```

### Regular Backups

```bash
#!/bin/bash
# Create daily backup script

# Set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Create timestamped backup
python scripts/export_mlflow.py --output-dir ~/mlflow_backups

# Optional: Clean up old backups (keep last 7 days)
find ~/mlflow_backups -name "mlflow_backup_*.zip" -mtime +7 -delete
```

## 📋 Examples

### Export

```bash
$ python scripts/export_mlflow.py
======================================================================
🔄 MLflow Data Export
======================================================================

📦 Exporting experiments...
  ✓ Experiment: yolov11-finger-counting (ID: 1)
  ✓ Experiment: Default (ID: 0)

🏃 Exporting runs...
  ✓ Run: yolo11n_20260204_120110
  ✓ Run: yolo11s_20260203_153000

  Total runs exported: 2

🎯 Exporting registered models...
  ✓ Model: yolo11-finger-counter (2 versions)

📦 Creating backup archive: mlflow_backup_20260204_153045.zip

======================================================================
✅ Export Complete!
======================================================================
📁 Backup file: backups/mlflow_backup_20260204_153045.zip
📊 Size: 234.56 MB
📦 Experiments: 2
🏃 Runs: 2
🎯 Models: 1
```

### Import (Dry Run)

```bash
$ python scripts/import_mlflow.py --dry-run backup.zip
======================================================================
🔄 MLflow Data Import
🔍 DRY RUN MODE - No changes will be made
======================================================================

📦 Extracting backup: mlflow_backup_20260204_153045.zip
📅 Backup date: 2026-02-04T15:30:45
📊 Contains: 2 experiments, 2 runs, 1 models

🔍 Analyzing backup contents...

📦 Importing experiments...
  ⊘ Skipped (exists): Default
  ✓ Would create: yolov11-finger-counting

🏃 Importing runs...
  ✓ Would create: yolo11n_20260204_120110 (45 artifacts)
  ⊘ Skipped (exists): yolo11s_20260203_153000

🎯 Importing registered models...
  ⊘ Model exists: yolo11-finger-counter
    ✓ Would create version 3

======================================================================
🔍 Dry Run Summary (No changes made)
======================================================================
📦 Experiments: 1 created, 1 skipped
🏃 Runs: 1 created, 1 skipped
🎯 Models: 0 created
📦 Model Versions: 1 created, 0 skipped
📁 Artifacts: 45 uploaded, 0 skipped
```

## 💡 Tips

1. **Regular Backups**: Set up a cron job for automated backups
2. **Version Control**: Keep backups in version-controlled storage (git-lfs, S3, etc.)
3. **Test Restores**: Periodically test your backups with `--dry-run`
4. **Disk Space**: Large experiments can create big backups - monitor disk usage
5. **Compression**: Backups are already compressed (ZIP_DEFLATED)

## 🐛 Troubleshooting

### Import Fails with "Run not found"

Model versions require their parent runs to exist. Import the backup containing those runs first.

### "Experiment already exists" error

Use `--overwrite` flag or delete the existing experiment manually.

### Out of disk space

```bash
# Check backup size before creating
du -sh backups/

# Clean up old backups
find backups/ -name "*.zip" -mtime +30 -delete
```

### Artifacts not uploading

Ensure the MLflow artifact store is accessible and has sufficient space.

## 📚 Related

- [MLflow Tracking Documentation](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- Main project README: `../Readme.md`
