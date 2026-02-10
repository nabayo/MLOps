#!/bin/bash
# Import MLflow data from a backup zip file using Docker

set -e

if [ -z "$1" ]; then
    echo "❌ Error: Backup file not specified"
    echo ""
    echo "Usage:"
    echo "  ./scripts/docker_import.sh <backup_file.zip> [--dry-run] [--overwrite]"
    echo ""
    echo "Examples:"
    echo "  ./scripts/docker_import.sh my_backup.zip"
    echo "  ./scripts/docker_import.sh my_backup.zip --dry-run"
    echo "  ./scripts/docker_import.sh my_backup.zip --overwrite"
    echo ""
    exit 1
fi

BACKUP_FILE="$1"
shift  # Remove first argument

# Check if file exists
if [ ! -f "backups/$BACKUP_FILE" ]; then
    echo "❌ Error: Backup file not found: backups/$BACKUP_FILE"
    echo ""
    echo "Available backups:"
    ls -1 backups/*.zip 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "🔄 Importing MLflow data from: $BACKUP_FILE"
echo ""

# Run import with remaining arguments
export BACKUP_FILE="$BACKUP_FILE"

# Use docker compose (modern) with BuildKit enabled
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose run --rm \
    -e BACKUP_FILE="$BACKUP_FILE" \
    import python scripts/import_mlflow.py /app/backups/"$BACKUP_FILE" "$@"

echo ""
echo "✅ Import complete!"
